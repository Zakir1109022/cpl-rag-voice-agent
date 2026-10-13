"""Safety guard for the browser tools.

Pure logic — no Playwright imports. All checks return either:
  - None  : the action is permitted
  - str   : a short, TTS-friendly error message the LLM will read aloud

The browser tools never raise; failures are returned as strings. This module
matches that contract.
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import socket
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Iterable
from urllib.parse import urlparse

# DNS lookups during URL safety checks should not block the LiveKit audio
# loop. 2 seconds is generous on healthy infra and short enough to keep
# voice latency tolerable.
_DNS_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Configuration snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyPolicy:
    """All policy knobs the SafetyGuard needs. Built once from Settings."""

    allowed_domains_csv: str = ""
    allow_private: bool = False
    allow_eval: bool = True
    download_dir: str = "/tmp/browser_downloads"
    max_actions_per_turn: int = 25
    eval_max_length: int = 2000

    @property
    def allowed_domains(self) -> list[str]:
        return [d.strip() for d in self.allowed_domains_csv.split(",") if d.strip()]


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


@dataclass
class SafetyGuard:
    policy: SafetyPolicy
    _action_counts: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # URL validation
    # ------------------------------------------------------------------

    def check_url(self, url: str) -> str | None:
        """Validate a URL before navigation.

        Returns None if the URL is permitted, or a short error string the
        LLM can speak to the user.
        """
        if not url or not isinstance(url, str):
            return "Navigation blocked: empty or invalid URL."

        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return (
                f"Navigation blocked: only http and https URLs are allowed "
                f"(got '{parsed.scheme or '<none>'}'."
            )

        host = (parsed.hostname or "").lower()
        if not host:
            return "Navigation blocked: URL has no hostname."

        # Allowlist — only enforced when the operator configured one.
        allowed = self.policy.allowed_domains
        if allowed and not any(fnmatch(host, pat.lower()) for pat in allowed):
            return (
                f"Navigation blocked: '{host}' is not in the allowed "
                f"domains list."
            )

        # SSRF — block private/loopback/link-local IPs unless explicitly allowed.
        if not self.policy.allow_private and self._host_is_private(host):
            return (
                f"Navigation blocked: '{host}' resolves to a private or "
                f"loopback IP range. Set BROWSER_ALLOW_PRIVATE=1 to override."
            )

        return None

    def _host_is_private(self, host: str) -> bool:
        """True if `host` (a hostname or literal IP) is in a private/loopback range.

        We resolve hostnames to IPs because attackers can use DNS to point
        `evil.example.com` at `10.0.0.5`. Only the first resolved address is
        checked — sufficient for the SSRF guard; full DNS-rebinding hardening
        is out of scope for v1.
        """
        # Literal IP first (cheap path).
        try:
            ip = ipaddress.ip_address(host)
            return self._ip_is_private(ip)
        except ValueError:
            pass

        # Hostname → resolve (with a hard timeout to keep the audio loop
        # responsive against slow / hostile DNS).
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(socket.getaddrinfo, host, None)
                infos = future.result(timeout=_DNS_TIMEOUT_S)
        except (socket.gaierror, concurrent.futures.TimeoutError, OSError):
            # Can't resolve in time — fail open at the network layer. The
            # browser will error and surface that to the user.
            return False

        for info in infos:
            sockaddr = info[4]
            ip_str = sockaddr[0]
            try:
                if self._ip_is_private(ipaddress.ip_address(ip_str)):
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _ip_is_private(ip: ipaddress._BaseAddress) -> bool:
        if isinstance(ip, ipaddress.IPv4Address):
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            )
        # IPv6
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )

    # ------------------------------------------------------------------
    # JavaScript evaluation
    # ------------------------------------------------------------------

    _EVAL_BLOCKLIST = (
        "import",         # python eval injection
        "<script",        # HTML smuggling
        "document.cookie",  # trivial cookie exfil
    )

    def check_eval(self, expression: str) -> str | None:
        if not self.policy.allow_eval:
            return (
                "JavaScript evaluation is disabled. "
                "Set BROWSER_ALLOW_EVAL=1 to enable browser_evaluate."
            )
        if not expression or not isinstance(expression, str):
            return "JavaScript evaluation blocked: empty expression."
        if len(expression) > self.policy.eval_max_length:
            return (
                f"JavaScript evaluation blocked: expression exceeds "
                f"{self.policy.eval_max_length} characters."
            )
        lowered = expression.lower()
        for token in self._EVAL_BLOCKLIST:
            if token in lowered:
                return (
                    f"JavaScript evaluation blocked: expression contains "
                    f"disallowed token '{token}'."
                )
        return None

    # ------------------------------------------------------------------
    # Download path sandbox
    # ------------------------------------------------------------------

    def check_download_path(self, path: str | None) -> str | None:
        """Constrain a download destination to `policy.download_dir`.

        When `path` is None, the browser session picks a default inside the
        sandbox. When provided, the resolved path must be inside the sandbox
        and free of `..` traversal.
        """
        if path is None:
            return None

        if not isinstance(path, str) or not path.strip():
            return "Download blocked: path is empty."

        if ".." in path.split("/") or ".." in path.split("\\"):
            return "Download blocked: path traversal is not allowed."

        sandbox = self.policy.download_dir.rstrip("/").rstrip("\\")
        normalized = path.replace("\\", "/").rstrip("/")

        # Reject absolute paths outside the sandbox.
        if normalized.startswith("/") and not normalized.startswith(sandbox + "/") and normalized != sandbox:
            return (
                f"Download blocked: only files inside '{sandbox}' are allowed."
            )

        return None

    # ------------------------------------------------------------------
    # Per-turn action counter
    # ------------------------------------------------------------------

    def can_act(self, room_id: str) -> str | None:
        if not room_id:
            return "Action blocked: no active room."
        if self._action_counts.get(room_id, 0) >= self.policy.max_actions_per_turn:
            return (
                f"Action cap reached for this turn "
                f"({self.policy.max_actions_per_turn} browser actions). "
                f"Wait for the customer to speak again, or escalate to a "
                f"human agent."
            )
        return None

    def note_action(self, room_id: str) -> None:
        self._action_counts[room_id] = self._action_counts.get(room_id, 0) + 1

    def reset_actions(self, room_id: str) -> None:
        self._action_counts.pop(room_id, None)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_guard_from_settings(settings) -> SafetyGuard:
    """Build a SafetyGuard from the project's `Settings` instance."""
    return SafetyGuard(
        policy=SafetyPolicy(
            allowed_domains_csv=getattr(settings, "browser_allowed_domains", "") or "",
            allow_private=bool(getattr(settings, "browser_allow_private", False)),
            allow_eval=bool(getattr(settings, "browser_allow_eval", True)),
            download_dir=str(getattr(settings, "browser_download_dir", "/tmp/browser_downloads")),
            max_actions_per_turn=int(getattr(settings, "browser_max_actions_per_turn", 25)),
            eval_max_length=2000,
        )
    )


__all__ = ["SafetyGuard", "SafetyPolicy", "build_guard_from_settings"]
