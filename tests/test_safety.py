"""Unit tests for the safety layer. Pure logic — no Playwright required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_safety():
    """Load safety.py directly so we don't pull in langchain_agent.__init__."""
    spec = importlib.util.spec_from_file_location(
        "browser_safety", Path(__file__).resolve().parent.parent / "langchain_agent" / "browser" / "safety.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("browser_safety", mod)
    spec.loader.exec_module(mod)
    return mod


safety = _load_safety()
SafetyGuard = safety.SafetyGuard
SafetyPolicy = safety.SafetyPolicy


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------


PRIVATE_HOSTS = [
    "http://10.0.0.1",
    "http://172.16.5.5",
    "http://192.168.1.1",
    "http://127.0.0.1",
    "http://169.254.169.254",  # AWS metadata
    "http://0.0.0.0",
    "http://[::1]",
    "http://[fc00::1]",
    "http://[fe80::1]",
]


@pytest.mark.parametrize("url", PRIVATE_HOSTS)
def test_ssrf_block_private_ips(url):
    g = SafetyGuard(policy=SafetyPolicy(allow_private=False))
    err = g.check_url(url)
    assert err is not None
    assert "private" in err.lower() or "loopback" in err.lower()


def test_ssrf_allows_public_with_empty_allowlist():
    g = SafetyGuard(policy=SafetyPolicy(allowed_domains_csv="", allow_private=False))
    # example.com should pass when allowlist is empty (public, non-private).
    assert g.check_url("https://example.com/") is None


def test_ssrf_allows_private_when_allow_private_true():
    g = SafetyGuard(policy=SafetyPolicy(allow_private=True))
    for url in ("http://10.0.0.1", "http://127.0.0.1", "http://[::1]"):
        assert g.check_url(url) is None


def test_allowlist_blocks_non_matching():
    g = SafetyGuard(policy=SafetyPolicy(allowed_domains_csv="*.cplphones.com,example.com"))
    assert g.check_url("https://www.cplphones.com/plans") is None
    assert g.check_url("https://example.com/") is None
    assert g.check_url("https://evil.com/") is not None
    assert g.check_url("https://www.cplphones.com.evil.com/") is not None


def test_rejects_non_http_schemes():
    g = SafetyGuard(policy=SafetyPolicy())
    for url in ("ftp://example.com/", "file:///etc/passwd", "javascript:alert(1)", "data:text/html,foo"):
        err = g.check_url(url)
        assert err is not None, url
        assert "blocked" in err.lower()


def test_rejects_empty_or_invalid_url():
    g = SafetyGuard(policy=SafetyPolicy())
    for url in ("", None, "not a url"):
        # "not a url" parses but has no scheme/host — we expect a block.
        err = g.check_url(url)  # type: ignore[arg-type]
        assert err is not None


# ---------------------------------------------------------------------------
# Eval gate
# ---------------------------------------------------------------------------


def test_eval_default_allows_simple_expression():
    g = SafetyGuard(policy=SafetyPolicy(allow_eval=True))
    assert g.check_eval("1 + 1") is None
    assert g.check_eval("document.title") is None


def test_eval_gate_off():
    g = SafetyGuard(policy=SafetyPolicy(allow_eval=False))
    err = g.check_eval("1 + 1")
    assert err is not None
    assert "disabled" in err.lower()


def test_eval_blocklist():
    g = SafetyGuard(policy=SafetyPolicy(allow_eval=True))
    for expr in ("import os", "document.cookie", "<script>alert(1)</script>"):
        err = g.check_eval(expr)
        assert err is not None, expr
        assert "blocked" in err.lower()


def test_eval_length_cap():
    g = SafetyGuard(policy=SafetyPolicy(allow_eval=True, eval_max_length=100))
    assert g.check_eval("x" * 100) is None
    assert g.check_eval("x" * 101) is not None


# ---------------------------------------------------------------------------
# Download path sandbox
# ---------------------------------------------------------------------------


def test_download_path_within_sandbox():
    g = SafetyGuard(policy=SafetyPolicy(download_dir="/tmp/browser_downloads"))
    assert g.check_download_path("/tmp/browser_downloads/foo.pdf") is None
    assert g.check_download_path("/tmp/browser_downloads/sub/bar.pdf") is None


def test_download_path_rejects_traversal():
    g = SafetyGuard(policy=SafetyPolicy(download_dir="/tmp/browser_downloads"))
    err = g.check_download_path("/tmp/browser_downloads/../etc/passwd")
    assert err is not None
    assert "traversal" in err.lower() or "blocked" in err.lower()


def test_download_path_rejects_escape():
    g = SafetyGuard(policy=SafetyPolicy(download_dir="/tmp/browser_downloads"))
    err = g.check_download_path("/etc/passwd")
    assert err is not None
    assert "blocked" in err.lower()


def test_download_path_none_is_allowed():
    """`None` means 'pick a default in the sandbox'; that's always OK."""
    g = SafetyGuard(policy=SafetyPolicy())
    assert g.check_download_path(None) is None


# ---------------------------------------------------------------------------
# Per-turn action cap
# ---------------------------------------------------------------------------


def test_action_cap_trips_at_limit():
    g = SafetyGuard(policy=SafetyPolicy(max_actions_per_turn=3))
    assert g.can_act("roomA") is None
    g.note_action("roomA")
    assert g.can_act("roomA") is None
    g.note_action("roomA")
    g.note_action("roomA")
    err = g.can_act("roomA")
    assert err is not None
    assert "cap" in err.lower()


def test_action_reset_clears_counter():
    g = SafetyGuard(policy=SafetyPolicy(max_actions_per_turn=2))
    g.note_action("roomA")
    g.note_action("roomA")
    assert g.can_act("roomA") is not None
    g.reset_actions("roomA")
    assert g.can_act("roomA") is None


def test_action_cap_is_per_room():
    g = SafetyGuard(policy=SafetyPolicy(max_actions_per_turn=1))
    g.note_action("roomA")
    assert g.can_act("roomA") is not None
    assert g.can_act("roomB") is None  # different room, separate counter


def test_action_cap_requires_room_id():
    g = SafetyGuard(policy=SafetyPolicy())
    err = g.can_act("")
    assert err is not None
