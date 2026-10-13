"""BrowserSession — one Chromium Browser per worker, fresh BrowserContext per room.

This module owns the Playwright lifecycle. Playwright's *sync* API is
thread-affine and must never be entered from a thread that has a running
asyncio loop (it raises "It looks like you are using Playwright Sync API
inside the asyncio loop"). To satisfy both constraints every blocking
Playwright call is dispatched to a single dedicated worker thread
(`_BrowserWorker`); the LiveKit event loop and the tool wrapper threads
only ever submit jobs and await results. The worker runs jobs serially —
one at a time — because Playwright sync objects cannot be touched from
two threads simultaneously.

Lifecycle:
    start()                     — launch Chromium (idempotent, lazy)
    stop()                      — close all contexts + browser
    context_for(room_id)        — get-or-create a per-room context
    drop_context(room_id)       — close + remove a room's context

All other methods are sync_* helpers invoked from the tools layer. They
always return a `str` the LLM can read aloud — never raise.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

# Playwright sync API. We import lazily inside start() so the rest of this
# module is importable in environments where Playwright's browsers are not
# yet installed (e.g., during `uv sync` in a slim CI image).
playwright_sync_api = None  # set on first start()


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _RoomContext:
    """A Playwright BrowserContext plus metadata for one LiveKit room."""

    room_id: str
    context: Any  # playwright.sync_api.BrowserContext
    pages: list[Any] = field(default_factory=list)  # type: ignore[type-arg]
    current_index: int = 0
    last_used_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Settings snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrowserLaunchSettings:
    """The subset of `Settings` that BrowserSession.start() reads."""

    headless: bool = True
    max_contexts: int = 8
    idle_timeout_s: int = 300
    default_timeout_ms: int = 15000
    action_timeout_ms: int = 5000
    viewport_w: int = 1280
    viewport_h: int = 720
    locale: str = "en-IN"
    timezone: str = "Asia/Kolkata"
    user_agent: str = ""
    proxy: str = ""
    download_dir: str = "/tmp/browser_downloads"


# ---------------------------------------------------------------------------
# Playwright worker thread
# ---------------------------------------------------------------------------


class _BrowserWorker:
    """A single thread that owns the Playwright sync API.

    Playwright's sync objects are bound to the thread that entered
    `sync_playwright()`, and greenlets cannot switch across threads, so
    every blocking call must be made from exactly one thread. The worker
    thread also never runs an asyncio loop, satisfying sync API's "not
    inside a running loop" requirement.

    `call()` is the only public entry point: it runs `fn(*args)` on the
    worker thread and blocks the *calling* thread until it completes,
    re-raising any exception it raised. Jobs are processed serially.
    """

    def __init__(self) -> None:
        self._jobs: "queue.Queue[tuple]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_started()
        response: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self._jobs.put((fn, args, kwargs, response))
        ok, value = response.get()
        if not ok:
            raise value
        return value

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="playwright-worker",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            fn, args, kwargs, response = self._jobs.get()
            if fn is None:
                response.put((True, None))
                return
            try:
                response.put((True, fn(*args, **kwargs)))
            except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
                response.put((False, exc))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class BrowserSession:
    """One Chromium Browser per worker process; one BrowserContext per room."""

    def __init__(self, settings: Any) -> None:
        self._settings = _coerce_settings(settings)
        self._browser: Any | None = None
        self._pw_ctx: Any | None = None
        self._contexts: OrderedDict[str, _RoomContext] = OrderedDict()
        self._started = False
        self._last_error: str | None = None
        # Every Playwright call and every mutation of self._contexts goes
        # through this single thread, so no separate lock is needed — the
        # worker job queue is the mutex.
        self._worker = _BrowserWorker()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch Chromium. Idempotent and safe to call from multiple coroutines."""
        await asyncio.to_thread(self._worker.call, self._start_worker)

    def _start_worker(self) -> None:
        """Launch Chromium on the worker thread. Idempotent."""
        if self._started and self._browser is not None:
            return

        global playwright_sync_api
        if playwright_sync_api is None:
            from playwright import sync_api as _sync_api  # type: ignore

            playwright_sync_api = _sync_api

        os.makedirs(self._settings.download_dir, exist_ok=True)

        launch_kwargs: dict[str, Any] = {
            "headless": self._settings.headless,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        }
        if self._settings.proxy:
            launch_kwargs["proxy"] = {"server": self._settings.proxy}

        try:
            # Enter the Playwright driver at most once per worker thread.
            # `sync_playwright().start()` binds a loop to this thread, so a
            # second call from the same thread raises "Playwright Sync API
            # inside the asyncio loop". We cache the driver and only
            # (re)launch the browser on retries — important after a
            # transient launch failure (e.g. the Chromium binary was not
            # installed yet): the retry reuses the driver instead of
            # re-entering the sync API.
            if self._pw_ctx is None:
                self._pw_ctx = playwright_sync_api.sync_playwright().start()
            browser = self._pw_ctx.chromium.launch(**launch_kwargs)
        except Exception as exc:
            self._last_error = f"failed to launch browser: {exc}"
            print(f"[BROWSER] {self._last_error}")
            # Leave _started False so callers can retry. Keep the driver so
            # the retry reuses it rather than starting a second one.
            self._browser = None
            return

        self._browser = browser
        self._started = True
        self._last_error = None
        print(f"[BROWSER] started (headless={self._settings.headless})")

    async def stop(self) -> None:
        """Close all room contexts and the browser. Safe to call multiple times."""
        await asyncio.to_thread(self._worker.call, self._teardown_worker)

    def _teardown_worker(self) -> None:
        """Close everything on the worker thread. Safe to call multiple times."""
        for room_id in list(self._contexts.keys()):
            self._close_context_locked(room_id)
        self._contexts.clear()

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as exc:
                print(f"[BROWSER] error closing browser: {exc}")
            self._browser = None

        if self._pw_ctx is not None:
            try:
                self._pw_ctx.stop()
            except Exception as exc:
                print(f"[BROWSER] error stopping playwright: {exc}")
            self._pw_ctx = None

        self._started = False
        print("[BROWSER] stopped")

    async def context_for(self, room_id: str) -> Any:
        """Get-or-create the BrowserContext for a LiveKit room.

        Evicts the oldest room's context if the pool is full.
        """
        if not room_id:
            raise ValueError("room_id is required")
        return await asyncio.to_thread(
            self._worker.call, self._context_for_worker, room_id
        )

    def _context_for_worker(self, room_id: str) -> Any:
        """Get-or-create the room context. Runs on the Playwright worker thread."""
        self._start_worker()
        if not self._started or self._browser is None:
            raise RuntimeError(
                f"browser failed to start: {self._last_error or 'unknown error'}"
            )

        existing = self._contexts.get(room_id)
        if existing is not None:
            existing.last_used_at = time.monotonic()
            self._contexts.move_to_end(room_id)
            return existing.context

        # Evict oldest if at capacity.
        while len(self._contexts) >= self._settings.max_contexts:
            oldest_id, _ = next(iter(self._contexts.items()))
            self._close_context_locked(oldest_id)

        ctx = self._new_context_locked()
        page = ctx.new_page()
        room_ctx = _RoomContext(
            room_id=room_id,
            context=ctx,
            pages=[page],
            current_index=0,
            last_used_at=time.monotonic(),
        )
        self._contexts[room_id] = room_ctx
        print(f"[BROWSER] new context for room={room_id} (pool={len(self._contexts)})")
        return ctx

    async def drop_context(self, room_id: str) -> None:
        await asyncio.to_thread(
            self._worker.call, self._close_context_locked, room_id
        )

    async def status(self) -> dict[str, Any]:
        def _snapshot() -> dict[str, Any]:
            return {
                "started": self._started,
                "contexts": list(self._contexts.keys()),
                "context_count": len(self._contexts),
                "max_contexts": self._settings.max_contexts,
                "last_error": self._last_error,
            }

        return await asyncio.to_thread(self._worker.call, _snapshot)

    # ------------------------------------------------------------------
    # Sync helpers — invoked by tools (via asyncio.to_thread) and always
    # dispatched to the Playwright worker thread.
    # ------------------------------------------------------------------

    def _require_context(self, room_id: str) -> _RoomContext:
        # Runs on the worker thread, which is the only thread that ever
        # touches the context map, so no locking is needed here.
        rc = self._contexts.get(room_id)
        if rc is None:
            raise RuntimeError(f"no browser context for room {room_id!r}")
        rc.last_used_at = time.monotonic()
        return rc

    def _current_page(self, room_id: str):
        rc = self._require_context(room_id)
        if not rc.pages:
            page = rc.context.new_page()
            rc.pages.append(page)
            rc.current_index = 0
        return rc.pages[rc.current_index]

    # ---- navigation / status -----------------------------------------

    def sync_navigate(self, room_id: str, url: str, wait_until: str, timeout_ms: int) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            timeout = timeout_ms or self._settings.default_timeout_ms
            try:
                page.goto(url, wait_until=wait_until, timeout=timeout)
            except Exception as exc:
                return f"Error: navigation to {url} failed: {exc}"
            return _summarize_page(page, action=f"navigate({url})", max_chars=1500)

        return self._worker.call(_run)

    def sync_snapshot(self, room_id: str, max_chars: int) -> str:
        from langchain_agent.browser.snapshot import page_snapshot

        def _run() -> str:
            page = self._current_page(room_id)
            return page_snapshot(page, max_chars=max_chars)

        return self._worker.call(_run)

    def sync_status(self, room_id: str) -> str:
        def _run() -> str:
            try:
                page = self._current_page(room_id)
                url = page.url
                title = page.title()
            except Exception as exc:
                url = f"<error: {exc}>"
                title = ""
            return (
                f"[Action: status]\n"
                f"[Room: {room_id}]\n"
                f"[Active URL: {url}]\n"
                f"[Title: {title}]\n"
                f"[Contexts in pool: {len(self._contexts)}/{self._settings.max_contexts}]\n"
                f"[Last browser error: {self._last_error or 'none'}]"
            )

        return self._worker.call(_run)

    # ---- interactions -------------------------------------------------

    def sync_click(self, room_id: str, ref: str, timeout_ms: int) -> str:
        from langchain_agent.browser.tools_helpers import resolve_locator

        def _run() -> str:
            page = self._current_page(room_id)
            timeout = timeout_ms or self._settings.action_timeout_ms
            try:
                locator = resolve_locator(page, ref)
                locator.first.click(timeout=timeout)
            except Exception as exc:
                return f"Error: could not click {ref!r}: {exc}"
            return _summarize_page(page, action=f"click({ref!r})", max_chars=1500)

        return self._worker.call(_run)

    def sync_type(
        self, room_id: str, ref: str, text: str, press_enter: bool, timeout_ms: int
    ) -> str:
        from langchain_agent.browser.tools_helpers import resolve_locator

        def _run() -> str:
            page = self._current_page(room_id)
            timeout = timeout_ms or self._settings.action_timeout_ms
            try:
                locator = resolve_locator(page, ref).first
                locator.fill(text, timeout=timeout)
                if press_enter:
                    locator.press("Enter", timeout=timeout)
            except Exception as exc:
                return f"Error: could not type into {ref!r}: {exc}"
            action = f"type({ref!r}, text={text!r}, press_enter={press_enter})"
            return _summarize_page(page, action=action, max_chars=1500)

        return self._worker.call(_run)

    def sync_press_key(self, room_id: str, key: str) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            try:
                page.keyboard.press(key)
            except Exception as exc:
                return f"Error: could not press key {key!r}: {exc}"
            return _summarize_page(page, action=f"press_key({key!r})", max_chars=1500)

        return self._worker.call(_run)

    def sync_hover(self, room_id: str, ref: str, timeout_ms: int) -> str:
        from langchain_agent.browser.tools_helpers import resolve_locator

        def _run() -> str:
            page = self._current_page(room_id)
            timeout = timeout_ms or self._settings.action_timeout_ms
            try:
                resolve_locator(page, ref).first.hover(timeout=timeout)
            except Exception as exc:
                return f"Error: could not hover {ref!r}: {exc}"
            return _summarize_page(page, action=f"hover({ref!r})", max_chars=1500)

        return self._worker.call(_run)

    def sync_select(self, room_id: str, ref: str, value: str, timeout_ms: int) -> str:
        from langchain_agent.browser.tools_helpers import resolve_locator

        def _run() -> str:
            page = self._current_page(room_id)
            timeout = timeout_ms or self._settings.action_timeout_ms
            try:
                resolve_locator(page, ref).first.select_option(value, timeout=timeout)
            except Exception as exc:
                return f"Error: could not select {value!r} in {ref!r}: {exc}"
            return _summarize_page(page, action=f"select({ref!r}, value={value!r})", max_chars=1500)

        return self._worker.call(_run)

    def sync_scroll(self, room_id: str, direction: str, amount: int) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            delta = max(1, int(amount)) * (1 if direction == "down" else -1)
            try:
                page.mouse.wheel(0, delta * 400)
            except Exception as exc:
                return f"Error: scroll failed: {exc}"
            return _summarize_page(page, action=f"scroll({direction}, {amount})", max_chars=1500)

        return self._worker.call(_run)

    def sync_wait_for(self, room_id: str, ref: str, timeout_ms: int) -> str:
        from langchain_agent.browser.tools_helpers import resolve_locator

        def _run() -> str:
            page = self._current_page(room_id)
            timeout = max(100, int(timeout_ms))
            try:
                resolve_locator(page, ref).first.wait_for(state="visible", timeout=timeout)
            except Exception as exc:
                return f"Error: timed out waiting for {ref!r}: {exc}"
            return _summarize_page(page, action=f"wait_for({ref!r})", max_chars=1500)

        return self._worker.call(_run)

    def sync_get_text(self, room_id: str, ref: str, max_chars: int) -> str:
        from langchain_agent.browser.snapshot import locator_text

        def _run() -> str:
            page = self._current_page(room_id)
            return locator_text(page, ref, max_chars=max_chars)

        return self._worker.call(_run)

    def sync_get_attribute(self, room_id: str, ref: str, attribute: str) -> str:
        from langchain_agent.browser.tools_helpers import resolve_locator

        def _run() -> str:
            page = self._current_page(room_id)
            try:
                value = resolve_locator(page, ref).first.get_attribute(attribute, timeout=2000)
            except Exception as exc:
                return f"Error: could not read attribute {attribute!r} from {ref!r}: {exc}"
            if value is None:
                return f"[Action: get_attribute({ref!r}, {attribute!r})]\n[Result: <attribute not present>]"
            return f"[Action: get_attribute({ref!r}, {attribute!r})]\n[Result: {value}]"

        return self._worker.call(_run)

    def sync_evaluate(self, room_id: str, expression: str) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            try:
                result = page.evaluate(expression)
            except Exception as exc:
                return f"Error: JavaScript evaluation failed: {exc}"
            try:
                rendered = json.dumps(result, default=str)[:2000]
            except (TypeError, ValueError):
                rendered = str(result)[:2000]
            return f"[Action: evaluate]\n[Result: {rendered}]"

        return self._worker.call(_run)

    # ---- tabs ---------------------------------------------------------

    def sync_list_tabs(self, room_id: str) -> str:
        def _run() -> str:
            rc = self._require_context(room_id)
            lines = [f"[Action: list_tabs]"]
            for i, p in enumerate(rc.pages):
                marker = "*" if i == rc.current_index else " "
                try:
                    title = p.title() or ""
                    url = p.url or ""
                except Exception:
                    title, url = "<error>", "<error>"
                lines.append(f"  {marker} [{i}] {title} — {url}")
            return "\n".join(lines)

        return self._worker.call(_run)

    def sync_open_tab(self, room_id: str, url: str) -> str:
        def _run() -> str:
            rc = self._require_context(room_id)
            try:
                page = rc.context.new_page()
                if url:
                    page.goto(url, timeout=self._settings.default_timeout_ms)
                rc.pages.append(page)
                rc.current_index = len(rc.pages) - 1
            except Exception as exc:
                return f"Error: could not open tab: {exc}"
            return f"[Action: open_tab(url={url!r})]\n[Now on tab {rc.current_index}]"

        return self._worker.call(_run)

    def sync_switch_tab(self, room_id: str, index: int) -> str:
        def _run() -> str:
            rc = self._require_context(room_id)
            if not (0 <= index < len(rc.pages)):
                return f"Error: tab index {index} out of range (have {len(rc.pages)} tabs)"
            rc.current_index = index
            page = rc.pages[index]
            return _summarize_page(page, action=f"switch_tab({index})", max_chars=1000)

        return self._worker.call(_run)

    def sync_close_page(self, room_id: str) -> str:
        def _run() -> str:
            rc = self._require_context(room_id)
            if not rc.pages:
                return "[Action: close_page]\n[No pages to close]"
            page = rc.pages.pop(rc.current_index)
            try:
                page.close()
            except Exception as exc:
                return f"Error: could not close page: {exc}"
            if not rc.pages:
                new_page = rc.context.new_page()
                rc.pages.append(new_page)
                rc.current_index = 0
            else:
                rc.current_index = max(0, min(rc.current_index, len(rc.pages) - 1))
            return f"[Action: close_page]\n[Remaining tabs: {len(rc.pages)}; active={rc.current_index}]"

        return self._worker.call(_run)

    # ---- screenshot / storage / cookies -------------------------------

    def sync_screenshot(self, room_id: str, path: str | None, full_page: bool) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            if not path:
                path = os.path.join(self._settings.download_dir, f"shot_{int(time.time()*1000)}.png")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            try:
                page.screenshot(path=path, full_page=bool(full_page))
            except Exception as exc:
                return f"Error: screenshot failed: {exc}"
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            return f"[Action: screenshot]\n[Path: {path}]\n[Bytes: {size}]\n[Full page: {bool(full_page)}]"

        return self._worker.call(_run)

    def sync_storage_get(self, room_id: str, kind: str, key: str | None) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            kind = (kind or "local").lower()
            try:
                if kind == "local":
                    value = page.evaluate("k => k ? localStorage.getItem(k) : JSON.stringify(localStorage)", key)
                elif kind == "session":
                    value = page.evaluate("k => k ? sessionStorage.getItem(k) : JSON.stringify(sessionStorage)", key)
                elif kind == "cookies":
                    cookies = page.context.cookies()
                    if key:
                        value = next((c for c in cookies if c.get("name") == key), None)
                    else:
                        value = cookies
                else:
                    return f"Error: unknown storage kind {kind!r}; use local|session|cookies"
            except Exception as exc:
                return f"Error: storage_get failed: {exc}"
            try:
                rendered = json.dumps(value, default=str)[:2000]
            except (TypeError, ValueError):
                rendered = str(value)[:2000]
            return f"[Action: storage_get(kind={kind!r}, key={key!r})]\n[Result: {rendered}]"

        return self._worker.call(_run)

    def sync_storage_set(self, room_id: str, kind: str, key: str, value: str) -> str:
        def _run() -> str:
            page = self._current_page(room_id)
            kind = (kind or "local").lower()
            try:
                if kind == "local":
                    page.evaluate("(args) => localStorage.setItem(args.k, args.v)", {"k": key, "v": value})
                elif kind == "session":
                    page.evaluate("(args) => sessionStorage.setItem(args.k, args.v)", {"k": key, "v": value})
                else:
                    return f"Error: storage_set only supports local|session, not {kind!r}"
            except Exception as exc:
                return f"Error: storage_set failed: {exc}"
            return f"[Action: storage_set(kind={kind!r}, key={key!r})]\n[Result: ok]"

        return self._worker.call(_run)

    def sync_storage_clear(self, room_id: str, kind: str) -> str:
        def _run() -> str:
            rc = self._require_context(room_id)
            kind = (kind or "all").lower()
            try:
                if kind in ("local", "all"):
                    for p in rc.pages:
                        try:
                            p.evaluate("() => localStorage.clear()")
                        except Exception:
                            pass
                if kind in ("session", "all"):
                    for p in rc.pages:
                        try:
                            p.evaluate("() => sessionStorage.clear()")
                        except Exception:
                            pass
                if kind in ("cookies", "all"):
                    rc.context.clear_cookies()
            except Exception as exc:
                return f"Error: storage_clear failed: {exc}"
            return f"[Action: storage_clear(kind={kind!r})]\n[Result: ok]"

        return self._worker.call(_run)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_context_locked(self) -> Any:
        s = self._settings
        kwargs: dict[str, Any] = {
            "viewport": {"width": s.viewport_w, "height": s.viewport_h},
            "locale": s.locale,
            "timezone_id": s.timezone,
            "accept_downloads": True,
        }
        if s.user_agent:
            kwargs["user_agent"] = s.user_agent
        return self._browser.new_context(**kwargs)

    def _close_context_locked(self, room_id: str) -> None:
        rc = self._contexts.pop(room_id, None)
        if rc is None:
            return
        try:
            for p in rc.pages:
                try:
                    p.close()
                except Exception:
                    pass
            rc.context.close()
        except Exception as exc:
            print(f"[BROWSER] error closing context for {room_id}: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_settings(settings: Any) -> BrowserLaunchSettings:
    """Build a BrowserLaunchSettings from a `Settings` instance or dataclass."""
    return BrowserLaunchSettings(
        headless=bool(getattr(settings, "browser_headless", True)),
        max_contexts=int(getattr(settings, "browser_max_contexts", 8)),
        idle_timeout_s=int(getattr(settings, "browser_idle_timeout_s", 300)),
        default_timeout_ms=int(getattr(settings, "browser_default_timeout_ms", 15000)),
        action_timeout_ms=int(getattr(settings, "browser_action_timeout_ms", 5000)),
        viewport_w=int(getattr(settings, "browser_viewport_w", 1280)),
        viewport_h=int(getattr(settings, "browser_viewport_h", 720)),
        locale=str(getattr(settings, "browser_locale", "en-IN")),
        timezone=str(getattr(settings, "browser_timezone", "Asia/Kolkata")),
        user_agent=str(getattr(settings, "browser_user_agent", "") or ""),
        proxy=str(getattr(settings, "browser_proxy", "") or ""),
        download_dir=str(getattr(settings, "browser_download_dir", "/tmp/browser_downloads")),
    )


def _summarize_page(page, action: str, max_chars: int) -> str:
    """Compose a compact post-action summary: action + URL + title + short text."""
    try:
        url = page.url or ""
    except Exception:
        url = "<error>"
    try:
        title = page.title() or ""
    except Exception:
        title = ""
    try:
        text = page.inner_text("body", timeout=2000) or ""
    except Exception:
        text = ""
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[truncated, full length {len(text)} chars]"
    parts = [f"[Action: {action}]", f"[URL: {url}]"]
    if title:
        parts.append(f"[Title: {title}]")
    if text:
        parts.append(f"[Visible text (truncated to {max_chars} chars):]\n{text}")
    else:
        parts.append("[Visible text: <empty>]")
    return "\n".join(parts)


__all__ = ["BrowserSession", "BrowserLaunchSettings"]
