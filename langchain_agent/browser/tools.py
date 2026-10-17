"""LangChain @tool definitions for browser control.

Each tool is a synchronous function returning `str`. Internally they call
the async `BrowserSession` helpers via `asyncio.run`, which themselves
dispatch the blocking Playwright sync API on a worker thread (so the
LiveKit audio loop is never starved).

Tools never raise — they return a short string the LLM can speak on any
failure (matches the contract of the existing tools in
`langchain_agent/tools.py`).

The current room id is set via `set_current_room()` by
`voice_agent/main.py:entrypoint` before each user turn. When no room is
active, every tool returns a polite error string.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from langchain_core.tools import tool

from langchain_agent.browser.session import BrowserSession
from langchain_agent.browser.tools_helpers import normalize_url as _normalize_url


# ---------------------------------------------------------------------------
# Per-room context plumbing
# ---------------------------------------------------------------------------

# A module-level global protected by a lock. We can't use a ContextVar
# because the LangChain agent graph invokes tools in a different asyncio
# task (and potentially a different thread) than the one LiveKit's
# `entrypoint` ran in — ContextVars don't propagate across that boundary.
#
# One LiveKit worker process services one room at a time (LiveKit calls
# `entrypoint` per-room, and the agent's session is bound to a single
# room). So a single global is correct, and a simple `threading.Lock`
# makes the writes safe against any future thread-pool contention.
_current_room_lock = threading.Lock()
_current_room: str | None = None


def set_current_room(room_id: str | None) -> None:
    """Called by voice_agent/main.py entrypoint to bind the active room."""
    global _current_room
    with _current_room_lock:
        _current_room = room_id


def get_current_room() -> str | None:
    with _current_room_lock:
        return _current_room


# Bound by langchain_agent/browser/__init__.py at import time.
session: BrowserSession | None = None


def bind_session(s: BrowserSession) -> None:
    global session
    session = s


def _require_room() -> tuple[BrowserSession, str] | str:
    if session is None:
        return "Browser session is not initialized."
    room_id = get_current_room()
    if not room_id:
        return "No active room — browser tools require a LiveKit room."
    return session, room_id


async def _dispatch(coro_factory):
    """Run an async coroutine factory, returning its string result.

    `coro_factory` is a no-arg callable that returns a coroutine. Wrapping
    it lets us reuse the same try/except around every dispatch site.
    """
    try:
        return await coro_factory()
    except Exception as exc:
        return f"Error: browser tool failed: {exc}"


def _run(coro_factory) -> str:
    """Sync wrapper. Used by every @tool below."""
    try:
        return asyncio.run(_dispatch(coro_factory))
    except Exception as exc:
        return f"Error: browser tool failed: {exc}"


# ---------------------------------------------------------------------------
# Helpers used by individual tools
# ---------------------------------------------------------------------------


async def _guarded(room_id: str, fn, *args, **kwargs):
    """Run a SafetyGuard.can_act check before calling the sync helper."""
    from langchain_agent.browser.safety import build_guard_from_settings
    from config import settings as _settings

    guard = build_guard_from_settings(_settings)
    err = guard.can_act(room_id)
    if err:
        return err
    guard.note_action(room_id)
    # The actual sync_* call is dispatched on a worker thread so the
    # audio loop is not blocked.
    return await asyncio.to_thread(fn, *args, **kwargs)


def _url_or_error(value: str) -> str | None:
    from langchain_agent.browser.safety import build_guard_from_settings
    from config import settings as _settings

    guard = build_guard_from_settings(_settings)
    return guard.check_url(value)


def _eval_or_error(expr: str) -> str | None:
    from langchain_agent.browser.safety import build_guard_from_settings
    from config import settings as _settings

    guard = build_guard_from_settings(_settings)
    return guard.check_eval(expr)


# ---------------------------------------------------------------------------
# Tools — navigation / status
# ---------------------------------------------------------------------------


@tool
def browser_navigate(url: str, wait_until: str = "load") -> str:
    """Open a URL in a REAL headless Chromium browser. Use whenever the
    customer says 'open', 'visit', 'go to', 'browse', or 'show' a website
    (e.g. 'open youtube.com', 'go to flipkart', 'browse amazon.in').
    Also use for any request that requires interacting with a webpage
    (checking an order, recharging an account, looking something up
    interactively, filling a form, logging in).

    Pass a full URL with scheme: 'https://www.youtube.com' (not 'youtube.com').
    The tool will add 'https://' automatically if the URL has no scheme.

    Returns the page title, final URL, and a brief top-of-page summary.
    Call browser_snapshot afterwards to see more of the page.

    `wait_until` can be "load" (default), "domcontentloaded", or "networkidle"."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    if wait_until not in ("load", "domcontentloaded", "networkidle"):
        return f"Error: invalid wait_until {wait_until!r}; use load|domcontentloaded|networkidle."

    url = _normalize_url(url)

    err = _url_or_error(url)
    if err:
        return err

    async def call():
        ctx_obj = await s.context_for(room_id)
        # Touching `ctx_obj` ensures the room has a context before we run.
        _ = ctx_obj
        return await _guarded(
            room_id,
            s.sync_navigate,
            room_id, url, wait_until, 0,  # 0 = use default
        )

    return _run(call)


@tool
def browser_snapshot(max_chars: int = 2000) -> str:
    """Return an accessibility-tree snapshot of the current page. Use this
    after browser_navigate to ground your understanding of the page before
    clicking or typing. The result is truncated to `max_chars`."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    max_chars = max(100, min(int(max_chars), 4000))

    async def call():
        return await _guarded(room_id, s.sync_snapshot, room_id, max_chars)

    return _run(call)


@tool
def browser_status() -> str:
    """Report session health: active page URL, room, contexts in the pool,
    and the last browser error. Call this to confirm a clean state, or to
    debug a stuck page."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    async def call():
        return await asyncio.to_thread(s.sync_status, room_id)

    return _run(call)


# ---------------------------------------------------------------------------
# Tools — interactions
# ---------------------------------------------------------------------------


@tool
def browser_click(selector_or_label: str) -> str:
    """Click an element on the current page. `selector_or_label` accepts:
      - a CSS selector: '#submit', 'button.primary', 'a[href="/login"]'
      - an accessibility ref: 'role:button:Recharge' (role:button + name)
      - a visible text fragment: 'Recharge now'

    Returns the page URL, title, and visible text after the click."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if not selector_or_label or not selector_or_label.strip():
        return "Error: selector_or_label is required."

    async def call():
        return await _guarded(
            room_id, s.sync_click, room_id, selector_or_label, 0
        )

    return _run(call)


@tool
def browser_type(selector_or_label: str, text: str, press_enter: bool = False) -> str:
    """Fill an input field. `selector_or_label` follows the same rules as
    browser_click. Set `press_enter=True` to submit the form afterwards."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if not selector_or_label or not selector_or_label.strip():
        return "Error: selector_or_label is required."

    async def call():
        return await _guarded(
            room_id, s.sync_type, room_id, selector_or_label, text, bool(press_enter), 0
        )

    return _run(call)


@tool
def browser_press_key(key: str) -> str:
    """Press a keyboard key on the focused element. Common keys: 'Enter',
    'Escape', 'Tab', 'ArrowDown', 'PageDown'. Use after browser_type to
    submit, or to dismiss a dialog."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if not key or not key.strip():
        return "Error: key is required."

    async def call():
        return await _guarded(room_id, s.sync_press_key, room_id, key)

    return _run(call)


@tool
def browser_scroll(direction: str = "down", amount: int = 3) -> str:
    """Scroll the viewport. `direction` is 'up' or 'down'. `amount` is the
    number of viewport-heights to scroll (1 = one full screen)."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if direction not in ("up", "down"):
        return f"Error: direction must be 'up' or 'down' (got {direction!r})."
    amount = max(1, min(int(amount), 20))

    async def call():
        return await _guarded(room_id, s.sync_scroll, room_id, direction, amount)

    return _run(call)


@tool
def browser_wait_for(selector_or_label: str, timeout_ms: int = 5000) -> str:
    """Wait for an element to appear (or to disappear when prefixed with
    'not:'). `timeout_ms` defaults to 5000; cap is 30000."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if not selector_or_label or not selector_or_label.strip():
        return "Error: selector_or_label is required."
    timeout = max(100, min(int(timeout_ms), 30000))

    async def call():
        return await _guarded(
            room_id, s.sync_wait_for, room_id, selector_or_label, timeout
        )

    return _run(call)


@tool
def browser_get_text(selector_or_label: str, max_chars: int = 2000) -> str:
    """Extract visible text from a specific element (not the whole body).
    Useful after clicking something to read the result."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if not selector_or_label or not selector_or_label.strip():
        return "Error: selector_or_label is required."
    max_chars = max(100, min(int(max_chars), 4000))

    async def call():
        return await _guarded(
            room_id, s.sync_get_text, room_id, selector_or_label, max_chars
        )

    return _run(call)


@tool
def browser_get_attribute(selector_or_label: str, attribute: str) -> str:
    """Read an HTML attribute (e.g. 'href', 'value', 'aria-label',
    'data-order-id') from an element. Returns the attribute value or
    '<attribute not present>'."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if not selector_or_label or not selector_or_label.strip():
        return "Error: selector_or_label is required."
    if not attribute or not attribute.strip():
        return "Error: attribute is required."

    async def call():
        return await _guarded(
            room_id, s.sync_get_attribute, room_id, selector_or_label, attribute
        )

    return _run(call)


# ---------------------------------------------------------------------------
# Tools — tabs
# ---------------------------------------------------------------------------


@tool
def browser_list_tabs() -> str:
    """List the currently open browser tabs with their title and URL.
    The active tab is marked with '*'."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    async def call():
        return await asyncio.to_thread(s.sync_list_tabs, room_id)

    return _run(call)


@tool
def browser_open_tab(url: str = "") -> str:
    """Open a new tab in the current browser context. If `url` is provided,
    navigate to it immediately. The new tab becomes the active page.

    Pass a full URL with scheme: 'https://www.youtube.com' (not 'youtube.com').
    The tool will add 'https://' automatically if the URL has no scheme."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if url:
        url = _normalize_url(url)
        err = _url_or_error(url)
        if err:
            return err

    async def call():
        return await _guarded(room_id, s.sync_open_tab, room_id, url)

    return _run(call)


@tool
def browser_switch_tab(index: int) -> str:
    """Switch the active tab by zero-based index. Use browser_list_tabs to
    see the current tabs and their indices first."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    async def call():
        return await asyncio.to_thread(s.sync_switch_tab, room_id, int(index))

    return _run(call)


@tool
def browser_close_page() -> str:
    """Close the currently active tab. If it was the last tab, a fresh blank
    tab is opened in its place."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    async def call():
        return await asyncio.to_thread(s.sync_close_page, room_id)

    return _run(call)


# ---------------------------------------------------------------------------
# Tools — screenshot / evaluate
# ---------------------------------------------------------------------------


@tool
def browser_screenshot(path: str = "", full_page: bool = False) -> str:
    """Save a PNG screenshot of the current page. When `path` is empty the
    file is saved in the sandboxed download directory. Returns the saved
    path and file size."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    from langchain_agent.browser.safety import build_guard_from_settings
    from config import settings as _settings

    guard = build_guard_from_settings(_settings)
    err = guard.check_download_path(path or None)
    if err:
        return err

    async def call():
        return await _guarded(
            room_id, s.sync_screenshot, room_id, path or None, bool(full_page)
        )

    return _run(call)


@tool
def browser_evaluate(js_expression: str) -> str:
    """Run a JavaScript expression in the page and return its JSON result.
    Useful for reading computed state, but do NOT use for actions — prefer
    browser_click / browser_type. Disabled when BROWSER_ALLOW_EVAL=0."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx

    err = _eval_or_error(js_expression)
    if err:
        return err

    async def call():
        return await _guarded(room_id, s.sync_evaluate, room_id, js_expression)

    return _run(call)


# ---------------------------------------------------------------------------
# Tools — storage / cookies
# ---------------------------------------------------------------------------


@tool
def browser_storage_get(kind: str = "local", key: str = "") -> str:
    """Read storage: 'local' (localStorage), 'session' (sessionStorage), or
    'cookies'. When `key` is empty, returns the entire store as JSON.
    Truncated to 2000 characters."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if kind not in ("local", "session", "cookies"):
        return f"Error: kind must be local|session|cookies (got {kind!r})."

    async def call():
        return await asyncio.to_thread(
            s.sync_storage_get, room_id, kind, key or None
        )

    return _run(call)


@tool
def browser_storage_set(kind: str, key: str, value: str) -> str:
    """Write a key/value into localStorage or sessionStorage ('local'|'session').
    Cookies cannot be set with this method."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if kind not in ("local", "session"):
        return f"Error: kind must be local|session (got {kind!r})."
    if not key:
        return "Error: key is required."

    async def call():
        return await asyncio.to_thread(
            s.sync_storage_set, room_id, kind, key, value
        )

    return _run(call)


@tool
def browser_storage_clear(kind: str = "all") -> str:
    """Clear browser storage. `kind` is 'local', 'session', 'cookies', or
    'all'. Use with care — this erases customer session data and forces a
    re-login on most sites."""
    ctx = _require_room()
    if isinstance(ctx, str):
        return ctx
    s, room_id = ctx
    if kind not in ("local", "session", "cookies", "all"):
        return f"Error: kind must be local|session|cookies|all (got {kind!r})."

    async def call():
        return await asyncio.to_thread(s.sync_storage_clear, room_id, kind)

    return _run(call)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


BROWSER_TOOLS = [
    browser_navigate,
    browser_snapshot,
    browser_click,
    browser_type,
    browser_press_key,
    browser_scroll,
    browser_wait_for,
    browser_get_text,
    browser_get_attribute,
    browser_list_tabs,
    browser_open_tab,
    browser_switch_tab,
    browser_close_page,
    browser_screenshot,
    browser_evaluate,
    browser_storage_get,
    browser_storage_set,
    browser_storage_clear,
    browser_status,
]


__all__ = [
    "BROWSER_TOOLS",
    "bind_session",
    "set_current_room",
    "get_current_room",
    "session",
]