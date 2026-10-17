"""Integration tests for the browser tools.

These tests need a real Chromium and a running HTTP server. They exercise
the sync helpers on `BrowserSession` directly (the `@tool` wrappers are
thin shims and are covered by the safety tests + manual smoke).

The full LLM-driven voice loop is tested manually via the LiveKit CLI.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import requires_chromium, sandbox_site  # noqa: F401


# ---------------------------------------------------------------------------
# Load browser modules without importing langchain_agent.__init__
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def browser_modules():
    base = ROOT / "langchain_agent" / "browser"
    safety = _load("browser_safety", base / "safety.py")
    tools_helpers = _load("browser_tools_helpers", base / "tools_helpers.py")
    snapshot = _load("browser_snapshot", base / "snapshot.py")
    session_mod = _load("browser_session", base / "session.py")
    return {
        "safety": safety,
        "tools_helpers": tools_helpers,
        "snapshot": snapshot,
        "session": session_mod,
    }


# ---------------------------------------------------------------------------
# A small fake settings object — only the attrs BrowserSession reads.
# ---------------------------------------------------------------------------


@dataclass
class FakeSettings:
    browser_headless: bool = True
    browser_max_contexts: int = 4
    browser_idle_timeout_s: int = 300
    browser_default_timeout_ms: int = 5000
    browser_action_timeout_ms: int = 3000
    browser_viewport_w: int = 1280
    browser_viewport_h: int = 720
    browser_locale: str = "en-US"
    browser_timezone: str = "UTC"
    browser_user_agent: str = ""
    browser_proxy: str = ""
    browser_download_dir: str = ""  # set per-test in tmp_path


# ---------------------------------------------------------------------------
# Session fixture: a BrowserSession that talks to a real Chromium.
# ---------------------------------------------------------------------------


@pytest.fixture
async def browser_session(browser_modules, tmp_path, sandbox_site):  # noqa: F811
    settings = FakeSettings(browser_download_dir=str(tmp_path / "dl"))
    os.makedirs(settings.browser_download_dir, exist_ok=True)
    session = browser_modules["session"].BrowserSession(settings)
    await session.start()
    try:
        yield session, sandbox_site, settings, browser_modules
    finally:
        await session.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_page(session, room_id: str):
    """Create a context + page, return the page (sync — runs in a thread)."""
    import asyncio
    ctx = asyncio.run(session.context_for(room_id))
    page = ctx.pages[0]
    return page, ctx


def _on_worker(session, fn):
    """Run a Playwright call on the session's dedicated worker thread.

    Playwright's sync objects are thread-affine: inspecting a page from the
    test thread raises `greenlet.error: Cannot switch to a different thread`.
    """
    return session._worker.call(fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_chromium
def test_status_reports_pool(browser_session):
    session, *_ = browser_session
    page, _ = _new_page(session, "roomA")
    out = session.sync_status("roomA")
    assert "[Action: status]" in out
    assert "[Room: roomA]" in out
    assert "[Contexts in pool: 1" in out
    _on_worker(session, page.close)


@requires_chromium
def test_navigate(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomN")
    out = session.sync_navigate("roomN", site["index_url"], "load", 5000)
    assert "[Action: navigate" in out
    assert "Sandbox Index" in out
    assert site["index_url"] in out


@requires_chromium
def test_snapshot_returns_accessibility_tree(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomS")
    session.sync_navigate("roomS", site["index_url"], "load", 5000)
    out = session.sync_snapshot("roomS", 4000)
    assert "[Action: snapshot]" in out
    assert "Sandbox Site" in out or "heading" in out.lower() or "link" in out.lower()


@requires_chromium
def test_click_by_css(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomC")
    session.sync_navigate("roomC", site["index_url"], "load", 5000)
    out = session.sync_click("roomC", "#link-form", 3000)
    assert "[Action: click" in out
    assert "form.html" in out


@requires_chromium
def test_click_by_text(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomCt")
    session.sync_navigate("roomCt", site["index_url"], "load", 5000)
    # Visible text fragment match
    out = session.sync_click("roomCt", "Form", 3000)
    assert "form.html" in out


@requires_chromium
def test_type_and_press_enter(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomT")
    session.sync_navigate("roomT", site["form_url"], "load", 5000)
    out = session.sync_type("roomT", "#name", "Rajesh", press_enter=False, timeout_ms=3000)
    assert "[Action: type" in out
    # Verify the input has the value
    val = _on_worker(session, lambda: page.locator("#name").input_value())
    assert val == "Rajesh"


@requires_chromium
def test_select_option(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomSel")
    session.sync_navigate("roomSel", site["form_url"], "load", 5000)
    out = session.sync_select("roomSel", "#plan", "premium", 3000)
    assert "[Action: select" in out
    val = _on_worker(session, lambda: page.locator("#plan").input_value())
    assert val == "premium"


@requires_chromium
def test_evaluate(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomE")
    session.sync_navigate("roomE", site["index_url"], "load", 5000)
    out = session.sync_evaluate("roomE", "1 + 41")
    assert "[Action: evaluate]" in out
    assert "42" in out


@requires_chromium
def test_get_text_and_attribute(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomG")
    session.sync_navigate("roomG", site["index_url"], "load", 5000)
    text = session.sync_get_text("roomG", "h1", 200)
    assert "Sandbox Site" in text
    attr = session.sync_get_attribute("roomG", "#link-form", "href")
    assert "form.html" in attr


@requires_chromium
def test_open_and_switch_and_close_tab(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomT2")
    session.sync_navigate("roomT2", site["index_url"], "load", 5000)
    out = session.sync_open_tab("roomT2", site["form_url"])
    assert "[Action: open_tab" in out
    listing = session.sync_list_tabs("roomT2")
    # Tabs are listed with their zero-based index; the new tab is the active one.
    assert "[1]" in listing
    assert "form.html" in listing
    switch = session.sync_switch_tab("roomT2", 0)
    assert "[Action: switch_tab" in switch
    close = session.sync_close_page("roomT2")
    assert "[Action: close_page]" in close


@requires_chromium
def test_storage_local_set_get_clear(browser_session):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomSt")
    session.sync_navigate("roomSt", site["index_url"], "load", 5000)
    session.sync_storage_set("roomSt", "local", "token", "abc123")
    out = session.sync_storage_get("roomSt", "local", "token")
    assert "abc123" in out
    clear = session.sync_storage_clear("roomSt", "local")
    assert "[Result: ok]" in clear
    out2 = session.sync_storage_get("roomSt", "local", "token")
    assert "null" in out2.lower() or out2.endswith("null") or '"token"' in out2


@requires_chromium
def test_screenshot(browser_session, tmp_path):
    session, site, *_ = browser_session
    page, _ = _new_page(session, "roomSh")
    session.sync_navigate("roomSh", site["index_url"], "load", 5000)
    out = session.sync_screenshot("roomSh", None, full_page=False)
    assert "[Action: screenshot]" in out
    assert "Path:" in out
    # Pick the saved path out of the output and assert it exists.
    path_line = [l for l in out.splitlines() if l.startswith("[Path:")][0]
    path = path_line.split(":", 1)[1].strip().rstrip("]")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


@requires_chromium
def test_per_room_isolation(browser_session):
    """Cookies / storage set in room A must not be visible in room B."""
    session, site, *_ = browser_session
    page_a, _ = _new_page(session, "roomA")
    page_b, _ = _new_page(session, "roomB")
    session.sync_navigate("roomA", site["index_url"], "load", 5000)
    session.sync_navigate("roomB", site["index_url"], "load", 5000)
    session.sync_storage_set("roomA", "local", "secret", "A_token")

    out_b = session.sync_storage_get("roomB", "local", "secret")
    # roomB should not see roomA's localStorage
    assert "A_token" not in out_b


@requires_chromium
def test_lru_eviction(browser_session):
    """When max_contexts is reached, opening a new room evicts the oldest."""
    session, site, settings, _ = browser_session
    # max_contexts from FakeSettings is 4
    rooms = [f"r{i}" for i in range(5)]
    pages = {}
    for r in rooms:
        pages[r], _ = _new_page(session, r)
    status = session.sync_status(rooms[-1])
    assert "[Contexts in pool: 4" in status  # capped at 4


# ---------------------------------------------------------------------------
# URL normalization (regression: double-scheme navigation bug)
# ---------------------------------------------------------------------------


def test_normalize_url_keeps_existing_scheme(browser_modules):
    """A URL that already has a scheme must be passed through unchanged.

    The old implementation split on '/' before testing for '://', so the
    '//' in 'https://example.com' split the scheme away and every full URL
    became 'https://https://example.com' → ERR_NAME_NOT_RESOLVED, and the
    voice agent told customers it could not use a browser.
    """
    normalize_url = browser_modules["tools_helpers"].normalize_url
    assert normalize_url("https://www.example.com") == "https://www.example.com"
    assert normalize_url("http://a.b/c") == "http://a.b/c"
    assert normalize_url("  https://youtube.com/watch  ") == "https://youtube.com/watch"


def test_normalize_url_adds_https_to_bare_domains(browser_modules):
    normalize_url = browser_modules["tools_helpers"].normalize_url
    assert normalize_url("youtube.com") == "https://youtube.com"
    assert normalize_url("www.google.com") == "https://www.google.com"
    assert normalize_url("") == ""


@requires_chromium
def test_sync_helpers_create_context_on_demand(browser_session):
    """A tool call for a room without a warmed context must create one
    instead of failing with 'no browser context for room ...'."""
    session, site, *_ = browser_session
    out = session.sync_navigate("roomLazy", site["index_url"], "load", 5000)
    assert "[Action: navigate" in out
    assert "Sandbox Index" in out
    tabs = session.sync_list_tabs("roomLazy")
    assert "Sandbox Index" in tabs
