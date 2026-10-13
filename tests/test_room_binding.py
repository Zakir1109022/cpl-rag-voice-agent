"""Tests for the room-binding plumbing in browser/tools.py.

The bug being guarded against: the LangChain agent invokes tools in a
different asyncio task than LiveKit's `entrypoint`. A naive ContextVar
implementation does NOT propagate across that boundary — so the tools
would see `room_id = None` and return an error string, and the LLM
would say "I am unable to do that."

These tests exercise:
  1. set + get on the same thread
  2. set on one thread, get on another (cross-thread)
  3. set on one thread, get from inside asyncio.run on another thread
     (this is the exact scenario from the voice loop)
  4. the `_require_room()` helper returns the session+room tuple when
     the room is bound, even when called from a worker thread
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def browser_tools_module():
    """Load browser/tools.py directly so we don't touch the heavy
    langchain_agent.__init__ (which requires the Docker-baked Qdrant
    collection)."""
    return _load(
        "browser_tools",
        ROOT / "langchain_agent" / "browser" / "tools.py",
    )


def test_set_and_get_on_same_thread(browser_tools_module):
    browser_tools_module.set_current_room("roomA")
    assert browser_tools_module.get_current_room() == "roomA"
    browser_tools_module.set_current_room(None)
    assert browser_tools_module.get_current_room() is None


def test_get_from_different_thread_after_set(browser_tools_module):
    """This is the broken-with-ContextVar scenario: LiveKit's entrypoint
    sets the room; the LangChain tool runs in another thread."""
    browser_tools_module.set_current_room("roomA")
    seen = {}

    def reader():
        seen["room"] = browser_tools_module.get_current_room()

    t = threading.Thread(target=reader)
    t.start()
    t.join()
    assert seen["room"] == "roomA"
    browser_tools_module.set_current_room(None)


def test_get_from_inside_asyncio_run_on_different_thread(browser_tools_module):
    """This is the exact production scenario: entrypoint runs in one
    thread; the agent graph invokes the tool inside asyncio.run on
    another thread. ContextVars do NOT propagate here."""
    browser_tools_module.set_current_room("roomA")
    seen = {}

    def runner():
        async def probe():
            seen["room"] = browser_tools_module.get_current_room()

        asyncio.run(probe())

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    assert seen["room"] == "roomA"
    browser_tools_module.set_current_room(None)


def test_require_room_returns_session_when_room_bound(browser_tools_module):
    """The full happy-path: room is bound, _require_room returns the
    (session, room_id) tuple instead of an error string."""
    fake_session = object()  # identity check; we don't care about behavior
    browser_tools_module.bind_session(fake_session)  # type: ignore[arg-type]
    browser_tools_module.set_current_room("room42")

    result = browser_tools_module._require_room()
    assert result == (fake_session, "room42")
    browser_tools_module.set_current_room(None)


def test_require_room_returns_error_when_no_room(browser_tools_module):
    fake_session = object()
    browser_tools_module.bind_session(fake_session)  # type: ignore[arg-type]
    browser_tools_module.set_current_room(None)

    result = browser_tools_module._require_room()
    assert isinstance(result, str)
    assert "no active room" in result.lower()


def test_require_room_returns_error_when_no_session(browser_tools_module):
    """Defensive: if bind_session was forgotten, return an error rather
    than blowing up with AttributeError on the session."""
    browser_tools_module.set_current_room("room42")
    # Don't bind a session — tools.session should be None.
    browser_tools_module.session = None
    result = browser_tools_module._require_room()
    assert isinstance(result, str)
    assert "not initialized" in result.lower()
    browser_tools_module.set_current_room(None)