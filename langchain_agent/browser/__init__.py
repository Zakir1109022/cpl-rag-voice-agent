"""Browser subpackage — Playwright-backed LangChain tools for the voice agent.

Public surface:
    BrowserSession  — singleton, bound on first import
    BROWSER_TOOLS   — list of @tool functions to register with the agent
    set_current_room / get_current_room — bound by voice_agent/main.py
"""

from __future__ import annotations

from config import settings as _settings

from langchain_agent.browser.session import BrowserSession

# Construct a session eagerly but launch Chromium lazily — the first
# `context_for()` call will trigger `start()`. This keeps `import
# langchain_agent.tools` cheap and side-effect-free when BROWSER_ENABLED=0.
_session = BrowserSession(_settings)

from langchain_agent.browser import tools as _tools  # noqa: E402  (after _session)

_tools.bind_session(_session)


# Re-export public symbols.
session = _session
BROWSER_TOOLS = _tools.BROWSER_TOOLS
set_current_room = _tools.set_current_room
get_current_room = _tools.get_current_room

__all__ = [
    "BrowserSession",
    "session",
    "BROWSER_TOOLS",
    "set_current_room",
    "get_current_room",
]
