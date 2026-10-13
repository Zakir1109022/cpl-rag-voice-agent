"""Shared helpers used by the browser tools.

Kept separate from `tools.py` to avoid the cost of importing all `@tool`
decorators when other modules (e.g. `session.py`, `snapshot.py`) only
need the helper functions.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool  # noqa: F401  (re-exported for tools.py)


# ---------------------------------------------------------------------------
# Selector resolution
# ---------------------------------------------------------------------------

_ROLE_PREFIX = "role:"
_TEXT_FALLBACK_KEYWORDS = (
    "button", "link", "menuitem", "tab", "checkbox", "radio",
    "textbox", "combobox", "option", "switch", "row", "columnheader",
)


def _looks_like_css(ref: str) -> bool:
    """Heuristic: does `ref` look like a CSS selector?"""
    if not ref:
        return False
    if ref.startswith(("#", ".", "[", "a ", "div ", "button ", "input ", "form ")):
        return True
    if re.search(r"^[a-zA-Z][\w-]*\s*[\[\.:>#]", ref):
        return True
    return False


def resolve_locator(page: Any, ref: str) -> Any:
    """Return a Playwright `Locator` for `ref`.

    Resolution order:
      1. CSS selector (when ref looks like one)
      2. Explicit `role:<role>:<name>`  e.g. `role:button:Recharge`
      3. Role-only  `role:<role>`       e.g. `role:link` (first match)
      4. Visible text (substring, case-insensitive)

    Locators are lazy — they don't fail until acted on.
    """
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("empty selector/label")

    # 1. CSS
    if _looks_like_css(ref):
        return page.locator(ref)

    # 2 / 3. role:role:name
    if ref.lower().startswith(_ROLE_PREFIX):
        rest = ref[len(_ROLE_PREFIX):]
        if ":" in rest:
            role, name = rest.split(":", 1)
            return page.get_by_role(role.strip(), name=name.strip())
        return page.get_by_role(rest.strip())

    # 4. Visible text
    return page.get_by_text(ref, exact=False)


__all__ = ["resolve_locator"]
