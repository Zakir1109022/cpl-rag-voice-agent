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

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def normalize_url(value: str) -> str:
    """Add https:// when the caller passes a bare domain like 'youtube.com'.

    Leaves alone anything that already carries a scheme (http://, https://,
    ftp://, ...) — the safety layer rejects non-http(s) schemes with a spoken
    error. The scheme check must look for '://', not just split on '/': the
    '//' of 'https://example.com' is itself a '/' separator, so a naive
    `split('/', 1)[0]` test sees only 'https:' and would prepend a second
    'https://', producing 'https://https://example.com' (ERR_NAME_NOT_RESOLVED).
    """
    v = (value or "").strip()
    if not v:
        return v
    if _SCHEME_RE.match(v):
        return v  # already has a scheme
    return f"https://{v}"


_ROLE_PREFIX = "role:"
_TEXT_FALLBACK_KEYWORDS = (
    "button", "link", "menuitem", "tab", "checkbox", "radio",
    "textbox", "combobox", "option", "switch", "row", "columnheader",
)

# Bare tag names ("h1", "div", "input") are valid CSS selectors. Compared
# case-sensitively: "h1" is CSS, "Form" is still treated as visible text.
_HTML_TAGS = frozenset(
    {
        "a", "abbr", "article", "aside", "b", "body", "br", "button", "canvas",
        "code", "dd", "details", "div", "dl", "dt", "em", "fieldset", "figcaption",
        "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head",
        "header", "hr", "html", "i", "iframe", "img", "input", "label", "li",
        "main", "nav", "ol", "option", "p", "pre", "section", "select", "small",
        "span", "strong", "summary", "table", "tbody", "td", "textarea", "tfoot",
        "th", "thead", "tr", "ul", "video",
    }
)


def _looks_like_css(ref: str) -> bool:
    """Heuristic: does `ref` look like a CSS selector?"""
    if not ref:
        return False
    if ref in _HTML_TAGS:
        return True
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


__all__ = ["resolve_locator", "normalize_url"]
