"""DOM-to-text summarizer for the browser tools.

The voice agent's LLM is the consumer of these strings. They are returned
as ToolMessage content, summarized aloud by the LLM (SpeechOnlyGraph in
voice_agent/main.py filters raw tool output from TTS).

The accessibility tree is preferred over inner_text because it carries
roles + names, which gives the LLM a much stronger grounding signal when
deciding which element to click next.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def page_snapshot(page, max_chars: int = 2000) -> str:
    """Return a spoken-friendly summary of the current page.

    Format:
        [Action: snapshot]
        [Title: ...]
        [URL: ...]
        [Accessibility tree (truncated to NNNN chars):]
        <tree>

    Falls back to `page.inner_text("body")` when the accessibility snapshot
    is empty (e.g., very simple pages where Playwright returns no AX nodes).
    """
    title = _safe_title(page)
    url = _safe_url(page)
    tree_text = _accessibility_snapshot(page)
    if not tree_text:
        tree_text = _inner_text(page, selector=None, max_chars=max_chars)

    return _format(
        action="snapshot",
        title=title,
        url=url,
        body=tree_text,
        max_chars=max_chars,
    )


def visible_text(page, selector: str | None = None, max_chars: int = 2000) -> str:
    """inner_text of a specific locator (or the whole body when no selector)."""
    title = _safe_title(page)
    url = _safe_url(page)
    body = _inner_text(page, selector=selector, max_chars=max_chars)

    return _format(
        action=f"get_text(selector={selector!r})" if selector else "get_text(body)",
        title=title,
        url=url,
        body=body,
        max_chars=max_chars,
    )


def locator_text(page, selector_or_label: str, max_chars: int = 2000) -> str:
    """Resolve a ref to a locator and return its inner_text."""
    from langchain_agent.browser.tools_helpers import resolve_locator  # local to avoid cycle

    try:
        locator = resolve_locator(page, selector_or_label)
        text = locator.inner_text(timeout=2000)
    except Exception as exc:
        return f"Error: could not read text from {selector_or_label!r}: {exc}"

    title = _safe_title(page)
    url = _safe_url(page)
    body = _truncate(text or "", max_chars)
    return _format(
        action=f"get_text(ref={selector_or_label!r})",
        title=title,
        url=url,
        body=body,
        max_chars=max_chars,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _safe_title(page) -> str:
    try:
        return page.title() or ""
    except Exception:
        return ""


def _safe_url(page) -> str:
    try:
        return page.url or ""
    except Exception:
        return ""


def _accessibility_snapshot(page) -> str:
    """Render the accessibility tree as a flat, indented string.

    Returns "" when the tree is unavailable or empty.
    """
    try:
        snap = page.accessibility.snapshot()
    except Exception:
        return ""
    if not snap:
        return ""
    return _render_ax_node(snap, depth=0)


def _render_ax_node(node: dict[str, Any], depth: int) -> str:
    """Render a Playwright accessibility-snapshot node tree as a flat string.

    Each line is "<indent><role>: <name>". Disabled / hidden nodes are
    omitted to keep the output LLM-friendly.
    """
    lines: list[str] = []

    role = node.get("role") or ""
    name = node.get("name") or ""
    value = node.get("value")
    description = node.get("description")

    # Skip pure structural nodes with no name (e.g., "generic" wrappers).
    if role or name:
        line = f"{'  ' * depth}{role}" if role else f"{'  ' * depth}{name}"
        if name and role:
            line += f": {name}"
        if value and isinstance(value, str) and value != name:
            line += f"  [value={value!r}]"
        if description:
            line += f"  ({description})"
        lines.append(line)

    for child in node.get("children", []) or []:
        if not isinstance(child, dict):
            continue
        sub = _render_ax_node(child, depth + 1)
        if sub:
            lines.append(sub)

    return "\n".join(lines)


def _inner_text(page, selector: str | None, max_chars: int) -> str:
    """Return inner_text of the body or a specific selector, truncated."""
    try:
        if selector:
            text = page.locator(selector).first.inner_text(timeout=2000)
        else:
            text = page.inner_text("body", timeout=2000)
    except Exception as exc:
        return f"(could not read text: {exc})"
    return _truncate(text or "", max_chars)


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated, full length {len(text)} chars]"


def _format(action: str, title: str, url: str, body: str, max_chars: int) -> str:
    """Render the standard tool-output block."""
    parts = [
        f"[Action: {action}]",
    ]
    if title:
        parts.append(f"[Title: {title}]")
    if url:
        parts.append(f"[URL: {url}]")
    if body:
        parts.append(f"[Content (truncated to {max_chars} chars):]\n{body}")
    else:
        parts.append("[Content: <empty>]")
    return "\n".join(parts)


__all__ = ["page_snapshot", "visible_text", "locator_text"]
