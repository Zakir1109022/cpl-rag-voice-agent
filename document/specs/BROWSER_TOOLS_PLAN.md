# Browser Tools — Playwright Integration for the Cpl Voice Agent

## Goal

Give the voice agent real browser control through a curated set of LangChain
`@tool` functions backed by Playwright / Chromium. The LLM drives actions
through tool calls; `SpeechOnlyGraph` (in `voice_agent/main.py`) strips
`ToolMessage` events from `astream`, so raw page state never reaches TTS —
only the LLM's spoken summary.

## Architecture

```
LiveKit worker process (long-lived)
    │
    │ @AgentServer(setup_fnc=...) called once per worker process
    ▼
BrowserSession
    ├── browser: chromium Browser (1x)
    ├── contexts: OrderedDict[room_id -> BrowserContext]  (LRU, max N)
    └── safety: SafetyGuard
        │
        │ per-room fresh BrowserContext
        ▼
LangChain @tool  ──►  asyncio.run  ──►  asyncio.to_thread  ──►  page.<op>
                                                              │
                                                              └─► str result
                                                                     │
                                                                     ▼
                                                          SpeechOnlyGraph (filters)
                                                                     │
                                                                     ▼
                                                            LLM summarization
                                                                     │
                                                                     ▼
                                                                 TTS
```

### Lifecycle hooks (LiveKit 1.4.5)

- **`AgentServer(setup_fnc=_setup_browser_session)`** — `_setup_browser_session`
  runs **once per worker process** during `JobProcess` setup. It registers an
  `atexit` handler that calls `BrowserSession.stop()` (closes all contexts +
  the browser) when the worker exits.
- **Per-room**: `entrypoint(ctx)` calls `await session.context_for(room_id)`
  to get-or-create a fresh `BrowserContext`, sets the thread-local room id
  for the duration of the call, and `drop_context(room_id)` in a `finally`
  block so cookies/storage cannot leak to the next customer.

### Why per-room BrowserContext

A `BrowserContext` owns cookies, localStorage, sessionStorage, IndexedDB
and the cookie jar. Sharing one across customers is a privacy/PII leak.
The corrected design: **one `Browser` per worker process, one fresh
`BrowserContext` per LiveKit room**. The browser is shared because
launching Chromium is expensive (~1–2s); contexts are cheap to create.

### Sync ↔ async bridge

Tools in `langchain_agent/tools.py` are synchronous (the existing
contract). Playwright's sync API is blocking. To avoid blocking the
LiveKit audio loop:

1. The `@tool` function calls `asyncio.run(_async_helper())`.
2. `_async_helper` calls `await asyncio.to_thread(session.sync_*, ...)`.
3. The sync `session.sync_*` calls the Playwright sync API on a worker thread.

The result is one tool call ≈ one thread-pool job ≈ one Playwright
sync call. The audio loop is never blocked for more than the duration of
a Python attribute lookup.

## Tool Inventory (18 tools)

All tools are synchronous, return `str`, never raise, and follow the
docstring-as-routing-signal pattern. The LLM uses the docstring to
decide which tool to call.

| # | Tool | Purpose |
|---|---|---|
| 1 | `browser_navigate(url, wait_until="load")` | Open URL |
| 2 | `browser_snapshot(max_chars=2000)` | Accessibility-tree snapshot |
| 3 | `browser_click(selector_or_label)` | Click by CSS, role+name, or text |
| 4 | `browser_type(selector_or_label, text, press_enter=False)` | Fill input |
| 5 | `browser_press_key(key)` | Keyboard key (Enter, Escape, Tab) |
| 6 | `browser_scroll(direction="down", amount=3)` | Scroll viewport |
| 7 | `browser_wait_for(selector_or_label, timeout_ms=5000)` | Wait for element |
| 8 | `browser_get_text(selector_or_label, max_chars=2000)` | Read text |
| 9 | `browser_get_attribute(selector_or_label, attribute)` | Read attribute |
| 10 | `browser_list_tabs()` | List open tabs |
| 11 | `browser_open_tab(url="")` | New tab |
| 12 | `browser_switch_tab(index)` | Switch active tab |
| 13 | `browser_close_page()` | Close current tab |
| 14 | `browser_screenshot(path="", full_page=False)` | Save PNG |
| 15 | `browser_evaluate(js_expression)` | Run JS (gated) |
| 16 | `browser_storage_get(kind, key="")` | Read localStorage / sessionStorage / cookies |
| 17 | `browser_storage_set(kind, key, value)` | Write storage |
| 18 | `browser_storage_clear(kind="all")` | Clear storage / cookies |
| 19 | `browser_status()` | Session health |

The `selector_or_label` argument resolves in this order:

1. **CSS selector** — `#submit`, `button.primary`, `a[href="/login"]`
2. **Accessibility ref** — `role:button:Recharge` → `page.get_by_role("button", name="Recharge")`
3. **Visible text** — `Recharge now` → `page.get_by_text("Recharge now", exact=False)`

## Safety

The `SafetyGuard` in `langchain_agent/browser/safety.py` enforces five
guardrails. Every check returns `None` (allowed) or a short string the
LLM can speak. **No exceptions are raised.**

| Guard | Default | Env var | Behavior |
|---|---|---|---|
| **URL allowlist** | empty (allow all + warn) | `BROWSER_ALLOWED_DOMAINS` | CSV globs. Non-matching host returns a spoken error. |
| **SSRF block** | deny | `BROWSER_ALLOW_PRIVATE=0` | Blocks 10/8, 172.16/12, 192.168/16, 169.254/16, 0.0.0.0, ::1, fc00::/7, fe80::/10. DNS lookups have a 2s timeout. |
| **JS eval gate** | allow | `BROWSER_ALLOW_EVAL=1` | Off → `browser_evaluate` always errors. On → 2000-char cap, blocklist (`import`, `<script`, `document.cookie`). |
| **Download sandbox** | `/tmp/browser_downloads` | `BROWSER_DOWNLOAD_DIR` | Rejects `..` traversal and absolute paths outside the sandbox. |
| **Per-turn action cap** | 25 | `BROWSER_MAX_ACTIONS_PER_TURN` | Incremented by every tool. Reset on each new user turn. |

### SSRF check details

```
check_url("http://10.0.0.1")              -> "private or loopback IP range"
check_url("http://169.254.169.254")        -> "private or loopback IP range"  (AWS metadata)
check_url("http://[::1]")                  -> "private or loopback IP range"
check_url("https://example.com/")          -> None  (allowed)
check_url("ftp://example.com/")            -> "only http and https are allowed"
```

The guard resolves hostnames to IPs before deciding. A DNS lookup is
wrapped in a `concurrent.futures` future with a 2-second timeout so a
slow or hostile DNS server cannot stall the audio loop.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `BROWSER_ENABLED` | `1` | Master switch. `0` removes the browser tools from the agent entirely. |
| `BROWSER_HEADLESS` | `1` | Headless Chromium. |
| `BROWSER_MAX_CONTEXTS` | `8` | LRU eviction cap on per-room contexts. |
| `BROWSER_IDLE_TIMEOUT_S` | `300` | Idle reset (informational; LRU does the actual eviction). |
| `BROWSER_ALLOWED_DOMAINS` | `` | CSV globs. Empty = allow all + log warning. |
| `BROWSER_ALLOW_PRIVATE` | `0` | Allow navigation to private IPs. |
| `BROWSER_ALLOW_EVAL` | `1` | Permit `browser_evaluate`. |
| `BROWSER_DEFAULT_TIMEOUT_MS` | `15000` | Default nav timeout. |
| `BROWSER_ACTION_TIMEOUT_MS` | `5000` | Default action timeout. |
| `BROWSER_VIEWPORT_W` / `_H` | `1280` / `720` | Default viewport. |
| `BROWSER_LOCALE` | `en-IN` | Browser locale. |
| `BROWSER_TIMEZONE` | `Asia/Kolkata` | Browser timezone. |
| `BROWSER_USER_AGENT` | `` | UA override. Empty = Playwright default. |
| `BROWSER_PROXY` | `` | HTTP/SOCKS proxy URL. |
| `BROWSER_DOWNLOAD_DIR` | `/tmp/browser_downloads` | Sandboxed download path. |
| `BROWSER_MAX_ACTIONS_PER_TURN` | `25` | Per-room per-turn action cap. |

## SYSTEM_PROMPT Extension

When `BROWSER_ENABLED=1`, the system prompt in
`langchain_agent/tools.py` gains a `BROWSER TOOLS` section instructing
the LLM to:

- Always `browser_navigate` first, then `browser_snapshot` to ground.
- Never read page text aloud verbatim.
- Keep spoken responses under ~30 words.
- Use `role:button:Name` syntax for role+name refs.
- Offer to escalate on browser-tool errors.

When `BROWSER_ENABLED=0`, the section is omitted and the browser tools
are not registered with the agent.

## File Layout

```
langchain_agent/browser/
├── __init__.py        # public surface: session, BROWSER_TOOLS, set_current_room
├── safety.py          # SafetyGuard — pure logic, no Playwright
├── snapshot.py        # accessibility-tree + visible-text summarizers
├── session.py         # BrowserSession — one Browser, per-room Contexts
├── tools_helpers.py   # resolve_locator(page, ref) — CSS / role / text fallback
└── tools.py           # 18 @tool definitions

tests/
├── __init__.py
├── conftest.py        # sandbox_site fixture, requires_chromium marker
├── test_safety.py     # 26 unit tests — no Playwright needed
└── test_browser_tools.py  # 14 integration tests — require Chromium
```

## Voice Pipeline Impact

- `SpeechOnlyGraph` already filters `ToolMessage` from TTS. **No change.**
- Tool result strings can be 2000 chars. The LLM reads them as
  `ToolMessage.content`, then produces a short spoken summary.
- Latency: 0.3–2s per tool call. `max_endpointing_delay=6.0s` (in
  `voice_agent/main.py`) tolerates this — the LLM only emits a spoken
  reply after the tool chain finishes.

## Packaging

`pyproject.toml`:
```toml
dependencies = [
    ...existing...,
    "playwright>=1.47",
]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]
```

`Dockerfile` build stage: adds ~20 system packages for Chromium and runs
`playwright install chromium` to bake the browser binary into the image.
Total image size grows by ~300MB.

Non-root `appuser` runs headless Chromium with:
```
--no-sandbox --disable-dev-shm-usage --disable-gpu
```

Local dev:
```bash
uv sync
uv run playwright install chromium
uv run pytest tests/ -v
```

## Tests

### `tests/test_safety.py` (26 tests, no Playwright)

Covers all five guardrails:
- 9 parametric SSRF cases (10/8, 172.16/12, 192.168/16, 127/8, 169.254/16, 0.0.0.0, ::1, fc00::/7, fe80::/10)
- Allowlist match / non-match / glob escaping
- Scheme allowlist (only http/https)
- Empty / invalid URL
- Eval gate (default, off, blocklist, length cap)
- Download path (within sandbox, traversal, escape, `None`)
- Per-turn action cap (trips at limit, reset, per-room isolation, requires room id)

All 26 pass in this repo's Windows env without Chromium.

### `tests/test_browser_tools.py` (14 tests, requires Chromium)

Auto-skipped when Chromium is not installed. Covers:
- Status reports pool size
- Navigate to sandbox site
- Snapshot returns accessibility tree
- Click by CSS / by visible text
- Type and press-enter
- Select dropdown option
- Evaluate JavaScript
- Get text and attribute
- Open / switch / close tab
- Storage local set / get / clear
- Screenshot
- **Per-room isolation** (cookie/storage in roomA not visible in roomB)
- **LRU eviction** (5 rooms, max_contexts=4 → pool caps at 4)

### Manual smoke checklist (voice loop)

1. "Open example.com" → `browser_navigate` → `browser_snapshot` →
   "I see Example Domain, a simple page used in examples. Want me to
   do anything there?"
2. "Click the 'More information...' link" → `browser_click("More information...")` →
   "Now on the IANA page. Anything else?"
3. "Recharge my number on the Cpl portal" (no real portal yet) →
   `browser_navigate("https://portal.cplphones.com")` → 404 → graceful
   spoken fallback offering escalation.
4. "Try to go to my router" (private IP) → `browser_navigate("http://192.168.1.1")`
   → "I can't access that — private network addresses are blocked for
   safety. Can I help with something else?"

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM reads page text aloud verbatim | SYSTEM_PROMPT rule + `max_chars=2000` cap + truncated `[truncated, full length NNNN chars]` suffix |
| Headless Chromium in container | `--no-sandbox` for `appuser`; ~300MB image growth is acceptable |
| Tool selection accuracy past 20 tools | Curated to 18 + `browser_*` prefix in prompt clusters them |
| Multi-step flow latency (10–20s) | `max_endpointing_delay=6.0s`; LLM emits one spoken reply after the chain finishes |
| Stale page state | LRU eviction + per-room `drop_context` in `finally` |
| Cookie/storage leakage | **Per-room `BrowserContext`** (the main correctness fix over the original draft) |
| DNS hangs in URL check | 2s timeout via `concurrent.futures` future |

## Out of Scope (v1)

- Firefox / WebKit (Chromium only)
- Per-user persistent profile on disk (`user_data_dir`) — v2
- CAPTCHA solving — escalate instead
- Visual diffing / screenshot comparison
- Network mocking, video recording, trace export — v2 if needed
