# Cpl Phones Voice Agent — Project Overview

A real-time, voice-based customer support agent for **Cpl Phones**, a fictional Indian telecom carrier. Customers speak to the agent over a live audio stream; the agent answers from Cpl's own documentation, looks up account balances and network status, recommends plans, escalates to a human, and can fall back to a web search for general-knowledge questions.

## Stack

| Layer | Technology |
|---|---|
| Voice runtime | LiveKit Agents 1.4.5 |
| Speech-to-text | Deepgram Nova-3 (multilingual) |
| Text-to-speech | Inworld TTS-2, voice "Ashley" |
| LLM | Groq (`openai/gpt-oss-120b`, temperature 0) |
| Agent framework | LangChain 1.x `create_agent` + `langchain-plugins-langchain` adapter |
| Retrieval | Qdrant (local on-disk, packaged into the Docker image) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Document parsing | Docling + `HierarchicalChunker` |
| Web search | `duckduckgo-search` (DuckDuckGo, no API key) |
| Browser automation | Playwright (Chromium, one `Browser` per worker, fresh `BrowserContext` per room) |
| Config | `pydantic-settings` reading `.env` |
| Packaging | uv + multi-stage Docker build, non-root user |

## Architecture

```
                          LiveKit Cloud room
                                |
                       +--------v--------+
                       |  voice_agent    |  Deepgram STT  -> Inworld TTS
                       |   (main.py)     |  SpeechOnlyGraph filters tool output
                       +--------+--------+
                                | LLMAdapter
                                v
                       +-----------------+         +-----------------+
                       | langchain_agent |  -----> |   Groq LLM      |
                       |   (tools.py)    |         |  (gpt-oss-120b) |
                       +--------+--------+         +-----------------+
                                |
        +-------+-------+-------+-------+-------+-------+
        |       |       |       |       |       |       |
     search   check   check   get_plan get_plan escalate web
     _plans   _acct   _net    _rec     _detail  _support search
        |
        v
   +---------+        +------------+
   | Qdrant  | <----- | embedder   |  (sentence-transformers)
   | (local) |
   +---------+
```

The LiveKit worker drives a `SpeechOnlyGraph` wrapper around the LangChain graph that strips `ToolMessage` events from the `astream` before TTS, so raw tool output is never spoken. The LLM reads the tool result, summarizes it, and that summary is what the user hears.

## Repository Layout

```
cpl-voice-agent/
├── config.py                  # pydantic-settings: LiveKit, Groq, Qdrant, voice
├── ingest.py                  # Build-time: PDFs -> Docling -> chunks -> Qdrant
├── livekit.toml               # LiveKit project subdomain + agent ID
├── pyproject.toml             # All dependencies (uv-managed)
├── uv.lock                    # Resolved lockfile
├── Dockerfile                 # Multi-stage build, bakes Qdrant index in
├── README.md
├── docs/
│   ├── cpl-plans.pdf          # Source documents for RAG
│   ├── cpl-billing-faq.pdf
│   ├── cpl-network-coverage.pdf
│   └── specs/                 # This document lives here
├── langchain_agent/
│   ├── __init__.py            # Re-exports cpl_agent, qdrant, embedder
│   ├── tools.py               # @tool definitions + SYSTEM_PROMPT + agent graph
│   ├── rag.py                 # retrieve() + build_context() + standalone rag()
│   └── browser/               # Playwright-backed browser tools
│       ├── __init__.py        # Public surface
│       ├── safety.py          # URL allowlist, SSRF, eval gate, download sandbox, action cap
│       ├── session.py         # BrowserSession: 1 Browser, per-room BrowserContext
│       ├── snapshot.py        # Accessibility-tree + visible-text summarizer
│       ├── tools_helpers.py   # resolve_locator(page, ref) — CSS/role/text fallback
│       └── tools.py           # 18 @tool definitions
├── voice_agent/
│   └── main.py                # LiveKit entrypoint, STT/TTS pipeline
├── data/                      # Built Qdrant index lives here at runtime
└── .env.example               # Template for required env vars
```

## Tool Inventory

Defined as `@tool` from `langchain_core.tools` in `langchain_agent/tools.py` and `langchain_agent/browser/tools.py`. Every tool is a synchronous function returning a `str`; the LLM uses the docstring as the routing signal.

| Tool | Source | Purpose |
|---|---|---|
| `search_plans(query)` | RAG over Cpl PDFs | Plan details, pricing, validity, comparisons |
| `check_account_balance(phone_number)` | `MOCK_CUSTOMER_DB` | Balance, data, voice, SMS usage |
| `check_network_status(region)` | `MOCK_NETWORK_STATUS` | 4G/5G coverage by Indian city |
| `get_plan_recommendation(usage_type, data_gb, voice_min)` | `MOCK_PLANS_DB` | Suggest a plan from usage pattern |
| `get_plan_details(plan_name)` | `MOCK_PLANS_DB` | Detailed info on a named plan |
| `escalate_to_support(reason)` | Side effect | Print ticket, return confirmation string |
| `web_search(query, max_results=5)` | DuckDuckGo | Open-domain fallback for general knowledge |

### Browser tools (Playwright)

Registered when `BROWSER_ENABLED=1`. Backed by `BrowserSession` in
`langchain_agent/browser/session.py` — one Chromium `Browser` per worker
process, fresh `BrowserContext` per LiveKit room. The full list and
safety policy live in `docs/specs/BROWSER_TOOLS_PLAN.md`.

| Tool | Purpose |
|---|---|
| `browser_navigate(url, wait_until="load")` | Open URL |
| `browser_snapshot(max_chars=2000)` | Accessibility-tree snapshot |
| `browser_click(selector_or_label)` | Click by CSS / role+name / text |
| `browser_type(selector_or_label, text, press_enter=False)` | Fill an input |
| `browser_press_key(key)` | Keyboard key (Enter, Escape, Tab) |
| `browser_scroll(direction, amount)` | Scroll viewport |
| `browser_wait_for(selector_or_label, timeout_ms=5000)` | Wait for an element |
| `browser_get_text(selector_or_label, max_chars=2000)` | Read text from an element |
| `browser_get_attribute(selector_or_label, attribute)` | Read HTML attribute |
| `browser_list_tabs()` | List open tabs |
| `browser_open_tab(url="")` | New tab |
| `browser_switch_tab(index)` | Switch active tab |
| `browser_close_page()` | Close current tab |
| `browser_screenshot(path="", full_page=False)` | Save PNG |
| `browser_evaluate(js_expression)` | Run JS (gated by `BROWSER_ALLOW_EVAL`) |
| `browser_storage_get(kind, key="")` | Read localStorage / sessionStorage / cookies |
| `browser_storage_set(kind, key, value)` | Write storage |
| `browser_storage_clear(kind="all")` | Clear storage / cookies |
| `browser_status()` | Session health |

Safety guards: URL allowlist, SSRF block, JS eval gate, download sandbox, per-turn action cap. See `BROWSER_TOOLS_PLAN.md` for details.

Mock DBs are in-module dicts; the RAG index is the only piece of state that requires ingestion. The agent refuses to start if the Qdrant collection is empty (`validate_qdrant_collection` at import time).

## RAG Pipeline

Build-time only (`ingest.py`):

1. Scan `docs/*.pdf` with `Path(docs_dir).glob("*.pdf")`.
2. Parse with Docling `DocumentConverter`.
3. Chunk with `HierarchicalChunker(max_characters=settings.chunk_size=500)`, prepending `meta.headings` as a breadcrumb.
4. Embed with `sentence-transformers/all-MiniLM-L6-v2` (downloaded into the image at build time).
5. Upsert into a local Qdrant collection `telecom_docs` (cosine distance) at `/app/data/qdrant`.

Runtime (`langchain_agent/rag.py`):

- `retrieve(client, embedder, query, top_k=5)` — embed the query, `query_points`, return payloads with `score`.
- `build_context(chunks)` — join as `[Source i]\n{chunk_text}` separated by `\n\n---\n\n`.

The `rag()` end-to-end helper in `rag.py` is a CLI-only convenience (run with `python langchain_agent/rag.py "question"`); the voice agent calls `retrieve` + `build_context` directly inside `search_plans`.

The Docker build verifies the index non-empty before tagging the production image (`points_count > 0` assertion in `Dockerfile`).

## Voice Pipeline

`voice_agent/main.py` wires the LiveKit session:

- STT: `inference.STT(model="deepgram/nova-3", language="multi")`
- TTS: `inference.TTS(model="inworld/inworld-tts-2", voice="Ashley")`
- Turn detection: STT-based endpointing, `min_endpointing_delay=0.5s`, `max=6.0s`
- LLM: `lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(cpl_agent))`
- First turn prompt: *"Greet the user as Cpl Phone's customer assistant and ask how you can help."*

`SpeechOnlyGraph` is a thin async wrapper that filters out `ToolMessage` events from the LangGraph `astream`, so only the LLM's final conversational reply reaches TTS. Tool results are seen by the LLM as `ToolMessage.content` and summarized.

## Configuration

All settings come from `.env` via `pydantic-settings` (case-insensitive, `extra="ignore"`). LiveKit and Groq values are also exported into `os.environ` for SDKs that read env directly.

| Env var | Default | Purpose |
|---|---|---|
| `LIVEKIT_URL` | — (required) | WebSocket URL of LiveKit project |
| `LIVEKIT_API_KEY` | — (required) | LiveKit API key |
| `LIVEKIT_API_SECRET` | — (required) | LiveKit API secret |
| `GROQ_API_KEY` | — (required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model id |
| `DOCS_DIR` | `docs` | PDF source directory for ingestion |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformers model id |
| `COLLECTION_NAME` | `telecom_docs` | Qdrant collection name |
| `CHUNK_SIZE` | `500` | Hierarchical chunker max characters |
| `QDRANT_URL` / `QDRANT_API_KEY` | `None` | Optional cloud Qdrant (unused in Docker) |
| `QDRANT_PATH` | `/app/data/qdrant` | Local Qdrant path (used in Docker) |
| `STT_MODEL` | `deepgram/nova-3` | STT model id |
| `STT_LANGUAGE` | `multi` | STT language code |
| `TTS_MODEL` | `inworld/inworld-tts-2` | TTS model id |
| `TTS_VOICE` | `Ashley` | TTS voice id |
| `BROWSER_ENABLED` | `1` | Master switch for browser tools |
| `BROWSER_HEADLESS` | `1` | Headless Chromium |
| `BROWSER_MAX_CONTEXTS` | `8` | LRU cap on per-room BrowserContexts |
| `BROWSER_IDLE_TIMEOUT_S` | `300` | Idle reset window |
| `BROWSER_ALLOWED_DOMAINS` | `` | CSV globs; empty = allow all + warn |
| `BROWSER_ALLOW_PRIVATE` | `0` | Allow private IP ranges (SSRF escape hatch) |
| `BROWSER_ALLOW_EVAL` | `1` | Permit `browser_evaluate` |
| `BROWSER_DEFAULT_TIMEOUT_MS` | `15000` | Default nav timeout |
| `BROWSER_ACTION_TIMEOUT_MS` | `5000` | Default action timeout |
| `BROWSER_VIEWPORT_W` / `_H` | `1280` / `720` | Default viewport |
| `BROWSER_LOCALE` | `en-IN` | Browser locale |
| `BROWSER_TIMEZONE` | `Asia/Kolkata` | Browser timezone |
| `BROWSER_USER_AGENT` | `` | UA override; empty = Playwright default |
| `BROWSER_PROXY` | `` | HTTP/SOCKS proxy URL |
| `BROWSER_DOWNLOAD_DIR` | `/tmp/browser_downloads` | Sandboxed download path |
| `BROWSER_MAX_ACTIONS_PER_TURN` | `25` | Per-turn action cap |

## Running

### Local development

```bash
uv sync
cp .env.example .env             # fill in LiveKit + Groq credentials
uv run python ingest.py          # build the local Qdrant index
uv run python -m voice_agent.main dev   # register worker + auto-dispatch to rooms
```

Use `dev` (or `start` for production) so the worker registers with LiveKit and is
auto-dispatched into rooms. `console` mode runs the worker unregistered against a
simulated local `console-room`; it is for terminal-only testing and will never join
a real room, so the web frontend fails with "Agent did not join the room."

### Docker

```bash
docker build -t cpl-voice-agent .
docker run --env-file .env cpl-voice-agent
```

The build runs `ingest.py` against placeholder credentials and asserts the Qdrant collection is non-empty before producing the runtime image. The Qdrant index is **packaged into the image**; no external Qdrant service is required at runtime.

## MCP Servers

The project ships project-level MCP config for editor/LLM tooling to query the official LiveKit documentation while working in this repo. No API keys, npm/pip packages, or env vars are required — it's a remote HTTP server.

| File | Tool | Server |
|---|---|---|
| `.puku/settings.json` | puku-cli | `https://docs.livekit.io/mcp` (key: `livekit-docs`) |
| `.vscode/mcp.json` | VS Code | `https://docs.livekit.io/mcp` (key: `livekit-docs`) |

To add it to other clients (Cursor, Claude Code, Codex, Gemini CLI, etc.) use the same URL — see <https://docs.livekit.io/reference/developer-tools/docs-mcp/>.

## Constraints & Conventions

- Responses are spoken aloud — keep them short. The system prompt enforces "single sentence preferred, max 2 sentences" and forbids tables/diagrams.
- Indian currency (₹) only.
- Phone numbers must be exactly 10 digits; `check_account_balance` validates and returns a polite error string on failure.
- Never invent plan, pricing, or coverage data — the LLM is instructed to escalate when in doubt.
- Tool functions never raise; they return a string the LLM can speak.
- `web_search` is the only external HTTP integration; it is rate-limited by DuckDuckGo and may return zero results, which the LLM handles as a graceful fallback.
