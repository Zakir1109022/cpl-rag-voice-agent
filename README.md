# Cpl Phones Voice Agent 🎙️📱

## Voice Agent

A real-time voice-based customer support agent for **Cpl Phones** — a fictional Indian telecom company. Customers speak to an AI assistant over a live audio stream that can look up account balances, answer plan questions using RAG over PDF docs, check network coverage, recommend plans, and escalate to human support. Built with **LiveKit Agents**, **LangChain + Groq**, **Docling**, and **Qdrant**.

---

## Getting Started

### Prerequisites

- Python `>= 3.12`
- [LiveKit Cloud](https://livekit.io/) credentials (URL, API Key, API Secret)
- [Groq](https://console.groq.com/keys) API Key

### Installation

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your LiveKit and Groq credentials

# Ingest PDF documents into vector DB (Windows PowerShell)
.\.venv\Scripts\python.exe ingest.py

# Start the voice agent and connect it to LiveKit (Windows PowerShell)
.\.venv\Scripts\python.exe -m voice_agent.main dev

# On macOS/Linux, use .venv/bin/python instead of .venv\Scripts\python.exe
```

`dev` (or `start` in production) registers a worker with LiveKit and auto-dispatches
it to new rooms, so the agent joins any room the frontend connects to.

> **Do not use `console` mode with the web frontend.** `console` runs the worker
> *unregistered* and only simulates a local `console-room`, so it never joins a real
> LiveKit room. The frontend waits 20s, then fails with "Agent did not join the room."
> Use `console` only for terminal-only testing, e.g. `... -m voice_agent.main console`.


---

## Example Prompts

| Try saying… | What happens |
|---|---|
| *"What plans do you offer?"* | Searches plan docs via RAG |
| *"What's the 5G coverage in Bangalore?"* | Returns network status by region |
| *"I want to speak to a human"* | Creates a support ticket and escalates |

---
Copyright©️ Cloud Production Ltd. All rights reserved.
