import asyncio

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobProcess, inference
from livekit.plugins import langchain as lk_langchain
from langchain_core.messages import ToolMessage

from config import settings

_ = load_dotenv()


# ---------------------------------------------------------------------------
# ToolMessage filter for SpeechOnlyGraph
# ---------------------------------------------------------------------------

def _is_tool_token(item: object) -> bool:
    """True for the (ToolMessage, metadata) pairs astream emits from the tools node."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the graph so tool return values never reach TTS."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


# ---------------------------------------------------------------------------
# Browser session lifecycle
# ---------------------------------------------------------------------------


def _setup_browser_session(proc: JobProcess) -> None:
    """Called once per LiveKit worker process.

    CRITICAL: this function runs inside LiveKit's `initialize_process_timeout`
    (default 10s). Any import here triggers `langchain_agent.__init__.py`,
    which in turn instantiates the embedder and validates the Qdrant
    collection — work that may legitimately take longer than the timeout
    on a cold start. So we deliberately DO NOT import `langchain_agent.*`
    here. Browser wiring is deferred to `entrypoint`, which runs in the
    same worker but outside the init handshake.

    What this function does:
      - If browser tools are disabled, log and return (fast).
      - Otherwise, register an `atexit` handler that lazily imports the
        browser session at process-exit time (imports there are safe
        because the handshake is already complete and the worker is
        tearing down).
    """
    if not settings.browser_enabled:
        print("[BROWSER] disabled (BROWSER_ENABLED=0) — skipping setup")
        return

    print(f"[BROWSER] setup for JobProcess pid={proc.pid}")

    try:
        import atexit

        def _shutdown() -> None:
            # Lazy import here: the worker is already tearing down, so any
            # cost is paid by the exit path, not the init handshake.
            try:
                import asyncio
                from langchain_agent.browser import session as browser_session

                asyncio.run(browser_session.stop())
            except Exception as exc:
                print(f"[BROWSER] shutdown error: {exc}")

        atexit.register(_shutdown)
    except Exception as exc:
        # Browser is optional. A failure here must never crash the worker
        # init handshake — that would take down the whole voice agent.
        print(f"[BROWSER] setup failed (continuing without browser): {exc}")


# ---------------------------------------------------------------------------
# Voice agent
# ---------------------------------------------------------------------------


class CplAgent(Agent):
    """Cpl Phones voice agent backed by a LangChain agent graph."""

    def __init__(self) -> None:
        from langchain_agent import cpl_agent

        super().__init__(
            instructions="You are the Cpl Phones customer support voice assistant.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(cpl_agent)),
        )


# Agent dispatch is configured via AGENT_NAME (see config.py / .env). Empty
# registers the worker for automatic dispatch. A non-empty value (which must
# match frontend/.env.local AGENT_NAME) enables explicit dispatch, isolating a
# local dev worker from a deployed one on the same LiveKit project to avoid
# duplicate-worker dispatch races.
server = AgentServer(setup_fnc=_setup_browser_session)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: agents.JobContext):
    # Bind the per-room browser context for the duration of this session.
    # When BROWSER_ENABLED=0, these are no-ops and every browser_* tool
    # returns a polite error string.
    from langchain_agent.browser import (
        set_current_room,
        get_current_room,
        session as browser_session,
    )
    from langchain_agent.browser.safety import build_guard_from_settings

    room_id = ctx.room.name
    set_current_room(room_id)
    safety = build_guard_from_settings(settings)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_detection="stt",
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0,
    )

    browser_warmup: asyncio.Task[None] | None = None
    try:
        # Start the session first so the agent participant joins the room
        # immediately. Cold-start work (Chromium launch, sentence-transformers
        # model load) must not delay the join, otherwise the frontend's
        # agent-connect timeout fires before the agent appears ("Agent did not
        # join the room.") and the session ends.
        await session.start(agent=CplAgent(), room=ctx.room)

        # Reset the per-turn browser-action cap so a long-running session
        # doesn't hit the cap on a previous turn.
        safety.reset_actions(room_id)

        # Warm the per-room BrowserContext (lazily launches Chromium) in the
        # background so the greeting isn't blocked on it. The finally block
        # awaits this before dropping the context to avoid a create/drop race.
        if settings.browser_enabled:
            browser_warmup = asyncio.create_task(browser_session.context_for(room_id))

        await session.generate_reply(
            user_input="Greet the user as Cpl Phone's customer assistant and ask how you can help."
        )
    finally:
        # Ensure the warm-up finished before dropping, so drop_context cannot
        # race the lazy Chromium launch and leave a BrowserContext behind.
        if browser_warmup is not None:
            try:
                await browser_warmup
            except Exception as exc:
                print(f"[BROWSER] context init failed for {room_id}: {exc}")

        # Drop the room's BrowserContext so cookies/storage cannot leak
        # to the next customer. The browser itself stays alive for the
        # lifetime of the worker process.
        if settings.browser_enabled:
            try:
                await browser_session.drop_context(room_id)
            except Exception as exc:
                print(f"[BROWSER] drop_context error for {room_id}: {exc}")
        set_current_room(None)


if __name__ == "__main__":
    agents.cli.run_app(server)
