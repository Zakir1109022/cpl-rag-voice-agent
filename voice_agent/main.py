import asyncio
import threading

from dotenv import load_dotenv

from livekit import agents, rtc
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
# Heavy-import warm-up
# ---------------------------------------------------------------------------

# `import langchain_agent` executes langchain_agent/__init__.py -> tools.py,
# which at module scope loads the SentenceTransformer model, opens the Qdrant
# client, and validates the collection. That work is multi-second, CPU-bound,
# and synchronous. If the first import happens inside the async `entrypoint`,
# it blocks the LiveKit event loop; the Rust core then logs
# "signal_event taking too much time" and can drop the signal socket
# mid-call. Warm it on a plain worker thread instead (see
# `_setup_worker_process`) so the entrypoint's import is a sys.modules hit.
_warmup_lock = threading.Lock()
_warmup_done = False


def _warm_heavy_imports() -> None:
    """Import the heavy langchain_agent stack off the event loop.

    Idempotent and safe to call from a worker thread or via
    ``asyncio.to_thread``. Errors are swallowed here so the worker init
    handshake cannot fail; the entrypoint re-runs the import and surfaces
    the same error, preserving fail-fast behavior for a broken RAG index.
    """
    global _warmup_done
    with _warmup_lock:
        if _warmup_done:
            return

        import time

        started = time.perf_counter()
        try:
            import langchain_agent  # noqa: F401  heavy: embedder + qdrant + graph
            import langchain_agent.browser  # noqa: F401
        except Exception as exc:
            print(f"[WARMUP] langchain_agent import failed: {exc}")
            return

        _warmup_done = True
        print(f"[WARMUP] langchain_agent ready in {time.perf_counter() - started:.2f}s")


# ---------------------------------------------------------------------------
# Worker process setup
# ---------------------------------------------------------------------------


def _setup_worker_process(proc: JobProcess) -> None:
    """Called once per LiveKit worker process, before any job's event loop.

    Runs inside LiveKit's `initialize_process_timeout` (default 10s), so this
    must return quickly. Two things happen here:

      - The heavy `langchain_agent` import (embedder + Qdrant validation) is
        kicked off on a background thread. It populates `sys.modules`
        concurrently with the init handshake and the first job, so the
        entrypoint's import is a cache hit instead of a multi-second block
        on the event loop.
      - If browser tools are enabled, register an `atexit` handler that
        lazily imports the browser session at process-exit time (imports
        there are safe because the handshake is already complete and the
        worker is tearing down).
    """
    # Warm imports off the event loop without blocking the init handshake:
    # the daemon thread keeps running for the process lifetime.
    threading.Thread(
        target=_warm_heavy_imports, name="langchain-warmup", daemon=True
    ).start()

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
# `num_idle_processes` defaults to 0 in dev mode, so the first job of a run
# has no pre-warmed process and logs "no warmed process available for job,
# waiting for one to be created" while it cold-spawns (slow with this project's
# imports). Keep one process warm so the first call joins promptly.
server = AgentServer(setup_fnc=_setup_worker_process, num_idle_processes=1)


def _reason_name(reason: object) -> str:
    """Map a LiveKit DisconnectReason enum value to a readable name."""
    try:
        return rtc.DisconnectReason.Name(reason)
    except (TypeError, ValueError):
        return str(reason)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: agents.JobContext):
    # Off-load the heavy langchain_agent import (SentenceTransformer + Qdrant)
    # from the event loop. _setup_worker_process normally has this warmed
    # already, so this is a sys.modules cache hit; awaiting it here also closes
    # the race where a job arrives before the warm-up thread finishes, so the
    # loop never blocks on the import during a live call.
    await asyncio.to_thread(_warm_heavy_imports)

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

    # Log *why* the agent's connection ends. The LiveKit core otherwise only
    # surfaces raw signal errors (e.g. "server sent leave during resumption");
    # these handlers name the actual reason (DUPLICATE_IDENTITY, ROOM_DELETED,
    # PARTICIPANT_REMOVED, ...) so connection drops are diagnosable.
    @ctx.room.on("disconnected")
    def _on_room_disconnected(reason: object) -> None:
        print(f"[LIVEKIT] room {room_id} disconnected reason={_reason_name(reason)}")

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        print(
            f"[LIVEKIT] participant {participant.identity} left room {room_id} "
            f"reason={_reason_name(participant.disconnect_reason)}"
        )

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

    @session.on("close")
    def _on_session_close(ev: object) -> None:
        reason = getattr(getattr(ev, "reason", None), "value", None) or getattr(
            ev, "reason", None
        )
        print(f"[LIVEKIT] session closed room {room_id} reason={reason} error={getattr(ev, 'error', None)}")

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
