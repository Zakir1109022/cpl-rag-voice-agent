from dotenv import load_dotenv

from livekit import agents
from langchain_core.messages import ToolMessage
from livekit.agents import Agent, AgentServer, AgentSession, inference
from livekit.plugins import langchain as lk_langchain

from config import settings

_ = load_dotenv()


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


class CplAgent(Agent):
    """Cpl Phones voice agent backed by a LangChain agent graph."""

    def __init__(self) -> None:
        from langchain_agent import cpl_agent

        super().__init__(
            instructions="You are the Cpl Phones customer support voice assistant.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(cpl_agent)),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
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

    await session.start(agent=CplAgent(), room=ctx.room)

    await session.generate_reply(user_input="Greet the user as Cpl Phone's customer assistant and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(server)