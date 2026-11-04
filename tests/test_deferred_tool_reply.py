from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall, LLMStream
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.events import ConversationItemAddedEvent, SpeechCreatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

# how long the async tool keeps working after its update; its result lands while the
# agent that called it is still switching away
TOOL_WORK = 0.5


@function_tool
async def send_link(ctx: RunContext) -> str:
    """Send a link."""
    await ctx.update("sending the link")
    await asyncio.sleep(TOOL_WORK)
    return "link sent"


class _Second(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="second agent")


class _First(Agent):
    def __init__(self, *, with_send_link: bool) -> None:
        super().__init__(instructions="first agent", tools=[send_link] if with_send_link else [])

    @function_tool
    async def handoff(self) -> Agent:
        """Hand off to the second agent."""
        await asyncio.sleep(0.05)  # finish after send_link's update, so the handoff is kept
        return _Second()


class _WaitTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="wait task")

    async def on_enter(self) -> None:
        await asyncio.sleep(TOOL_WORK * 2)
        self.complete(None)


class _Parent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent agent", tools=[send_link])

    @function_tool
    async def collect(self) -> str:
        """Run a task."""
        await _WaitTask()
        return "collected"


class _AskTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="ask task")

    async def on_enter(self) -> None:
        await asyncio.sleep(TOOL_WORK * 2)  # keeps the first run open past send_link's work

    @function_tool
    async def finish(self) -> None:
        """Finish the task."""
        self.complete(None)


class _Asker(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="asker agent", tools=[send_link])

    @function_tool
    async def ask(self) -> str:
        """Ask the user."""
        await _AskTask()
        return "asked"


def _outputs(agent: Agent) -> list[str]:
    return [i.output for i in agent.chat_ctx.items if i.type == "function_call_output"]


async def _run(
    agent: Agent,
    other_tool: str,
    caplog: pytest.LogCaptureFixture,
    *,
    session_tools: list[AsyncToolset] | None = None,
) -> tuple[list[str], Agent]:
    """Run one turn whose LLM step calls ``send_link`` alongside ``other_tool``. Returns the
    agent current at each ``generate_reply`` speech, and the agent current at the end."""
    calls = [
        FunctionToolCall(name=name, arguments="{}", call_id=f"call_{name}")
        for name in ("send_link", other_tool)
    ]
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            # the draining agent's tool reply outlasts the async tool's work
            FakeLLMResponse(input="", content="checking", ttft=0.1, duration=TOOL_WORK * 2),
        ]
    )
    async with AgentSession(llm=llm, tools=session_tools or []) as sess:
        speakers: list[str] = []

        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            if ev.source == "generate_reply":
                speakers.append(type(sess.current_agent).__name__)

        sess.on("speech_created", _on_speech_created)
        await sess.start(agent)

        with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
            await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
            await asyncio.sleep(TOOL_WORK * 4)

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, errors
        return speakers, sess.current_agent


async def test_reply_during_handoff_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """An activity-scoped async tool that finishes while its agent drains must not raise."""
    first = _First(with_send_link=True)
    speakers, current = await _run(first, "handoff", caplog)
    assert speakers == ["_First"]  # only the user turn
    assert isinstance(current, _Second)
    assert "link sent" in _outputs(first)
    assert any("owning activity closed" in r.getMessage() for r in caplog.records)


async def test_reply_during_agent_task_waits_for_resume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An async tool that finishes under an AgentTask is announced by its own agent once
    the task returns, never by the task."""
    parent = _Parent()
    speakers, current = await _run(parent, "collect", caplog)
    assert speakers == ["_Parent", "_Parent"]  # the user turn, then the deferred reply
    assert current is parent
    assert "link sent" in _outputs(parent)


async def test_reply_under_agent_task_does_not_block_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reply waiting for an AgentTask that needs another user turn must not hold the
    current run open, or the next run can never start."""
    calls = [
        FunctionToolCall(name=name, arguments="{}", call_id=f"call_{name}")
        for name in ("send_link", "ask")
    ]
    finish = [FunctionToolCall(name="finish", arguments="{}", call_id="call_finish")]
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            FakeLLMResponse(input="done", content="", ttft=0.1, duration=0.1, tool_calls=finish),
        ]
    )
    asker = _Asker()
    async with AgentSession(llm=llm) as sess:
        speakers: list[str] = []

        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            if ev.source == "generate_reply":
                speakers.append(type(sess.current_agent).__name__)

        sess.on("speech_created", _on_speech_created)
        await sess.start(asker)

        with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
            await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
            assert isinstance(sess.current_agent, _AskTask)
            await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, errors
        assert sess.current_agent is asker
        assert speakers[-1] == "_Asker"  # the deferred reply, after the task returns
        assert "link sent" in _outputs(asker)


async def test_session_scoped_reply_follows_handoff(caplog: pytest.LogCaptureFixture) -> None:
    """A session-scoped async tool that finishes during a handoff is announced by the
    next agent."""
    first = _First(with_send_link=False)
    toolset = AsyncToolset(id="links", tools=[send_link])
    speakers, current = await _run(first, "handoff", caplog, session_tools=[toolset])
    assert isinstance(current, _Second)
    assert speakers == ["_First", "_Second"]
    assert "link sent" in _outputs(current)


async def test_session_scoped_reply_is_dropped_on_close() -> None:
    """A session-scoped reply still pending when the session closes is dropped, and the
    close completes."""
    calls = [FunctionToolCall(name="send_link", arguments="{}", call_id="call_send_link")]
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            # keeps the agent busy past the tool's work, so the reply waits for idle
            FakeLLMResponse(input="busy", content="still going", ttft=0.1, duration=10.0),
        ]
    )
    toolset = AsyncToolset(id="links", tools=[send_link])
    sess = AgentSession(llm=llm, tools=[toolset])
    await sess.start(_First(with_send_link=False))
    await sess.run(user_input="go")
    sess.generate_reply(user_input="busy")
    await asyncio.sleep(TOOL_WORK * 2)  # the tool has finished; its reply waits for idle
    assert toolset._executor._reply_task is not None
    assert not toolset._executor._reply_task.done()

    await asyncio.wait_for(sess.aclose(), timeout=5.0)


class _OwedReplyStream(LLMStream):
    def __init__(
        self,
        llm: FakeLLM,
        *,
        chat_ctx: Any,
        tools: Any,
        conn_options: Any,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)

    async def _run(self) -> None:
        await asyncio.sleep(0.1)
        self._event_ch.send_nowait(
            ChatChunk(
                id="owed_reply",
                delta=ChoiceDelta(role="assistant", content="It is sunny."),
            )
        )


class _OwedReplyLLM(FakeLLM):
    """FakeLLM that can answer the executor's instruction-only deferred reply."""

    def chat(
        self,
        *,
        chat_ctx: Any,
        tools: Any = None,
        conn_options: Any = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        if any(
            item.type == "function_call_output" and item.output == "link sent"
            for item in chat_ctx.items
        ):
            return _OwedReplyStream(
                self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
            )
        return super().chat(chat_ctx=chat_ctx, tools=tools, conn_options=conn_options, **kwargs)


async def test_owed_deferred_reply_keeps_agent_thinking() -> None:
    """A speech ending while a deferred tool reply is owed must not reach listening.

    The deferred reply cannot start until the agent is idle; if the speech-end state
    ignores the pending update, clients see a listening edge before the tool answer.
    """
    calls = [FunctionToolCall(name="send_link", arguments="{}", call_id="call_send_link")]
    llm = _OwedReplyLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go", content="one moment", ttft=0.1, duration=0.1, tool_calls=calls
            ),
            # the busy reply outlasts the tool, so the tool reply stays queued
            FakeLLMResponse(input="busy", content="still going", ttft=0.1, duration=1.0),
            FakeLLMResponse(input="link sent", content="It is sunny.", ttft=0.1, duration=0.1),
        ]
    )
    agent = _First(with_send_link=True)
    async with AgentSession(llm=llm) as sess:
        states: list[str] = []
        answered = asyncio.Event()

        sess.on("agent_state_changed", lambda ev: states.append(ev.new_state))

        def _on_item(ev: ConversationItemAddedEvent) -> None:
            if getattr(ev.item, "text_content", None) == "It is sunny.":
                answered.set()

        sess.on("conversation_item_added", _on_item)
        await sess.start(agent)
        await sess.run(user_input="go")
        sess.generate_reply(user_input="busy")
        await asyncio.wait_for(answered.wait(), timeout=10.0)
        await asyncio.sleep(0.2)

        assert sess.agent_state == "listening"
        speaking = [i for i, state in enumerate(states) if state == "speaking"]
        assert len(speaking) >= 3, states
        between = states[speaking[-2] + 1 : speaking[-1]]
        assert "listening" not in between, states
