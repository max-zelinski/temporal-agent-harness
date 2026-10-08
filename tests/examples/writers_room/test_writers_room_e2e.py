# ABOUTME: Deterministic end-to-end of the writers-room example: the whole agent tree —
# showrunner -> writers room (research/screenwriter/critic loop) -> preproduction
# (parallel) -> file writer — runs in a time-skipping Temporal env with scripted models
# (TestModel / FunctionModel in place of OpenAI) and stub tool activities, so the full
# pipeline executes with no OPENAI_API_KEY and no network. This is what proves the four
# workflow classes, the generated subagent toolsets, and the loop/parallel orchestration
# actually wire together.
#
# Run with: uv run --group examples pytest tests/examples/writers_room -v

from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta
from typing import Any

# The example builds its staff and showrunner agents at module import, and the OpenAI
# provider resolves eagerly — so an OPENAI_API_KEY must exist before the imports below
# (and before the sandbox loads its own copies). It is never sent anywhere: every model
# call in this file is dispatched by activity name to the scripted test models.
os.environ.setdefault("OPENAI_API_KEY", "writers-room-test")

import pytest_asyncio
from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig

from temporal_agent_harness.harness import agent
from temporal_agent_harness.harness.agent_protocol import (
    SEND_AGENT_MESSAGE_UPDATE,
    TURN_EVENTS_TOPIC,
    AgentConfig,
    AgentEvent,
    AgentEventType,
    AgentMessage,
    AgentMessageReply,
)
from temporal_agent_harness.plugin import AgentHarnessPlugin

from examples.writers_room import staff, workflow
from examples.writers_room.workflow import (
    FileWriterWorkflow,
    FilmConceptTeamWorkflow,
    PreproductionTeamWorkflow,
    WritersRoomWorkflow,
)

_STUB_TIMEOUT = ActivityConfig(start_to_close_timeout=timedelta(seconds=10))


# ---------------------------------------------------------------------------
# Stub tool activities — same activity names as the real tools, so the test worker
# answers them without touching Wikipedia or the filesystem. The workflow-side toolset
# is still built from the real tools, so the model-facing schemas are unchanged.
# ---------------------------------------------------------------------------


@agent.activity_tool_defn(
    inherently_safe=True, name="search_wikipedia", activity_config=_STUB_TIMEOUT
)
async def stub_search_wikipedia(query: str) -> str:
    return f"Ada Lovelace (1815-1852): English mathematician, wrote the first algorithm for Babbage's Analytical Engine. (stub for {query!r})"


_WRITTEN: dict[str, str] = {}


@agent.activity_tool_defn(
    inherently_safe=True, name="write_pitch_file", activity_config=_STUB_TIMEOUT
)
async def stub_write_pitch_file(filename: str, contents: str) -> str:
    _WRITTEN[filename] = contents
    return f"/tmp/{filename}.md"


# ---------------------------------------------------------------------------
# Scripted models for the leaf agents. The streamed model-request activity needs each
# model's stream_function (or TestModel's built-in streaming), so every FunctionModel
# gets both: the function produces the ModelResponse, the stream variant replays it as
# stream items.
# ---------------------------------------------------------------------------


async def _replay_as_stream(response: ModelResponse):
    for i, part in enumerate(response.parts):
        if isinstance(part, TextPart):
            yield part.content
        elif isinstance(part, ToolCallPart):
            args = part.args if isinstance(part.args, str) else json.dumps(part.args)
            yield {
                i: DeltaToolCall(
                    name=part.tool_name,
                    json_args=args,
                    tool_call_id=part.tool_call_id,
                )
            }


def _scripted(respond) -> FunctionModel:
    """A FunctionModel whose streamed variant replays ``respond``'s ModelResponse as
    stream items — the streamed model-request activity needs a stream_function, the
    plain one a function; giving both keeps either path scripted."""

    async def sfn(messages, info: AgentInfo):
        async for item in _replay_as_stream(respond(messages, info)):
            yield item

    return FunctionModel(function=respond, stream_function=sfn)


_CRITIC_CALLS: list[int] = []


def _critic_response(messages, info: AgentInfo) -> ModelResponse:
    # Reject the first draft, approve the second — exercising one full loop iteration.
    approved = len(_CRITIC_CALLS) >= 1
    _CRITIC_CALLS.append(1)
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={
                    "approved": approved,
                    "notes": "Compelling arc." if approved else "Tighten act two.",
                },
                tool_call_id=f"critic-{len(_CRITIC_CALLS)}",
            )
        ]
    )


def _screenwriter_response(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={
                    "logline": "She counted the future.",
                    "treatment": "Act I: Ada meets Babbage. Act II: the engine, the notes, "
                    "the wager. Act III: the first program, lost then found.",
                },
                tool_call_id="screenwriter-1",
            )
        ]
    )


# ---------------------------------------------------------------------------
# The scripted showrunner: replays the fixed pipeline — each response issues the next
# tool call, reading the subagent handle out of the start_* tool returns.
# ---------------------------------------------------------------------------


def _tool_returns(messages) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    ]


def _last_result(messages, tool_name: str) -> Any:
    for part in reversed(_tool_returns(messages)):
        if part.tool_name == tool_name:
            content = part.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    pass
            return content
    return None


def _call(step: int, name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name=name, args=args, tool_call_id=f"showrunner-{step}")
        ]
    )


def _showrunner_response(messages, info: AgentInfo) -> ModelResponse:
    """One scripted step per model call, in pipeline order."""
    n = len(_tool_returns(messages))  # completed tool calls so far
    wr = _last_result(messages, "start_writers_room")
    pp = _last_result(messages, "start_preproduction")
    fw = _last_result(messages, "start_file_writer")
    pitch = _last_result(messages, "writers_room_pitch") or {}
    prepro = _last_result(messages, "preproduction_plan") or {}
    filed = _last_result(messages, "file_writer_write") or {}

    match n:
        case 0:
            return _call(n, "start_writers_room", {})
        case 1:
            return _call(
                n,
                "writers_room_pitch",
                {"subagent": wr, "request": {"subject": "Ada Lovelace"}},
            )
        case 2:
            return _call(n, "stop_writers_room", {"subagent": wr})
        case 3:
            return _call(n, "start_preproduction", {})
        case 4:
            return _call(
                n,
                "preproduction_plan",
                {
                    "subagent": pp,
                    "request": {
                        "logline": pitch["logline"],
                        "draft": pitch["draft"],
                    },
                },
            )
        case 5:
            return _call(n, "stop_preproduction", {"subagent": pp})
        case 6:
            return _call(n, "start_file_writer", {})
        case 7:
            return _call(
                n,
                "file_writer_write",
                {
                    "subagent": fw,
                    "package": {
                        "subject": "Ada Lovelace",
                        "logline": pitch["logline"],
                        "draft": pitch["draft"],
                        "box_office": prepro["box_office"],
                        "casting": prepro["casting"],
                    },
                },
            )
        case 8:
            return _call(n, "stop_file_writer", {"subagent": fw})
        case _:
            return ModelResponse(
                parts=[
                    TextPart(
                        content=(
                            f"Done. {filed.get('title')} — "
                            f"'{pitch.get('logline')}' — filed at {filed.get('path')} "
                            f"after {pitch.get('iterations')} critique cycles."
                        )
                    )
                ]
            )


def _test_cast() -> tuple[FunctionModel, dict[str, Any]]:
    """The scripted cast: a showrunner FunctionModel plus per-role staff models."""
    test_staff = staff.build_staff(
        {
            "researcher": TestModel(
                call_tools=["search_wikipedia"],
                custom_output_text="Ada Lovelace: mathematician; wrote the first "
                "algorithm for Babbage's Analytical Engine; her notes contain the "
                "first published computer program.",
            ),
            "screenwriter": _scripted(_screenwriter_response),
            "critic": _scripted(_critic_response),
            "box_office": TestModel(
                call_tools=[], custom_output_text="$60-120M worldwide, per comps."
            ),
            "casting": TestModel(
                call_tools=[], custom_output_text="Ada: an actor with quiet intensity."
            ),
            "file_writer": TestModel(
                call_tools=["write_pitch_file"],
                custom_output_args={
                    "title": "The Analytical Engine",
                    "path": "/tmp/the-analytical-engine.md",
                },
            ),
        }
    )
    showrunner_model = _scripted(_showrunner_response)
    showrunner = workflow.build_showrunner(showrunner_model)
    return showrunner_model, test_staff, showrunner


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def env_and_queue():
    """A time-skipping env plus one worker hosting the whole agent tree (exactly like
    the example's worker.py, minus the real model/tools).

    The scripted cast is injected through WORKER REGISTRATION alone: inside the
    sandbox, ``staff``/``workflow`` are fresh restricted copies whose TemporalAgents
    only dispatch model and tool calls to activities on the worker *by name*. So the
    test just registers the scripted agents' activities via ``AgentPlugin`` — every
    ``TemporalAgent.run`` in workflow code then lands on the test models.
    """
    _, test_staff, showrunner = _test_cast()

    env = await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    )
    # The subagent toolsets pin children to workflow.TASK_QUEUE, so the worker must
    # poll that queue — the time-skipping env is per-test, so a fixed name is safe.
    task_queue = workflow.TASK_QUEUE
    async with Worker(
        env.client,
        task_queue=task_queue,
        workflows=[
            FilmConceptTeamWorkflow,
            WritersRoomWorkflow,
            PreproductionTeamWorkflow,
            FileWriterWorkflow,
        ],
        plugins=[
            # PydanticAIPlugin on the worker supplies the sandbox passthrough modules
            # (httpx/openai/...) the providers need — without it, constructing an Agent
            # on a provider-string model fails inside the sandbox.
            PydanticAIPlugin(),
            *(AgentPlugin(a) for a in [showrunner, *test_staff.values()]),
            AgentHarnessPlugin(tools=[stub_search_wikipedia, stub_write_pitch_file]),
        ],
    ):
        try:
            yield env, task_queue
        finally:
            await env.shutdown()


async def test_writers_room_end_to_end(env_and_queue):
    """'Ada Lovelace' in -> a filed pitch out, through every stage of the tree."""
    env, task_queue = env_and_queue
    _CRITIC_CALLS.clear()
    _WRITTEN.clear()

    handle = await env.client.start_workflow(
        FilmConceptTeamWorkflow.run,
        AgentConfig(),
        id=f"FilmConceptTeam-{uuid.uuid4()}",
        task_queue=task_queue,
    )
    await handle.execute_update(
        SEND_AGENT_MESSAGE_UPDATE,
        AgentMessage(type="ask", payload={"text": "Ada Lovelace"}),
        result_type=AgentMessageReply,
    )

    # Read the parent turn's stream: the reply plus the subagent lifecycle events.
    stream = WorkflowStreamClient.create(env.client, handle.id)
    reply: str | None = None
    seen: set[str] = set()
    async for item in stream.subscribe(
        topics=[TURN_EVENTS_TOPIC], from_offset=0, result_type=AgentEvent
    ):
        envelope: AgentEvent = item.data
        seen.add(envelope.event.type)
        if envelope.event.type == AgentEventType.MESSAGE_HANDLER_END:
            reply = envelope.event.output.get("text")
        if envelope.event.type == AgentEventType.TURN_END:
            break

    assert reply is not None, "the showrunner's turn ended without a reply"
    assert "The Analytical Engine" in reply
    assert "She counted the future" in reply
    assert "2 critique cycles" in reply

    # The loop iterated: critic ran twice (reject, then approve).
    assert len(_CRITIC_CALLS) == 2

    # The file writer's tool ran — with the full package assembled into the document...
    # (the stub captured the contents the model passed).
    assert len(_WRITTEN) == 1
