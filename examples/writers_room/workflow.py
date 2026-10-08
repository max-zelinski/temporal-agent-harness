"""A writers room on the harness: an ADK-style multi-agent film-concept team.

Four harness agents reproduce the boxes in the diagram — the difference from ADK is only
WHERE the orchestration lives:

* :class:`FilmConceptTeamWorkflow` is the SequentialAgent and the user-facing greeter.
  It is a MODEL-DRIVEN agent (Pydantic AI on the harness): the model converses with the
  user to get a historical subject, then sequences the three stages through generated
  subagent tools — ``writers_room_pitch``, ``preproduction_plan``, ``file_writer_write`` —
  each call a turn on a real child agent workflow (see ``agent.subagent_toolset``).

* :class:`WritersRoomWorkflow` is the LoopAgent — and it is NOT model-driven. Its
  ``pitch`` handler is plain workflow code: one researcher call, then a Python ``for``
  loop alternating screenwriter and critic until the critic approves (bounded at
  ``MAX_ITERATIONS``). The iteration is deterministic: the critic's structured
  ``Critique`` is the exit condition, not a hint to a parent model.

* :class:`PreproductionTeamWorkflow` is the ParallelAgent. One handler, one
  ``asyncio.gather`` — the box-office analyst and the casting director run concurrently
  (two model-request activities in flight at once).

* :class:`FileWriterWorkflow` is the file writer: a small model-driven agent with a
  real ``write_pitch_file`` tool that lands the finished pitch on the worker's disk.

The six leaf agents (researcher, screenwriter, critic, box_office, casting, file_writer)
are module-level ``TemporalAgent``s in ``staff.py`` — invoked by the containers exactly as
an ADK container invokes its sub-agents, each model call a durable activity.
"""

from __future__ import annotations

import asyncio

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import TemporalAgent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model

    from temporal_agent_harness.ai_sdks.pydantic_ai_harness import (
        HarnessDeps,
        build_harness_toolset,
        harness_event_stream_handler,
    )
    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import (
        AgentConfig,
        MidTurn,
        TextMessage,
        TextReply,
        ToolApprovalPolicy,
    )
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner

    from . import staff
    from .models import (
        FileWriteResult,
        PitchRequest,
        PitchResult,
        PreproductionInput,
        PreproductionResult,
        ScreenplayPackage,
    )


TASK_QUEUE = "writers-room"
DEFAULT_MODEL = staff.DEFAULT_MODEL

# The writers-room loop's safety bound: the critic gets this many chances to approve a
# draft before the room ships whatever it has.
MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# LoopAgent: writers_room — researcher -> [screenwriter <-> critic] x N
# ---------------------------------------------------------------------------


@agent.defn(name="WritersRoom")
class WritersRoomWorkflow:
    """The LoopAgent: drafts, critiques, iterates — deterministically, in workflow code."""

    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            # The staff tools (wikipedia search) are inherently safe; nothing else reaches
            # a tool call inside this workflow.
            approval_policy_default=ToolApprovalPolicy.allow_inherently_safe(),
        )

    @agent.accepts
    async def pitch(self, request: PitchRequest) -> PitchResult:
        """Develop a film concept for `request.subject`: research it, then iterate a
        screenwriter/critic loop until the critic approves the treatment. Returns the
        logline, the approved draft, and how many cycles it took."""
        deps = HarnessDeps(runner=self._runner)

        # [researcher] researches using Wikipedia — one shot, its brief feeds every draft.
        research = await staff.STAFF["researcher"].run(
            f"Research the historical subject {request.subject!r} for a feature film.",
            deps=deps,
        )
        brief = str(research.output)

        # [screenwriter] writes a draft -> [critic] offers feedback -> repeat until the
        # critic decides to exit the loop (or we hit the bound). In ADK the loop body and
        # exit are agent-internal; here the loop is just workflow code.
        notes = ""
        treatment = None
        iterations = MAX_ITERATIONS
        for iterations in range(1, MAX_ITERATIONS + 1):
            written = await staff.STAFF["screenwriter"].run(
                "Research brief:\n"
                f"{brief}\n\n"
                f"Critic notes to address (empty for a first draft): {notes}",
                deps=deps,
            )
            treatment = written.output
            verdict = await staff.STAFF["critic"].run(
                f"Review this film treatment.\n\nLogline: {treatment.logline}\n\n"
                f"Treatment:\n{treatment.treatment}",
                deps=deps,
            )
            if verdict.output.approved:
                break
            notes = verdict.output.notes

        assert treatment is not None
        return PitchResult(
            logline=treatment.logline,
            draft=treatment.treatment,
            iterations=iterations,
        )


# ---------------------------------------------------------------------------
# ParallelAgent: preproduction_team — box_office_researcher + casting_agent
# ---------------------------------------------------------------------------


@agent.defn(name="PreproductionTeam")
class PreproductionTeamWorkflow:
    """The ParallelAgent: two leaf agents on one ``asyncio.gather``."""

    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            approval_policy_default=ToolApprovalPolicy.allow_inherently_safe(),
        )

    @agent.accepts
    async def plan(self, request: PreproductionInput) -> PreproductionResult:
        """Take the approved concept and produce the preproduction analyses: a
        box-office estimate and casting suggestions, researched concurrently. Returns
        both reports."""
        brief = f"Logline: {request.logline}\n\nTreatment:\n{request.draft}"
        # Two TemporalAgent.run calls in flight at once — two model-request activities,
        # awaited together. In ADK ParallelAgent fans sub-agents out; a gather is the
        # same thing in workflow code.
        box_office, casting = await asyncio.gather(
            staff.STAFF["box_office"].run(
                f"Estimate this film's box office.\n\n{brief}",
                deps=HarnessDeps(runner=self._runner),
            ),
            staff.STAFF["casting"].run(
                f"Suggest casting for this film's principal characters.\n\n{brief}",
                deps=HarnessDeps(runner=self._runner),
            ),
        )
        return PreproductionResult(
            box_office=str(box_office.output), casting=str(casting.output)
        )


# ---------------------------------------------------------------------------
# file_writer — a small tool-using agent that writes the pitch document
# ---------------------------------------------------------------------------


@agent.defn(name="FileWriter")
class FileWriterWorkflow:
    """The file writer: aggregates the package and writes it to a file."""

    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            approval_policy_default=ToolApprovalPolicy.allow_inherently_safe(),
        )

    @agent.accepts
    async def write(self, package: ScreenplayPackage) -> FileWriteResult:
        """Assemble the full pitch document for the film concept in `package` — subject,
        logline, treatment, box-office analysis, casting — and write it to a markdown
        file. Returns the document's title and the path it was written to."""
        result = await staff.STAFF["file_writer"].run(
            "Assemble and file this film-concept pitch.\n\n"
            f"Subject: {package.subject}\n"
            f"Logline: {package.logline}\n\n"
            f"Treatment:\n{package.draft}\n\n"
            f"Box-office analysis:\n{package.box_office}\n\n"
            f"Casting:\n{package.casting}",
            deps=HarnessDeps(runner=self._runner),
        )
        return result.output


# ---------------------------------------------------------------------------
# SequentialAgent: film_concept_team — greeter + model-driven sequencer
# ---------------------------------------------------------------------------

# The subagent toolsets, generated statically from each container's @agent.accepts
# handlers. The showrunner model drives the three stages through these:
#   start_writers_room / writers_room_pitch / stop_writers_room
#   start_preproduction / preproduction_plan / stop_preproduction
#   start_file_writer / file_writer_write / stop_file_writer
_SUBAGENT_TOOLS = [
    *agent.subagent_toolset(
        WritersRoomWorkflow, key="writers_room", task_queue=TASK_QUEUE
    ),
    *agent.subagent_toolset(
        PreproductionTeamWorkflow, key="preproduction", task_queue=TASK_QUEUE
    ),
    *agent.subagent_toolset(
        FileWriterWorkflow, key="file_writer", task_queue=TASK_QUEUE
    ),
]
_SUBAGENT_TOOL_NAMES = [t.__name__ for t in _SUBAGENT_TOOLS]

_TEAM_TOOLSET, _TEAM_TOOL_CONFIG = build_harness_toolset(
    _SUBAGENT_TOOLS, id="film-team"
)

SHOWRUNNER_INSTRUCTIONS = f"""\
You are the showrunner of a film-concept team: a greeter, then a fixed three-stage \
pipeline, conversing naturally along the way.

1. Greet the user and ask them for a historical subject for the film — a real person, \
place, or event (e.g. "Ada Lovelace"). If they already named one, confirm it and start.
2. WRITERS ROOM: call `start_writers_room` for a handle, then `writers_room_pitch` passing \
the handle as `subagent` and `request` = {{'subject': <the subject>}}. The room \
researches the subject, then iterates a draft/critique loop until the critic approves — \
you don't drive that loop. It returns `logline`, `draft`, and `iterations` (the number \
of critique cycles it ran). Then `stop_writers_room` with the handle.
3. PREPRODUCTION: `start_preproduction`, then `preproduction_plan` with the handle as \
`subagent` and `request` = {{'logline': <logline>, 'draft': <the FULL draft text>}}. It \
returns `box_office` and `casting` analyses (computed in parallel inside the team). Then \
`stop_preproduction`.
4. FILE WRITER: `start_file_writer`, then `file_writer_write` with the handle as \
`subagent` and `package` = {{'subject', 'logline', 'draft', 'box_office', 'casting'}} — \
pass the full texts from the earlier stages, not summaries. It returns `title` and \
`path`. Then `stop_file_writer`.
5. Wrap up: report the title, the logline, how many critique cycles the room ran, and \
where the pitch document was filed.

Announce each stage in a sentence or two as it starts. If a call fails, report the \
error plainly rather than retrying the same call."""


def build_showrunner(model: str | Model) -> TemporalAgent:
    """Build the showrunner TemporalAgent on `model` (a provider string, or a `Model`
    instance such as TestModel in tests)."""
    return TemporalAgent(
        Agent(
            model,
            instructions=SHOWRUNNER_INSTRUCTIONS,
            deps_type=HarnessDeps,
            toolsets=[_TEAM_TOOLSET],
        ),
        name="film_concept_team",
        event_stream_handler=harness_event_stream_handler,
        tool_activity_config=_TEAM_TOOL_CONFIG,
    )


# Built once at module load — its activities are registered on the worker via
# AgentPlugin (see worker.py). The model comes through `staff.DEFAULT_MODEL` (a
# pass-through module attribute) so a test can swap in a scripted model by rebinding
# that one name before the worker loads this module.
SHOWRUNNER = build_showrunner(staff.DEFAULT_MODEL)


@agent.defn(name="FilmConceptTeam")
class FilmConceptTeamWorkflow:
    """The SequentialAgent: a conversational showrunner sequencing the three stages."""

    @agent.init
    def __init__(self, config: AgentConfig) -> None:
        self._runner = AgentWorkflowRunner(
            config,
            stream=WorkflowStream(),
            # The showrunner's only tools are the generated subagent tools: allow-list
            # them by name (plus anything inherently safe) rather than skipping the gate
            # entirely — the pipeline runs unattended, but nothing else can.
            approval_policy_default=ToolApprovalPolicy.allow_tools(
                _SUBAGENT_TOOL_NAMES, also_inherently_safe=True
            ),
        )
        self._history: list[ModelMessage] = []

    @agent.accepts(mid_turn=MidTurn.ENQUEUE)
    async def ask(self, message: TextMessage) -> TextReply:
        """Chat with the showrunner. It greets you and asks for a historical subject,
        then runs the film-concept pipeline: writers room, preproduction, file writer —
        narrating each stage and finishing with the title and the filed pitch path."""
        result = await SHOWRUNNER.run(
            message.text,
            deps=HarnessDeps(runner=self._runner),
            message_history=self._history,
        )
        self._history = result.all_messages()
        return TextReply(text=str(result.output))
