"""The film-concept team's staff: the six leaf Pydantic AI agents.

Each is a ``TemporalAgent`` — the upstream Pydantic AI Temporal plugin wraps a plain
``Agent`` so its model requests run as Temporal activities — wired with the harness's
``harness_event_stream_handler`` so streamed tokens publish onto whichever turn drives it.
None of them is a workflow: they are invoked from the container workflows in
``workflow.py``, the same way an ADK Sequential/Loop/Parallel agent invokes the agents it
contains. The containers supply the per-run runner through
``deps=HarnessDeps(runner=...)`` (see ``pydantic_ai_harness``), so these agents are built
ONCE and shared by every concurrent workflow.

``build_staff`` takes the model so a test (or an operator) can swap every agent's model in
one place — a provider string for production, a ``Model`` instance such as
``pydantic_ai.models.test.TestModel`` for tests.
"""

from __future__ import annotations

from typing import Mapping

# NOTE on imports: this module is only ever imported INSIDE workflow.py's
# `workflow.unsafe.imports_passed_through()` block (workflow side) or directly by
# worker.py/test code (non-sandbox side), so the pydantic_ai imports below need no
# pass-through of their own — they are ordinary imports here.

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.models import Model

from temporal_agent_harness.ai_sdks.pydantic_ai_harness import (
    HarnessDeps,
    build_harness_toolset,
    harness_event_stream_handler,
)

from .models import Critique, FileWriteResult, Treatment
from .tools import search_wikipedia, write_pitch_file

DEFAULT_MODEL = "openai:gpt-5.1"

RESEARCHER_INSTRUCTIONS = """\
You are a film researcher. Given a historical subject, call the `search_wikipedia` tool \
to gather material, then write a tight research brief: the key facts, the most dramatic \
episodes, the central conflict, and two or three vivid details a screenwriter could build \
scenes around. About 150 words. Call the tool — don't rely on memory."""

SCREENWRITER_INSTRUCTIONS = """\
You are a screenwriter. Given a research brief about a historical subject, write a \
feature-film concept: a one-sentence logline plus a three-act treatment of about 150 \
words (setup, escalation, resolution). When you receive critic notes, rewrite the \
treatment to address every one of them, keeping the same format."""

CRITIC_INSTRUCTIONS = """\
You are a brutally honest development critic reviewing a film treatment. Approve only if \
it is genuinely compelling: a clear protagonist, escalating stakes, a satisfying arc, and \
a reason this story matters. If you approve, set `approved` and leave one line of praise \
in `notes`. Otherwise leave `approved` false and give up to three specific, actionable \
notes for the next draft."""

BOX_OFFICE_INSTRUCTIONS = """\
You are a box-office analyst. Given a logline and treatment for a historical film, name \
two or three comparable released films and estimate a worldwide box-office range with a \
one-line rationale. About 80 words."""

CASTING_INSTRUCTIONS = """\
You are a casting director. Given a logline and treatment for a historical film, propose \
casting for the two or three principal characters with one line on why each fits. \
About 80 words."""

FILE_WRITER_INSTRUCTIONS = """\
You are a development executive. You receive the full package for a film concept: the \
subject, logline, treatment, box-office analysis, and casting. Assemble a polished \
markdown pitch document — a `##` section each for the logline, the synopsis, the market \
analysis, and the casting — and save it by calling the `write_pitch_file` tool with a \
short slug for `filename` (e.g. "the-analytical-engine") and the document as `contents`. \
Invent a memorable title. Return the title you chose and the path the tool reports."""


def _durable(
    inner: Agent,
    name: str,
    tool_config: dict | None = None,
) -> TemporalAgent:
    """Wrap one staff agent for Temporal, wiring the harness streaming seam."""
    return TemporalAgent(
        inner,
        name=name,
        event_stream_handler=harness_event_stream_handler,
        tool_activity_config=tool_config,  # run harness tools in-workflow, not in an activity
    )


def build_staff(model: str | Model | Mapping[str, str | Model]) -> dict[str, TemporalAgent]:
    """Build the six leaf agents. `model` is one model for all of them, or a
    ``{staff_name: model}`` mapping to give individual agents their own (as a test does to
    script each role differently)."""

    def pick(name: str) -> str | Model:
        return model.get(name, DEFAULT_MODEL) if isinstance(model, Mapping) else model

    # Harness tools adapted onto the SDK. build_harness_toolset returns the matching
    # tool_activity_config that disables Pydantic AI's per-tool activity wrapper, so the
    # calls run in-workflow where the harness approval gate and tool events live.
    research_toolset, research_tool_config = build_harness_toolset(
        [search_wikipedia], id="research-tools"
    )
    file_toolset, file_tool_config = build_harness_toolset(
        [write_pitch_file], id="file-tools"
    )

    return {
        "researcher": _durable(
            Agent(
                pick("researcher"),
                instructions=RESEARCHER_INSTRUCTIONS,
                deps_type=HarnessDeps,
                toolsets=[research_toolset],
            ),
            "researcher",
            research_tool_config,
        ),
        "screenwriter": _durable(
            Agent(
                pick("screenwriter"),
                instructions=SCREENWRITER_INSTRUCTIONS,
                deps_type=HarnessDeps,
                output_type=Treatment,
            ),
            "screenwriter",
        ),
        "critic": _durable(
            Agent(
                pick("critic"),
                instructions=CRITIC_INSTRUCTIONS,
                deps_type=HarnessDeps,
                output_type=Critique,
            ),
            "critic",
        ),
        "box_office": _durable(
            Agent(
                pick("box_office"),
                instructions=BOX_OFFICE_INSTRUCTIONS,
                deps_type=HarnessDeps,
            ),
            "box_office_researcher",
        ),
        "casting": _durable(
            Agent(
                pick("casting"),
                instructions=CASTING_INSTRUCTIONS,
                deps_type=HarnessDeps,
            ),
            "casting_agent",
        ),
        "file_writer": _durable(
            Agent(
                pick("file_writer"),
                instructions=FILE_WRITER_INSTRUCTIONS,
                deps_type=HarnessDeps,
                toolsets=[file_toolset],
                output_type=FileWriteResult,
            ),
            "file_writer",
            file_tool_config,
        ),
    }


# The production staff, built once at module load on the default model. Container
# workflows reference it as `staff.STAFF` at call time. In the workflow sandbox this
# module is a restricted copy — the agents built there only dispatch model/tool calls
# to worker-registered activities by name, so what actually runs is whatever agents
# the worker registers (a real cast in production, scripted models in tests).
STAFF = build_staff(DEFAULT_MODEL)
