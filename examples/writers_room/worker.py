"""Worker for the writers-room example.

Run from the repo root with:
    uv run --group examples python -m examples.writers_room.worker

One worker, one task queue, all four agents: the showrunner (FilmConceptTeam) and the
three containers it starts as subagents (WritersRoom, PreproductionTeam, FileWriter).
A single worker hosts the whole tree because the showrunner's generated subagent tools
start children on this same queue.

Plugins, like the other examples:

  * ``PydanticAIPlugin`` on the CLIENT — the Pydantic-compatible data converter and the
    workflow-sandbox passthroughs the SDK needs.
  * ``AgentHarnessPlugin(tools=[search_wikipedia, write_pitch_file])`` on the CLIENT —
    registers the two activity tools' worker-side bodies AND the subagent-turn activity
    every ``subagent_toolset``-driving agent needs.
  * one ``AgentPlugin`` per TemporalAgent on the WORKER — registers each durable agent's
    activities (model request/stream, event_stream_handler, call_tool). Seven agents:
    the showrunner plus the six staff members.

Env vars (set in .env.local — see .env.example):
    TEMPORAL_CONFIG_FILE / TEMPORAL_PROFILE   Temporal connection profile
    OPENAI_API_KEY                            required — the staff call the OpenAI API
    WRITERS_ROOM_TASK_QUEUE                   task queue to poll (default: writers-room)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from temporal_agent_harness.plugin import AgentHarnessPlugin

from . import staff
from .tools import search_wikipedia, write_pitch_file
from .workflow import (
    SHOWRUNNER,
    TASK_QUEUE,
    FileWriterWorkflow,
    FilmConceptTeamWorkflow,
    PreproductionTeamWorkflow,
    WritersRoomWorkflow,
)

# Every TemporalAgent whose activities this worker registers: the showrunner plus the
# six staff members. The names are unique (they prefix each activity name).
ALL_AGENTS = [SHOWRUNNER, *staff.STAFF.values()]


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    task_queue = os.environ.get("WRITERS_ROOM_TASK_QUEUE", TASK_QUEUE)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("error: OPENAI_API_KEY env var not set")

    # PydanticAIPlugin supplies the Pydantic-compatible data converter + sandbox
    # passthroughs; AgentHarnessPlugin (last) leaves that converter in place and adds the
    # harness's own requirements — including the two tool activities and the
    # run_subagent_turn activity the showrunner's subagent tools dispatch through.
    connect_config = ClientConfig.load_client_connect_config()
    client = await Client.connect(
        **connect_config,
        plugins=[
            PydanticAIPlugin(),
            AgentHarnessPlugin(tools=[search_wikipedia, write_pitch_file]),
        ],
    )

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[
            FilmConceptTeamWorkflow,
            WritersRoomWorkflow,
            PreproductionTeamWorkflow,
            FileWriterWorkflow,
        ],
        plugins=[AgentPlugin(durable) for durable in ALL_AGENTS],
    )
    print(
        f"Writers-room worker ready: "
        f"profile={os.environ.get('TEMPORAL_PROFILE', 'default')!r} "
        f"address={connect_config.get('target_host')} "
        f"namespace={connect_config.get('namespace')} "
        f"taskQueue={task_queue}",
        flush=True,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
