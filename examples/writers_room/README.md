# Writers room (ADK-style multi-agent team on Pydantic AI)

A film-concept team reproducing the classic ADK `film_concept_team` diagram — greeter →
writers-room loop → parallel preproduction → file writer — on the harness's
**Pydantic AI** integration (`temporal_agent_harness.ai_sdks.pydantic_ai_harness`).

```
FilmConceptTeam            — SequentialAgent + greeter (model-driven, user-facing)
  └─ WritersRoom           — LoopAgent (deterministic workflow code)
  │     researcher ──► [screenwriter ◄─► critic] × up to 3
  └─ PreproductionTeam     — ParallelAgent (asyncio.gather)
  │     box_office_researcher ∥ casting_agent
  └─ FileWriter            — model-driven agent with a write_pitch_file tool
```

## What it demonstrates

- **The ADK agent shapes, mapped onto the harness.** The three *containers* are real
  harness agent workflows (`@agent.defn`); the six *leaf* agents are ordinary
  `TemporalAgent`s invoked inside them — the same way an ADK `SequentialAgent` /
  `LoopAgent` / `ParallelAgent` invokes the agents it contains.
- **Agents as subagents.** The showrunner doesn't know the containers' internals: it
  drives them through generated tools (`start_writers_room`, `writers_room_pitch`,
  `stop_writers_room`, …) produced by `agent.subagent_toolset`, adapted onto Pydantic AI
  with `build_harness_toolset`. Each stage is a real child workflow with its own turn
  stream — a UI can mount it live via the `workflow_id` in the `subagent_started` event.
- **LoopAgent = a Python loop.** `WritersRoomWorkflow.run` calls the screenwriter and
  critic `TemporalAgent`s in a `for` loop and exits when the critic's *structured output*
  (`Critique.approved`) says so — the loop is deterministic workflow code, not a model's
  improvisation.
- **ParallelAgent = `asyncio.gather`.** `PreproductionTeamWorkflow.run` gathers the
  box-office and casting `TemporalAgent.run` calls — two model-request activities in
  flight concurrently.
- **Structured outputs.** The screenwriter returns a `Treatment`, the critic a
  `Critique`, the file writer a `FileWriteResult` — Pydantic AI `output_type`s that
  become the pipeline's typed hand-offs.
- **Tools inside a subagent.** The researcher's `search_wikipedia` and the file writer's
  `write_pitch_file` are harness activity tools (`@agent.activity_tool_defn`) adapted
  onto their leaf agents — approval gating and `tool_start`/`tool_end` events included,
  resolved against each child workflow's own runner via `HarnessDeps`.

## Layout

| File | Role |
|---|---|
| `models.py` | The typed hand-offs between stages (`PitchRequest`, `Treatment`, `Critique`, `PitchResult`, `PreproductionResult`, `ScreenplayPackage`, `FileWriteResult`). |
| `tools.py` | `search_wikipedia` + `write_pitch_file` — activity tools (real I/O, worker-side bodies). |
| `staff.py` | The six leaf `TemporalAgent`s, built by `build_staff(model)`. The module-level `STAFF` is what the containers call. |
| `workflow.py` | The four `@agent.defn` containers: `FilmConceptTeam`, `WritersRoom`, `PreproductionTeam`, `FileWriter`. |
| `worker.py` | One worker, one task queue: all four workflows + `AgentPlugin` per `TemporalAgent`, plus `AgentHarnessPlugin(tools=[...])`. |
| `agents.toml` | Registry entry for the showrunner (the only user-facing agent) in the shared web UI. |

## Run it

Prereq: from the repo root, `cp .env.example .env.local` and set `OPENAI_API_KEY` (and
your Temporal connection profile). Then, each in its own terminal:

```sh
just temporal          # 1. local Temporal dev server (or bring your own)
just session-manager   # 2. packaged session-manager worker
just server            # 3. serves API + UI on http://localhost:8000
just worker            # 4. the writers-room worker (all four agents)
```

Open http://localhost:8000, pick **Writers Room**, and chat — e.g. *"Ada Lovelace"*. The
showrunner greets you, then narrates each stage: the room researches on Wikipedia,
drafts, and iterates against the critic; preproduction fans out; the file writer writes
`writers_room_output/<slug>.md` next to the worker and the showrunner reports the path.

To gate the pipeline on human approvals instead, start the session with a stricter
`ToolApprovalPolicy` (the showrunner's default allow-lists only its subagent tools), or
remove entries from `allow_tools(...)` in `workflow.py`.

Without `just`, the equivalent commands (from the repo root):

```sh
uv run --group examples temporal-agent-harness session-manager
uv run --group examples temporal-agent-harness serve examples/writers_room/agents.toml --host 0.0.0.0 --port 8000
uv run --group examples python -m examples.writers_room.worker
```

## Test it

`just test` (or `uv run --group examples pytest tests/examples/writers_room -v`) runs a
fully deterministic end-to-end of the whole graph in a time-skipping Temporal env: every
staff member and the showrunner run on scripted `TestModel`/`FunctionModel` models, so
the pipeline — greeter, subagent start/run/stop calls, the critic loop, the parallel
fan-out, the file write — executes with no API key and no network.
