"""Worker-side activity tools for the writers-room example.

Both tools do real I/O — one HTTP call, one filesystem write — so they are durable,
activity-backed tools (``@agent.activity_tool_defn``), never inline in a workflow. The
researcher's ``search_wikipedia`` is the ``[wikipedia tool]`` box in the diagram;
``write_pitch_file`` is the file writer's write.

The tool schema the model sees is derived from each function's signature and docstring. The
worker registers each activity body by handing both tools to
``AgentHarnessPlugin(tools=...)`` (see ``worker.py``).

No ``from __future__ import annotations`` — activity args/returns cross Temporal's data
converter, and stringized annotations can trip its type resolution (mirrors the other
examples' activities modules).
"""

from datetime import timedelta
from pathlib import Path

import httpx
from temporalio.workflow import ActivityConfig

from temporal_agent_harness.harness import agent

_TOOL_TIMEOUT = ActivityConfig(start_to_close_timeout=timedelta(seconds=30))
_HTTP_TIMEOUT = 10.0

# Where the file writer writes, relative to the worker's working directory.
OUTPUT_DIR = Path("writers_room_output")


@agent.activity_tool_defn(
    inherently_safe=True,
    activity_config=_TOOL_TIMEOUT,
)
async def search_wikipedia(query: str) -> str:
    """Search Wikipedia for `query` and return the summary of the best-matching article.

    Args:
        query: a person, place, or topic, e.g. "Ada Lovelace".
    """
    async with httpx.AsyncClient() as client:
        # Resolve the query to an article title, then fetch that article's summary.
        opensearch = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            timeout=_HTTP_TIMEOUT,
        )
        opensearch.raise_for_status()
        titles = opensearch.json()[1]
        if not titles:
            return f"No Wikipedia article found for {query!r}."
        summary = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{titles[0]}",
            timeout=_HTTP_TIMEOUT,
        )
        summary.raise_for_status()
        data = summary.json()
    extract = data.get("extract") or "(no summary available)"
    return f"{data.get('title', titles[0])}: {extract}"


@agent.activity_tool_defn(
    inherently_safe=True,
    activity_config=_TOOL_TIMEOUT,
)
async def write_pitch_file(filename: str, contents: str) -> str:
    """Write `contents` to a markdown file under the worker's writers_room_output directory
    and return the absolute path written. `filename` is a short slug, e.g. "the-analytical-engine".
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in filename.lower()
    ).strip("_")
    path = OUTPUT_DIR / f"{safe or 'pitch'}.md"
    path.write_text(contents)
    return str(path.resolve())
