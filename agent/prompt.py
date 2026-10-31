"""The system prompt and task message the agent starts with.

The agent replies with ONE JSON object per turn (an "action"), e.g.
    {"thought": "check the deploy list", "tool": "grep", "args": {"pattern": "inventory", "path": "/deploys.csv"}}

We use plain-text JSON actions instead of the server's tool-calling feature so that:
  - every byte the model sees is in the messages we build (easy to test for prefix changes),
  - it works on any OpenAI-compatible server with no tool-parser flags,
  - time-to-first-token is measured on real text, not on a parser's buffered output.
"""
from __future__ import annotations

from env.tools import TOOL_SCHEMAS


def tool_lines(order: list[str] | None = None) -> str:
    """One line per tool. `order` exists only so the P5 hygiene bug can shuffle it."""
    by_name = {t["function"]["name"]: t["function"] for t in TOOL_SCHEMAS}
    names = order or [t["function"]["name"] for t in TOOL_SCHEMAS]
    rows = []
    for n in names:
        f = by_name[n]
        params = ", ".join(f"{p}: {v['type']}" for p, v in f["parameters"]["properties"].items())
        rows.append(f"- {n}({params}): {f['description']}")
    return "\n".join(rows)


def system_prompt(header: str = "", tool_order: list[str] | None = None) -> str:
    """`header` is empty for every clean policy; P5 puts a timestamp there."""
    return (
        f"{header}"
        "You are an on-call engineer investigating an incident in a small company's systems.\n"
        "You can only see the systems through these tools:\n"
        f"{tool_lines(tool_order)}\n\n"
        "Start with ls('/') and /README.md if you need orientation. Log files are large: prefer grep\n"
        "to find lines, then read around the line numbers it returns.\n\n"
        "Reply with exactly ONE JSON object and nothing else, in this form:\n"
        '{"thought": "<one short sentence>", "tool": "<ls|read|grep|submit>", "args": {...}}\n'
        'Example: {"thought": "find OOM events", "tool": "grep", '
        '"args": {"pattern": "OOMKilled", "path": "/logs/search/2026-10-03.log"}}\n'
        "When you know the answer, call submit with args {\"answer\": \"...\"} in the exact format the\n"
        "question asks for. You have a limited number of turns."
    )


def task_message(task: dict) -> str:
    return f"Question: {task['question']}"


def observation_message(tool: str, output: str) -> str:
    return f"[{tool} result]\n{output}"