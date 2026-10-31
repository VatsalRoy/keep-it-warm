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

DATA_GUIDE = """\
How the data is organised (these are the ONLY files; do not guess other paths):
- /logs/<service>/<YYYY-MM-DD>.log  one file per service per UTC day. Services: api-gateway, auth,
  payments, orders, inventory, search, notifications, shipping. Dates: 2026-10-01 .. 2026-10-05.
- /deploys.csv   every deploy, sorted by time. Columns: deploy_id,service,timestamp,author,commit,notes
- /oncall.yaml   schedule -> "<date>" -> <service> -> primary / secondary
- /runbooks/*.md remediation steps, one section per error code ("## <code>: ...")

Every log line looks like:
  <timestamp> <LEVEL> <service> key=value key=value ...
  2026-10-09T08:00:00.000Z ERROR api-gateway req=r-1234 method=POST path=/v1/checkout status=502 latency_ms=926 code=E5002 msg="dependency failed: orders"
Useful keys: status=<HTTP code>, req=<request id> (the same id appears in every service that handled
the request), code=<error code>, msg=..., event=<name> for non-request events (for example
event=OOMKilled, event=config_reload, event=job_failed).
Code E5002 means "a dependency failed": the msg names the service that failed next, so follow the
chain until you reach a service whose error is NOT E5002 - that is where the failure originated."""


def system_prompt(header: str = "", tool_order: list[str] | None = None) -> str:
    """`header` is empty for every clean policy; P5 puts a timestamp there."""
    return (
        f"{header}"
        "You are an on-call engineer investigating an incident in a small company's systems.\n"
        "You can only see the systems through these tools:\n"
        f"{tool_lines(tool_order)}\n\n"
        f"{DATA_GUIDE}\n\n"
        "How to work:\n"
        "- grep takes ONE file and a regex matched against raw lines, e.g. 'status=5\\d\\d', 'event=OOMKilled',\n"
        "  'req=r-1234 '. To search several services or days, grep each file in turn.\n"
        "- read returns numbered lines; read whole small files (deploys.csv, oncall.yaml) in 60-line pages.\n"
        "- Timestamps are ISO-8601 UTC, so they sort as text. 'Preceded' means strictly earlier.\n"
        "- If a call returns an error or 0 matches, change the path or pattern; do not repeat it.\n"
        "- Never submit 'unknown' or 'none': every question has an answer in the data. Keep investigating.\n"
        "- Write your thought in English.\n\n"
        "Reply with exactly ONE JSON object and nothing else, in this form:\n"
        '{"thought": "<one short sentence>", "tool": "<ls|read|grep|submit>", "args": {...}}\n'
        'Example: {"thought": "find OOM events", "tool": "grep", '
        '"args": {"pattern": "event=OOMKilled", "path": "/logs/search/2026-10-03.log"}}\n'
        "When you know the answer, call submit with args {\"answer\": \"...\"} in the exact format the\n"
        "question asks for. You have at most 25 turns."
    )

def task_message(task: dict) -> str:
    return f"Question: {task['question']}"


def observation_message(tool: str, output: str) -> str:
    return f"[{tool} result]\n{output}"