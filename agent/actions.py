"""Parse the model's reply into (tool, args). Small models add extra text, so be forgiving."""
from __future__ import annotations

import json

TOOLS = {"ls", "read", "grep", "submit"}


class ActionError(ValueError):
    pass


def _first_json_object(text: str) -> str:
    """Return the first balanced {...} block, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        raise ActionError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ActionError("unbalanced JSON object")


def parse_action(text: str) -> tuple[str, dict]:
    try:
        obj = json.loads(_first_json_object(text))
    except json.JSONDecodeError as e:
        raise ActionError(f"invalid JSON: {e.msg}") from e
    tool = obj.get("tool") or obj.get("name")
    args = obj.get("args") if "args" in obj else obj.get("arguments", {})
    if tool not in TOOLS:
        raise ActionError(f"unknown tool {tool!r}; use one of ls, read, grep, submit")
    if not isinstance(args, dict):
        raise ActionError("args must be a JSON object")
    return tool, args