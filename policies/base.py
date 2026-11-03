"""A context policy decides which messages are sent to the model on each turn.

`history` is the full, append-only record of the episode. Each entry is a dict with
"role", "content" and "kind" (system | task | action | observation). The loop strips
"kind" before sending. A policy must NOT modify `history`; it returns a new list.

A fresh policy object is created for every episode, so policies may keep small amounts of
state (e.g. when they last folded). Policies that make their own model calls (summaries)
put them in `self.calls` so the loop can log and bill them like any other request.
"""
from __future__ import annotations


class Policy:
    name = "base"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], object]] = []  # (messages sent, Reply) for extra model calls
        self.notes: list[dict] = []  # anything worth saving with the episode (e.g. P4's summaries)

    def build(self, history: list[dict], turn: int) -> list[dict]:
        raise NotImplementedError

    def drain_calls(self) -> list[tuple[list[dict], object]]:
        calls, self.calls = self.calls, []
        return calls


def pairs(history: list[dict]) -> list[int]:
    """Indices of 'action' messages; each starts one agent step (action + its observation)."""
    return [i for i, m in enumerate(history) if m["kind"] == "action"]


MASK = "[output elided to save context; re-run the tool if you need it again]"


def masked(m: dict) -> dict:
    """An observation with its bulky body replaced by a short note (first line kept for orientation)."""
    first = m["content"].split("\n", 1)[0]
    return {**m, "content": f"{first}\n{MASK}"}