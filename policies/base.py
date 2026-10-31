"""A context policy decides which messages are sent to the model on each turn.

`history` is the full, append-only record of the episode. Each entry is a dict with
"role", "content" and "kind" (system | task | action | observation). The loop strips
"kind" before sending. A policy must NOT modify `history`; it returns a new list.
"""
from __future__ import annotations


class Policy:
    name = "base"

    def build(self, history: list[dict], turn: int) -> list[dict]:
        raise NotImplementedError