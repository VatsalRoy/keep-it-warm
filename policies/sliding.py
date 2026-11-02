"""P1: sliding window. Keep the system prompt, every question, and only the last K agent steps.

Once there are more than K steps, the oldest kept step changes EVERY turn, so the prompt
changes right after the system prompt and questions: the cache breaks every turn.
"""
from __future__ import annotations

from policies.base import Policy, pairs


class SlidingWindow(Policy):
    def __init__(self, k: int = 6) -> None:
        super().__init__()
        self.k = k
        self.name = f"P1_window{k}"

    def build(self, history: list[dict], turn: int) -> list[dict]:
        starts = pairs(history)
        if len(starts) <= self.k:
            return list(history)
        cutoff = starts[-self.k]  # first message of the oldest step we keep
        return [m for i, m in enumerate(history) if i >= cutoff or m["kind"] in ("system", "task")]