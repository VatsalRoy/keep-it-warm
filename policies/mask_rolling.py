"""P2: rolling observation masking. Tool outputs older than the last K are replaced by a short note.

The agent's own actions stay, so it remembers WHAT it did, just not the bulky outputs.
Every new tool output pushes one more old output out of the window, so the prompt changes
at that (early) position every turn: the cache breaks every turn.
"""
from __future__ import annotations

from policies.base import Policy, masked


class RollingMask(Policy):
    def __init__(self, k: int = 4) -> None:
        super().__init__()
        self.k = k
        self.name = f"P2_mask{k}"

    def build(self, history: list[dict], turn: int) -> list[dict]:
        obs = [i for i, m in enumerate(history) if m["kind"] == "observation"]
        old = set(obs[:-self.k]) if len(obs) > self.k else set()
        return [masked(m) if i in old else m for i, m in enumerate(history)]