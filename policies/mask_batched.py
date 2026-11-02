"""P3: batched observation masking. Same edit as P2, but only every N turns.

Between folds the policy is append-only, so the cache stays warm. On a fold turn it masks
every tool output except the last K, in one go: one cache break per N turns instead of N.
"""
from __future__ import annotations

from policies.base import Policy, masked


class BatchedMask(Policy):
    def __init__(self, every: int = 6, k: int = 4) -> None:
        super().__init__()
        self.every, self.k = every, k
        self.n_masked = 0  # how many of the oldest observations are currently masked
        self.name = f"P3_batch{every}"

    def build(self, history: list[dict], turn: int) -> list[dict]:
        obs = [i for i, m in enumerate(history) if m["kind"] == "observation"]
        if turn % self.every == 0:  # fold turn: catch up in one edit
            self.n_masked = max(self.n_masked, len(obs) - self.k)
        old = set(obs[:self.n_masked])
        return [masked(m) if i in old else m for i, m in enumerate(history)]