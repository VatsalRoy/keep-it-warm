"""P0: send the full history every turn. Append-only, so the prompt prefix never changes."""
from __future__ import annotations

from policies.base import Policy


class FullHistory(Policy):
    name = "P0_full"

    def build(self, history: list[dict], turn: int) -> list[dict]:
        return list(history)