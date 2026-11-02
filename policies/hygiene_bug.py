"""P5: the "harmless" hygiene bug. Full history (like P0) but the system prompt starts with
the current time, which changes every turn. One changed line at token ~5 means the cache
can never be reused, even though nothing important changed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from agent.prompt import system_prompt
from policies.base import Policy

T0 = datetime(2026, 10, 6, 9, 0, 0)


class HygieneBug(Policy):
    name = "P5_timestamp"

    def build(self, history: list[dict], turn: int) -> list[dict]:
        now = T0 + timedelta(seconds=7 * turn)  # deterministic stand-in for datetime.now()
        header = f"Current time: {now:%Y-%m-%d %H:%M:%S} UTC\n"
        return [{**history[0], "content": system_prompt(header=header)}] + list(history[1:])