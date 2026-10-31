"""Price sheets (USD per 1M tokens) and per-request cost.

A: GPT-OSS-120B-style sheet from the AKV paper (Table 6): mild 2x cache discount.
B: premium-API sheet (Claude Opus 4.7 nominal rates in CarryOn, Table 8): steep 10x discount.
"""
from __future__ import annotations

SHEETS = {
    "A": {"miss": 0.150, "hit": 0.075, "out": 0.600},
    "B": {"miss": 5.00, "hit": 0.50, "out": 25.00},
}


def cost(uncached: int, cached: int, output: int, sheet: str) -> float:
    """True cost: cached input is billed at the cheaper 'hit' price."""
    p = SHEETS[sheet]
    return (uncached * p["miss"] + cached * p["hit"] + output * p["out"]) / 1e6


def naive_cost(prompt_tokens: int, output: int, sheet: str) -> float:
    """What you'd estimate from token counts alone, ignoring the cache (H1's 'naive' column)."""
    p = SHEETS[sheet]
    return (prompt_tokens * p["miss"] + output * p["out"]) / 1e6