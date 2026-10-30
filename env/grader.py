"""Exact-match grading. No LLM judge.

Answers are comma- or newline-separated items. Each item is normalized (case, whitespace,
surrounding quotes/backticks, trailing period) and compared to the gold list as:
  exact   - a single item, must be equal
  int     - the first integer in the answer must equal the gold integer
  ordered - same items, same order
  set     - same items, any order
"""
from __future__ import annotations

import re


def norm(item: str) -> str:
    s = item.strip()
    s = re.sub(r"^(final\s+)?answer\s*:\s*", "", s, flags=re.I)
    for _ in range(2):  # twice, so "`cmd`." and ".`cmd`" both clean up
        s = s.strip().rstrip(".").strip().strip("`'\"")
    return re.sub(r"\s+", " ", s).lower()


def split_items(answer: str) -> list[str]:
    parts = re.split(r"[,\n]|\s->\s|\s→\s", answer)
    return [n for n in (norm(p) for p in parts) if n]


def grade(task: dict, submitted: str | None) -> bool:
    if not submitted:
        return False
    gold = [norm(g) for g in task["answer"]]
    match = task["match"]
    if match == "int":
        m = re.search(r"-?\d+", submitted)
        return bool(m) and int(m.group()) == int(gold[0])
    if match == "exact":
        return norm(submitted) == gold[0]
    items = split_items(submitted)
    if match == "ordered":
        return items == gold
    if match == "set":
        return sorted(items) == sorted(gold)
    raise ValueError(f"unknown match type {match}")