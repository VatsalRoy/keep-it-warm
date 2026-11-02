"""Group the 24 tasks into 12 "sessions" of 4 questions each.

A session is one long conversation: the agent answers question 1, then question 2 is added to
the same history, and so on, like an on-call engineer working through a ticket queue.

Construction (so position effects can be measured cleanly, for H4):
  round 1 (sessions s01-s06): a seeded shuffle of all 24 tasks, cut into 6 sessions of 4.
  round 2 (sessions s07-s12): every task again, at the MIRRORED position (0<->3, 1<->2) and
                              with different companions.
So each task is asked exactly twice: once early in a session and once late.

    python -m env.make_sessions
"""
from __future__ import annotations

import argparse
import json
import random

SESSION_LEN = 4


def build_sessions(task_ids: list[str], seed: int = 42) -> list[dict]:
    rng = random.Random(seed * 1000 + 99)
    ids = sorted(task_ids)
    rng.shuffle(ids)
    n = len(ids) // SESSION_LEN
    round1 = [ids[i * SESSION_LEN:(i + 1) * SESSION_LEN] for i in range(n)]
    for _ in range(1000):  # retry until no round-2 session repeats a round-1 pair of companions
        round2 = [[None] * SESSION_LEN for _ in range(n)]
        for p in range(SESSION_LEN):
            col = [s[p] for s in round1]
            rng.shuffle(col)
            for k, t in enumerate(col):
                round2[k][SESSION_LEN - 1 - p] = t
        pairs1 = {frozenset((a, b)) for s in round1 for a in s for b in s if a != b}
        if not any(frozenset((a, b)) in pairs1 for s in round2 for a in s for b in s if a != b):
            break
    sessions = round1 + round2
    return [{"id": f"s{i + 1:02d}", "tasks": s} for i, s in enumerate(sessions)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="tasks/sessions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ids = [json.loads(line)["id"] for line in open(a.tasks, encoding="utf-8")]
    sessions = build_sessions(ids, a.seed)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    for s in sessions:
        print(s["id"], " | ".join(s["tasks"]))


if __name__ == "__main__":
    main()