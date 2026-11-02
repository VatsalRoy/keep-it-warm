"""Quick report for one run folder (turns.jsonl + episodes.jsonl).

    python scripts/summarize_run.py runs/baseline_sessions
    python scripts/summarize_run.py runs/baseline_sessions --episode s03
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.pricing import cost, naive_cost  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--episode", help="session id (or task id) to show turn by turn")
    a = ap.parse_args()
    turns, eps = load(Path(a.run) / "turns.jsonl"), load(Path(a.run) / "episodes.jsonl")
    by_ep = defaultdict(list)
    for r in turns:
        by_ep[r["episode_id"]].append(r)

    print(f"{'session':<14} {'correct':>7} {'turns':>5} {'end':<16} {'peak_ctx':>8} {'cached':>7} "
          f"{'$A':>8} {'$B':>8} {'naive$B':>8} {'wall_s':>7}")
    tot = defaultdict(float)
    for e in eps:
        rs = by_ep[e["episode_id"]]
        p = sum(r["prompt_tokens"] for r in rs)
        c = sum(r["cached_tokens"] for r in rs)
        o = sum(r["completion_tokens"] for r in rs)
        peak = max((r["active_context_tokens"] for r in rs), default=0)
        ca, cb, nb = cost(p - c, c, o, "A"), cost(p - c, c, o, "B"), naive_cost(p, o, "B")
        print(f"{e['session_id']:<14} {e['n_correct']:>3}/{e['n_questions']:<3} {e['turns']:>5} {e['end_reason']:<16} "
              f"{peak:>8} {c / max(p, 1):>7.0%} {ca:>8.4f} {cb:>8.4f} {nb:>8.4f} {e['wall_s']:>7}")
        for k, v in [("p", p), ("c", c), ("a", ca), ("b", cb), ("nb", nb), ("wall", e["wall_s"]),
                     ("ok", e["n_correct"]), ("q", e["n_questions"]), ("peak", peak)]:
            tot[k] += v
    n = max(len(eps), 1)
    solved = max(tot["ok"], 1)
    print(f"\n{len(eps)} sessions, {tot['q']:.0f} questions | accuracy {tot['ok'] / max(tot['q'], 1):.0%} "
          f"| mean peak context {tot['peak'] / n:,.0f} tokens | cached {tot['c'] / max(tot['p'], 1):.0%}")
    print(f"cost: sheet A ${tot['a']:.4f}, sheet B ${tot['b']:.4f} (naive B, ignoring cache: ${tot['nb']:.4f})")
    print(f"cost per solved question: sheet A ${tot['a'] / solved:.5f}, sheet B ${tot['b'] / solved:.4f}")
    print(f"GPU wall time {tot['wall'] / 60:.1f} min total, {tot['wall'] / n:.0f} s per session on average")

    qs = [q for e in eps for q in e["questions"]]
    for key in ["tier", "position", "end"]:
        groups = defaultdict(list)
        for x in qs:
            groups[x[key]].append(x["correct"])
        print(f"accuracy by {key}: " + ", ".join(f"{k}={sum(v)}/{len(v)}" for k, v in sorted(groups.items())))

    ends = defaultdict(int)
    for e in eps:
        ends[e["end_reason"]] += 1
    print("session end reasons: " + ", ".join(f"{k}={v}" for k, v in ends.items()))

    # Prefix check: for an append-only policy, a prompt should never change before the previous end.
    broken = [r for r in turns if r["first_changed_token_idx"] is not None
              and r["first_changed_token_idx"] < r["prev_prompt_tokens"]]
    print(f"\nprefix edits (first_changed < previous length): {len(broken)} of "
          f"{sum(r['first_changed_token_idx'] is not None for r in turns)} turns")
    for r in broken[:5]:
        print(f"  {r['session_id']} turn {r['turn']}: changed at token {r['first_changed_token_idx']} "
              f"of {r['prev_prompt_tokens']}")
    bad = [r for r in turns if not r["ok"]]
    print(f"unparseable replies: {len(bad)}" + (f" (e.g. {bad[0].get('error')})" if bad else ""))

    if a.episode:
        e = next((x for x in eps if x["session_id"] == a.episode), None)
        if e is None:
            sys.exit(f"\nno episode {a.episode!r} in this run")
        print(f"\n{a.episode}: turn | q | prompt | cached | uncached | first_changed/prev | ttft_ms | action")
        for r in by_ep[e["episode_id"]]:
            print(f"  {r['turn']:>3} | {r['question_idx']} | {r['prompt_tokens']:>6} | {r['cached_tokens']:>6} | "
                  f"{r['uncached_tokens']:>6} | {r['first_changed_token_idx']}/{r['prev_prompt_tokens']} | "
                  f"{r['ttft_ms']:>8} | {r['action']}")
        for x in e["questions"]:
            print(f"  Q{x['position'] + 1} {x['task_id']:<16} {'OK ' if x['correct'] else 'BAD'} turns={x['turns']:<3} "
                  f"end={x['end']:<20} answer={x['submitted']!r}")


if __name__ == "__main__":
    main()