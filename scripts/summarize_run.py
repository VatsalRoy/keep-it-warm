"""Quick report for one run folder (turns.jsonl + episodes.jsonl).

    python scripts/summarize_run.py runs/baseline_smoke
    python scripts/summarize_run.py runs/baseline_smoke --episode t3-rootcause-1
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
    ap.add_argument("--episode", help="task id to show turn by turn")
    a = ap.parse_args()
    turns, eps = load(Path(a.run) / "turns.jsonl"), load(Path(a.run) / "episodes.jsonl")
    by_ep = defaultdict(list)
    for r in turns:
        by_ep[r["episode_id"]].append(r)

    print(f"{'task':<16} {'ok':<4} {'turns':>5} {'end':<16} {'peak_ctx':>8} {'cached':>7} "
          f"{'$A':>8} {'$B':>8} {'naive$B':>8} {'wall_s':>7}")
    tot = defaultdict(float)
    for e in eps:
        rs = by_ep[e["episode_id"]]
        p = sum(r["prompt_tokens"] for r in rs)
        c = sum(r["cached_tokens"] for r in rs)
        o = sum(r["completion_tokens"] for r in rs)
        peak = max((r["active_context_tokens"] for r in rs), default=0)
        ca, cb, nb = cost(p - c, c, o, "A"), cost(p - c, c, o, "B"), naive_cost(p, o, "B")
        print(f"{e['task_id']:<16} {'yes' if e['correct'] else 'NO':<4} {e['turns']:>5} {e['end_reason']:<16} "
              f"{peak:>8} {c / max(p, 1):>7.0%} {ca:>8.4f} {cb:>8.4f} {nb:>8.4f} {e['wall_s']:>7}")
        for k, v in [("p", p), ("c", c), ("o", o), ("a", ca), ("b", cb), ("nb", nb), ("wall", e["wall_s"]),
                     ("ok", e["correct"]), ("turns", e["turns"]), ("peak", peak)]:
            tot[k] += v
    n = max(len(eps), 1)
    print(f"\n{len(eps)} episodes | accuracy {tot['ok'] / n:.0%} | mean turns {tot['turns'] / n:.1f} | "
          f"mean peak context {tot['peak'] / n:,.0f} tokens | cached {tot['c'] / max(tot['p'], 1):.0%}")
    print(f"cost: sheet A ${tot['a']:.4f}, sheet B ${tot['b']:.4f} (naive B, ignoring cache: ${tot['nb']:.4f})")
    print(f"GPU wall time {tot['wall'] / 60:.1f} min total, {tot['wall'] / n:.0f} s per episode on average")

    # Prefix check: for an append-only policy, a prompt should never change before the previous end.
    broken = [r for r in turns if r["first_changed_token_idx"] is not None
              and r["first_changed_token_idx"] < r["prev_prompt_tokens"]]
    print(f"\nprefix edits (first_changed < previous length): {len(broken)} of "
          f"{sum(r['first_changed_token_idx'] is not None for r in turns)} turns")
    for r in broken[:5]:
        print(f"  {r['task_id']} turn {r['turn']}: changed at token {r['first_changed_token_idx']} "
              f"of {r['prev_prompt_tokens']}")
    bad = [r for r in turns if not r["ok"]]
    print(f"unparseable replies: {len(bad)}" + (f" (e.g. {bad[0].get('error')})" if bad else ""))

    if a.episode:
        e = next((x for x in eps if x["task_id"] == a.episode), None)
        if e is None:
            sys.exit(f"\nno episode for task {a.episode!r} in this run")
        print(f"\n{a.episode}: turn | prompt | cached | uncached | first_changed/prev | ttft_ms | action")
        for r in by_ep[e["episode_id"]]:
            print(f"  {r['turn']:>3} | {r['prompt_tokens']:>6} | {r['cached_tokens']:>6} | {r['uncached_tokens']:>6} | "
                  f"{r['first_changed_token_idx']}/{r['prev_prompt_tokens']} | {r['ttft_ms']:>8} | {r['action']}")


if __name__ == "__main__":
    main()