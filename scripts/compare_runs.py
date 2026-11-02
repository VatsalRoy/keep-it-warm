"""Side-by-side table of several run folders (one per policy).

    python scripts/compare_runs.py runs/fake_full runs/fake_window runs/fake_mask ...
    python scripts/compare_runs.py --sessions s01,s05,s07,s12 runs/main/P0_full runs/main/P5_timestamp

The first folder is the reference (normally P0): the last columns show the change vs it.
--sessions restricts every folder to the same sessions, so subsets are compared fairly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.pricing import cost, naive_cost  # noqa: E402


def stats(run: Path, only: set[str] | None = None) -> dict:
    turns = [json.loads(line) for line in open(run / "turns.jsonl", encoding="utf-8")]
    eps = [json.loads(line) for line in open(run / "episodes.jsonl", encoding="utf-8")]
    if only:
        turns = [r for r in turns if r["session_id"] in only]
        eps = [e for e in eps if e["session_id"] in only]
    p = sum(r["prompt_tokens"] for r in turns)
    c = sum(r["cached_tokens"] for r in turns)
    o = sum(r["completion_tokens"] for r in turns)
    solved = sum(e["n_correct"] for e in eps)
    asked = sum(e["n_questions"] for e in eps)
    peaks = {}
    for r in turns:
        peaks[r["episode_id"]] = max(peaks.get(r["episode_id"], 0), r["active_context_tokens"])
    return {
        "policy": turns[0]["policy"] if turns else run.name, "sessions": len(eps), "acc": solved / max(asked, 1),
        "turns": len(turns), "peak": sum(peaks.values()) / max(len(peaks), 1), "prompt": p, "cached": c / max(p, 1),
        "naiveB": naive_cost(p, o, "B"), "A": cost(p - c, c, o, "A"), "B": cost(p - c, c, o, "B"),
        "solved": solved, "wall_min": sum(e["wall_s"] for e in eps) / 60,
    }


def main() -> None:
    args = sys.argv[1:]
    only = None
    if args and args[0] == "--sessions":
        only, args = set(args[1].split(",")), args[2:]
    rows = [stats(Path(d), only) for d in args]
    ref = rows[0]
    print(f"{'policy':<14} {'sess':>4} {'acc':>5} {'turns':>5} {'peak':>7} {'tokens':>9} {'cached':>6} "
          f"{'naive$B':>8} {'$A':>7} {'$B':>7} {'tokens':>7} {'naive$B':>7} {'true$A':>7} {'true$B':>7} {'GPU min':>7}")
    print(f"{'':<14} {'':>4} {'':>5} {'':>5} {'':>7} {'':>9} {'':>6} {'':>8} {'':>7} {'':>7} "
          f"{'--- change vs ' + ref['policy'] + ' ---':>31}")
    for r in rows:
        d = lambda k: f"{(r[k] / ref[k] - 1):+.0%}" if ref[k] else "n/a"  # noqa: E731
        print(f"{r['policy']:<14} {r['sessions']:>4} {r['acc']:>5.0%} {r['turns']:>5} {r['peak']:>7,.0f} {r['prompt']:>9,} "
              f"{r['cached']:>6.0%} {r['naiveB']:>8.3f} {r['A']:>7.3f} {r['B']:>7.3f} {d('prompt'):>7} {d('naiveB'):>7} "
              f"{d('A'):>7} {d('B'):>7} {r['wall_min']:>7.1f}")


if __name__ == "__main__":
    main()