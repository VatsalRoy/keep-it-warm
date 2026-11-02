"""Run the agent on sessions (or single tasks) under one policy and save telemetry.

Real model (on Kaggle, with vLLM running):
    python scripts/run_episodes.py --policy full --out runs/main/P0            # all 12 sessions
    python scripts/run_episodes.py --policy full --sessions s01,s02 --out runs/x
    python scripts/run_episodes.py --policy full --tasks t2-oom-2 --out runs/x  # one-question episodes

Offline dry run on a laptop (fake model that replays the oracle's tool calls):
    python scripts/run_episodes.py --fake --policy full --out runs/fake_sessions

Writes <out>/turns.jsonl (one row per turn) and <out>/episodes.jsonl (one row per session).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python scripts/x.py` finds our packages

from agent.client import FakeClient, VLLMClient  # noqa: E402
from agent.loop import run_episode  # noqa: E402
from agent.pricing import cost  # noqa: E402
from env.oracle import Oracle  # noqa: E402
from env.tools import IncidentDesk  # noqa: E402
from policies.full import FullHistory  # noqa: E402
from policies.hygiene_bug import HygieneBug  # noqa: E402
from policies.mask_batched import BatchedMask  # noqa: E402
from policies.mask_rolling import RollingMask  # noqa: E402
from policies.sliding import SlidingWindow  # noqa: E402
from policies.summary import SummaryFold  # noqa: E402

# name -> factory(args, client). A NEW policy object is made for every episode.
POLICIES = {
    "full": lambda a, c: FullHistory(),                                  # P0
    "window": lambda a, c: SlidingWindow(k=a.k or 6),                    # P1
    "mask": lambda a, c: RollingMask(k=a.k or 4),                        # P2
    "batch": lambda a, c: BatchedMask(every=a.every, k=a.k or 4),        # P3
    "summary": lambda a, c: SummaryFold(c, threshold=a.threshold),       # P4
    "timestamp": lambda a, c: HygieneBug(),                              # P5
}


def oracle_script(desk: IncidentDesk, task: dict) -> list[str]:
    """The oracle's tool calls, written as the JSON actions a perfect model would send."""
    o = Oracle(desk)
    answer = o.solve(task)
    acts = [json.dumps({"thought": "next step", "tool": n, "args": a}) for n, a in o.t.log]
    return acts + [json.dumps({"thought": "done", "tool": "submit", "args": {"answer": answer}})]


def load_episodes(a) -> list[tuple[str, list[dict]]]:
    """Returns [(session_id, [task, ...]), ...]."""
    tasks = {t["id"]: t for t in (json.loads(line) for line in open(a.tasks_file, encoding="utf-8"))}
    if a.tasks:  # one-question episodes
        eps = [(tid, [tasks[tid]]) for tid in a.tasks.split(",")]
    else:
        sessions = [json.loads(line) for line in open(a.sessions_file, encoding="utf-8")]
        if a.sessions:
            wanted = a.sessions.split(",")
            sessions = [s for s in sessions if s["id"] in wanted]
        eps = [(s["id"], [tasks[t] for t in s["tasks"]]) for s in sessions]
    return eps[:a.limit] if a.limit else eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="full", choices=sorted(POLICIES))
    ap.add_argument("--tasks-file", default="tasks/tasks.jsonl")
    ap.add_argument("--sessions-file", default="tasks/sessions.jsonl")
    ap.add_argument("--sessions", help="comma-separated session ids (default: all)")
    ap.add_argument("--tasks", help="comma-separated task ids, each run as its own 1-question episode")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--k", type=int, help="window size (P1, default 6) or outputs kept (P2/P3, default 4)")
    ap.add_argument("--every", type=int, default=6, help="P3: mask every N turns")
    ap.add_argument("--threshold", type=int, default=8192, help="P4: summarize above this many tokens")
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--world", default="world")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--fake", action="store_true", help="offline fake model (replays the oracle)")
    ap.add_argument("--fake-max-context", type=int, default=28672)
    ap.add_argument("--no-reset", action="store_true", help="don't empty the cache between episodes")
    ap.add_argument("--resume", action="store_true", help="skip sessions already saved in <out>/episodes.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    desk = IncidentDesk(a.world)
    episodes = load_episodes(a)
    client = FakeClient(a.fake_max_context) if a.fake else VLLMClient(a.base, a.model, temperature=a.temperature)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.resume and (out / "episodes.jsonl").exists():
        done = {(e["session_id"], e["seed"]) for e in map(json.loads, open(out / "episodes.jsonl", encoding="utf-8"))}
        print(f"resume: {len(done)} episode(s) already saved in {out}, skipping them")

    with open(out / "turns.jsonl", "a") as ft, open(out / "episodes.jsonl", "a") as fe:
        for seed in [int(s) for s in a.seeds.split(",")]:
            for sid, tasks in episodes:
                if (sid, seed) in done:
                    continue
                if a.fake:
                    client.load_script([act for t in tasks for act in oracle_script(desk, t)])
                if not a.no_reset and not client.reset_cache():
                    sys.exit("Could not reset the prefix cache. Start vLLM with VLLM_SERVER_DEV_MODE=1, "
                             "or pass --no-reset (results will then depend on episode order).")
                policy = POLICIES[a.policy](a, client)
                rows, ep = run_episode(tasks, desk, client, policy, seed=seed, session_id=sid)
                for r in rows:
                    ft.write(json.dumps(r) + "\n")
                fe.write(json.dumps(ep) + "\n")
                ft.flush()
                fe.flush()
                prompt = sum(r["prompt_tokens"] for r in rows)
                cached = sum(r["cached_tokens"] for r in rows)
                outp = sum(r["completion_tokens"] for r in rows)
                peak = max((r["active_context_tokens"] for r in rows), default=0)
                marks = "".join("✓" if r["correct"] else "✗" for r in ep["questions"])
                print(f"{sid:<14} s{seed} {ep['n_correct']}/{ep['n_questions']} {marks:<5} turns={ep['turns']:<3} "
                      f"end={ep['end_reason']:<16} peak_ctx={peak:<6} cached={cached / max(prompt, 1):.0%} "
                      f"$B={cost(prompt - cached, cached, outp, 'B'):.4f} wall={ep['wall_s']}s")


if __name__ == "__main__":
    main()