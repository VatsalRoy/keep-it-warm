"""Run the agent on a set of tasks under one policy and save telemetry.

Real model (on Kaggle, with vLLM running):
    python scripts/run_episodes.py --policy full --tasks t1-errcount-1,t2-oom-2,t3-trace-1 --out runs/baseline_smoke

Offline dry run on a laptop (fake model that replays the oracle's tool calls):
    python scripts/run_episodes.py --fake --policy full --tier 3 --out runs/fake_smoke

Writes <out>/turns.jsonl (one row per turn) and <out>/episodes.jsonl (one row per episode).
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

POLICIES = {"full": FullHistory}


def oracle_script(desk: IncidentDesk, task: dict) -> list[str]:
    """The oracle's tool calls, written as the JSON actions a perfect model would send."""
    o = Oracle(desk)
    answer = o.solve(task)
    acts = [json.dumps({"thought": "next step", "tool": n, "args": a}) for n, a in o.t.log]
    return acts + [json.dumps({"thought": "done", "tool": "submit", "args": {"answer": answer}})]


def load_tasks(path: str, ids: str | None, tier: int | None, limit: int | None) -> list[dict]:
    tasks = [json.loads(line) for line in open(path, encoding="utf-8")]
    if ids:
        wanted = ids.split(",")
        tasks = [t for t in tasks if t["id"] in wanted]
    if tier:
        tasks = [t for t in tasks if t["tier"] == tier]
    return tasks[:limit] if limit else tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="full", choices=sorted(POLICIES))
    ap.add_argument("--tasks-file", default="tasks/tasks.jsonl")
    ap.add_argument("--tasks", help="comma-separated task ids")
    ap.add_argument("--tier", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--world", default="world")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--fake", action="store_true", help="offline fake model (replays the oracle)")
    ap.add_argument("--no-reset", action="store_true", help="don't empty the cache between episodes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    desk = IncidentDesk(a.world)
    tasks = load_tasks(a.tasks_file, a.tasks, a.tier, a.limit)
    client = FakeClient() if a.fake else VLLMClient(a.base, a.model, temperature=a.temperature)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "turns.jsonl", "a") as ft, open(out / "episodes.jsonl", "a") as fe:
        for seed in [int(s) for s in a.seeds.split(",")]:
            for task in tasks:
                if a.fake:
                    client.load_script(oracle_script(desk, task))
                if not a.no_reset and not client.reset_cache():
                    sys.exit("Could not reset the prefix cache. Start vLLM with VLLM_SERVER_DEV_MODE=1, "
                             "or pass --no-reset (results will then depend on episode order).")
                rows, ep = run_episode(task, desk, client, POLICIES[a.policy](), seed=seed,
                                       max_turns=a.max_turns)
                for r in rows:
                    ft.write(json.dumps(r) + "\n")
                fe.write(json.dumps(ep) + "\n")
                ft.flush()
                fe.flush()
                prompt = sum(r["prompt_tokens"] for r in rows)
                cached = sum(r["cached_tokens"] for r in rows)
                outp = sum(r["completion_tokens"] for r in rows)
                print(f"{'OK ' if ep['correct'] else 'BAD'} {task['id']:<16} s{seed} turns={ep['turns']:<3} "
                      f"end={ep['end_reason']:<16} peak_ctx={max((r['active_context_tokens'] for r in rows), default=0):<6} "
                      f"cached={cached / max(prompt, 1):.0%} $B={cost(prompt - cached, cached, outp, 'B'):.4f} "
                      f"wall={ep['wall_s']}s  answer={ep['submitted']!r}")


if __name__ == "__main__":
    main()