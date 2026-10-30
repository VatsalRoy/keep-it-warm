"""Turn the planted clues into 24 tasks with exact, checkable answers.

The answers are computed from the generator's in-memory world (the "ground truth").
env/oracle.py later re-derives every answer using only the agent's tools, which proves
each task is solvable and the ground truth is right.

    python -m env.make_tasks --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

from env.generate_world import (DATES, DEPENDENCY_CODE, RUNBOOKS, SERVICES, Deploy, World,
                                _write, fmt_ts, hash_tree)


def last_deploy_before(w: World, service: str, ts: datetime) -> Deploy:
    cands = [d for d in w.deploys if d.service == service and d.ts < ts]
    assert cands, f"no deploy to {service} before {ts}"
    return cands[-1]


def build_tasks(w: World) -> list[dict]:
    rng = random.Random(w.seed * 1000 + 7)
    tasks: list[dict] = []

    def task(tid, tier, template, question, answer, match, clue, fmt):
        tasks.append({
            "id": tid, "tier": tier, "template": template,
            "question": f"{question}\nSubmit with submit() in this format: {fmt}",
            "answer": answer, "match": match, "clue_turn_expected": clue,
        })

    five_xx = re.compile(r"status=5\d\d")
    # Tier 1 --------------------------------------------------------------
    for i, d in enumerate(rng.sample(DATES, 3)):
        counts = {s: sum(bool(five_xx.search(l.body)) for l in w.logs[(s, d)]) for s in SERVICES}
        best = max(counts.values())
        winners = [s for s, c in counts.items() if c == best]
        assert len(winners) == 1, (d, counts)
        task(f"t1-most5xx-{i + 1}", 1, "most_5xx",
             f"Which service logged the most HTTP 5xx responses on {d}?",
             winners, "exact", "late", "<service name>")
    pairs = rng.sample([(s, d) for s in SERVICES for d in DATES], 3)
    for i, (s, d) in enumerate(pairs):
        n = sum(l.level == "ERROR" for l in w.logs[(s, d)])
        task(f"t1-errcount-{i + 1}", 1, "count_errors",
             f"How many log lines at level ERROR did {s} write on {d}?",
             [str(n)], "int", "early", "<integer>")
    for i, (s, d) in enumerate(rng.sample([(s, d) for s in SERVICES for d in DATES], 2)):
        task(f"t1-oncall-{i + 1}", 1, "oncall",
             f"Who was the primary on-call engineer for {s} on {d}?",
             [w.oncall[d][s][0]], "exact", "early", "<handle>")
    # Tier 2 --------------------------------------------------------------
    for i, p in enumerate(w.plants["oom"]):
        s = p["service"]
        first = min((l for d in DATES for l in w.logs[(s, d)] if "event=OOMKilled" in l.body),
                    key=lambda l: l.ts)
        dep = last_deploy_before(w, s, first.ts)
        date = first.ts.strftime("%Y-%m-%d")
        task(f"t2-oom-{i + 1}", 2, "oom_deploy_oncall",
             f"{s} has been getting OOMKilled. Find its very first OOMKilled event. Which deploy to {s} "
             f"most recently preceded it, and who was the primary on-call for {s} that day?",
             [dep.deploy_id, w.oncall[date][s][0]], "ordered", "early", "<deploy_id>, <handle>")
    for i, p in enumerate(w.plants["config"]):
        s, d = p["service"], p["date"]
        reload = next(l for l in w.logs[(s, d)] if "event=config_reload" in l.body)
        key = re.search(r"key=(\S+)", reload.body).group(1)
        new = re.search(r"new=(\S+)", reload.body).group(1)
        deps = [x for x in w.deploys if x.service == s and x.notes == f"config: set {key}={new}"]
        assert len(deps) == 1
        task(f"t2-config-{i + 1}", 2, "config_change",
             f"On {d}, {s} reloaded its configuration. Which key changed, what was the new value, and "
             f"which deploy shipped that change?",
             [key, new, deps[0].deploy_id], "ordered", "mid", "<key>, <new_value>, <deploy_id>")
    for i, p in enumerate(w.plants["runbook"]):
        s, d, hms = p["service"], p["date"], p["time"]
        line = next(l for l in w.logs[(s, d)] if fmt_ts(l.ts)[11:19] == hms)
        code = re.search(r"code=(E\d+)", line.body).group(1)
        rb = next(f for f, (_, codes) in RUNBOOKS.items() if code in codes)
        cmd = RUNBOOKS[rb][1][code][1][0].strip("`")
        task(f"t2-runbook-{i + 1}", 2, "runbook_first_step",
             f"At {d} {hms} UTC, {s} logged an error. According to the runbooks, what is the first "
             f"remediation command for that error code?",
             [cmd], "exact", "mid", "<command, without backticks>")
    # Tier 3 --------------------------------------------------------------
    for i, p in enumerate(w.plants["trace"]):
        rid, d = p["rid"], p["date"]
        hits = sorted((l.ts, s) for s in SERVICES for l in w.logs[(s, d)] if f"req={rid} " in l.body)
        task(f"t3-trace-{i + 1}", 3, "trace_request",
             f"Request {rid} was handled on {d}. List every service it touched, in the order it reached them.",
             [s for _, s in hits], "ordered", "mid", "<service>, <service>, ...")
    for i, p in enumerate(w.plants["rootcause"]):
        rid, d = p["rid"], p["date"]
        hits = [(l, s) for s in SERVICES for l in w.logs[(s, d)] if f"req={rid} " in l.body]
        origin_line, origin = next((l, s) for l, s in hits if f"code={DEPENDENCY_CODE}" not in l.body)
        dep = last_deploy_before(w, origin, origin_line.ts)
        task(f"t3-rootcause-{i + 1}", 3, "root_cause",
             f"Request {rid} failed on {d}. Find the service where the failure originated (not a dependency "
             f"error). Which deploy to that service most recently preceded the failure, and who was that "
             f"service's primary on-call that day?",
             [origin, dep.deploy_id, w.oncall[d][origin][0]], "ordered", "early",
             "<service>, <deploy_id>, <handle>")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--world", default="world")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    a = ap.parse_args()
    w = World(a.seed).build()
    w.write(Path(a.world))
    tasks = build_tasks(w)
    out = Path(a.tasks)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, "".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks))
    manifest = {"seed": a.seed, "world_sha256": hash_tree(Path(a.world)), "n_tasks": len(tasks)}
    _write(out.parent / "world_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()