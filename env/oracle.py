"""Scripted solver that answers every task using ONLY the agent's tools.

It never touches the generator's internals, so if it agrees with tasks.jsonl we know
(a) each answer is recoverable from what the agent can see, and (b) the generator's
ground truth is right. It also records the trajectory size, to sanity-check how many
tokens a near-perfect agent would accumulate.

    python -m env.oracle --world world --tasks tasks/tasks.jsonl
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import dataclass, field

import yaml

from env.grader import grade
from env.tools import READ_MAX_LINES, IncidentDesk

ROW = re.compile(r"^\s*(\d+)  (.*)$")
HEADER = re.compile(r"lines (\d+)-(\d+) of (\d+)")


@dataclass
class Trace:
    calls: int = 0
    chars: int = 0
    log: list = field(default_factory=list)


class Oracle:
    def __init__(self, desk: IncidentDesk):
        self.desk = desk
        self.t = Trace()

    # tool wrappers that count the size of what the agent would see
    def _call(self, name: str, **args) -> str:
        out = self.desk.call(name, args)
        self.t.calls += 1
        self.t.chars += len(out)
        self.t.log.append((name, args))
        return out

    def ls(self, path: str) -> list[str]:
        out = self._call("ls", path=path)
        return [r.split()[0].rstrip("/") for r in out.splitlines()[1:]]

    def read_all(self, path: str) -> list[str]:
        lines, start = [], 1
        while True:
            out = self._call("read", path=path, start_line=start, n_lines=READ_MAX_LINES)
            head, *rows = out.splitlines()
            lines += [ROW.match(r).group(2) for r in rows]
            _, end, total = map(int, HEADER.search(head).groups())
            if end >= total:
                return lines
            start = end + 1

    def grep(self, pattern: str, path: str) -> tuple[int, list[str]]:
        out = self._call("grep", pattern=pattern, path=path)
        head, *rows = out.splitlines()
        n = int(re.search(r": (\d+) matching lines", head).group(1))
        return n, [ROW.match(r).group(2) for r in rows if ROW.match(r)]

    # parsed views
    def deploys(self) -> list[dict]:
        return list(csv.DictReader(io.StringIO("\n".join(self.read_all("/deploys.csv")))))

    def oncall(self, service: str, date: str) -> str:
        return yaml.safe_load("\n".join(self.read_all("/oncall.yaml")))["schedule"][date][service]["primary"]

    def last_deploy(self, service: str, ts: str) -> str:
        cands = [d for d in self.deploys() if d["service"] == service and d["timestamp"] < ts]
        return max(cands, key=lambda d: d["timestamp"])["deploy_id"]

    # ------------------------------------------------------------------ #
    def solve(self, task: dict) -> str:
        q = task["question"]
        date = re.search(r"\d{4}-\d{2}-\d{2}", q)
        date = date.group() if date else None
        services = self.ls("/logs")
        tpl = task["template"]

        if tpl == "most_5xx":
            counts = {s: self.grep(r"status=5\d\d", f"/logs/{s}/{date}.log")[0] for s in services}
            return max(counts, key=counts.get)

        if tpl == "count_errors":
            s = re.search(r"did (\S+) write", q).group(1)
            return str(self.grep(r"Z ERROR ", f"/logs/{s}/{date}.log")[0])

        if tpl == "oncall":
            s = re.search(r"for (\S+) on", q).group(1)
            return self.oncall(s, date)

        if tpl == "oom_deploy_oncall":
            s = q.split()[0]
            for f in sorted(self.ls(f"/logs/{s}")):
                n, rows = self.grep("OOMKilled", f"/logs/{s}/{f}")
                if n:
                    ts = rows[0].split()[0]
                    return f"{self.last_deploy(s, ts)}, {self.oncall(s, ts[:10])}"
            raise RuntimeError("no OOM found")

        if tpl == "config_change":
            s = re.search(r", (\S+) reloaded", q).group(1)
            _, rows = self.grep("config_reload", f"/logs/{s}/{date}.log")
            key = re.search(r"key=(\S+)", rows[0]).group(1)
            new = re.search(r"new=(\S+)", rows[0]).group(1)
            _, deps = self.grep(re.escape(f"{key}={new}"), "/deploys.csv")
            dep = [d for d in deps if d.split(",")[1] == s][0].split(",")[0]
            return f"{key}, {new}, {dep}"

        if tpl == "runbook_first_step":
            hms = re.search(r"\d{2}:\d{2}:\d{2}", q).group()
            s = re.search(r"UTC, (\S+) logged", q).group(1)
            _, rows = self.grep(f"T{hms}", f"/logs/{s}/{date}.log")
            code = re.search(r"code=(E\d+)", rows[0]).group(1)
            for rb in self.ls("/runbooks"):
                if self.grep(f"## {code}", f"/runbooks/{rb}")[0]:
                    text = self.read_all(f"/runbooks/{rb}")
                    i = next(i for i, l in enumerate(text) if l.startswith(f"## {code}"))
                    step = next(l for l in text[i:] if l.startswith("1. "))
                    return re.search(r"`([^`]+)`", step).group(1)
            raise RuntimeError("code not in runbooks")

        if tpl in ("trace_request", "root_cause"):
            rid = re.search(r"r-\d{4}", q).group()
            hits = []
            for s in services:
                _, rows = self.grep(rf"req={rid} ", f"/logs/{s}/{date}.log")
                hits += [(r.split()[0], s, r) for r in rows]
            hits.sort()
            if tpl == "trace_request":
                return ", ".join(s for _, s, _ in hits)
            ts, origin, _ = next(h for h in hits if "code=E5002" not in h[2])
            return f"{origin}, {self.last_deploy(origin, ts)}, {self.oncall(origin, date)}"

        raise ValueError(f"unknown template {tpl}")


def run(world: str, tasks_path: str) -> list[dict]:
    desk = IncidentDesk(world)
    out = []
    for line in open(tasks_path, encoding="utf-8"):
        task = json.loads(line)
        o = Oracle(desk)
        ans = o.solve(task)
        out.append({"id": task["id"], "tier": task["tier"], "answer": ans, "ok": grade(task, ans),
                    "tool_calls": o.t.calls, "approx_tokens": o.t.chars // 4})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="world")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    a = ap.parse_args()
    res = run(a.world, a.tasks)
    for r in res:
        print(f"{'OK ' if r['ok'] else 'BAD'} {r['id']:<20} calls={r['tool_calls']:<3} "
              f"~tokens={r['approx_tokens']:<6} {r['answer']}")
    print(f"\n{sum(r['ok'] for r in res)}/{len(res)} solved by oracle")


if __name__ == "__main__":
    main()