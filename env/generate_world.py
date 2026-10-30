"""Seeded generator for the "Incident Desk" world.

Builds a small fake company: 8 microservices, 5 days of logs, a deploy history,
an on-call schedule and a few runbooks. Signals are planted on purpose (error
spikes, OOM kills, config changes, cross-service request traces) so that every
task has one exact, checkable answer.

Same seed -> byte-identical world. Usage:

    python -m env.generate_world --seed 42 --out world
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
# Static vocabulary
# --------------------------------------------------------------------------- #
SERVICES = [
    "api-gateway", "auth", "payments", "orders",
    "inventory", "search", "notifications", "shipping",
]
DATES = [f"2026-10-0{d}" for d in range(1, 6)]
PEOPLE = [
    "priya.nair", "marco.rossi", "lena.vogel", "sam.okafor", "yuki.tanaka",
    "diego.alvarez", "hana.kim", "omar.haddad", "ines.costa", "tom.brennan",
    "aisha.bello", "jonas.berg",
]
ENDPOINTS = {
    "api-gateway": [("GET", "/v1/health"), ("POST", "/v1/checkout"), ("GET", "/v1/products")],
    "auth": [("POST", "/v1/token"), ("GET", "/v1/verify")],
    "payments": [("POST", "/v1/charge"), ("POST", "/v1/refund")],
    "orders": [("POST", "/v1/orders"), ("GET", "/v1/orders/{id}")],
    "inventory": [("GET", "/v1/stock"), ("POST", "/v1/reserve")],
    "search": [("GET", "/v1/search"), ("GET", "/v1/suggest")],
    "notifications": [("POST", "/v1/email"), ("POST", "/v1/push")],
    "shipping": [("POST", "/v1/quote"), ("POST", "/v1/label")],
}
JOBS = ["reconcile", "reindex", "flush-outbox", "rotate-logs", "sync-catalog"]
DEPLOY_NOTES = [
    "release", "bump dependencies", "feature flag cleanup", "logging tweaks",
    "refactor handlers", "add metrics", "perf: batch writes", "fix nil check",
    "update base image", "tighten input validation",
]
OOM_NOTES = ["raise worker threads", "enable result caching", "load full catalog in memory",
             "increase batch size"]
CONFIG_KEYS = [
    ("db.pool_size", ["50", "8", "80"]),
    ("http.timeout_ms", ["3000", "250", "5000"]),
    ("cache.ttl_s", ["600", "5", "1200"]),
    ("worker.concurrency", ["16", "2", "32"]),
    ("retry.max_attempts", ["3", "0", "5"]),
]
DEPENDENCY_CODE = "E5002"

# runbook file -> (title, {code: (description, [steps])}); step 1 is always a command.
RUNBOOKS: dict[str, tuple[str, dict[str, tuple[str, list[str]]]]] = {
    "database.md": ("Database connectivity", {
        "E2041": ("connection pool exhausted", [
            "`opsctl db pool-flush --primary`",
            "Check `db_pool_in_use` on the service dashboard; it should drop below 70% within 2 minutes.",
            "If it does not, look for long-running transactions with `opsctl db top --sort duration`.",
            "Open an incident if more than one service is affected.",
        ]),
        "E2047": ("read replica lag above threshold", [
            "`opsctl db failover-read --to replica-b`",
            "Confirm replica-b lag is under 2s.",
            "Page the database team if both replicas are lagging.",
        ]),
    }),
    "memory.md": ("Memory pressure and OOM", {
        "E3010": ("heap pressure / GC thrash", [
            "`opsctl heap dump --live`",
            "Attach the dump to the incident ticket.",
            "If the last deploy changed caching or batch sizes, roll it back.",
        ]),
        "E3019": ("native memory leak suspected", [
            "`opsctl pod recycle --oldest`",
            "Watch RSS for 15 minutes; if it grows linearly, escalate to the owning team.",
        ]),
    }),
    "network.md": ("Network and dependencies", {
        "E4002": ("upstream connect timeout", [
            "`opsctl mesh retry-budget --raise 20`",
            "Check whether the upstream service is deploying right now.",
            "Lower the budget back after the incident.",
        ]),
        "E4013": ("TLS handshake failure", [
            "`opsctl certs rotate --edge`",
            "Verify certificate expiry with `opsctl certs list`.",
        ]),
    }),
    "auth.md": ("Authentication", {
        "E6101": ("token signing key mismatch", [
            "`opsctl keys sync --jwks`",
            "Invalidate cached tokens older than the last key rotation.",
        ]),
        "E6107": ("rate limited by identity provider", [
            "`opsctl idp backoff --minutes 15`",
            "Ask the IdP vendor for a quota increase if this repeats.",
        ]),
    }),
    "queues.md": ("Queues and background jobs", {
        "E7020": ("consumer lag", [
            "`opsctl queue scale-consumers --by 2`",
            "Confirm lag is decreasing on the queue dashboard.",
        ]),
        "E7031": ("poison message", [
            "`opsctl queue dlq-move --head`",
            "Inspect the message in the dead-letter queue before replaying.",
        ]),
    }),
    "cache.md": ("Caching", {
        "E8004": ("cache stampede", [
            "`opsctl cache warm --hot-keys`",
            "Enable request coalescing if it is off.",
        ]),
        "E8011": ("eviction storm", [
            "`opsctl cache resize --step up`",
            "Check for an unexpected change in key cardinality.",
        ]),
    }),
}
CODE_MSG = {code: desc for _, (_, codes) in RUNBOOKS.items() for code, (desc, _) in codes.items()}
CODE_MSG[DEPENDENCY_CODE] = "dependency failed"
ALL_CODES = sorted(c for c in CODE_MSG if c != DEPENDENCY_CODE)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    ts: datetime
    level: str
    body: str


@dataclass
class Deploy:
    ts: datetime
    service: str
    author: str
    commit: str
    notes: str
    deploy_id: str = ""


def day_start(date: str) -> datetime:
    return datetime.fromisoformat(date + "T00:00:00+00:00")


def fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def render(service: str, line: Line) -> str:
    return f"{fmt_ts(line.ts)} {line.level:<5} {service} {line.body}"


class World:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.logs: dict[tuple[str, str], list[Line]] = {(s, d): [] for s in SERVICES for d in DATES}
        self.deploys: list[Deploy] = []
        self.oncall: dict[str, dict[str, tuple[str, str]]] = {}
        self.used_rids: set[str] = set()
        self.plants: dict[str, list[dict]] = {}

    # ---- helpers ---------------------------------------------------------- #
    def rand_time(self, date: str, h0: int = 0, h1: int = 24) -> datetime:
        secs = self.rng.randint(h0 * 3600, h1 * 3600 - 1)
        return day_start(date) + timedelta(seconds=secs, milliseconds=self.rng.randint(0, 999))

    def bg_rid(self) -> str:
        while True:
            rid = "r-" + self.rng.choice("abcdef") + "".join(self.rng.choice("0123456789abcdef") for _ in range(5))
            if rid not in self.used_rids:
                self.used_rids.add(rid)
                return rid

    def endpoint(self, service: str) -> tuple[str, str]:
        method, path = self.rng.choice(ENDPOINTS[service])
        return method, path.replace("{id}", str(self.rng.randint(10000, 99999)))

    def req_body(self, service: str, rid: str, status: int, latency: int, extra: str = "") -> str:
        method, path = self.endpoint(service)
        body = f"req={rid} method={method} path={path} status={status} latency_ms={latency}"
        return body + (" " + extra if extra else "")

    def error_extra(self, code: str, msg: str | None = None) -> str:
        return f'code={code} msg="{msg or CODE_MSG[code]}"'

    def add(self, service: str, date: str, line: Line) -> None:
        self.logs[(service, date)].append(line)

    def add_deploy(self, ts: datetime, service: str, notes: str) -> Deploy:
        d = Deploy(ts, service, self.rng.choice(PEOPLE),
                   "".join(self.rng.choice("0123456789abcdef") for _ in range(7)), notes)
        self.deploys.append(d)
        return d

    # ---- background ------------------------------------------------------- #
    def gen_background(self) -> None:
        rng = self.rng
        for s in SERVICES:
            for d in DATES:
                for _ in range(rng.randint(220, 320)):
                    ts = self.rand_time(d)
                    r = rng.random()
                    rid = self.bg_rid()
                    if r < 0.012:
                        code = rng.choice(ALL_CODES)
                        self.add(s, d, Line(ts, "ERROR", self.req_body(
                            s, rid, rng.choice([500, 502, 503, 504]), rng.randint(300, 4000),
                            self.error_extra(code))))
                    elif r < 0.018:
                        code = rng.choice(ALL_CODES)
                        self.add(s, d, Line(ts, "ERROR", f"event=job_failed job={rng.choice(JOBS)} "
                                                         + self.error_extra(code)))
                    elif r < 0.06:
                        self.add(s, d, Line(ts, "WARN", self.req_body(
                            s, rid, rng.choice([400, 401, 404, 409, 429]), rng.randint(3, 90))))
                    else:
                        self.add(s, d, Line(ts, "INFO", self.req_body(
                            s, rid, rng.choice([200, 200, 200, 201]), rng.randint(3, 180))))
        for d in DATES:
            for _ in range(rng.randint(12, 16)):
                self.add_deploy(self.rand_time(d, 6, 22), rng.choice(SERVICES), rng.choice(DEPLOY_NOTES)
                                + f" v1.{rng.randint(0, 40)}.{rng.randint(0, 9)}")
        for d in DATES:
            self.oncall[d] = {}
            for s in SERVICES:
                p, q = rng.sample(PEOPLE, 2)
                self.oncall[d][s] = (p, q)

    
    # ---- planted signals -------------------------------------------------- #
    def plant_spikes(self) -> None:
        rng = self.rng
        self.plants["spike"] = []
        for d in DATES:
            s = rng.choice(SERVICES)
            start = self.rand_time(d, 2, 20)
            self.add_deploy(start - timedelta(minutes=rng.randint(3, 25)), s,
                            f"release v2.{rng.randint(0, 20)}.0")
            code = rng.choice(ALL_CODES)
            for _ in range(rng.randint(14, 22)):
                ts = start + timedelta(seconds=rng.randint(0, 45 * 60), milliseconds=rng.randint(0, 999))
                self.add(s, d, Line(ts, "ERROR", self.req_body(
                    s, self.bg_rid(), rng.choice([500, 503]), rng.randint(800, 5000), self.error_extra(code))))
            self.plants["spike"].append({"service": s, "date": d})

    def plant_ooms(self) -> None:
        rng = self.rng
        self.plants["oom"] = []
        for s in rng.sample(["payments", "orders", "inventory", "search", "shipping"], 3):
            d = rng.choice(DATES[1:4])
            t = self.rand_time(d, 3, 20)
            self.add_deploy(t - timedelta(minutes=rng.randint(20, 90)), s,
                            rng.choice(OOM_NOTES) + f" v3.{rng.randint(0, 9)}.0")
            # distractor: a deploy to another service closer to the OOM
            other = rng.choice([x for x in SERVICES if x != s])
            self.add_deploy(t - timedelta(minutes=rng.randint(2, 15)), other, "bump dependencies v1.9.1")
            # distractor: a later hotfix to the same service
            self.add_deploy(t + timedelta(minutes=rng.randint(30, 120)), s, "hotfix: lower memory footprint")
            n = rng.randint(2, 4)
            for i in range(n):
                ts = t + timedelta(minutes=i * rng.randint(4, 30), milliseconds=rng.randint(0, 999))
                self.add(s, d, Line(ts, "WARN", f"event=OOMKilled pod={s}-{rng.randint(1000, 9999):x} "
                                                f"container=app memory_limit=512Mi"))
            # a later OOM on the following day too (only the first one matters)
            nd = DATES[DATES.index(d) + 1]
            self.add(s, nd, Line(self.rand_time(nd, 1, 10), "WARN",
                                 f"event=OOMKilled pod={s}-{rng.randint(1000, 9999):x} container=app memory_limit=512Mi"))
            self.plants["oom"].append({"service": s})

    def plant_configs(self) -> None:
        rng = self.rng
        self.plants["config"] = []
        services = rng.sample(SERVICES, 3)
        keys = rng.sample(CONFIG_KEYS, 3)
        for s, (key, (old, new, other)) in zip(services, keys):
            d = rng.choice(DATES[:3])
            r = self.rand_time(d, 4, 20)
            self.add_deploy(r - timedelta(minutes=rng.randint(2, 15)), s, f"config: set {key}={new}")
            self.add(s, d, Line(r, "INFO", f"event=config_reload key={key} old={old} new={new}"))
            d2 = rng.choice(DATES[DATES.index(d) + 1:])
            r2 = self.rand_time(d2, 4, 20)
            self.add_deploy(r2 - timedelta(minutes=rng.randint(2, 15)), s, f"config: set {key}={other}")
            self.add(s, d2, Line(r2, "INFO", f"event=config_reload key={key} old={new} new={other}"))
            self.plants["config"].append({"service": s, "date": d})

    def plant_traces(self) -> None:
        rng = self.rng
        self.plants["trace"], self.plants["rootcause"] = [], []
        rids = [f"r-{n}" for n in rng.sample(range(1000, 9999), 8)]
        others = [s for s in SERVICES if s != "api-gateway"]
        for i, rid in enumerate(rids):
            failed = i >= 4
            d = rng.choice(DATES)
            t = self.rand_time(d, 1, 22)
            path = ["api-gateway"] + rng.sample(others, rng.randint(3, 5))
            times = []
            for _ in path:
                times.append(t)
                t = t + timedelta(milliseconds=rng.randint(2, 40))
            if not failed:
                for s, ts in zip(path, times):
                    self.add(s, d, Line(ts, "INFO", self.req_body(s, rid, 200, rng.randint(5, 200))))
                self.plants["trace"].append({"rid": rid, "date": d})
            else:
                origin = path[-1]
                code = rng.choice(ALL_CODES)
                self.add(origin, d, Line(times[-1], "ERROR", self.req_body(
                    origin, rid, 503, rng.randint(200, 3000), self.error_extra(code))))
                for j in range(len(path) - 1):
                    self.add(path[j], d, Line(times[j], "ERROR", self.req_body(
                        path[j], rid, 502, rng.randint(300, 3500),
                        self.error_extra(DEPENDENCY_CODE, f"dependency failed: {path[j + 1]}"))))
                self.add_deploy(times[0] - timedelta(minutes=rng.randint(30, 240)), origin,
                                rng.choice(DEPLOY_NOTES) + f" v4.{rng.randint(0, 9)}.0")
                self.plants["rootcause"].append({"rid": rid, "date": d})

    def plant_runbook_errors(self) -> None:
        """Planted last so the chosen second is unique in its file."""
        rng = self.rng
        self.plants["runbook"] = []
        for rb in rng.sample(sorted(RUNBOOKS), 2):
            code = rng.choice(sorted(RUNBOOKS[rb][1]))
            s, d = rng.choice(SERVICES), rng.choice(DATES)
            taken = {fmt_ts(l.ts)[11:19] for l in self.logs[(s, d)]}
            while True:
                ts = self.rand_time(d, 1, 23)
                if fmt_ts(ts)[11:19] not in taken:
                    break
            self.add(s, d, Line(ts, "ERROR", self.req_body(
                s, self.bg_rid(), 500, rng.randint(400, 3000), self.error_extra(code))))
            self.plants["runbook"].append({"service": s, "date": d, "time": fmt_ts(ts)[11:19]})

    def finalize(self) -> None:
        for k in self.logs:
            self.logs[k].sort(key=lambda l: l.ts)
        self.deploys.sort(key=lambda x: x.ts)
        for i, dep in enumerate(self.deploys):
            dep.deploy_id = f"d-{1001 + i}"

    def build(self) -> "World":
        self.gen_background()
        self.plant_spikes()
        self.plant_ooms()
        self.plant_configs()
        self.plant_traces()
        self.plant_runbook_errors()
        self.finalize()
        return self

    
    # ---- writing ---------------------------------------------------------- #
    def write(self, root: Path) -> None:
        if root.exists():
            shutil.rmtree(root)
        (root / "logs").mkdir(parents=True)
        for (s, d), lines in self.logs.items():
            p = root / "logs" / s
            p.mkdir(exist_ok=True)
            _write(p / f"{d}.log", "\n".join(render(s, l) for l in lines) + "\n")
        rows = ["deploy_id,service,timestamp,author,commit,notes"]
        rows += [f"{x.deploy_id},{x.service},{fmt_ts(x.ts)},{x.author},{x.commit},{x.notes}" for x in self.deploys]
        _write(root / "deploys.csv", "\n".join(rows) + "\n")
        oc = ["# On-call schedule. Each shift covers the full UTC day.", "schedule:"]
        for d in DATES:
            oc.append(f'  "{d}":')
            for s in SERVICES:
                p, q = self.oncall[d][s]
                oc.append(f"    {s}:")
                oc.append(f"      primary: {p}")
                oc.append(f"      secondary: {q}")
        _write(root / "oncall.yaml", "\n".join(oc) + "\n")
        (root / "runbooks").mkdir()
        for fname, (title, codes) in RUNBOOKS.items():
            _write(root / "runbooks" / fname, _runbook_md(title, codes))
        _write(root / "README.md", WORLD_README)


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _runbook_md(title: str, codes: dict[str, tuple[str, list[str]]]) -> str:
    out = [f"# Runbook: {title}", "",
           "Use this runbook when a service logs one of the error codes below. Always note the time you",
           "started remediation in the incident channel. Commands assume you are logged in to the ops",
           "bastion with `opsctl` configured for the production cluster.", ""]
    for code, (desc, steps) in codes.items():
        out += [f"## {code}: {desc}", "", "### Remediation", ""]
        out += [f"{i}. {'Run ' if i == 1 else ''}{st}" for i, st in enumerate(steps, 1)]
        out += ["", "### Background", "",
                f"{desc.capitalize()} is usually a symptom, not a root cause. Check the deploy history for",
                "recent changes to the affected service before escalating.", ""]
    out += ["## Escalation", "", "If remediation does not help within 30 minutes, page the secondary on-call",
            "listed in `oncall.yaml` and open a SEV-2 incident.", ""]
    return "\n".join(out)


WORLD_README = """# Incident Desk

Operational data for 8 services over 2026-10-01 .. 2026-10-05.

- `logs/<service>/<date>.log` - one line per event, sorted by time.
  Format: `<timestamp> <LEVEL> <service> key=value ...`
  For requests, the timestamp is the moment the service received the request.
  `req=` is the request ID; the same ID appears in every service that handled it.
  Errors carry `code=` and `msg=`. Code E5002 means a dependency failed; the msg names it.
- `deploys.csv` - every production deploy, sorted by time.
- `oncall.yaml` - primary and secondary on-call per service per day (UTC days).
- `runbooks/*.md` - remediation steps per error code.
"""


def hash_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="world")
    a = ap.parse_args()
    World(a.seed).build().write(Path(a.out))
    print(f"seed={a.seed}  world_sha256={hash_tree(Path(a.out))}")


if __name__ == "__main__":
    main()