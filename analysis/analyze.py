"""Phase 7 analysis: turn the raw run folders into tables, charts and failure tags.

    python analysis/analyze.py                 # reads runs/main, writes results/
    python analysis/analyze.py --runs runs/main --out results --boot 2000

Outputs
  results/summary.csv, results/summary.md    one row per policy, with 95% bootstrap CIs
  results/naive_vs_true.md                   H1: cost from token counts vs cost with cache pricing
  results/subsets.md                         P5 and the cadence sweep vs P0 on the same sessions
  results/h4_position.md                     accuracy early vs late in a session, and by clue position
  results/failures.csv, results/failure_tags.md   questions P0 got right that a policy got wrong
  results/figures/headline.png               accuracy vs $ per solved question, sheet A and B
  results/figures/where_the_money_goes.png   cached vs uncached tokens per turn, one session
  results/figures/cadence.png                P3: how often to mask

Statistics: the unit of resampling is the SESSION (12 of them), because the 4 questions in a
session share one conversation and are not independent. Comparisons with P0 are PAIRED: each
bootstrap sample draws the same sessions for both policies.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.pricing import SHEETS  # noqa: E402
from env.grader import norm, split_items  # noqa: E402

# Display name -> run folders. Several folders = repeated runs of the same policy (pooled).
MAIN = {
    "P0 full": ["P0_full", "P0_full_v2"],
    "P1 window": ["P1_window6"],
    "P2 mask": ["P2_mask4"],
    "P3 batch": ["P3_batch6"],
    "P4 summary": ["P4_summary8k", "P4_summary8k_v2"],
}
P5_SET = ["s01", "s05", "s07", "s12"]
HEAVY = ["s02", "s03", "s08", "s10"]
SUBSETS = {  # name -> (folders, sessions); always compared with P0 on the same sessions
    "P5 timestamp": (["P5_timestamp"], P5_SET),
    "P3 every 3": (["P3_batch3"], HEAVY),
    "P3 every 6": (["P3_batch6"], HEAVY),
    "P3 every 10": (["P3_batch10"], HEAVY),
}

# Colours (reference palette, light mode): identity is also carried by direct labels.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def load_policy(runs: Path, folders: list[str], sessions: list[str] | None = None) -> list[dict]:
    """One record per (repeat, session): its questions plus summed token/cost stats."""
    out = []
    for rep, folder in enumerate(f for f in folders if (runs / f / "episodes.jsonl").exists()):
        turns = defaultdict(list)
        for r in read_jsonl(runs / folder / "turns.jsonl"):
            turns[r["episode_id"]].append(r)
        for e in read_jsonl(runs / folder / "episodes.jsonl"):
            if sessions and e["session_id"] not in sessions:
                continue
            rs = turns[e["episode_id"]]
            p = sum(r["prompt_tokens"] for r in rs)
            c = sum(r["cached_tokens"] for r in rs)
            o = sum(r["completion_tokens"] for r in rs)
            out.append({
                "rep": rep, "folder": folder, "sid": e["session_id"], "episode": e, "turn_rows": rs,
                "prompt": p, "cached": c, "uncached": p - c, "output": o,
                "solved": e["n_correct"], "asked": e["n_questions"],
                "peak": max((r["active_context_tokens"] for r in rs), default=0),
                "wall": e["wall_s"], "ttft": [r["ttft_ms"] for r in rs if r.get("action") != "summarize"],
                "logical": sum(r["logical_tokens"] for r in rs),
            })
    return out


def money(recs: list[dict], sheet: str, naive: bool = False) -> float:
    p = SHEETS[sheet]
    if naive:
        return sum(r["prompt"] * p["miss"] + r["output"] * p["out"] for r in recs) / 1e6
    return sum(r["uncached"] * p["miss"] + r["cached"] * p["hit"] + r["output"] * p["out"] for r in recs) / 1e6


def per_solved(recs: list[dict], sheet: str) -> float:
    solved = sum(r["solved"] for r in recs)
    return money(recs, sheet) / solved if solved else float("nan")


def accuracy(recs: list[dict]) -> float:
    asked = sum(r["asked"] for r in recs)
    return sum(r["solved"] for r in recs) / asked if asked else float("nan")


# --------------------------------------------------------------------------- #
# Bootstrap (session-level, paired with P0)
# --------------------------------------------------------------------------- #
def bootstrap(pol: list[dict], ref: list[dict], n: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    by_p, by_r = defaultdict(list), defaultdict(list)
    for r in pol:
        by_p[r["sid"]].append(r)
    for r in ref:
        by_r[r["sid"]].append(r)
    sids = sorted(set(by_p) & set(by_r)) or sorted(by_p)
    stats = defaultdict(list)
    for _ in range(n):
        pick = rng.choice(sids, size=len(sids), replace=True)
        P = [r for s in pick for r in by_p[s]]
        R = [r for s in pick for r in by_r.get(s, [])]
        stats["acc"].append(accuracy(P))
        stats["d_acc"].append(accuracy(P) - accuracy(R) if R else np.nan)
        for sh in SHEETS:
            ps, rs = per_solved(P, sh), per_solved(R, sh) if R else np.nan
            stats[f"ps_{sh}"].append(ps)
            stats[f"ratio_{sh}"].append(ps / rs if rs and np.isfinite(rs) and rs > 0 else np.nan)
    ci = lambda k: tuple(np.nanpercentile(stats[k], [2.5, 97.5]))  # noqa: E731
    return {k: ci(k) for k in stats}


# --------------------------------------------------------------------------- #
# Per-question helpers (H4 and failure tags)
# --------------------------------------------------------------------------- #
def question_actions(transcript: list[dict]) -> list[list[tuple]]:
    """Tool calls grouped by question: [[(tool, args_json), ...], ...]."""
    groups: list[list[tuple]] = []
    for m in transcript:
        if m["kind"] == "task":
            groups.append([])
        elif m["kind"] == "action" and groups:
            c = m["content"]
            try:
                a = json.loads(c[c.find("{"):c.rfind("}") + 1])
                groups[-1].append((a.get("tool"), json.dumps(a.get("args"), sort_keys=True)))
            except (ValueError, AttributeError):
                groups[-1].append((None, None))
    return groups


def auto_tags(policy: str, q: dict, q0: dict, rec: dict, task: dict) -> list[str]:
    tags = []
    if q["end"] == "question_turn_limit":
        tags.append("turn_limit")
    if q["end"] in ("context_overflow", "not_reached"):
        tags.append(q["end"])
    sub = q["submitted"] or ""
    if re.search(r"\b(unknown|none)\b", sub, re.I):
        tags.append("gave_up")
    gold = [norm(g) for g in task["answer"]]
    if sub and task["match"] == "ordered" and len(gold) > 1 and sorted(split_items(sub)) == sorted(gold):
        tags.append("wrong_order")            # right items, wrong order (e.g. a request trace)
    elif sub and all(g in norm(sub) for g in gold) and not q["correct"]:
        tags.append("format_slip")            # right content plus extra text, e.g. "Run `cmd`"
    acts = question_actions(rec["episode"]["transcript"])
    calls = acts[q["position"]] if q["position"] < len(acts) else []
    tool_calls = [c for c in calls if c[0] in ("ls", "read", "grep")]
    if len(tool_calls) != len(set(tool_calls)):
        tags.append("repeated_calls")
    if q["turns"] <= 2 and q0["turns"] > 2:
        tags.append("answered_fast")
    rows = [r for r in rec["turn_rows"] if r["question_idx"] == q["position"]]
    if any(r.get("action") == "summarize" for r in rec["turn_rows"] if r["question_idx"] <= q["position"]):
        tags.append("after_summary")
    if any(r["first_changed_token_idx"] is not None and r["first_changed_token_idx"] < r["prev_prompt_tokens"]
           and r.get("action") != "summarize" for r in rows):
        tags.append("context_edited_during_q")
    return tags or ["other"]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_without_overlap(fig, points) -> None:
    """Direct labels: try a few positions around each dot and keep the first that hits nothing."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = {}
    offsets = [(7, 6), (7, -14), (-7, 6), (-7, -14), (7, 18), (-7, -26)]
    for ax, x, y, text in points:
        boxes = taken.setdefault(id(ax), [])
        for dx, dy in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK,
                            ha="left" if dx > 0 else "right",
                            fontweight="bold" if text.startswith("P0") else "normal")
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in boxes) or (dx, dy) == offsets[-1]:
                boxes.append(bb)
                break
            t.remove()


def chart_headline(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    placed = []
    for ax, sh, title in zip(axes, ["A", "B"], ["Sheet A: cache hit = 50% of miss price",
                                                "Sheet B: cache hit = 10% of miss price"]):
        style(ax)
        for r in rows:
            x, y = r[f"per_solved_{sh}"] * 1000, r["acc"] * 100
            xlo, xhi = (v * 1000 for v in r[f"per_solved_{sh}_ci"])
            ylo, yhi = (v * 100 for v in r["acc_ci"])
            c = BLUE if r["policy"].startswith("P0") else NEUTRAL
            ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], fmt="none",
                        ecolor=c, elinewidth=1, alpha=0.45, capsize=0)
            ax.scatter([x], [y], s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            placed.append((ax, x, y, r["policy"]))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("$ per solved question (thousandths of a dollar)", fontsize=9, color=INK2)
    axes[0].set_ylabel("Accuracy (% of 48 questions)", fontsize=9, color=INK2)
    fig.tight_layout()
    label_without_overlap(fig, placed)
    fig.suptitle("Better = up and to the left. Bars = 95% bootstrap CI (sessions resampled).",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_money(recs_by_policy: dict, sid: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    names = [n for n in recs_by_policy if any(r["sid"] == sid for r in recs_by_policy[n])]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.0 * len(names) + 0.6), sharex=True, sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        rec = next(r for r in recs_by_policy[name] if r["sid"] == sid)
        rows = rec["turn_rows"]
        x = np.arange(1, len(rows) + 1)
        cached = np.array([r["cached_tokens"] for r in rows]) / 1000
        unc = np.array([r["uncached_tokens"] for r in rows]) / 1000
        style(ax)
        ax.bar(x, cached, width=0.8, color=BLUE, edgecolor=SURFACE, linewidth=1, label="cached (cheap)")
        ax.bar(x, unc, bottom=cached, width=0.8, color=ORANGE, edgecolor=SURFACE, linewidth=1,
               label="uncached / re-read (full price)")
        for i, r in enumerate(rows):
            if r.get("action") == "summarize":
                ax.annotate("summary", (x[i], cached[i] + unc[i]), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7, color=INK2)
        pct = rec["uncached"] / max(rec["prompt"], 1)
        ax.set_title(f"{name}: {rec['prompt'] / 1000:.0f}K input tokens, {pct:.0%} uncached, "
                     f"{rec['solved']}/{rec['asked']} correct", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("K tokens", fontsize=8, color=INK2)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[-1].set_xlabel(f"Request number in session {sid}", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_cadence(points: list[tuple], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    style(ax)
    ns = [p[0] for p in points]
    ax.axhline(0, color=INK2, linewidth=1)
    for idx, (label, color) in [(1, ("input tokens vs P0", NEUTRAL)), (2, ("true cost, sheet B, vs P0", ORANGE))]:
        ys = [p[idx] * 100 for p in points]
        ax.plot(ns, ys, color=color, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ax.annotate(label, (ns[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xticks(ns)
    ax.set_xlabel("P3: mask old tool outputs every N turns", fontsize=9, color=INK2)
    ax.set_ylabel("Change vs P0 (%)", fontsize=9, color=INK2)
    ax.set_xlim(min(ns) - 1, max(ns) + 6)
    ax.set_title("P3 batched masking: how often to mask (4 longest sessions)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/main")
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--money-session", default="s10")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tasks = {t["id"]: t for t in read_jsonl(ROOT / "tasks" / "tasks.jsonl")}

    recs = {name: load_policy(runs, folders) for name, folders in MAIN.items()}
    recs = {k: v for k, v in recs.items() if v}
    ref = recs["P0 full"]
    print("loaded:", {k: f"{len({r['rep'] for r in v})} run(s), {len(v)} sessions" for k, v in recs.items()})

    # ---- 1. summary table with CIs ----------------------------------------- #
    rows = []
    for name, rs in recs.items():
        b = bootstrap(rs, ref, a.boot)
        ttft = np.array([t for r in rs for t in r["ttft"]])
        row = {
            "policy": name, "runs": len({r["rep"] for r in rs}), "sessions": len(rs),
            "questions": sum(r["asked"] for r in rs), "solved": sum(r["solved"] for r in rs),
            "acc": accuracy(rs), "acc_ci": b["acc"], "d_acc": accuracy(rs) - accuracy(ref), "d_acc_ci": b["d_acc"],
            "turns_per_q": sum(len(r["turn_rows"]) for r in rs) / sum(r["asked"] for r in rs),
            "mean_peak": np.mean([r["peak"] for r in rs]),
            "cached_pct": sum(r["cached"] for r in rs) / sum(r["prompt"] for r in rs),
            "input_tokens": sum(r["prompt"] for r in rs) / len({r["rep"] for r in rs}),
            "uncached_tokens": sum(r["uncached"] for r in rs) / len({r["rep"] for r in rs}),
            "ttft_mean_ms": float(ttft.mean()), "ttft_p95_ms": float(np.percentile(ttft, 95)),
            "gpu_min_per_run": sum(r["wall"] for r in rs) / 60 / len({r["rep"] for r in rs}),
        }
        for sh in SHEETS:
            row[f"naive_{sh}"] = money(rs, sh, naive=True) / row["runs"]
            row[f"true_{sh}"] = money(rs, sh) / row["runs"]
            row[f"per_solved_{sh}"] = per_solved(rs, sh)
            row[f"per_solved_{sh}_ci"] = b[f"ps_{sh}"]
            row[f"ratio_{sh}"] = per_solved(rs, sh) / per_solved(ref, sh)
            row[f"ratio_{sh}_ci"] = b[f"ratio_{sh}"]
        rows.append(row)

    flat = []
    for r in rows:
        f = {}
        for k, v in r.items():
            if isinstance(v, tuple):
                f[k + "_lo"], f[k + "_hi"] = round(float(v[0]), 6), round(float(v[1]), 6)
            else:
                f[k] = round(float(v), 6) if isinstance(v, (float, np.floating)) else v
        flat.append(f)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    pct = lambda x: f"{x:.0%}"  # noqa: E731
    ci = lambda t, f=pct: f"[{f(t[0])}, {f(t[1])}]"  # noqa: E731
    usd = lambda x: f"${x:.4f}"  # noqa: E731
    xx = lambda x: f"{x:.2f}x"  # noqa: E731
    md = md_table(
        ["Policy", "Runs", "Accuracy [95% CI]", "vs P0 (pts) [CI]", "Turns/q", "Cached", "$/solved A",
         "$/solved B", "B vs P0 [CI]", "TTFT p95"],
        [[r["policy"], r["runs"], f"{pct(r['acc'])} {ci(r['acc_ci'])}",
          f"{r['d_acc'] * 100:+.0f} [{r['d_acc_ci'][0] * 100:+.0f}, {r['d_acc_ci'][1] * 100:+.0f}]",
          f"{r['turns_per_q']:.1f}", pct(r["cached_pct"]), usd(r["per_solved_A"]), usd(r["per_solved_B"]),
          f"{xx(r['ratio_B'])} {ci(r['ratio_B_ci'], xx)}", f"{r['ttft_p95_ms'] / 1000:.1f} s"] for r in rows])
    (out / "summary.md").write_text(md)
    print("\n" + md)

    # ---- 2. naive vs true (H1) ---------------------------------------------- #
    p0 = rows[0]
    nv = []
    for r in rows:
        nv.append([r["policy"], f"{r['input_tokens'] / p0['input_tokens'] - 1:+.0%}",
                   f"{r['naive_A'] / p0['naive_A'] - 1:+.0%}", f"{r['true_A'] / p0['true_A'] - 1:+.0%}",
                   f"{r['naive_B'] / p0['naive_B'] - 1:+.0%}", f"{r['true_B'] / p0['true_B'] - 1:+.0%}"])
    nvt = md_table(["Policy", "Input tokens", "Naive cost A", "True cost A", "Naive cost B", "True cost B"], nv)
    (out / "naive_vs_true.md").write_text("Change vs P0 (total cost per run of 12 sessions).\n\n" + nvt)
    print(nvt)

    # ---- 3. subsets: P5 and cadence ------------------------------------------ #
    sub_rows, cadence = [], []
    for name, (folders, sessions) in SUBSETS.items():
        pol = load_policy(runs, folders, sessions)
        base = load_policy(runs, MAIN["P0 full"], sessions)
        if not pol:
            continue
        reps = len({r["rep"] for r in base})
        d_tok = (sum(r["prompt"] for r in pol) / len({r["rep"] for r in pol})) / (sum(r["prompt"] for r in base) / reps) - 1
        d_b = (money(pol, "B") / len({r["rep"] for r in pol})) / (money(base, "B") / reps) - 1
        d_a = (money(pol, "A") / len({r["rep"] for r in pol})) / (money(base, "A") / reps) - 1
        sub_rows.append([name, ",".join(sessions), f"{accuracy(pol):.0%} vs {accuracy(base):.0%}",
                         f"{sum(r['cached'] for r in pol) / sum(r['prompt'] for r in pol):.0%}",
                         f"{d_tok:+.0%}", f"{d_a:+.0%}", f"{d_b:+.0%}",
                         f"{per_solved(pol, 'B') / per_solved(base, 'B'):.2f}x"])
        m = re.match(r"P3 every (\d+)", name)
        if m:
            cadence.append((int(m.group(1)), d_tok, d_b))
    st = md_table(["Policy", "Sessions", "Accuracy (vs P0)", "Cached", "Tokens vs P0", "True A vs P0",
                   "True B vs P0", "$/solved B vs P0"], sub_rows)
    (out / "subsets.md").write_text(st)
    print(st)

    # ---- 4. H4: early vs late, and clue position ----------------------------- #
    h4 = []
    for name, rs in recs.items():
        qs = [q for r in rs for q in r["episode"]["questions"]]
        early = [q["correct"] for q in qs if q["position"] <= 1]
        late = [q["correct"] for q in qs if q["position"] >= 2]
        by_clue = defaultdict(list)
        for q in qs:
            by_clue[q["clue_turn_expected"]].append(q["correct"])
        h4.append([name, f"{np.mean(early):.0%}", f"{np.mean(late):.0%}", f"{(np.mean(late) - np.mean(early)) * 100:+.0f}"]
                  + [f"{np.mean(by_clue[k]):.0%} (n={len(by_clue[k])})" for k in ("early", "mid", "late")])
    h4t = md_table(["Policy", "Q1-2 accuracy", "Q3-4 accuracy", "late - early (pts)", "clue early", "clue mid",
                    "clue late"], h4)
    (out / "h4_position.md").write_text(h4t)
    print(h4t)

    # ---- 5. failures vs P0 + extra turns ------------------------------------ #
    p0_q = {}
    for r in ref:
        for q in r["episode"]["questions"]:
            p0_q.setdefault((r["sid"], q["position"]), []).append(q)
    fails, tag_counts, extra = [], defaultdict(lambda: defaultdict(int)), []
    for name, rs in recs.items():
        if name == "P0 full":
            continue
        dturns = []
        for r in rs:
            for q in r["episode"]["questions"]:
                q0s = p0_q.get((r["sid"], q["position"]), [])
                if not q0s:
                    continue
                dturns.append(q["turns"] - np.mean([x["turns"] for x in q0s]))
                q0 = q0s[0]
                if any(x["correct"] for x in q0s) and not q["correct"]:
                    tags = auto_tags(name, q, q0, r, tasks[q["task_id"]])
                    for t in tags:
                        tag_counts[name][t] += 1
                    tag_counts[name]["_lost"] += 1
                    fails.append({"policy": name, "run": r["folder"], "session": r["sid"], "q": q["position"] + 1,
                                  "task": q["task_id"], "turns": q["turns"], "p0_turns": q0["turns"],
                                  "submitted": q["submitted"], "gold": ", ".join(tasks[q["task_id"]]["answer"]),
                                  "auto_tags": ";".join(tags), "manual_tag": ""})
        extra.append((name, float(np.mean(dturns)) if dturns else 0.0, sum(1 for d in dturns if d > 0), len(dturns)))
    if fails:
        with open(out / "failures.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fails[0]))
            w.writeheader()
            w.writerows(fails)
    all_tags = sorted({t for c in tag_counts.values() for t in c if t != "_lost"})
    ft = md_table(["Policy", "Lost vs P0"] + all_tags,
                  [[n, c["_lost"]] + [c.get(t, 0) for t in all_tags] for n, c in tag_counts.items()])
    et = md_table(["Policy", "Extra turns per question vs P0", "Questions that took longer"],
                  [[n, f"{d:+.2f}", f"{k}/{m}"] for n, d, k, m in extra])
    (out / "failure_tags.md").write_text("Questions answered correctly under P0 but not under the policy.\n"
                                         "Tags are automatic and can overlap; refine them by hand in failures.csv.\n\n"
                                         + ft + "\n" + et)
    print(ft + "\n" + et)

    # ---- 6. charts ---------------------------------------------------------- #
    chart_headline(rows, out / "figures" / "headline.png")
    chart_money({k: [r for r in v if r["rep"] == max(x["rep"] for x in v)] for k, v in recs.items()},
                a.money_session, out / "figures" / "where_the_money_goes.png")
    if cadence:
        chart_cadence(sorted(cadence), out / "figures" / "cadence.png")
    print(f"wrote {out}/summary.csv, *.md and figures/*.png")


if __name__ == "__main__":
    main()