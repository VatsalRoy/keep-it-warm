"""Agent Cache Bill Viewer: replay recorded agent runs and see what each request really cost.

Runs on a free CPU Space: nothing here calls a model; it replays traces recorded on a GPU.
Data: traces.json.gz, built by space/build_data.py from the project's runs/ folder.
"""
from __future__ import annotations

import gzip
import html
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHEETS = {
    "A: cache hit = 50% of miss price": {"miss": 0.150, "hit": 0.075, "out": 0.600},
    "B: cache hit = 10% of miss price": {"miss": 5.00, "hit": 0.50, "out": 25.00},
}
MASK = "[output elided to save context; re-run the tool if you need it again]"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"
REPO = "https://github.com/VatsalRoy/keep-it-warm"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> dict:
    local = Path(os.environ.get("TRACES_PATH", Path(__file__).parent / "data" / "traces.json.gz"))
    if not local.exists():
        from huggingface_hub import hf_hub_download
        local = Path(hf_hub_download(repo_id=os.environ.get("TRACES_REPO", "VatsalRoy/keep-it-warm-traces"),
                                     filename="traces.json.gz", repo_type="dataset"))
    with gzip.open(local, "rt", encoding="utf-8") as f:
        return json.load(f)


DATA = load_data()
POLICIES = list(DATA["policies"])
P0 = POLICIES[0]


def sessions_for(policy: str) -> list[str]:
    return sorted(DATA["policies"][policy]["sessions"])


def cost(r: dict, p: dict) -> float:
    return (r["uncached_tokens"] * p["miss"] + r["cached_tokens"] * p["hit"] + r["completion_tokens"] * p["out"]) / 1e6


# --------------------------------------------------------------------------- #
# Replay: which messages did the policy send on agent step k? (mirrors policies/*.py)
# --------------------------------------------------------------------------- #
def actions(tr: list[dict]) -> list[int]:
    return [i for i, m in enumerate(tr) if m["kind"] == "action"]


def history_before(tr: list[dict], k: int) -> list[dict]:
    """The full history right before the agent's k-th action (1-based)."""
    return tr[:actions(tr)[k - 1]]


def masked(m: dict) -> dict:
    return {**m, "content": m["content"].split("\n", 1)[0] + "\n" + MASK}


def sent_messages(kind: str, tr: list[dict], notes: list[dict], k: int) -> tuple[list[dict], int]:
    """Returns (messages sent with a 'status' field, number of history messages left out)."""
    h = history_before(tr, k)
    tag = lambda ms, s="kept": [{**m, "status": s} for m in ms]  # noqa: E731
    if kind in ("full", "timestamp"):
        out = tag(h)
        if kind == "timestamp":
            out[0]["status"] = "changes every turn (timestamp on line 1)"
        return out, 0
    if kind.startswith("window"):
        n = int(kind[6:])
        starts = [i for i, m in enumerate(h) if m["kind"] == "action"]
        if len(starts) <= n:
            return tag(h), 0
        keep = [m for i, m in enumerate(h) if i >= starts[-n] or m["kind"] in ("system", "task")]
        return tag(keep), len(h) - len(keep)
    if kind.startswith("mask") or kind.startswith("batch"):
        keep_last = 4
        obs = [i for i, m in enumerate(h) if m["kind"] == "observation"]
        if kind.startswith("mask"):
            n_masked = max(0, len(obs) - keep_last)
        else:  # batched: the mask frontier only moves on fold turns
            every, n_masked = int(kind[5:]), 0
            for t in range(1, k + 1):
                if t % every == 0:
                    ht = history_before(tr, t)
                    n_masked = max(n_masked, sum(m["kind"] == "observation" for m in ht) - keep_last)
        old = set(obs[:n_masked])
        return [({**masked(m), "status": "masked"} if i in old else {**m, "status": "kept"})
                for i, m in enumerate(h)], 0
    if kind == "summary":
        done = [n for n in notes if n["turn"] <= k]
        if not done:
            return tag(h), 0
        n = done[-1]
        question = next(m["content"] for m in reversed(tr[:n["history_len"]]) if m["kind"] == "task")
        summary = {"role": "user", "kind": "summary", "status": "summary",
                   "content": f"Summary of your work so far:\n{n['summary'].strip()}\n\n"
                              f"You are still working on this:\n{question}"}
        rest = h[n["history_len"]:]
        return [{**h[0], "status": "kept"}, summary] + tag(rest), n["history_len"] - 1
    raise ValueError(kind)


def first_change(prev: list[dict], cur: list[dict]) -> int | None:
    """Index of the first sent message that differs from the previous request, or None if append-only."""
    for i, (a, b) in enumerate(zip(prev, cur)):
        if a["content"] != b["content"]:
            return i
    return None if len(cur) >= len(prev) else len(cur)


# --------------------------------------------------------------------------- #
# Tab 1: replay
# --------------------------------------------------------------------------- #
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def session_chart(policy: str, sid: str, sheet: str, step: int):
    s = DATA["policies"][policy]["sessions"][sid]
    rows, price = s["rows"], SHEETS[sheet]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for ax in (a1, a2):
        style(ax)
    x = list(range(1, len(rows) + 1))
    c = [r["cached_tokens"] / 1000 for r in rows]
    u = [r["uncached_tokens"] / 1000 for r in rows]
    a1.bar(x, c, color=BLUE, edgecolor=SURFACE, linewidth=1, label="cached (cheap)")
    a1.bar(x, u, bottom=c, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="uncached (full price)")
    for i, r in enumerate(rows):
        if r["action"] == "summarize":
            a1.annotate("summary", (x[i], c[i] + u[i]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    a1.set_ylabel("K input tokens", fontsize=9, color=INK2)
    a1.legend(loc="upper left", fontsize=8, frameon=False)
    run = 0.0
    cum = []
    for r in rows:
        run += cost(r, price)
        cum.append(run * 100)
    a2.plot(x, cum, color=ORANGE if policy != P0 else BLUE, linewidth=2, label=policy.split(" (")[0])
    if policy != P0 and sid in DATA["policies"][P0]["sessions"]:
        run0, cum0 = 0.0, []
        for r in DATA["policies"][P0]["sessions"][sid]["rows"]:
            run0 += cost(r, price)
            cum0.append(run0 * 100)
        a2.plot(range(1, len(cum0) + 1), cum0, color=NEUTRAL, linewidth=2, label="P0 full history")
    a2.set_ylabel("cumulative cost (US cents)", fontsize=9, color=INK2)
    a2.set_xlabel("request number in the session", fontsize=9, color=INK2)
    a2.legend(loc="upper left", fontsize=8, frameon=False)
    # mark the selected agent step
    req = [i for i, r in enumerate(rows) if r["action"] != "summarize"]
    if 1 <= step <= len(req):
        for ax in (a1, a2):
            ax.axvline(req[step - 1] + 1, color=INK2, linewidth=1, linestyle=":")
    fig.tight_layout()
    return fig


def session_summary(policy: str, sid: str, sheet: str) -> str:
    s = DATA["policies"][policy]["sessions"][sid]
    rows, price = s["rows"], SHEETS[sheet]
    total = sum(cost(r, price) for r in rows)
    p = sum(r["prompt_tokens"] for r in rows)
    cached = sum(r["cached_tokens"] for r in rows)
    solved = sum(q["correct"] for q in s["questions"])
    marks = " ".join(("✅" if q["correct"] else "❌") + f" Q{q['position'] + 1}" for q in s["questions"])
    line = (f"**{policy}**, session **{sid}**: {len(rows)} requests, {p:,} input tokens, "
            f"**{cached / max(p, 1):.0%} cached**, **${total:.4f}** under sheet {sheet[0]}  \n{marks} "
            f"({solved}/4 correct)")
    if policy != P0 and sid in DATA["policies"][P0]["sessions"]:
        r0 = DATA["policies"][P0]["sessions"][sid]["rows"]
        t0 = sum(cost(r, price) for r in r0)
        p0 = sum(r["prompt_tokens"] for r in r0)
        line += (f"  \nvs **P0 full history** on the same session: tokens **{p / p0 - 1:+.0%}**, "
                 f"cost **{total / t0 - 1:+.0%}**")
    return line


def step_view(policy: str, sid: str, step: int) -> tuple[str, str]:
    pol = DATA["policies"][policy]
    s = pol["sessions"][sid]
    tr, notes, kind = s["transcript"], s["policy_notes"], pol["kind"]
    n_steps = len(actions(tr))
    step = max(1, min(int(step), n_steps))
    sent, dropped = sent_messages(kind, tr, notes, step)
    prev = sent_messages(kind, tr, notes, step - 1)[0] if step > 1 else []
    fc = first_change(prev, sent) if prev else None
    rows = [r for r in s["rows"] if r["action"] != "summarize"]
    r = rows[step - 1] if step - 1 < len(rows) else None
    head = f"### Agent step {step} of {n_steps}\n"
    if r:
        f, pv = r["first_changed_token_idx"], r["prev_prompt_tokens"]
        where = ("first request" if f is None else
                 "**append-only**: nothing before the new part changed" if f >= pv else
                 f"**changed at token {f:,}** of the previous {pv:,}; everything after that is re-read")
        head += (f"{r['prompt_tokens']:,} input tokens: **{r['cached_tokens']:,} cached**, "
                 f"**{r['uncached_tokens']:,} uncached** · time to first token {r['ttft_ms'] / 1000:.1f} s  \n"
                 f"Prompt change vs previous request: {where}")
        if r.get("action"):
            head += f" · the agent then called **`{r['action']}`**"
    if dropped:
        head += f"  \n{dropped} older message(s) were left out of this request."
    items = []
    for i, m in enumerate(sent):
        text = m["content"]
        preview = html.escape(text[:600] + (" …" if len(text) > 600 else ""))
        badge = {"kept": "", "masked": "✂️ masked", "summary": "📝 summary"}.get(m["status"], "⚠️ " + m["status"])
        first = fc is not None and i == fc
        border = ORANGE if first else ("#c9c8c2" if m["status"] == "kept" else BLUE)
        note = " · <b style='color:%s'>first message that changed since the previous request</b>" % ORANGE if first else ""
        items.append(
            f"<div style='border-left:4px solid {border};padding:6px 10px;margin:6px 0;background:{SURFACE}'>"
            f"<div style='font-size:12px;color:{INK2}'>#{i + 1} · {m['role']} · {m['kind']} {badge}{note}</div>"
            f"<pre style='white-space:pre-wrap;font-size:12px;margin:4px 0 0;color:{INK}'>{preview}</pre></div>")
    return head, "".join(items)


def on_select(policy, sid, sheet):
    sids = sessions_for(policy)
    sid = sid if sid in sids else sids[0]
    n = len(actions(DATA["policies"][policy]["sessions"][sid]["transcript"]))
    head, msgs = step_view(policy, sid, 1)
    return (gr.Dropdown(choices=sids, value=sid), gr.Slider(minimum=1, maximum=max(n, 1), step=1, value=1),
            session_summary(policy, sid, sheet), session_chart(policy, sid, sheet, 1), head, msgs)


def on_step(policy, sid, sheet, step):
    head, msgs = step_view(policy, sid, step)
    return session_chart(policy, sid, sheet, int(step)), head, msgs


# --------------------------------------------------------------------------- #
# Tab 2: what-if calculator (same mechanics, your numbers)
# --------------------------------------------------------------------------- #
def simulate(policy: str, S: int, T: int, a: int, o: int, m: int, K: int, keep: int, N: int) -> dict:
    """Per-task tokens for an agent that takes T steps, each adding an action (a tokens) and a tool
    output (o tokens). Cached = the longest unchanged prefix vs the previous request (+ the previous
    action, which the server already computed while generating it)."""
    prev, n_masked = None, 0
    tot = {"cached": 0, "uncached": 0, "output": 0}
    for t in range(1, T + 1):
        steps = list(range(1, t))  # steps already in the history
        if policy == "P1":
            steps = steps[-K:]
        segs = [("sys", 0, S)]
        if policy == "P3" and t % N == 0:
            n_masked = max(n_masked, len(steps) - keep)
        for s in steps:
            segs.append((f"a{s}", 0, a))
            masked_now = (policy == "P2" and s <= len(range(1, t)) - keep) or (policy == "P3" and s <= n_masked)
            segs.append((f"o{s}", 1 if masked_now else 0, m if masked_now else o))
        if policy == "P5":
            segs[0] = ("sys", t, S)
        cached = 0
        if prev is not None:
            for x, y in zip(prev + [(f"a{t - 1}", 0, a)], segs):
                if x != y:
                    break
                cached += y[2]
        total = sum(sg[2] for sg in segs)
        tot["cached"] += min(cached, total)
        tot["uncached"] += total - min(cached, total)
        tot["output"] += a
        prev = segs
    return tot


def what_if(S, T, a, o, K, keep, N, miss, hit, out, tasks):
    S, T, a, o, K, keep, N = (int(v) for v in (S, T, a, o, K, keep, N))
    names = {"P0": "P0 full history", "P1": f"P1 sliding window ({K} steps)",
             "P2": f"P2 rolling masking (keep {keep})", "P3": f"P3 batched masking (every {N})",
             "P5": "P5 timestamp bug"}
    res = {}
    for p in names:
        r = simulate(p, S, T, a, o, 25, K, keep, N)
        r["cost"] = (r["uncached"] * miss + r["cached"] * hit + r["output"] * out) / 1e6 * tasks
        r["tokens"] = r["cached"] + r["uncached"]
        res[p] = r
    base = res["P0"]
    lines = ["| Policy | Input tokens / task | Cached | Monthly cost | vs P0 |", "|---|---|---|---|---|"]
    for p, r in res.items():
        lines.append(f"| {names[p]} | {r['tokens']:,} ({r['tokens'] / base['tokens'] - 1:+.0%}) | "
                     f"{r['cached'] / max(r['tokens'], 1):.0%} | ${r['cost']:,.2f} | {r['cost'] / base['cost'] - 1:+.0%} |")
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURFACE)
    style(ax)
    labels = [names[p].split(" (")[0] for p in res]
    vals = [r["cost"] for r in res.values()]
    ax.barh(labels[::-1], vals[::-1], color=[BLUE if l.startswith("P0") else NEUTRAL for l in labels[::-1]],
            edgecolor=SURFACE)
    for i, v in enumerate(vals[::-1]):
        ax.annotate(f"${v:,.0f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlabel("projected monthly cost (USD)", fontsize=9, color=INK2)
    ax.grid(True, axis="x", color=GRID)
    ax.set_xlim(0, max(vals) * 1.2)
    fig.tight_layout()
    note = ("Simplified model: every step adds one action and one tool output of fixed size; masked outputs "
            "shrink to ~25 tokens; cache blocks, minimum cacheable lengths and cache lifetimes are ignored. "
            "It shows the mechanism, not a quote.")
    return "\n".join(lines) + "\n\n" + note, fig


# --------------------------------------------------------------------------- #
ABOUT = f"""
### What you're looking at
AI agents resend their whole history every turn. Servers cache the **prefix** of a prompt, so
re-sent text is cheap, but only if it is **byte-identical from the start**. Context policies that
trim, mask or summarize the history edit earlier parts of the prompt, and everything after an edit
is re-read at full price.

These traces come from a real experiment: Qwen3-4B on vLLM (prefix caching on, Kaggle T4 GPUs),
12 sessions of 4 exact-answer questions about a fake company's logs, deploys and on-call schedule.
Every request's cached and uncached tokens were recorded by the server.

**Headline:** every trimming policy cut tokens by 22–46%, but under a 10× cache discount each one cost
1.5–2.6× more per solved question than keeping the full history. One timestamp line at the top of the
prompt cost 6.6×.

Code, data and paper: [{REPO}]({REPO})
"""

with gr.Blocks(title="Agent Cache Bill Viewer") as demo:
    gr.Markdown("# Agent Cache Bill Viewer\nReplay real agent runs and see which tokens were cached, "
                "which were re-read at full price, and what each request cost.")
    with gr.Tab("Replay a session"):
        with gr.Row():
            dd_policy = gr.Dropdown(POLICIES, value=POLICIES[1] if len(POLICIES) > 1 else P0, label="Policy")
            dd_session = gr.Dropdown(sessions_for(POLICIES[0]), value=sessions_for(POLICIES[0])[0], label="Session")
            rd_sheet = gr.Radio(list(SHEETS), value=list(SHEETS)[1], label="Price sheet")
        md_summary = gr.Markdown()
        plot = gr.Plot()
        sl_step = gr.Slider(1, 10, step=1, value=1, label="Agent step: see exactly what was sent")
        md_step = gr.Markdown()
        html_msgs = gr.HTML()
        sel = [dd_policy, dd_session, rd_sheet]
        outs = [dd_session, sl_step, md_summary, plot, md_step, html_msgs]
        for c in sel:
            c.change(on_select, sel, outs)
        sl_step.change(on_step, sel + [sl_step], [plot, md_step, html_msgs])
        demo.load(on_select, sel, outs)
    with gr.Tab("What-if calculator"):
        gr.Markdown("Plug in your own agent's numbers. Same mechanism as the experiment, simplified.")
        with gr.Row():
            n_S = gr.Number(800, label="System prompt tokens", precision=0)
            n_T = gr.Number(25, label="Steps per task", precision=0)
            n_a = gr.Number(60, label="Tokens per action (model output)", precision=0)
            n_o = gr.Number(1200, label="Tokens per tool output", precision=0)
        with gr.Row():
            n_K = gr.Number(6, label="P1: window size (steps)", precision=0)
            n_keep = gr.Number(4, label="P2/P3: tool outputs kept", precision=0)
            n_N = gr.Number(6, label="P3: mask every N steps", precision=0)
            n_tasks = gr.Number(10000, label="Tasks per month", precision=0)
        with gr.Row():
            n_miss = gr.Number(5.00, label="$ / 1M uncached input")
            n_hit = gr.Number(0.50, label="$ / 1M cached input")
            n_out = gr.Number(25.00, label="$ / 1M output")
        btn = gr.Button("Estimate", variant="primary")
        md_res = gr.Markdown()
        plot_res = gr.Plot()
        ins = [n_S, n_T, n_a, n_o, n_K, n_keep, n_N, n_miss, n_hit, n_out, n_tasks]
        btn.click(what_if, ins, [md_res, plot_res])
        demo.load(what_if, ins, [md_res, plot_res])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()