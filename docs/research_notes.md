# Research notes: Keep It Warm

## Research question

For a long-horizon tool-using agent, which context-management policy gives
the best accuracy per dollar once prefix-cache pricing is counted, and how
much depends on where and how often the context is edited?

## In my own words

## Hypotheses (written before running anything)

- H1: Counting tokens only, rolling trimming/masking looks 30-60% cheaper than
full history. Counting cache misses, most of that saving disappears, and
some policies cost more than full history.
- H2: Batching edits (mask every N turns) recovers most of the saving and
keeps the cache warm.
- H3: A prompt hygiene bug (timestamp at the top of the system prompt) drops
the cache hit rate to ~0 and costs more than any policy choice.
- H4: Threshold summarization hurts accuracy most on tasks whose key clue
appeared mid-trajectory.

## Price sheets (USD per 1M tokens)

| Sheet | Cache miss | Cache hit | Output | Discount |
|---|---|---|---|---|
| A (cheap, mild discount) | 0.150 | 0.075 | 0.600 | 2x |
| B (premium, steep discount) | 5.00 | 0.50 | 25.00 | 10x |

Cost formula per request:
uncached_input * p_miss + cached_input * p_hit + output * p_out

## Positioning (how my project differs)

The AKV paper changes the *server* to avoid re-reading the prompt. I'm asking
how much a practitioner can recover using only *client-side* policies on a
standard vLLM server, which an engineer can do without touching the serving stack.

## Paper notes (claim / evidence / what I'll test)

### Cache Engineering for Agents (arXiv 2610.28577)

- Claim:
- Evidence:
- What I'll test:

### Pack Light, CarryOn (arXiv 2610.18784)

- Claim:
- Evidence:
- What I'll test:

### vLLM Automatic Prefix Caching docs

- Key fact: caching works in blocks (often 16 tokens); a hit only counts for
whole blocks that match exactly from token 0.
- Metrics I'll read: vllm:prefix_cache_hits, vllm:prefix_cache_queries

## Setup (Phase 3)

- Hardware: Kaggle, 1x Tesla T4 (16 GB), driver CUDA 13.0
- vLLM 0.32.0 (V1 engine), Qwen/Qwen3-4B-Instruct-2507, dtype float16 (cast from bf16)
- Attention backend: TRITON_ATTN (FlashAttention 2 needs compute capability >= 8.0; T4 is 7.5)
- Kaggle fix: `pip uninstall -y torchaudio` after installing vLLM (CUDA mismatch crash)
- Cache smoke test, 2 independent sessions: warm request cached 5,520 / ~5,550 tokens both times; TTFT ~14 s cold vs ~0.23 s warm (~60x); a 1-char edit at the start -> 0 cached.
- Cold prefill throughput: ~400 tokens/s (relevant for experiment time budget)

## Pilot run 1 (P0, 5 tasks, prompt v1)

- Accuracy 1/5; mean peak context ~3K tokens; 31 s/episode; 0 prefix edits in 45 turns.
- Failures: guessed paths (access.log x7), guessed formats ("HTTP 5xx", "inventory reload"),
stopped at first E5002 hop, picked a deploy AFTER the failure, gave up with "unknown",
thoughts drifted into Chinese.
- Fix (prompt v2): data guide + working rules in the system prompt; tool errors suggest ls().
- Real tokenizer: ~60 tokens per log line (a 50-line read = 3,154 tokens).
- Prompt is FROZEN after pilot 2. No further edits before the main experiment.

# Phase 4 Log: Building and validating the agent baseline (STAR)

**Project:** Keep It Warm: does trimming an agent's context save money once prompt caching is counted?
**Phase:** 4 (4A to 4D), Oct 30 to Nov 1, 2026
**Hardware:** Kaggle, 1x Tesla T4 (16 GB), vLLM 0.32.0, Qwen3-4B-Instruct-2507 (fp16)

---

## S: Situation

After Phases 1–3 I had:
- a reproducible fake "Incident Desk" world: 8 services, 5 days of logs, deploys, on-call schedule, runbooks;
- 24 tasks with exact answers, all solved by a tool-only oracle (24/24);
- proof on a real GPU that vLLM's prefix cache works: a repeated 5.5K-token prefix was 99.4% cached
  and its first token came ~60x faster, while a single changed character dropped cached tokens to 0.

What was missing was an **agent**: something that solves the tasks with tools while recording, every
turn, how many tokens were sent, how many were cached, and what it cost. Without that, no context
policy could be compared.

## T: Task

Build the agent loop and an official **P0 (full history) baseline** that meets four conditions:

| # | Condition | Why it matters |
|---|---|---|
| 1 | Accuracy roughly 40–70% | Policies must be able to both lose and keep accuracy |
| 2 | Contexts long enough to trim (original target 15–40K tokens) | Otherwise every policy behaves like P0 |
| 3 | P0 truly append-only on the real model (0 prefix edits) | P0 must be a clean, cache-friendly reference |
| 4 | Runtime that fits a free GPU budget | The full experiment has ~6 policies |

Also: test everything offline first, so GPU time is only spent on code that already works.

## A: Action

### 4A: Agent loop, built and tested without a GPU
- Wrote the loop: ask the model for an action, run the tool, append the result, repeat.
- **Design decisions:**
  - The model replies with plain-text **JSON actions**, not vLLM's tool-calling feature. Every
    byte the model sees is then in messages I build (testable), no parser flags are needed, and
    time-to-first-token is measured on real text.
  - The history is **append-only**; a *policy* decides what is sent each turn. This keeps "logical
    tokens" (what full history would cost) available for the naive-vs-true cost comparison.
  - Per-turn telemetry: prompt / cached / uncached / completion tokens, TTFT, and
    **first_changed_token_idx**, which shows where each prompt first differs from the previous one.
- Built a **FakeClient** that replays the oracle's actions and simulates vLLM's cache (16-token
  blocks with chained fingerprints), so the whole pipeline runs on a laptop.
- **A test caught a design flaw:** at temperature 0, running the same task twice gave 99% cached
  tokens, which were leftovers from the previous run, not real savings. Fix: empty the cache
  before every episode (`/reset_prefix_cache`, enabled with `VLLM_SERVER_DEV_MODE=1`). The
  runner stops with an error if the reset fails, so biased numbers can't slip through.
- 54 tests passing.

### 4B: Pilot 1 on the real model (5 tasks)
| Accuracy | Mean peak context | Cached | Prefix edits | Time / episode |
|---|---|---|---|---|
| 1/5 (20%) | ~3K tokens | 81% | 0 / 45 turns | 31 s |

I read every transcript and sorted the failures into types:
- **Guessed paths:** tried `/logs/<service>/access.log` 7 times in a row without listing the folder.
- **Guessed formats:** searched for `HTTP 5xx` and `inventory reload`; the logs use `status=503`
  and `event=config_reload`.
- **Gave up:** submitted `unknown` with turns left.
- **Reasoning slips:** stopped at the first hop of a dependency chain; picked a deploy *after* the failure.
- **Language drift:** the agent's "thoughts" switched to Chinese partway through.

Root cause: **the agent had no map of the data**, so it guessed.

### 4C: Prompt v2, then freeze it (24 tasks)
- Added a **data guide** to the system prompt (file layout, log line format with an example, key
  fields, how E5002 dependency chains work) plus working rules ("preceded means strictly earlier",
  "never submit unknown", "change the pattern after 0 matches", "think in English").
- Made tool errors suggest the next step, e.g. `no such file ... (use ls('/logs/api-gateway'))`.
- **Caught a leak in my own fix:** the first example log line used request `r-3227`, the real
  request in task t3-rootcause-1. Replaced it with a made-up ID and added a test that fails if any
  task ID appears in the prompt.
- **Froze the prompt** after this pilot, so it can't be tuned toward one policy later.

| Accuracy | Mean peak context | Prefix edits | Time / episode |
|---|---|---|---|
| 11/24 (46%) | ~4K tokens | 0 / 149 turns | 35 s |

- **Failure audit:** all 13 failures were genuine model mistakes (paging through files, comparing
  timestamps, ordering services by time). None was a task or grader bug.
- **New problem:** a *better* agent finished each question in ~7 turns, so contexts stayed tiny.
  A 6-turn window or a 16K summary threshold would never trigger, and the policies couldn't be
  told apart.

### 4D: Redesign the episode as a multi-question session
- Changed the unit of work from "1 question" to a **session of 4 questions in one conversation**,
  like an on-call agent working a ticket queue. Context from earlier questions keeps piling up.
- Built 12 sessions from the 24 tasks. Each task appears **twice, at mirrored positions**
  (1st/4th, 2nd/3rd) with different companions, so early and late versions of the *same* question
  can be compared (a clean test for H4).
- Added a per-question turn cap (20), per-question grading, and "cost per solved question" as the
  main metric.
- Set the max context to 28,672 tokens, based on the server's measured KV capacity (29,728 tokens).
- 62 tests passing.

## R: Result

### Official P0 baseline: 12 sessions, 48 questions
| Metric | Value | Condition |
|---|---|---|
| Accuracy | **44%** (21/48); Tier 1: 11/16, Tier 2: 8/16, Tier 3: 2/16 | 1 ✅ |
| Peak context, mean / max | **11.1K / 21K** tokens, 0 overflows | 2 ⚠️ partly (see below) |
| Prefix edits under P0 | **0 of 264 turns** | 3 ✅ |
| Cached share of input | **93%** | |
| Cost, sheet B: naive vs true | $8.92 vs **$1.89** (4.7x gap) | |
| GPU time | 33 min total, 163 s per session | 4 ✅ |

Condition 2 is partly met. Contexts grew almost 3x, which is enough for the window, masking and
8K-summary policies to trigger, but the mean stays below the original 15–40K target. I record this
as a deviation and adjusted P4's threshold from 16K to 8K.

### Findings along the way
1. **Qwen's chat template is append-only:** 0 prefix edits across 458 real turns in three runs.
2. **vLLM also caches generated tokens:** the model's own reply was already cached when it came
   back as history (cached tokens exceeded the previous prompt's length).
3. **Log lines are expensive:** ~60 tokens per line (a 50-line read = 3,154 tokens), against my
   estimate of 40.
4. **A miss costs more time the later it happens:** re-reading ~3,650 new tokens took 11 s at
   1.6K context, 28 s at 7.9K, and 43 s at 13.4K (about 330 down to 85 tokens/s on the T4).
   Prices are per token, but latency grows with position.
5. **Long context hurts even without trimming:** 4th-position questions scored 3/12 against 6/12
   for positions 1–3 (small sample, to be confirmed in Phase 6).
6. **Stale context misleads:** in one session the agent answered an on-call question in one turn,
   without any tool call, from a wrong belief it had formed during an earlier question.

### Lessons learned
- **Test offline first.** The fake model plus cache simulator caught the cache-reuse bias and a
  prompt leak before they cost GPU time or corrupted results.
- **Read transcripts, not just scores.** "20% accuracy" said nothing; the transcripts showed the
  agent lacked a map of the data.
- **A better agent can break the experiment.** Improving accuracy shortened contexts, so the
  benchmark design itself had to change to keep the research question testable.
- **Write deviations down as they happen:** prompt frozen after pilot 2, the session framing line
  (prompt v3), the 8K summary threshold, single seed at temperature 0.

### Status
P0 baseline accepted. Next: Phase 5, the context policies P1–P5 with prefix-stability tests.

## Phase 6: Main experiment results (raw, before analysis)

**Date:** 2026-11-02 / 03
**Setup:** Kaggle 2x Tesla T4, one vLLM 0.32.0 server per GPU (ports 8000/8001),
Qwen3-4B-Instruct-2507 fp16, max-model-len 28,672, prefix caching on, cache reset before
every session, temperature 0, one seed, concurrency 1 per server. Prompt v3 (frozen).
**Workload:** 12 sessions x 4 questions = 48 questions per full run.
**Runtime:** 7 jobs, 0 failures. Matrix wall time ~3 h on 2 GPUs in parallel.

### Main table (12 sessions, sheet A = 2x cache discount, sheet B = 10x)
| Policy | Acc | Turns | Mean peak ctx | Cached | Tokens vs P0 | True $A vs P0 | True $B vs P0 | GPU min |
|---|---|---|---|---|---|---|---|---|
| P0 full history | 44% (21/48) | 276 | 11,132 | 93% | — | — | — | 32.6 |
| P1 window (K=6) | 38% (18/48) | 295 | 6,226 | 37% | -46% | -14% | **+96%** | 47.2 |
| P2 rolling mask (K=4) | 31% (15/48) | 337 | 6,994 | 61% | -23% | +1% | **+87%** | 52.5 |
| P3 batched mask (every 6) | 35% (17/48) | 305 | 7,695 | 84% | -22% | -11% | +22% | 40.0 |
| P4 summary at 8K (run 1) | 35% (17/48) | 282 | 8,649 | 90% | -28% | -22% | -6% | 32.0 |
| P4 summary at 8K (run 2) | 40% (19/48) | 306 | — | — | — | — | ~-4% | ~33 |

### Cost per solved question
| Policy | Sheet A | Sheet B | B vs P0 |
|---|---|---|---|
| P0 | $0.0070 | **$0.090** | 1.0x |
| P1 | $0.0071 | $0.205 | 2.3x |
| P2 | $0.0099 | $0.235 | 2.6x |
| P3 | $0.0077 | $0.136 | 1.5x |
| P4 (avg of 2 runs, 36/96) | ~$0.0068 | ~$0.100 | 1.1x |

### Subsets (compared with P0 on the same sessions)
- **P5 timestamp bug** (s01, s05, s07, s12): cached **0%** (P0: 94%). True cost +113% (A),
  **+413% (B)**. $ per solved question (B): $0.329 vs $0.050 = **6.6x**. Accuracy 7/16 vs 9/16.
- **P3 cadence sweep** (s02, s03, s08, s10; true $B vs P0): every 3 turns **+61%**, every 6
  +14%, every 10 **+2%**. Tokens -11% / -30% / -36%. Accuracy 4/16, 4/16, 5/16 vs P0 7/16.
  Masking every 3 turns also used the most turns (151 vs 108).

### Preliminary verdicts (confidence intervals pending in Phase 7)
- **H1 supported.** Token savings of 22–46% became cost INCREASES of 22–96% under sheet B.
  Under sheet A the savings shrank (P1: -46% tokens -> -14% cost; P2: -23% tokens -> +1% cost).
  Whether trimming saves money depends on the cache discount.
- **H2 partly supported.** Batching cut the penalty (P2 +87% -> P3 +22% under B; P3 is cheaper
  than P0 under A) and less frequent folding was cheaper (3 -> 6 -> 10: +61% -> +14% -> +2%),
  but no batched variant beat P0 under sheet B.
- **H3 supported.** One timestamp line (+413% under B) cost more than any policy choice (max +96%).
- **H4 open.** Every policy lowered accuracy (31–40% vs 44%). The mechanisms are below; the
  effect size needs Phase 7's confidence intervals.

### Transcript findings
- **Masking hides results the agent still needs.** P2 s11 Q2: after grepping all 8 services, the
  earlier counts (including the winner, orders with 19) were masked; the agent answered api-gateway.
- **Masking causes repeated work.** P2 s05 Q2 repeated the same failing searches until the
  20-turn limit, because the evidence that they had failed was masked. P2 used 7.0 turns per
  question vs 5.8 for P0 (+21%) and hit the question turn limit twice (P0: never).
- **Summaries turn beliefs into "facts".** With summary texts saved (run 2, 7 folds):
  - s03: the right timestamp was paired with the wrong deploy ID (d-1040 instead of d-1039),
    and the agent submitted d-1040.
  - s02: "most recent deploy before 08:39 is d-1025 (14:05)", a reasoning error stored as fact.
  - s11: "under 2026-10-05 ... at line 23"; line 23 is the Oct 1 section.
  - s10: the agent re-checked after the summary and answered correctly, the one helpful case.
- **Summaries invite skipping verification.** P4 run 1 s03 Q3 answered in 2 turns from
  remembered deploy facts (P0 took 9 turns and was right).

### Noise floor (unplanned finding)
- P4 run twice with identical code and temperature 0: **6/48 question outcomes flipped**
  (17 vs 19 correct); 6 of 12 sessions diverged, one at the very first action.
- Cause: GPU floating-point results are not bit-identical between runs; when two next tokens are
  nearly tied, a tiny difference changes the choice and the conversation diverges from there.
- Consequence: accuracy differences of about 2–3 questions are within noise. Cost is far more
  stable (true $B: $1.778 vs $1.809, 2% apart).

### Deviations from the original plan (for the README)
- Episodes are 4-question sessions (Phase 4D), not single tasks.
- P4 threshold is 8K instead of 16K (contexts peak ~11K on average).
- One seed at temperature 0 (a second seed at T=0 would be identical by design); P4 run twice
  to measure noise instead.
- P5 and the cadence sweep use 4-session subsets to stay within the GPU budget.
- P4 run 1 did not save summary texts (logging gap, fixed via `policy_notes`; run 2 has them).

### Open items for Phase 7
- Bootstrap 95% CIs for accuracy and $ per solved question (paired by question).
- Decide whether to rerun P0 (~20–35 GPU min) for a P0 noise estimate.
- Systematic failure tagging across all policies; count extra turns caused by masking.

## Phase 7: Final verdicts (session-level paired bootstrap, 2,000 resamples)

| | Verdict | Key number (95% CI) |
|---|---|---|
| H1 | Supported | $/solved under sheet B vs P0: P1 2.29x [1.64, 3.80], P2 2.62x [1.83, 5.12], P3 1.51x [1.06, 2.57] |
| H2 | Partly | Batching cut P2's 2.62x to 1.51x, still above P0; cadence 3/6/10 -> +61%/+14%/+2% cost (B) |
| H3 | Supported | Timestamp bug: 0% cached, 6.6x $/solved (4 sessions), above any policy (max 2.6x) |
| H4 | Not supported as stated | P0 itself loses 12 pts on Q3-4 vs Q1-2; no policy significantly worse; clue position confounded with tier |

- Accuracy: every policy below P0 (-6 to -12 pts), none significant (all CIs reach 0; P2 -12 [-27, +0]).
- P4 summary: the only policy tied with P0 on cost (1.11x [0.95, 1.33]) and the fastest
  (p95 TTFT 6.0 s vs 10.5 s), because its prompts stay short.
- Hidden cost of masking: P2 +1.27 turns per question (27/48 questions took longer).
- Failure tags (hand-read, P2 and P4): about 1/3 of P2's losses were caused by masking (lost clue x2,
  re-read loop x1); 1/9 of P4's by the summary (answered from memory). The rest match the
  run-to-run noise seen in Phase 6.
- Clearest example: P2 s05 Q3, "others have 0", where the "others" were the masked outputs,
  including the right answer (orders, 19).