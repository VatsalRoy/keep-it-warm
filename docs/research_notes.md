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