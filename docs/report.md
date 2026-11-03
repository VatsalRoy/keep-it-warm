# Keep It Warm: what context trimming really costs a tool-using agent

**Vatsal Roy · November 2026 · [code and data](https://github.com/VatsalRoy/keep-it-warm)**

## 1. Question

Long-running AI agents resend their whole conversation on every turn. To control the growing bill,
teams trim it: drop old turns, mask old tool outputs, or summarize. But inference servers and APIs
cache the *prefix* of a prompt, and any edit to an earlier message forces everything after it to be
re-processed at full price. So a policy that sends fewer tokens can still cost more.

**Research question:** for a long-horizon tool-using agent, which context policy gives the best
accuracy per dollar once prefix-cache pricing is counted, and how much depends on where and how
often the context is edited?

Recent work changes the *server* to avoid re-reading (AKV, arXiv 2610.28577). This project asks how
much a practitioner can get with *client-side policies only*, on a stock vLLM server.

## 2. Method

- **Benchmark.** A seeded fake company ("Incident Desk": 8 services, 5 days of logs, deploys,
  on-call, runbooks) with 24 exact-answer tasks in 3 difficulty tiers. A scripted oracle using only
  the agent's tools solves all 24. Tasks are grouped into 12 sessions of 4 questions in one
  conversation; each task appears once early and once late.
- **Agent.** Qwen3-4B-Instruct-2507 on vLLM 0.32.0 (Kaggle T4), JSON actions over 4 tools (ls,
  read, grep, submit), temperature 0. The system prompt was tuned in two pilot runs, then frozen.
- **Policies.** P0 full history; P1 sliding window (6 steps); P2 rolling masking of tool outputs
  (keep 4); P3 the same masking batched every 6 turns (sweep: 3, 6, 10); P4 summarize at 8K tokens;
  P5 full history plus a timestamp at the top of the system prompt.
- **Measurement.** Per request: prompt, cached and uncached tokens (from vLLM, cross-checked against
  its metrics counters), output tokens, time to first token, and the first token index where the
  prompt changed. The cache is reset before every session.
- **Cost.** Two price sheets: A (cache hit = 50% of miss price) and B (10%). The primary metric is
  **$ per solved question**. Uncertainty: 2,000-sample bootstrap over sessions, paired with P0.
- **Checks before GPU time.** A fake model with a simulated block cache, and a test asserting where
  each policy changes the prompt (79 tests).

## 3. Results

| Policy | Accuracy | Tokens vs P0 | True cost B vs P0 | $/solved B vs P0 [95% CI] |
|---|---|---|---|---|
| P0 full history | 44% | — | — | 1.00x |
| P1 sliding window | 38% | -46% | +96% | 2.29x [1.64, 3.80] |
| P2 rolling masking | 31% | -23% | +87% | 2.62x [1.83, 5.12] |
| P3 batched masking | 35% | -22% | +22% | 1.51x [1.06, 2.57] |
| P4 summary (2 runs) | 38% | -26% | -5% | 1.11x [0.95, 1.33] |
| P5 timestamp bug (4 sessions) | 44% vs 56% | +24% | +413% | 6.6x |

1. **Token savings do not survive cache pricing (H1, supported).** Under sheet B, the three policies
   that edit the prompt often cost 1.5–2.6x more per solved question, with all intervals excluding
   1.0x. Under sheet A the token savings shrank to between -20% and +1% in true cost. *Whether
   trimming saves money depends on the cache discount.*
2. **Edit less often (H2, partly supported).** Batching the same masking cut P2's 2.62x to 1.51x.
   Masking every 3 / 6 / 10 turns cost +61% / +14% / +2% vs P0. No batched variant beat full history
   under sheet B.
3. **Prompt hygiene dominates (H3, supported).** One timestamp line meant 0% cached and 6.6x cost
   per solved question, more than any policy choice.
4. **Summarization is the practical compromise.** P4 was the only policy statistically tied with
   P0 on cost (1.11x) and had the lowest tail latency (p95 time to first token 6.0 s vs 10.5 s),
   because its prompts stay short.
5. **Accuracy fell under every policy, but not significantly (H4, not supported as stated).**
   -6 to -12 points, all intervals reaching 0. P0 itself lost 12 points on questions 3–4 vs 1–2,
   and no policy lost significantly more.

**Why accuracy fell (hand-read failures).** About a third of masking's lost answers were caused by
masking. In one, the agent concluded "inventory has 7, search 2, notifications 3, others have 0": the
"others" were the masked outputs, including the right answer (19). Masking also cost +1.27 turns per
question in repeated work. Summaries recorded the agent's current beliefs as facts, including a right
timestamp paired with the wrong deploy ID, and the agent then answered without re-checking.

**Noise floor.** Running P4 twice with identical code at temperature 0 flipped 6 of 48 answers (GPU
floating-point results are not bit-identical between runs), while total cost differed by only 2%.
Cost conclusions are robust; accuracy differences of 2–3 questions are not.

## 4. Three takeaways for teams deploying agents

1. **Count uncached tokens, not tokens.** A context policy is only cheaper if it lowers
   *uncached* input. With a steep cache discount, the cache-friendly baseline (append-only history)
   is very hard to beat.
2. **If you must edit, edit rarely and late.** Batch edits, and keep the system prompt and tool list
   byte-identical across turns. Fold contexts in large, infrequent steps (like summarization) rather
   than small edits every turn.
3. **Audit prompt hygiene first.** Timestamps, request IDs or reordered tool lists near the top of a
   prompt can cost more than every other optimization combined, and they are a one-line fix.

## 5. Limitations

One 4B model, one synthetic world, 48 questions per policy and one seed (plus one repeat): accuracy
results are suggestive. Concurrency 1, so cache eviction under load is not tested. Two illustrative
price sheets; real providers add rules (minimum cacheable length, cache lifetime) not modelled here.
P5 and the cadence sweep used 4-session subsets. Next steps: KV-level dropping as in AKV, multi-user
load, larger models, and a real API bill as a check.