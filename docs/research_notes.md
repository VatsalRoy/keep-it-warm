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