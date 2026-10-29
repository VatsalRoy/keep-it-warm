# Research notes: Keep It Warm

## Research question
For a long-horizon tool-using agent, which context-management policy gives
the best accuracy per dollar once prefix-cache pricing is counted, and how
much depends on where and how often the context is edited?

## In my own words
<!-- 3-4 sentences: why can trimming an agent's history make it MORE expensive? -->

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