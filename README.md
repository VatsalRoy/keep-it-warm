# Keep It Warm

Does trimming an AI agent's context actually save money once prompt caching is counted?

## The idea in 3 rows

vLLM prefix cache, Qwen3-4B-Instruct-2507 on a Kaggle T4, ~5.5K-token prompt:

| Request | Prompt tokens | Cached tokens | Time to first token |
|---|---|---|---|
| 1. Cold (new prefix) | 5,553 | 0 | 14.4 s |
| 2. Warm (same prefix, different question) | 5,553 | 5,520 (99.4%) | 0.23 s |
| 3. Same prefix with **one character changed** at the start | 5,556 | 0 | 13.4 s |


One edited character near the start of a prompt throws away the whole cache.
Agent context policies that edit early messages do exactly that, every turn.

_Reproduce: `notebooks/01_cache_smoke_test.ipynb` · raw numbers: `results/apc_smoke_test.json`_

🚧 Work in progress: full results coming.