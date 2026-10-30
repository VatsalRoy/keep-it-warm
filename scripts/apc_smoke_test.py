"""Day-1 cache check: does vLLM's prefix cache actually work, and what breaks it?

Sends 3 requests to a running vLLM server (concurrency 1):
  1. cold   - a ~4K-token prefix + question A        -> expect ~0 cached tokens
  2. warm   - the SAME prefix + question B           -> expect ~prefix length cached
  3. edited - the prefix with ONE character changed near the start + question A
                                                      -> expect ~0 cached tokens again

Cached tokens are read two ways: the response's usage.prompt_tokens_details.cached_tokens
(needs --enable-prompt-tokens-details) and the change in the server's /metrics counters.

    python scripts/apc_smoke_test.py --base http://localhost:8000 --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

METRICS = ["vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:prompt_tokens_cached"]


def read_metrics(base: str) -> dict[str, float]:
    """Sum each counter across label sets. Prometheus adds a _total suffix to counters."""
    text = requests.get(f"{base}/metrics", timeout=10).text
    out = {m: 0.0 for m in METRICS}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for m in METRICS:
            hit = re.match(rf"^{re.escape(m)}(?:_total)?(?:\{{[^}}]*\}})?\s+(\S+)$", line)
            if hit:
                out[m] += float(hit.group(1))
    return out


def build_prefix(world: Path, run_tag: str, target_lines: int) -> str:
    """A realistic prefix: instructions + a block of real log lines from the world."""
    log = (world / "logs" / "payments" / "2026-10-02.log").read_text().splitlines()
    body = "\n".join(log[:target_lines])
    return (f"[run {run_tag}] You are an on-call assistant. Below is a log excerpt from the "
            f"payments service. Answer questions about it briefly.\n\n{body}")


def chat(base: str, model: str, system: str, question: str) -> dict:
    """One streaming request. Returns token counts and time to first token."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    with requests.post(f"{base}/v1/chat/completions", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                ttft = time.perf_counter() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens_usage": details.get("cached_tokens"),
        "ttft_ms": round((ttft or (time.perf_counter() - t0)) * 1000, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--world", default="world")
    ap.add_argument("--lines", type=int, default=100, help="log lines in the prefix (~40 tokens each)")
    ap.add_argument("--out", default="results/apc_smoke_test.json")
    a = ap.parse_args()

    # A fresh tag per run, so request 1 is truly cold even if you run the script twice.
    tag = f"{random.randrange(16**6):06x}"
    prefix = build_prefix(Path(a.world), tag, a.lines)
    # One-character edit near the very start: change the first letter of "You".
    edited = prefix.replace("You are", "Xou are", 1)
    assert edited != prefix and len(edited) == len(prefix)

    plan = [
        ("1 cold: new prefix", prefix, "How many lines have status=500? Reply with a number."),
        ("2 warm: same prefix", prefix, "What is the first request ID? Reply with the ID only."),
        ("3 edited: 1 char changed at start", edited, "How many lines have status=500? Reply with a number."),
    ]
    rows = []
    for label, system, question in plan:
        before = read_metrics(a.base)
        res = chat(a.base, a.model, system, question)
        time.sleep(0.5)  # let the server update its counters
        after = read_metrics(a.base)
        res["cache_hits_metric"] = int(after["vllm:prefix_cache_hits"] - before["vllm:prefix_cache_hits"])
        res["cache_queries_metric"] = int(after["vllm:prefix_cache_queries"] - before["vllm:prefix_cache_queries"])
        res["request"] = label
        rows.append(res)

    print("\n| Request | Prompt tokens | Cached (usage) | Cache hits (/metrics) | TTFT (ms) |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['request']} | {r['prompt_tokens']} | {r['cached_tokens_usage']} | "
              f"{r['cache_hits_metric']} | {r['ttft_ms']} |")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "run_tag": tag, "rows": rows}, indent=2) + "\n")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()