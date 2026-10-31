"""Model clients. Both return the same Reply, so the agent loop can't tell them apart.

VLLMClient - talks to a real vLLM server (streaming, so we can time the first token).
FakeClient - offline stand-in for testing on a laptop: replays a scripted list of actions
             and SIMULATES vLLM's prefix cache (16-token blocks, chained fingerprints).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

import requests

BLOCK = 16


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float
    cached_source: str  # "usage" or "metrics" - where cached_tokens came from


class ContextOverflow(Exception):
    """The prompt no longer fits in the server's max context length."""


def strip_kind(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# --------------------------------------------------------------------------- #
class VLLMClient:
    def __init__(self, base: str, model: str, max_tokens: int = 400, temperature: float = 0.0,
                 timeout: int = 900):
        self.base, self.model = base.rstrip("/"), model
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout

    def _cache_hits(self) -> float:
        text = requests.get(f"{self.base}/metrics", timeout=10).text
        return sum(float(m.group(1)) for m in re.finditer(
            r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+(\S+)$", text, re.M))

    def reset_cache(self) -> bool:
        """Empty the server's prefix cache so every episode starts cold. Needs the server to be
        started with VLLM_SERVER_DEV_MODE=1; returns False if the endpoint isn't available."""
        try:
            return requests.post(f"{self.base}/reset_prefix_cache", timeout=30).status_code == 200
        except requests.RequestException:
            return False

    def tokenize(self, messages: list[dict]) -> list[int]:
        r = requests.post(f"{self.base}/tokenize", timeout=60, json={
            "model": self.model, "messages": strip_kind(messages), "add_generation_prompt": True})
        r.raise_for_status()
        return r.json()["tokens"]

    def chat(self, messages: list[dict], seed: int | None = None) -> Reply:
        payload = {
            "model": self.model, "messages": strip_kind(messages),
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if seed is not None:
            payload["seed"] = seed
        hits_before = self._cache_hits()
        t0 = time.perf_counter()
        ttft, parts, usage = None, [], {}
        with requests.post(f"{self.base}/v1/chat/completions", json=payload, stream=True,
                           timeout=self.timeout) as r:
            if r.status_code == 400 and re.search(r"context length|too long|maximum", r.text, re.I):
                raise ContextOverflow(r.text[:300])
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
                    continue
                chunk = json.loads(raw[6:])
                if chunk.get("choices"):
                    piece = chunk["choices"][0].get("delta", {}).get("content") or ""
                    if piece and ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        e2e = time.perf_counter() - t0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        source = "usage"
        if cached is None:  # older servers: fall back to the /metrics counter delta
            time.sleep(0.2)
            cached, source = int(self._cache_hits() - hits_before), "metrics"
        return Reply("".join(parts), usage.get("prompt_tokens", 0), int(cached),
                     usage.get("completion_tokens", 0), round((ttft or e2e) * 1000, 1),
                     round(e2e * 1000, 1), source)


# --------------------------------------------------------------------------- #
class FakeClient:
    """Offline model for laptop tests. Tokens are approximated as 4-character chunks of a
    Qwen-style chat template, which is enough to show cache hits and misses."""

    def __init__(self, max_context: int = 16384):
        self.max_context = max_context
        self.script: list[str] = []
        self.cache: set[str] = set()

    def reset_cache(self) -> bool:
        self.cache.clear()
        return True

    def load_script(self, replies: list[str]) -> None:
        self.script = list(replies)

    def tokenize(self, messages: list[dict]) -> list[str]:
        toks: list[str] = []
        for m in messages:
            c = m["content"]
            toks += ["<|im_start|>", m["role"]] + [c[i:i + 4] for i in range(0, len(c), 4)] + ["<|im_end|>"]
        return toks + ["<|im_start|>", "assistant"]  # the "generation prompt": your turn, assistant

    def chat(self, messages: list[dict], seed: int | None = None) -> Reply:
        toks = self.tokenize(messages)
        if len(toks) > self.max_context:
            raise ContextOverflow(f"{len(toks)} tokens > {self.max_context}")
        h, cached, warm = hashlib.sha1(), 0, True
        for b in range(len(toks) // BLOCK):
            h.update("\x00".join(toks[b * BLOCK:(b + 1) * BLOCK]).encode())
            key = h.hexdigest()
            if warm and key in self.cache:
                cached += BLOCK
            else:
                warm = False
            self.cache.add(key)
        cached = min(cached, len(toks) - 1)  # like vLLM, always recompute the last token
        text = self.script.pop(0) if self.script else '{"tool": "submit", "args": {"answer": ""}}'
        uncached = len(toks) - cached
        ttft = 5 + uncached * 2.5  # ms; pretend ~400 tokens/s cold prefill, like the T4
        return Reply(text, len(toks), cached, max(1, len(text) // 4), round(ttft, 1),
                     round(ttft + 20, 1), "fake")