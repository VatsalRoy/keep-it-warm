"""P4: threshold summarization ("compaction").

While the prompt is under `threshold` tokens, the policy is append-only. When it would go over,
the policy asks the model to summarize everything so far, then continues from:
    [system prompt] + [summary + the current question] + (new steps, appended as usual)
One big cache miss per fold, warm in between. The summary request itself is a real model call,
so it is logged and billed (see Policy.calls).
"""
from __future__ import annotations

from policies.base import Policy

SUMMARY_REQUEST = (
    "Pause and summarize your work so far, for your own later use; the full history will be removed. "
    "Keep every concrete fact you may still need: file paths, line numbers, timestamps, request ids, "
    "deploy ids, names, answers already submitted, and what is still missing for the current question. "
    "Plain text, no JSON."
)


class SummaryFold(Policy):
    def __init__(self, client, threshold: int = 8192) -> None:
        super().__init__()
        self.client, self.threshold = client, threshold
        self.summary_msg: dict | None = None
        self.keep_from = 1  # history index from which messages are sent verbatim
        self.name = f"P4_summary{threshold // 1024}k"

    def _view(self, history: list[dict]) -> list[dict]:
        head = [history[0]] + ([self.summary_msg] if self.summary_msg else [])
        return head + list(history[self.keep_from:])

    def build(self, history: list[dict], turn: int) -> list[dict]:
        view = self._view(history)
        if len(self.client.tokenize(view)) <= self.threshold:
            return view
        request = view + [{"role": "user", "kind": "task", "content": SUMMARY_REQUEST}]
        reply = self.client.chat(request)
        self.calls.append((request, reply))
        question = next(m["content"] for m in reversed(history) if m["kind"] == "task")
        self.summary_msg = {"role": "user", "kind": "summary", "content":
                            f"Summary of your work so far:\n{reply.text.strip()}\n\n"
                            f"You are still working on this:\n{question}"}
        self.keep_from = len(history)
        return self._view(history)