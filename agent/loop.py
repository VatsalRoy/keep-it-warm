"""The agent loop: ask the model for an action, run the tool, repeat, and log every turn.

One call to run_episode() = one task solved (or not) under one policy and one seed.
Each turn produces one telemetry row; the episode produces one summary row.
"""
from __future__ import annotations

import time
import uuid

from agent.actions import ActionError, parse_action
from agent.client import ContextOverflow
from agent.prompt import observation_message, system_prompt, task_message
from env.grader import grade
from env.tools import IncidentDesk

MAX_TURNS = 25


def first_diff(prev: list, cur: list) -> int:
    """Index of the first token that differs. If `prev` is a prefix of `cur`, returns len(prev):
    the edit happened only after everything the cache already has (the ideal case)."""
    n = min(len(prev), len(cur))
    for i in range(n):
        if prev[i] != cur[i]:
            return i
    return n


def initial_history(task: dict, system: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": system or system_prompt(), "kind": "system"},
        {"role": "user", "content": task_message(task), "kind": "task"},
    ]


def run_episode(task: dict, desk: IncidentDesk, client, policy, seed: int = 0,
                max_turns: int = MAX_TURNS, system: str | None = None) -> tuple[list[dict], dict]:
    episode_id = f"{task['id']}__{policy.name}__s{seed}__{uuid.uuid4().hex[:6]}"
    history = initial_history(task, system)
    rows: list[dict] = []
    prev_tokens: list | None = None
    submitted, end_reason = None, "turn_limit"
    t_start = time.perf_counter()

    for turn in range(1, max_turns + 1):
        sent = policy.build(history, turn)
        sent_tokens = client.tokenize(sent)
        logical = len(sent_tokens) if sent == history else len(client.tokenize(history))
        try:
            reply = client.chat(sent, seed=seed)
        except ContextOverflow:
            end_reason = "context_overflow"
            break

        row = {
            "episode_id": episode_id, "task_id": task["id"], "tier": task["tier"],
            "policy": policy.name, "seed": seed, "turn": turn,
            "prompt_tokens": reply.prompt_tokens, "cached_tokens": reply.cached_tokens,
            "uncached_tokens": reply.prompt_tokens - reply.cached_tokens,
            "completion_tokens": reply.completion_tokens,
            "logical_tokens": logical, "active_context_tokens": len(sent_tokens),
            "first_changed_token_idx": None if prev_tokens is None else first_diff(prev_tokens, sent_tokens),
            "prev_prompt_tokens": None if prev_tokens is None else len(prev_tokens),
            "ttft_ms": reply.ttft_ms, "e2e_ms": reply.e2e_ms, "cached_source": reply.cached_source,
        }
        prev_tokens = sent_tokens
        history.append({"role": "assistant", "content": reply.text, "kind": "action"})

        try:
            tool, args = parse_action(reply.text)
            row.update(action=tool, ok=True)
        except ActionError as e:
            row.update(action=None, ok=False, error=str(e))
            history.append({"role": "user", "kind": "observation", "content": observation_message(
                "error", f"Could not parse your reply ({e}). Reply with exactly one JSON object.")})
            rows.append(row)
            continue

        rows.append(row)
        if tool == "submit":
            submitted, end_reason = str(args.get("answer", "")), "submitted"
            break
        output = desk.call(tool, args)
        history.append({"role": "user", "kind": "observation", "content": observation_message(tool, output)})

    summary = {
        "episode_id": episode_id, "task_id": task["id"], "tier": task["tier"],
        "clue_turn_expected": task["clue_turn_expected"], "policy": policy.name, "seed": seed,
        "turns": len(rows), "end_reason": end_reason, "submitted": submitted,
        "correct": grade(task, submitted), "wall_s": round(time.perf_counter() - t_start, 1),
        "transcript": [{"role": m["role"], "kind": m["kind"], "content": m["content"]} for m in history],
    }
    return rows, summary