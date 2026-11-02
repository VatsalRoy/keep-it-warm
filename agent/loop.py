"""The agent loop: ask the model for an action, run the tool, repeat, and log every turn.

One call to run_episode() = one SESSION: a list of 1..N questions answered in ONE conversation
under one policy and one seed. When the agent submits an answer, the next question is appended
to the same history, so context from earlier questions keeps piling up (that's what makes the
episode long-horizon). Each turn produces one telemetry row; each episode one summary row.
"""
from __future__ import annotations

import time
import uuid

from agent.actions import ActionError, parse_action
from agent.client import ContextOverflow
from agent.prompt import observation_message, question_message, system_prompt
from env.grader import grade
from env.tools import IncidentDesk

MAX_TURNS_PER_QUESTION = 20


def first_diff(prev: list, cur: list) -> int:
    """Index of the first token that differs. If `prev` is a prefix of `cur`, returns len(prev):
    the edit happened only after everything the cache already has (the ideal case)."""
    n = min(len(prev), len(cur))
    for i in range(n):
        if prev[i] != cur[i]:
            return i
    return n


def run_episode(tasks: list[dict] | dict, desk: IncidentDesk, client, policy, seed: int = 0,
                session_id: str | None = None, max_turns_per_question: int = MAX_TURNS_PER_QUESTION,
                system: str | None = None) -> tuple[list[dict], dict]:
    tasks = [tasks] if isinstance(tasks, dict) else list(tasks)
    n = len(tasks)
    session_id = session_id or tasks[0]["id"]
    episode_id = f"{session_id}__{policy.name}__s{seed}__{uuid.uuid4().hex[:6]}"
    history = [
        {"role": "system", "content": system or system_prompt(), "kind": "system"},
        {"role": "user", "content": question_message(tasks[0], 0, n), "kind": "task"},
    ]
    results = [{"task_id": t["id"], "tier": t["tier"], "template": t["template"], "position": i,
                "clue_turn_expected": t["clue_turn_expected"], "submitted": None, "correct": False,
                "turns": 0, "end": "not_reached"} for i, t in enumerate(tasks)]
    rows: list[dict] = []
    prev_tokens: list | None = None
    q, q_turns, turn, end_reason = 0, 0, 0, "completed"
    t_start = time.perf_counter()

    def next_question(prefix: str) -> bool:
        """Move on to the next question. Returns False when the session is over."""
        nonlocal q, q_turns
        q, q_turns = q + 1, 0
        if q >= n:
            return False
        history.append({"role": "user", "kind": "task", "content": f"{prefix}\n\n{question_message(tasks[q], q, n)}"})
        return True

    while q < n:
        turn += 1
        q_turns += 1
        task = tasks[q]
        sent = policy.build(history, turn)
        sent_tokens = client.tokenize(sent)
        logical = len(sent_tokens) if sent == history else len(client.tokenize(history))
        try:
            reply = client.chat(sent, seed=seed)
        except ContextOverflow:
            end_reason = "context_overflow"
            results[q]["end"] = "context_overflow"
            break

        row = {
            "episode_id": episode_id, "session_id": session_id, "task_id": task["id"], "tier": task["tier"],
            "question_idx": q, "policy": policy.name, "seed": seed, "turn": turn,
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
        results[q]["turns"] = q_turns

        try:
            tool, args = parse_action(reply.text)
            row.update(action=tool, ok=True)
        except ActionError as e:
            row.update(action=None, ok=False, error=str(e))
            rows.append(row)
            msg = observation_message("error", f"Could not parse your reply ({e}). Reply with exactly one JSON object.")
            if q_turns >= max_turns_per_question:
                results[q]["end"] = "question_turn_limit"
                if not next_question(f"{msg}\n[turn limit for this question reached; no answer recorded]"):
                    break
            else:
                history.append({"role": "user", "kind": "observation", "content": msg})
            continue

        rows.append(row)
        if tool == "submit":
            ans = str(args.get("answer", ""))
            results[q].update(submitted=ans, correct=grade(task, ans), end="submitted")
            if not next_question("[submit result]\nAnswer recorded."):
                break
            continue
        output = observation_message(tool, desk.call(tool, args))
        if q_turns >= max_turns_per_question:
            results[q]["end"] = "question_turn_limit"
            if not next_question(f"{output}\n[turn limit for this question reached; no answer recorded]"):
                break
        else:
            history.append({"role": "user", "kind": "observation", "content": output})

    summary = {
        "episode_id": episode_id, "session_id": session_id, "policy": policy.name, "seed": seed,
        "n_questions": n, "n_correct": sum(r["correct"] for r in results), "questions": results,
        "turns": len(rows), "end_reason": end_reason, "wall_s": round(time.perf_counter() - t_start, 1),
        "transcript": [{"role": m["role"], "kind": m["kind"], "content": m["content"]} for m in history],
    }
    return rows, summary