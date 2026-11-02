import json
from collections import Counter

import pytest

from agent.client import FakeClient
from agent.loop import run_episode
from env.generate_world import World
from env.make_sessions import SESSION_LEN, build_sessions
from env.make_tasks import build_tasks
from env.tools import IncidentDesk
from policies.full import FullHistory
from scripts.run_episodes import oracle_script


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    w = World(42).build()
    w.write(root)
    tasks = build_tasks(w)
    return IncidentDesk(root), {t["id"]: t for t in tasks}, build_sessions([t["id"] for t in tasks])


# ---- session construction ------------------------------------------------- #
def test_every_task_appears_twice_at_mirrored_positions(setup):
    _, tasks, sessions = setup
    assert len(sessions) == 12 and all(len(s["tasks"]) == SESSION_LEN for s in sessions)
    assert Counter(t for s in sessions for t in s["tasks"]) == {t: 2 for t in tasks}
    pos = {}
    for s in sessions:
        for i, t in enumerate(s["tasks"]):
            pos.setdefault(t, []).append(i)
    for t, (p1, p2) in pos.items():
        assert p1 + p2 == SESSION_LEN - 1, t  # 0<->3, 1<->2: once early, once late


def test_no_duplicates_inside_a_session(setup):
    _, _, sessions = setup
    assert all(len(set(s["tasks"])) == SESSION_LEN for s in sessions)


def test_sessions_are_deterministic(setup):
    _, tasks, sessions = setup
    assert build_sessions(list(tasks)) == sessions


def test_sessions_file_matches_builder(setup):
    _, _, sessions = setup
    on_disk = [json.loads(line) for line in open("tasks/sessions.jsonl", encoding="utf-8")]
    assert on_disk == sessions, "regenerate with: python -m env.make_sessions"


# ---- running a session ---------------------------------------------------- #
def test_fake_session_answers_all_questions_in_one_conversation(setup):
    desk, tasks, sessions = setup
    s = sessions[0]
    ts = [tasks[t] for t in s["tasks"]]
    client = FakeClient()
    client.load_script([a for t in ts for a in oracle_script(desk, t)])
    rows, ep = run_episode(ts, desk, client, FullHistory(), session_id=s["id"])
    assert ep["n_correct"] == SESSION_LEN and ep["end_reason"] == "completed"
    assert [q["position"] for q in ep["questions"]] == list(range(SESSION_LEN))
    assert sorted({r["question_idx"] for r in rows}) == list(range(SESSION_LEN))
    # questions 2..4 are appended to the SAME history, numbered, after "Answer recorded."
    task_msgs = [m["content"] for m in ep["transcript"] if m["kind"] == "task"]
    assert len(task_msgs) == SESSION_LEN
    assert task_msgs[0].startswith("You will be asked 4 questions")
    assert "Answer recorded." in task_msgs[1] and "Question 2 of 4" in task_msgs[1]
    # the context keeps growing across questions and the prefix never changes under P0
    assert rows[-1]["prompt_tokens"] > 3 * rows[0]["prompt_tokens"]
    assert all(r["first_changed_token_idx"] == r["prev_prompt_tokens"] for r in rows[1:])


def test_turn_limit_moves_on_to_next_question(setup):
    desk, tasks, _ = setup
    ts = [tasks["t1-errcount-1"], tasks["t1-errcount-2"]]
    client = FakeClient()
    stuck = ['{"tool": "ls", "args": {"path": "/"}}'] * 3
    client.load_script(stuck + oracle_script(desk, ts[1]))
    rows, ep = run_episode(ts, desk, client, FullHistory(), max_turns_per_question=3)
    q1, q2 = ep["questions"]
    assert q1["end"] == "question_turn_limit" and not q1["correct"]
    assert q2["end"] == "submitted" and q2["correct"]
    assert any("turn limit for this question reached" in m["content"] for m in ep["transcript"])


def test_overflow_marks_remaining_questions_not_reached(setup):
    desk, tasks, _ = setup
    ts = [tasks["t1-errcount-1"], tasks["t1-errcount-2"], tasks["t1-errcount-3"]]
    client = FakeClient(max_context=1500)
    client.load_script(oracle_script(desk, ts[0]) + ['{"tool": "read", "args": {"path": "/deploys.csv", "n_lines": 60}}'] * 5)
    rows, ep = run_episode(ts, desk, client, FullHistory())
    assert ep["end_reason"] == "context_overflow"
    assert [q["end"] for q in ep["questions"]] == ["submitted", "context_overflow", "not_reached"]