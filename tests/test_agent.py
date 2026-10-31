import json

import pytest

from agent.actions import ActionError, parse_action
from agent.client import ContextOverflow, FakeClient
from agent.loop import first_diff, run_episode
from agent.pricing import cost, naive_cost
from env.generate_world import World
from env.make_tasks import build_tasks
from env.tools import IncidentDesk
from policies.full import FullHistory
from scripts.run_episodes import oracle_script


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    w = World(42).build()
    w.write(root)
    return IncidentDesk(root), {t["id"]: t for t in build_tasks(w)}


# ---- action parsing ------------------------------------------------------- #
def test_parse_plain_json():
    assert parse_action('{"tool": "ls", "args": {"path": "/"}}') == ("ls", {"path": "/"})


def test_parse_json_with_chatter_and_braces_in_strings():
    text = 'Sure! {"thought": "look for {braces}", "tool": "grep", "args": {"pattern": "a{2}", "path": "/x"}} done'
    assert parse_action(text) == ("grep", {"pattern": "a{2}", "path": "/x"})


@pytest.mark.parametrize("bad", ["no json here", '{"tool": "fly", "args": {}}', '{"tool": "ls", "args": 3}', '{"tool": '])
def test_parse_rejects_bad_actions(bad):
    with pytest.raises(ActionError):
        parse_action(bad)


# ---- helpers -------------------------------------------------------------- #
def test_first_diff():
    assert first_diff([1, 2, 3], [1, 2, 3, 4]) == 3   # pure append: change starts after old end
    assert first_diff([1, 2, 3], [1, 9, 3, 4]) == 1   # edit at position 1


def test_cache_makes_it_cheaper():
    assert cost(0, 10_000, 0, "B") == pytest.approx(naive_cost(10_000, 0, "B") / 10)
    assert cost(0, 10_000, 0, "A") == pytest.approx(naive_cost(10_000, 0, "A") / 2)


# ---- full episodes with the fake model ------------------------------------ #
def test_fake_episode_solves_task_and_keeps_cache_warm(setup):
    desk, tasks = setup
    task = tasks["t2-oom-2"]
    client = FakeClient()
    client.load_script(oracle_script(desk, task))
    rows, ep = run_episode(task, desk, client, FullHistory())
    assert ep["correct"] and ep["end_reason"] == "submitted"
    assert ep["submitted"] == "d-1050, aisha.bello"
    for r in rows[1:]:
        # P0 only appends, so the prompt never changes before the previous end ...
        assert r["first_changed_token_idx"] == r["prev_prompt_tokens"]
        # ... so almost everything is cached (all full blocks of the previous prompt).
        assert r["cached_tokens"] >= (r["prev_prompt_tokens"] // 16) * 16 - 16
    json.dumps(rows), json.dumps(ep)  # everything must be JSON-serialisable


def test_unparseable_reply_is_logged_and_agent_continues(setup):
    desk, tasks = setup
    task = tasks["t1-errcount-1"]
    client = FakeClient()
    client.load_script(["I think I will look around first.", *oracle_script(desk, task)])
    rows, ep = run_episode(task, desk, client, FullHistory())
    assert rows[0]["ok"] is False and "error" in rows[0]
    assert ep["correct"]
    assert "Could not parse" in ep["transcript"][3]["content"]


def test_turn_limit(setup):
    desk, tasks = setup
    client = FakeClient()
    client.load_script(['{"tool": "ls", "args": {"path": "/"}}'] * 10)
    rows, ep = run_episode(tasks["t1-errcount-1"], desk, client, FullHistory(), max_turns=4)
    assert len(rows) == 4 and ep["end_reason"] == "turn_limit" and not ep["correct"]


def test_context_overflow_ends_episode(setup):
    desk, tasks = setup
    client = FakeClient(max_context=600)
    client.load_script(['{"tool": "read", "args": {"path": "/deploys.csv", "n_lines": 60}}'] * 10)
    rows, ep = run_episode(tasks["t1-errcount-1"], desk, client, FullHistory())
    assert ep["end_reason"] == "context_overflow"
    with pytest.raises(ContextOverflow):
        FakeClient(max_context=5).chat([{"role": "user", "content": "x" * 100}])


def test_system_prompt_leaks_no_task_ids(setup):
    """The prompt's examples must not contain any request id or deploy id used by a task."""
    import re

    from agent.prompt import system_prompt
    _, tasks = setup
    prompt = system_prompt()
    for t in tasks.values():
        for rid in re.findall(r"r-\d{4}", t["question"]):
            assert rid not in prompt, rid
        for item in t["answer"]:
            if re.fullmatch(r"d-\d{4}", item):
                assert item not in prompt, item