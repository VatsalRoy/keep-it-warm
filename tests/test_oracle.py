import json

import pytest

from env.generate_world import World
from env.make_tasks import build_tasks
from env.oracle import Oracle
from env.grader import grade
from env.tools import IncidentDesk


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    w = World(42).build()
    w.write(root)
    return IncidentDesk(root), build_tasks(w)


def test_task_counts_by_tier(setup):
    _, tasks = setup
    assert len(tasks) == 24
    assert {t["tier"] for t in tasks} == {1, 2, 3}
    assert len({t["id"] for t in tasks}) == 24  # ids are unique


def test_every_task_has_required_fields(setup):
    _, tasks = setup
    for t in tasks:
        assert set(t) >= {"id", "tier", "template", "question", "answer", "match", "clue_turn_expected"}
        assert t["clue_turn_expected"] in {"early", "mid", "late"}
        assert t["answer"], t["id"]


@pytest.mark.parametrize("i", range(24))
def test_oracle_solves_each_task(setup, i):
    desk, tasks = setup
    task = tasks[i]
    answer = Oracle(desk).solve(task)
    assert grade(task, answer), f"{task['id']}: oracle said {answer!r}, gold is {task['answer']}"


def test_tasks_file_is_deterministic(setup):
    _, tasks = setup
    again = build_tasks(World(42).build())
    assert json.dumps(tasks, sort_keys=True) == json.dumps(again, sort_keys=True)