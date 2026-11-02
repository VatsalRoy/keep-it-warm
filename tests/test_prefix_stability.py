"""WHERE does each policy change the prompt? Checked on a fake run before any GPU time is spent.

For every turn the loop records first_changed_token_idx (where this prompt first differs from the
previous one) and prev_prompt_tokens (how long the previous one was).
  first_changed == prev_prompt_tokens  -> append-only, the cache stays warm
  first_changed <  prev_prompt_tokens  -> an EARLIER part changed, everything after it is a cache miss
"""
import copy

import pytest

from agent.client import FakeClient
from agent.loop import run_episode
from env.generate_world import World
from env.make_sessions import build_sessions
from env.make_tasks import build_tasks
from env.tools import IncidentDesk
from policies.full import FullHistory
from policies.hygiene_bug import HygieneBug
from policies.mask_batched import BatchedMask
from policies.mask_rolling import RollingMask
from policies.sliding import SlidingWindow
from policies.summary import SummaryFold
from scripts.run_episodes import oracle_script


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    w = World(42).build()
    w.write(root)
    tasks = {t["id"]: t for t in build_tasks(w)}
    sessions = {s["id"]: s for s in build_sessions(list(tasks))}
    return IncidentDesk(root), [tasks[t] for t in sessions["s10"]["tasks"]]  # s10 = longest session


def run(world, make_policy):
    desk, ts = world
    client = FakeClient()
    client.load_script([a for t in ts for a in oracle_script(desk, t)])
    return run_episode(ts, desk, client, make_policy(client), session_id="s10")


def early(r) -> bool:
    return r["first_changed_token_idx"] is not None and r["first_changed_token_idx"] < r["prev_prompt_tokens"]


def test_p0_full_is_append_only(world):
    rows, ep = run(world, lambda c: FullHistory())
    assert ep["n_correct"] == 4
    assert not any(early(r) for r in rows)


def test_p1_window_breaks_every_turn_once_window_is_full(world):
    k = 6
    rows, ep = run(world, lambda c: SlidingWindow(k))
    assert ep["n_correct"] == 4
    assert not any(early(r) for r in rows[:k + 1])    # window not full yet: append-only
    assert all(early(r) for r in rows[k + 1:])        # then the oldest kept step changes every turn


def test_p2_rolling_mask_breaks_almost_every_turn(world):
    k = 4
    rows, ep = run(world, lambda c: RollingMask(k))
    later = rows[k + 1:]
    assert not any(early(r) for r in rows[:k + 1])
    assert sum(early(r) for r in later) >= 0.8 * len(later)  # only turns right after a submit escape
    assert ep["n_correct"] == 4


@pytest.mark.parametrize("every", [3, 6, 10])
def test_p3_batched_mask_breaks_only_on_fold_turns(world, every):
    rows, _ = run(world, lambda c: BatchedMask(every=every, k=4))
    breaks = [r["turn"] for r in rows if early(r)]
    assert breaks, "expected at least one fold"
    assert all(t % every == 0 for t in breaks)       # warm on every non-fold turn


def test_p4_summary_breaks_only_right_after_a_summary_call(world):
    rows, ep = run(world, lambda c: SummaryFold(c, threshold=1500))
    folds = [i for i, r in enumerate(rows) if r["action"] == "summarize"]
    assert len(folds) >= 2
    for i, r in enumerate(rows):
        if early(r):
            assert i - 1 in folds, f"turn {r['turn']} changed early without a fold"
    for i in folds:
        assert not early(rows[i])                    # the summary REQUEST itself extends the cached prompt
        assert rows[i + 1]["active_context_tokens"] < rows[i]["active_context_tokens"]  # and it shrinks the context
    assert not any(m["kind"] == "summary" for m in ep["transcript"])  # the full history itself is untouched


def test_p5_timestamp_breaks_at_the_very_start_every_turn(world):
    rows, _ = run(world, lambda c: HygieneBug())
    assert all(early(r) and r["first_changed_token_idx"] < 20 for r in rows[1:])
    assert all(r["cached_tokens"] == 0 for r in rows)


@pytest.mark.parametrize("make", [lambda c: FullHistory(), lambda c: SlidingWindow(3), lambda c: RollingMask(2),
                                  lambda c: BatchedMask(3, 2), lambda c: HygieneBug()])
def test_policies_never_modify_history(world, make):
    desk, ts = world
    client = FakeClient()
    client.load_script(oracle_script(desk, ts[0]))
    _, ep = run_episode(ts[:1], desk, client, FullHistory())
    history = [dict(m) for m in ep["transcript"]]
    before = copy.deepcopy(history)
    make(client).build(history, turn=12)
    assert history == before


def test_what_masking_and_windowing_keep(world):
    desk, ts = world
    client = FakeClient()
    client.load_script([a for t in ts for a in oracle_script(desk, t)])
    _, ep = run_episode(ts, desk, client, FullHistory())
    history = ep["transcript"]
    n_obs = sum(m["kind"] == "observation" for m in history)
    # P2 masks all but the last 4 tool outputs, keeps their first line, and keeps every action
    sent = RollingMask(4).build(history, turn=99)
    assert sum("output elided" in m["content"] for m in sent) == n_obs - 4
    assert all(m["content"].startswith("[") for m in sent if "output elided" in m["content"])
    assert sum(m["kind"] == "action" for m in sent) == sum(m["kind"] == "action" for m in history)
    # P1 keeps the system prompt and every question, but only the last 6 steps
    sent = SlidingWindow(6).build(history, turn=99)
    assert sum(m["kind"] == "task" for m in sent) == 4 and sent[0]["kind"] == "system"
    assert sum(m["kind"] == "action" for m in sent) == 6