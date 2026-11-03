"""The Space replays policies from transcripts. Check the replay matches what the real policy sent."""
import importlib
import json
import os
import sys

import pytest

pytest.importorskip("gradio")

from agent.client import FakeClient  # noqa: E402
from agent.loop import run_episode  # noqa: E402
from env.generate_world import World  # noqa: E402
from env.make_sessions import build_sessions  # noqa: E402
from env.make_tasks import build_tasks  # noqa: E402
from env.tools import IncidentDesk  # noqa: E402
from policies.full import FullHistory  # noqa: E402
from policies.hygiene_bug import HygieneBug  # noqa: E402
from policies.mask_batched import BatchedMask  # noqa: E402
from policies.mask_rolling import RollingMask  # noqa: E402
from policies.sliding import SlidingWindow  # noqa: E402
from policies.summary import SummaryFold  # noqa: E402
from scripts.run_episodes import oracle_script  # noqa: E402


class Recorder:
    """Wraps a policy and remembers exactly which messages it sent on every step."""
    def __init__(self, inner):
        self.inner, self.sent = inner, []
        self.name, self.notes = inner.name, inner.notes

    def build(self, history, turn):
        out = self.inner.build(history, turn)
        self.sent.append([m["content"] for m in out])
        return out

    def drain_calls(self):
        return self.inner.drain_calls()


CASES = [("full", lambda c: FullHistory()), ("window6", lambda c: SlidingWindow(6)), ("mask4", lambda c: RollingMask(4)),
         ("batch6", lambda c: BatchedMask(6, 4)), ("batch3", lambda c: BatchedMask(3, 4)),
         ("summary", lambda c: SummaryFold(c, threshold=1500)), ("timestamp", lambda c: HygieneBug())]


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    import gzip
    d = tmp_path_factory.mktemp("space")
    (d / "t.json.gz").write_bytes(gzip.compress(json.dumps(
        {"policies": {"P0 full history": {"kind": "full", "folder": "x", "sessions": {"s01": {
            "questions": [], "transcript": [{"role": "system", "kind": "system", "content": "x"},
                                            {"role": "assistant", "kind": "action", "content": "{}"}],
            "policy_notes": [], "rows": []}}}}}).encode()))
    os.environ["TRACES_PATH"] = str(d / "t.json.gz")
    sys.path.insert(0, "space")
    return importlib.import_module("app")


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    w = World(42).build()
    w.write(root)
    tasks = {t["id"]: t for t in build_tasks(w)}
    s10 = next(s for s in build_sessions(list(tasks)) if s["id"] == "s10")
    return IncidentDesk(root), [tasks[t] for t in s10["tasks"]]


@pytest.mark.parametrize("kind,make", CASES, ids=[c[0] for c in CASES])
def test_replay_matches_what_the_policy_sent(app_module, session, kind, make):
    desk, ts = session
    client = FakeClient()
    client.load_script([a for t in ts for a in oracle_script(desk, t)])
    rec = Recorder(make(client))
    _, ep = run_episode(ts, desk, client, rec, session_id="s10")
    tr, notes = ep["transcript"], ep["policy_notes"]
    assert len(rec.sent) == len(app_module.actions(tr))
    for k, real in enumerate(rec.sent, start=1):
        replay, _ = app_module.sent_messages(kind, tr, notes, k)
        if kind == "timestamp":   # the app shows the stored system prompt, the real one had a timestamp header
            real, replay = real[1:], replay[1:]
        assert [m["content"] for m in replay] == real, f"{kind}: step {k} differs"


def test_what_if_reproduces_the_qualitative_result(app_module):
    sim = {p: app_module.simulate(p, 800, 25, 60, 1200, 25, 6, 4, 6) for p in ["P0", "P1", "P2", "P3", "P5"]}
    tokens = {p: r["cached"] + r["uncached"] for p, r in sim.items()}
    b = lambda r: r["uncached"] * 5 + r["cached"] * 0.5  # noqa: E731
    assert tokens["P1"] < tokens["P0"] and tokens["P2"] < tokens["P0"]            # trimming sends fewer tokens ...
    assert b(sim["P1"]) > b(sim["P0"]) and b(sim["P2"]) > b(sim["P0"])            # ... but costs more (sheet B)
    assert b(sim["P3"]) < b(sim["P2"])                                            # batching helps
    assert sim["P5"]["cached"] == 0                                               # timestamp kills the cache