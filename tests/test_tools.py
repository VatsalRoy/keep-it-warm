import pytest

from env.generate_world import World
from env.tools import READ_MAX_LINES, TOOL_SCHEMAS, IncidentDesk


@pytest.fixture(scope="module")
def desk(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    World(42).build().write(root)
    return IncidentDesk(root)


def test_ls_root(desk):
    out = desk.call("ls", {"path": "/"})
    assert out.startswith("/:")
    for name in ["logs/", "runbooks/", "deploys.csv", "oncall.yaml"]:
        assert name in out


def test_read_returns_numbered_range(desk):
    out = desk.call("read", {"path": "/deploys.csv", "start_line": 2, "n_lines": 3})
    lines = out.splitlines()
    assert lines[0].startswith("== /deploys.csv (lines 2-4 of ")
    assert len(lines) == 1 + 3
    assert lines[1].strip().startswith("2  d-1001,")


def test_read_is_capped(desk):
    out = desk.call("read", {"path": "/logs/payments/2026-10-02.log", "n_lines": 500})
    assert len(out.splitlines()) == 1 + READ_MAX_LINES


def test_grep_finds_planted_oom(desk):
    out = desk.call("grep", {"pattern": "OOMKilled", "path": "/logs/inventory/2026-10-03.log"})
    assert ": 4 matching lines ==" in out
    assert "14:19:02.904Z" in out


def test_tools_are_deterministic(desk):
    args = {"path": "/logs/auth/2026-10-01.log", "start_line": 10, "n_lines": 40}
    assert desk.call("read", args) == desk.call("read", args)


def test_errors_come_back_as_text(desk):
    assert desk.call("read", {"path": "/nope.txt"}).startswith("error: no such file")
    assert desk.call("read", {"path": "/../../etc/passwd"}).startswith("error: path outside")
    assert desk.call("grep", {"pattern": "x", "path": "/logs"}).startswith("error:")
    assert desk.call("read", {}).startswith("error: bad arguments")
    assert desk.call("fly", {}).startswith("error: unknown tool")


def test_tool_order_is_fixed():
    assert [t["function"]["name"] for t in TOOL_SCHEMAS] == ["ls", "read", "grep", "submit"]