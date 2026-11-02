import json
import subprocess
import sys


def run(out, *extra):
    cmd = [sys.executable, "scripts/run_episodes.py", "--fake", "--policy", "batch", "--sessions", "s01,s02",
           "--out", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def test_resume_skips_finished_sessions(tmp_path):
    out = tmp_path / "P3"
    run(out)
    first = (out / "episodes.jsonl").read_text().splitlines()
    log = run(out, "--resume")
    assert "2 episode(s) already saved" in log
    assert (out / "episodes.jsonl").read_text().splitlines() == first  # nothing duplicated
    assert {json.loads(x)["session_id"] for x in first} == {"s01", "s02"}


def test_without_resume_a_rerun_appends(tmp_path):
    out = tmp_path / "P3"
    run(out)
    run(out)
    assert len((out / "episodes.jsonl").read_text().splitlines()) == 4  # why --resume matters