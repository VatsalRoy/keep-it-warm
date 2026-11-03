"""End-to-end check of analysis/analyze.py on a small fake experiment (no GPU)."""
import csv
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")


def sh(*args):
    subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)


def test_analysis_runs_and_p0_is_its_own_reference(tmp_path):
    runs = tmp_path / "runs"
    for policy, name, extra in [("full", "P0_full", []), ("window", "P1_window6", []), ("mask", "P2_mask4", []),
                                ("batch", "P3_batch6", []), ("summary", "P4_summary8k", []),
                                ("timestamp", "P5_timestamp", ["--sessions", "s01,s05,s07,s12"])]:
        sh("scripts/run_episodes.py", "--fake", "--policy", policy, "--sessions",
           extra[1] if extra else "s01,s05,s07,s12,s10", "--out", str(runs / name))
    out = tmp_path / "results"
    sh("analysis/analyze.py", "--runs", str(runs), "--out", str(out), "--boot", "50", "--money-session", "s10")
    rows = {r["policy"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    assert float(rows["P0 full"]["ratio_B"]) == pytest.approx(1.0)
    assert float(rows["P1 window"]["ratio_B"]) > 1.0          # the fake agent replays P0's steps, so P1 only adds misses
    assert float(rows["P1 window"]["cached_pct"]) < float(rows["P0 full"]["cached_pct"])
    for f in ["summary.md", "naive_vs_true.md", "subsets.md", "h4_position.md", "failure_tags.md",
              "figures/headline.png", "figures/where_the_money_goes.png"]:
        assert (out / f).exists(), f
    assert "P5 timestamp" in (out / "subsets.md").read_text()