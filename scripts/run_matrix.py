"""Run the main experiment: a list of jobs (policy + settings + sessions), one GPU group at a time.

Kaggle gives two T4 GPUs. We start one vLLM server per GPU (ports 8000 and 8001) and run one
group of jobs against each, in parallel:

    python scripts/run_matrix.py --group gpu0 --base http://localhost:8000 &
    python scripts/run_matrix.py --group gpu1 --base http://localhost:8001 &

Every job writes to runs/main/<job name>/ and is resumable: if the session dies, rerun the same
command and finished sessions are skipped. Use --fake to dry-run everything on a laptop.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = "s02,s03,s08,s10"        # longest sessions: where cadence matters most
P5_SET = "s01,s05,s07,s12"       # P5 re-reads EVERYTHING every turn, so a 4-session sample

# (job name, run_episodes.py arguments). Groups are balanced to ~equal GPU time.
MATRIX = {
    "gpu0": [
        ("P1_window6",   ["--policy", "window", "--k", "6"]),
        ("P4_summary8k", ["--policy", "summary", "--threshold", "8192"]),
        ("P5_timestamp", ["--policy", "timestamp", "--sessions", P5_SET]),
    ],
    "gpu1": [
        ("P2_mask4",     ["--policy", "mask", "--k", "4"]),
        ("P3_batch6",    ["--policy", "batch", "--every", "6", "--k", "4"]),
        ("P3_batch3",    ["--policy", "batch", "--every", "3", "--k", "4", "--sessions", HEAVY]),
        ("P3_batch10",   ["--policy", "batch", "--every", "10", "--k", "4", "--sessions", HEAVY]),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(MATRIX))
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out-root", default="runs/main")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    for name, args in MATRIX[a.group]:
        out = Path(a.out_root) / name
        cmd = [sys.executable, str(ROOT / "scripts" / "run_episodes.py"), *args,
               "--out", str(out), "--resume", "--base", a.base] + (["--fake"] if a.fake else [])
        t0 = time.time()
        print(f"[{a.group}] START {name}: {' '.join(args)}", flush=True)
        code = subprocess.call(cmd)
        print(f"[{a.group}] {'DONE' if code == 0 else f'FAILED (exit {code})'} {name} "
              f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[{a.group}] ALL JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()