"""Pack the recorded runs into one compact file for the Hugging Face Space.

    python space/build_data.py                 # reads runs/main, writes space/data/traces.json.gz

Only what the viewer needs is kept: per-request token counts and the full transcripts.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

# folder -> (label shown in the app, policy kind used to replay what was sent)
RUNS = {
    "P0_full": ("P0 full history", "full"),
    "P1_window6": ("P1 sliding window (6 steps)", "window6"),
    "P2_mask4": ("P2 rolling masking (keep 4)", "mask4"),
    "P3_batch6": ("P3 batched masking (every 6)", "batch6"),
    "P3_batch3": ("P3 batched masking (every 3)", "batch3"),
    "P3_batch10": ("P3 batched masking (every 10)", "batch10"),
    "P4_summary8k_v2": ("P4 summary at 8K", "summary"),
    "P5_timestamp": ("P5 timestamp bug", "timestamp"),
}
ROW_KEYS = ["turn", "question_idx", "action", "prompt_tokens", "cached_tokens", "uncached_tokens",
            "completion_tokens", "first_changed_token_idx", "prev_prompt_tokens", "ttft_ms"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/main")
    ap.add_argument("--out", default="space/data/traces.json.gz")
    a = ap.parse_args()
    data = {"policies": {}}
    for folder, (label, kind) in RUNS.items():
        d = Path(a.runs) / folder
        if not (d / "episodes.jsonl").exists():
            print(f"skip {folder} (not found)")
            continue
        rows = {}
        for line in open(d / "turns.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rows.setdefault(r["episode_id"], []).append({k: r.get(k) for k in ROW_KEYS})
        sessions = {}
        for line in open(d / "episodes.jsonl", encoding="utf-8"):
            e = json.loads(line)
            sessions[e["session_id"]] = {
                "questions": [{k: q[k] for k in ("task_id", "tier", "position", "submitted", "correct", "turns", "end")}
                              for q in e["questions"]],
                "transcript": e["transcript"], "policy_notes": e.get("policy_notes", []),
                "rows": rows.get(e["episode_id"], []),
            }
        data["policies"][label] = {"kind": kind, "folder": folder, "sessions": sessions}
        print(f"{folder:<18} {len(sessions):>2} sessions")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()