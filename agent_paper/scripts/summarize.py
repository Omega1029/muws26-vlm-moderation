"""summarize.py — one row per experiment tag: resolve rate next to the diagnostics that explain it.

  python scripts/summarize.py                 # markdown table to stdout
  python scripts/summarize.py --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swe_agent.diagnostics import aggregate  # noqa: E402

COLS = ["n", "resolve_rate", "submit_rate", "false_submit_rate", "empty_patch_rate",
        "verified_before_submit_rate", "valid_call_rate", "text_fallback_rate", "no_call_rate",
        "dominant_tool_share", "repeat_action_rate", "mean_steps", "mean_tokens"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)

    by_tag = defaultdict(list)
    for line in Path(a.runs).read_text().splitlines():
        r = json.loads(line)
        by_tag[r["tag"]].append(r)
    table = [{"tag": tag, **aggregate(rows)} for tag, rows in sorted(by_tag.items())]

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", *COLS], extrasaction="ignore")
            w.writeheader()
            w.writerows(table)
    print("| tag | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for row in table:
        print(f"| {row['tag']} | " + " | ".join(str(row.get(c)) for c in COLS) + " |")


if __name__ == "__main__":
    main()
