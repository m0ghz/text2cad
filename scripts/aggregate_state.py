#!/usr/bin/env python3
"""
Utility script to compute aggregate metrics from a partial evaluation state file.

The state file should be the newline-delimited JSON written by the evaluator
while processing samples (partial_state.jsonl). This script summarises the
partial run so progress can be inspected even if the evaluator did not finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from an evaluation partial state file."
    )
    parser.add_argument(
        "state_file",
        type=Path,
        help="Path to the partial_state.jsonl file generated during evaluation.",
    )
    return parser.parse_args()


def aggregate_state(state_path: Path) -> dict:
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    num_samples = 0
    final_score_sum = 0.0
    judge_score_sum = 0.0

    with state_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            num_samples += 1
            final_score_sum += float(record.get("final_score", 0.0))
            judge = record.get("judge", {})
            judge_score_sum += float(judge.get("score", 0.0))

    if num_samples == 0:
        avg_final = 0.0
        avg_judge = 0.0
    else:
        avg_final = final_score_sum / num_samples
        avg_judge = judge_score_sum / num_samples

    return {
        "num_samples": num_samples,
        "average_final_score": avg_final,
        "average_judge_score": avg_judge,
        "state_file": str(state_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    metrics = aggregate_state(args.state_file)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI utility
    raise SystemExit(main())
