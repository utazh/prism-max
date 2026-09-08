"""Compare one baseline/candidate PRISM result pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUMMARY_METRICS = (
    "accuracy",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "mean_response_ready_ms",
    "mean_selector_load_ms",
    "mean_selector_compute_ms",
    "mean_selector_wait_ms",
    "mean_selector_calls",
    "mean_promixed_period",
)

EXACTNESS_FIELDS = (
    "layer_token_selection_sha256",
    "generation_prediction",
    "label_first_token_prediction",
    "prediction",
    "correct",
    "selector_calls",
    "promixed_decisions",
    "promixed_p1_decisions",
    "promixed_p2_decisions",
    "promixed_p4_decisions",
    "promixed_p8_decisions",
)


def _records(directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (directory / "scored_records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def compare(baseline: Path, candidate: Path, task: str) -> dict:
    baseline_summary = json.loads(
        (baseline / "summary.json").read_text(encoding="utf-8")
    )["tasks"][task]
    candidate_summary = json.loads(
        (candidate / "summary.json").read_text(encoding="utf-8")
    )["tasks"][task]
    baseline_rows = _records(baseline)
    candidate_rows = _records(candidate)
    if [row["uid"] for row in baseline_rows] != [
        row["uid"] for row in candidate_rows
    ]:
        raise ValueError("baseline and candidate UID order differs")

    metrics = {}
    for name in SUMMARY_METRICS:
        before = float(baseline_summary[name])
        after = float(candidate_summary[name])
        metrics[name] = {
            "baseline": before,
            "candidate": after,
            "delta": after - before,
            "delta_percent": (
                None if before == 0 else 100.0 * (after - before) / before
            ),
        }
    differences = {
        field: sum(
            left.get(field) != right.get(field)
            for left, right in zip(baseline_rows, candidate_rows)
        )
        for field in EXACTNESS_FIELDS
    }
    return {
        "task": task,
        "samples": len(baseline_rows),
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "metrics": metrics,
        "record_differences": differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.baseline, args.candidate, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
