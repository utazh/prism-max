#!/usr/bin/env python3
"""Build a common 128-request table from frozen 50% SSD runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in LABEL=DIRECTORY form; repeatable and ordered.",
    )
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser.parse_args()


def parse_runs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        label, separator, directory = spec.partition("=")
        if not separator or not label or not directory:
            raise ValueError(f"invalid run specification: {spec!r}")
        parsed.append((label, Path(directory)))
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError("run labels must be unique")
    return parsed


def load_records(directory: Path) -> list[dict[str, Any]]:
    path = directory / "scored_records.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 128:
        raise ValueError(f"expected 128 records in {path}, found {len(rows)}")
    return rows


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.fmean(float(row.get(metric, 0.0)) for row in rows)


def main() -> int:
    args = parse_args()
    runs = parse_runs(args.run)
    table = []
    for label, directory in runs:
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        rows = load_records(directory)
        overall = summary["overall"]
        table.append(
            {
                "method": label,
                "accuracy": float(overall["accuracy"]),
                "mean_ttft_ms": float(overall["mean_ttft_ms"]),
                "p95_ttft_ms": float(overall["p95_ttft_ms"]),
                "mean_total_ssd_read_mib": mean_metric(
                    rows, "total_ssd_read_bytes"
                )
                / (1024**2),
                "mean_physical_prefetch_mib": mean_metric(
                    rows, "physical_prefetch_kv_bytes"
                )
                / (1024**2),
                "mean_critical_ssd_read_mib": mean_metric(
                    rows, "critical_ssd_read_bytes"
                )
                / (1024**2),
                "run_directory": str(directory),
            }
        )

    by_label = {row["method"]: row for row in table}
    if args.candidate_label not in by_label:
        raise ValueError(f"candidate label not found: {args.candidate_label}")
    candidate = by_label[args.candidate_label]
    for row in table:
        row["candidate_ttft_change_percent"] = (
            candidate["mean_ttft_ms"] / row["mean_ttft_ms"] - 1.0
        ) * 100.0
        row["candidate_accuracy_change_pp"] = (
            candidate["accuracy"] - row["accuracy"]
        ) * 100.0

    payload = {
        "scope": {
            "model": "Qwen2.5-7B-Instruct",
            "kv_budget": 0.50,
            "storage": "server SSD",
            "execution": "sequential batch-size-1 requests",
            "requests": 128,
        },
        "candidate": args.candidate_label,
        "methods": table,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "| Method | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | "
        "Total SSD (MiB) | Critical SSD (MiB) | Candidate TTFT vs method |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| {method} | {accuracy:.6f} | {mean_ttft_ms:.2f} | "
            "{p95_ttft_ms:.2f} | {mean_total_ssd_read_mib:.2f} | "
            "{mean_critical_ssd_read_mib:.2f} | "
            "{candidate_ttft_change_percent:+.2f}% |".format(**row)
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
