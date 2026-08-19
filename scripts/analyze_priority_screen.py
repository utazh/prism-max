#!/usr/bin/env python3
"""Audit an ABBA single-request prefetch-priority screen."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-a", required=True, type=Path)
    parser.add_argument("--priority-a", required=True, type=Path)
    parser.add_argument("--priority-b", required=True, type=Path)
    parser.add_argument("--control-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--minimum-mean-speedup-percent", type=float, default=2.0)
    parser.add_argument("--maximum-p95-regression-percent", type=float, default=1.0)
    return parser.parse_args()


def load_records(directory: Path) -> dict[str, dict[str, Any]]:
    path = directory / "scored_records.jsonl"
    records = {
        str(row["uid"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if not records:
        raise ValueError(f"no scored records in {path}")
    return records


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    values = [float(value) for value in values]
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def mean_metric(
    runs: Sequence[dict[str, dict[str, Any]]],
    metric: str,
) -> float:
    return statistics.fmean(
        float(row.get(metric, 0.0))
        for run in runs
        for row in run.values()
    )


def main() -> int:
    args = parse_args()
    named = {
        "control_a": load_records(args.control_a),
        "priority_a": load_records(args.priority_a),
        "priority_b": load_records(args.priority_b),
        "control_b": load_records(args.control_b),
    }
    uid_sets = {name: set(rows) for name, rows in named.items()}
    first_uids = next(iter(uid_sets.values()))
    if any(uids != first_uids for uids in uid_sets.values()):
        raise ValueError(
            "ABBA runs have different request UIDs: "
            + json.dumps({name: len(uids) for name, uids in uid_sets.items()})
        )
    uids = sorted(first_uids)
    controls = (named["control_a"], named["control_b"])
    priorities = (named["priority_a"], named["priority_b"])

    prediction_mismatches = sum(
        len({str(run[uid]["prediction"]) for run in named.values()}) != 1
        for uid in uids
    )
    selection_mismatches = sum(
        len(
            {
                str(run[uid]["layer_token_selection_sha256"])
                for run in named.values()
            }
        )
        != 1
        for uid in uids
    )
    correctness_mismatches = sum(
        len({bool(run[uid]["correct"]) for run in named.values()}) != 1
        for uid in uids
    )

    control_ttft = [
        statistics.fmean(float(run[uid]["ttft_ms"]) for run in controls)
        for uid in uids
    ]
    priority_ttft = [
        statistics.fmean(float(run[uid]["ttft_ms"]) for run in priorities)
        for uid in uids
    ]
    balanced_deltas = [
        candidate - reference
        for reference, candidate in zip(control_ttft, priority_ttft)
    ]
    control_mean = statistics.fmean(control_ttft)
    priority_mean = statistics.fmean(priority_ttft)
    mean_change_percent = (priority_mean / control_mean - 1.0) * 100.0
    control_p95 = percentile(control_ttft, 0.95)
    priority_p95 = percentile(priority_ttft, 0.95)
    p95_change_percent = (priority_p95 / control_p95 - 1.0) * 100.0
    delta_ci = bootstrap_mean_ci(
        balanced_deltas,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    forward_deltas = [
        float(named["priority_a"][uid]["ttft_ms"])
        - float(named["control_a"][uid]["ttft_ms"])
        for uid in uids
    ]
    reverse_deltas = [
        float(named["priority_b"][uid]["ttft_ms"])
        - float(named["control_b"][uid]["ttft_ms"])
        for uid in uids
    ]
    scheduler_metrics = {}
    for label in ("current", "next", "period"):
        for metric in ("submitted", "queue_wait_ms", "execution_ms"):
            key = f"prefetch_scheduler_{label}_{metric}"
            scheduler_metrics[f"control_mean_{label}_{metric}"] = mean_metric(
                controls, key
            )
            scheduler_metrics[f"priority_mean_{label}_{metric}"] = mean_metric(
                priorities, key
            )

    failed_or_cancelled = sum(
        int(row.get("prefetch_scheduler_total_failed", 0))
        + int(row.get("prefetch_scheduler_total_cancelled", 0))
        for run in named.values()
        for row in run.values()
    )
    integrity_ok = (
        prediction_mismatches == 0
        and selection_mismatches == 0
        and correctness_mismatches == 0
        and failed_or_cancelled == 0
    )
    speed_gate = (
        mean_change_percent <= -abs(args.minimum_mean_speedup_percent)
        and delta_ci[1] < 0.0
        and p95_change_percent <= args.maximum_p95_regression_percent
    )
    gate_pass = integrity_ok and speed_gate

    payload = {
        "scope": {
            "execution": "sequential batch-size-1 requests",
            "unique_requests": len(uids),
            "run_order": [
                "control_a",
                "priority_a",
                "priority_b",
                "control_b",
            ],
            "repetitions_per_variant": 2,
        },
        "integrity": {
            "prediction_mismatches": prediction_mismatches,
            "selection_hash_mismatches": selection_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "scheduler_failed_or_cancelled_jobs": failed_or_cancelled,
        },
        "order_balanced_ttft": {
            "control_mean_ms": control_mean,
            "priority_mean_ms": priority_mean,
            "priority_change_percent": mean_change_percent,
            "control_p95_ms": control_p95,
            "priority_p95_ms": priority_p95,
            "priority_p95_change_percent": p95_change_percent,
            "mean_paired_delta_ms": statistics.fmean(balanced_deltas),
            "median_paired_delta_ms": statistics.median(balanced_deltas),
            "paired_bootstrap_mean_delta_95ci_ms": delta_ci,
            "priority_faster_requests": sum(delta < 0 for delta in balanced_deltas),
            "forward_order_mean_delta_ms": statistics.fmean(forward_deltas),
            "reverse_order_mean_delta_ms": statistics.fmean(reverse_deltas),
        },
        "scheduler": scheduler_metrics,
        "gate": {
            "minimum_mean_speedup_percent": args.minimum_mean_speedup_percent,
            "maximum_p95_regression_percent": args.maximum_p95_regression_percent,
            "requires_bootstrap_upper_bound_below_zero": True,
            "pass": gate_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if gate_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
