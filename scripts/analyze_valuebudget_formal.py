#!/usr/bin/env python3
"""Compare a 128-request value-budget run with the frozen Phase-A+B baseline."""

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
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--expected-scale", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
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


def load_summary(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


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
    rng = random.Random(seed)
    values = [float(value) for value in values]
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def mean_metric(rows: dict[str, dict[str, Any]], metric: str) -> float:
    return statistics.fmean(float(row.get(metric, 0.0)) for row in rows.values())


def main() -> int:
    args = parse_args()
    baseline = load_records(args.baseline)
    candidate = load_records(args.candidate)
    baseline_summary = load_summary(args.baseline)
    candidate_summary = load_summary(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("formal baseline and candidate have different request UIDs")
    uids = sorted(baseline)

    prediction_mismatches = sum(
        baseline[uid]["prediction"] != candidate[uid]["prediction"]
        for uid in uids
    )
    selection_mismatches = sum(
        baseline[uid]["layer_token_selection_sha256"]
        != candidate[uid]["layer_token_selection_sha256"]
        for uid in uids
    )
    correctness_mismatches = sum(
        bool(baseline[uid]["correct"]) != bool(candidate[uid]["correct"])
        for uid in uids
    )
    scheduler_failures = sum(
        int(row.get("prefetch_scheduler_total_failed", 0))
        + int(row.get("prefetch_scheduler_total_cancelled", 0))
        for row in candidate.values()
    )
    feature_and_scale_ok = all(
        int(row.get("impress_value_ordered_prefetch", -1)) == 1
        and int(row.get("impress_value_prefetch_budget_scaled", -1)) == 1
        and int(row.get("impress_rolling_period_prefetch", -1)) == 0
        and math.isclose(
            float(row.get("impress_value_prefetch_budget_scale", math.nan)),
            args.expected_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in candidate.values()
    )
    warmup_ok = (
        int(baseline_summary["runtime"]["warmup_passes"]) == 1
        and int(candidate_summary["runtime"]["warmup_passes"]) == 1
    )

    baseline_ttft = [float(baseline[uid]["ttft_ms"]) for uid in uids]
    candidate_ttft = [float(candidate[uid]["ttft_ms"]) for uid in uids]
    deltas = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline_ttft, candidate_ttft)
    ]
    baseline_mean = statistics.fmean(baseline_ttft)
    candidate_mean = statistics.fmean(candidate_ttft)
    mean_change = (candidate_mean / baseline_mean - 1.0) * 100.0
    baseline_p95 = percentile(baseline_ttft, 0.95)
    candidate_p95 = percentile(candidate_ttft, 0.95)
    p95_change = (candidate_p95 / baseline_p95 - 1.0) * 100.0
    delta_ci = bootstrap_mean_ci(
        deltas,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    metrics = (
        "prefetch_wait_ms",
        "physical_prefetch_kv_bytes",
        "total_ssd_read_bytes",
        "inter_period_hit_tokens",
        "inter_period_missing_tokens",
        "inter_period_unused_tokens",
        "impress_period_prefetch_hit_tokens",
        "impress_period_prefetch_missing_tokens",
        "impress_period_prefetch_unused_tokens",
    )
    mechanism = {}
    for metric in metrics:
        mechanism[f"baseline_mean_{metric}"] = mean_metric(baseline, metric)
        mechanism[f"candidate_mean_{metric}"] = mean_metric(candidate, metric)

    integrity_ok = (
        len(uids) == 128
        and prediction_mismatches == 0
        and selection_mismatches == 0
        and correctness_mismatches == 0
        and scheduler_failures == 0
        and feature_and_scale_ok
        and warmup_ok
    )
    speed_gate = (
        mean_change <= -2.0
        and delta_ci[1] < 0.0
        and p95_change <= 1.0
    )
    payload = {
        "scope": {
            "execution": "sequential batch-size-1 requests",
            "requests": len(uids),
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "warmup_passes": 1,
        },
        "integrity": {
            "prediction_mismatches": prediction_mismatches,
            "selection_hash_mismatches": selection_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "scheduler_failed_or_cancelled_jobs": scheduler_failures,
            "feature_flags_and_scale_correct": feature_and_scale_ok,
            "matched_warmup_protocol": warmup_ok,
        },
        "quality": {
            "baseline_accuracy": mean_metric(baseline, "correct"),
            "candidate_accuracy": mean_metric(candidate, "correct"),
        },
        "paired_ttft": {
            "baseline_mean_ms": baseline_mean,
            "candidate_mean_ms": candidate_mean,
            "candidate_change_percent": mean_change,
            "baseline_p95_ms": baseline_p95,
            "candidate_p95_ms": candidate_p95,
            "candidate_p95_change_percent": p95_change,
            "mean_paired_delta_ms": statistics.fmean(deltas),
            "median_paired_delta_ms": statistics.median(deltas),
            "paired_bootstrap_mean_delta_95ci_ms": delta_ci,
            "candidate_faster_requests": sum(delta < 0.0 for delta in deltas),
        },
        "mechanism": mechanism,
        "gate": {
            "minimum_mean_speedup_percent": 2.0,
            "maximum_p95_regression_percent": 1.0,
            "requires_bootstrap_upper_bound_below_zero": True,
            "pass": integrity_ok and speed_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload["gate"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
