#!/usr/bin/env python3
"""Audit a two-variant ABBA screen over sequential batch-size-1 requests."""

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
    parser.add_argument("--reference-a", required=True, type=Path)
    parser.add_argument("--candidate-a", required=True, type=Path)
    parser.add_argument("--candidate-b", required=True, type=Path)
    parser.add_argument("--reference-b", required=True, type=Path)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument(
        "--feature-flag-metric",
        action="append",
        required=True,
        help="Metric that must be 0 for references and 1 for candidates; repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--minimum-mean-speedup-percent", type=float, default=2.0)
    parser.add_argument("--maximum-p95-regression-percent", type=float, default=1.0)
    parser.add_argument("--expected-candidate-timing-samples", type=int)
    parser.add_argument("--expected-candidate-value-budget-scale", type=float)
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
        "reference_a": load_records(args.reference_a),
        "candidate_a": load_records(args.candidate_a),
        "candidate_b": load_records(args.candidate_b),
        "reference_b": load_records(args.reference_b),
    }
    uid_sets = {name: set(rows) for name, rows in named.items()}
    first_uids = next(iter(uid_sets.values()))
    if any(uids != first_uids for uids in uid_sets.values()):
        raise ValueError(
            "ABBA runs have different request UIDs: "
            + json.dumps({name: len(uids) for name, uids in uid_sets.items()})
        )
    uids = sorted(first_uids)
    references = (named["reference_a"], named["reference_b"])
    candidates = (named["candidate_a"], named["candidate_b"])

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
    failed_or_cancelled = sum(
        int(row.get("prefetch_scheduler_total_failed", 0))
        + int(row.get("prefetch_scheduler_total_cancelled", 0))
        for run in named.values()
        for row in run.values()
    )

    reference_ttft = [
        statistics.fmean(float(run[uid]["ttft_ms"]) for run in references)
        for uid in uids
    ]
    candidate_ttft = [
        statistics.fmean(float(run[uid]["ttft_ms"]) for run in candidates)
        for uid in uids
    ]
    balanced_deltas = [
        candidate - reference
        for reference, candidate in zip(reference_ttft, candidate_ttft)
    ]
    reference_mean = statistics.fmean(reference_ttft)
    candidate_mean = statistics.fmean(candidate_ttft)
    mean_change_percent = (candidate_mean / reference_mean - 1.0) * 100.0
    reference_p95 = percentile(reference_ttft, 0.95)
    candidate_p95 = percentile(candidate_ttft, 0.95)
    p95_change_percent = (candidate_p95 / reference_p95 - 1.0) * 100.0
    delta_ci = bootstrap_mean_ci(
        balanced_deltas,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    forward_deltas = [
        float(named["candidate_a"][uid]["ttft_ms"])
        - float(named["reference_a"][uid]["ttft_ms"])
        for uid in uids
    ]
    reverse_deltas = [
        float(named["candidate_b"][uid]["ttft_ms"])
        - float(named["reference_b"][uid]["ttft_ms"])
        for uid in uids
    ]
    mechanism_metrics = {}
    for metric in (
        "impress_deferred_compute_samples",
        "impress_deferred_compute_pending",
        "impress_deferred_compute_pending_max",
        "impress_mean_prefetch_budget_seconds",
        "prefetch_wait_ms",
        "physical_prefetch_kv_bytes",
        "total_ssd_read_bytes",
        "inter_period_hit_tokens",
        "inter_period_missing_tokens",
        "inter_period_unused_tokens",
        "impress_period_prediction_mean_jaccard",
        "impress_period_prediction_precision",
        "impress_period_prediction_recall",
        "impress_period_prediction_mean_layer_distance",
        "impress_period_prefetch_hit_tokens",
        "impress_period_prefetch_missing_tokens",
        "impress_period_prefetch_unused_tokens",
        "impress_value_ordered_prefetch_jobs",
        "impress_value_prefetch_budget_scale",
    ):
        mechanism_metrics[f"reference_mean_{metric}"] = mean_metric(
            references, metric
        )
        mechanism_metrics[f"candidate_mean_{metric}"] = mean_metric(
            candidates, metric
        )

    expected_samples_ok = True
    if args.expected_candidate_timing_samples is not None:
        expected_samples_ok = all(
            int(row.get("impress_deferred_compute_samples", -1))
            == args.expected_candidate_timing_samples
            and int(row.get("impress_deferred_compute_pending", -1)) == 0
            for run in candidates
            for row in run.values()
        )
    reference_feature_disabled = all(
        int(row.get(metric, -1)) == 0
        for metric in args.feature_flag_metric
        for run in references
        for row in run.values()
    )
    candidate_feature_enabled = all(
        int(row.get(metric, -1)) == 1
        for metric in args.feature_flag_metric
        for run in candidates
        for row in run.values()
    )
    value_budget_scale_ok = True
    if args.expected_candidate_value_budget_scale is not None:
        expected_scale = float(args.expected_candidate_value_budget_scale)
        value_budget_scale_ok = all(
            math.isclose(
                float(row.get("impress_value_prefetch_budget_scale", math.nan)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for run in references
            for row in run.values()
        ) and all(
            math.isclose(
                float(row.get("impress_value_prefetch_budget_scale", math.nan)),
                expected_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for run in candidates
            for row in run.values()
        )
    integrity_ok = (
        prediction_mismatches == 0
        and selection_mismatches == 0
        and correctness_mismatches == 0
        and failed_or_cancelled == 0
        and expected_samples_ok
        and reference_feature_disabled
        and candidate_feature_enabled
        and value_budget_scale_ok
    )
    speed_gate = (
        mean_change_percent <= -abs(args.minimum_mean_speedup_percent)
        and delta_ci[1] < 0.0
        and p95_change_percent <= args.maximum_p95_regression_percent
    )
    gate_pass = integrity_ok and speed_gate

    payload = {
        "candidate": args.candidate_name,
        "scope": {
            "execution": "sequential batch-size-1 requests",
            "unique_requests": len(uids),
            "run_order": [
                "reference_a",
                "candidate_a",
                "candidate_b",
                "reference_b",
            ],
            "repetitions_per_variant": 2,
        },
        "integrity": {
            "prediction_mismatches": prediction_mismatches,
            "selection_hash_mismatches": selection_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "scheduler_failed_or_cancelled_jobs": failed_or_cancelled,
            "feature_flag_metrics": args.feature_flag_metric,
            "reference_feature_disabled": reference_feature_disabled,
            "candidate_feature_enabled": candidate_feature_enabled,
            "expected_candidate_timing_samples": expected_samples_ok,
            "expected_candidate_value_budget_scale": (
                args.expected_candidate_value_budget_scale
            ),
            "value_budget_scale_ok": value_budget_scale_ok,
        },
        "order_balanced_ttft": {
            "reference_mean_ms": reference_mean,
            "candidate_mean_ms": candidate_mean,
            "candidate_change_percent": mean_change_percent,
            "reference_p95_ms": reference_p95,
            "candidate_p95_ms": candidate_p95,
            "candidate_p95_change_percent": p95_change_percent,
            "mean_paired_delta_ms": statistics.fmean(balanced_deltas),
            "median_paired_delta_ms": statistics.median(balanced_deltas),
            "paired_bootstrap_mean_delta_95ci_ms": delta_ci,
            "candidate_faster_requests": sum(
                delta < 0 for delta in balanced_deltas
            ),
            "forward_order_mean_delta_ms": statistics.fmean(forward_deltas),
            "reverse_order_mean_delta_ms": statistics.fmean(reverse_deltas),
        },
        "mechanism": mechanism_metrics,
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
