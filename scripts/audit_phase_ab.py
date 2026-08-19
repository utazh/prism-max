#!/usr/bin/env python3
"""Audit matched PRISM phase A/B runs against the frozen HyperInfer baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics
from collections.abc import Iterable, Mapping
from typing import Any


MEAN_FIELDS = (
    "selected_tokens",
    "selected_kv_bytes",
    "physical_prefetch_tokens",
    "physical_prefetch_kv_bytes",
    "critical_ssd_read_bytes",
    "selector_key_bytes",
    "total_ssd_read_bytes",
    "inter_period_hit_tokens",
    "inter_period_missing_tokens",
    "inter_period_unused_tokens",
    "prefetch_wait_ms",
    "prefetch_elapsed_ms",
    "selector_load_ms",
    "selector_compute_ms",
    "impress_next_prefetch_jobs",
    "impress_period_prefetch_jobs",
    "impress_period_prefetch_tokens",
    "effective_mean_keep_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=pathlib.Path)
    parser.add_argument("--phase-a-dir", required=True, type=pathlib.Path)
    parser.add_argument("--phase-ab-dir", required=True, type=pathlib.Path)
    parser.add_argument("--reverse-phase-a-dir", type=pathlib.Path)
    parser.add_argument("--reverse-phase-ab-dir", type=pathlib.Path)
    parser.add_argument("--profile", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_730)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(run_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "scored_records.jsonl"
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        uid = str(row["uid"])
        if uid in records:
            raise ValueError(f"duplicate uid {uid!r} in {path}")
        records[uid] = row
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def mean_if_present(
    rows: Iterable[Mapping[str, Any]], field: str
) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def metrics(
    records: Mapping[str, Mapping[str, Any]], uids: Iterable[str]
) -> dict[str, Any]:
    rows = [records[uid] for uid in uids]
    ttfts = [float(row["ttft_ms"]) for row in rows]
    correct = sum(bool(row["correct"]) for row in rows)
    result: dict[str, Any] = {
        "samples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_ttft_ms": statistics.fmean(ttfts),
        "p95_ttft_ms": percentile(ttfts, 0.95),
    }
    for field in MEAN_FIELDS:
        value = mean_if_present(rows, field)
        if value is not None:
            key = (
                field
                if field == "effective_mean_keep_ratio"
                else f"mean_{field}"
            )
            result[key] = value
    return result


def bootstrap_mean_delta_ci(
    deltas: list[float], samples: int, seed: int
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    size = len(deltas)
    means = [
        statistics.fmean(deltas[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def exact_mcnemar_pvalue(wrong_to_correct: int, correct_to_wrong: int) -> float:
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(wrong_to_correct, correct_to_wrong) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def comparison(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    uids: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    ref_metrics = metrics(reference, uids)
    candidate_metrics = metrics(candidate, uids)
    deltas = [
        float(candidate[uid]["ttft_ms"]) - float(reference[uid]["ttft_ms"])
        for uid in uids
    ]
    wrong_to_correct = sum(
        not bool(reference[uid]["correct"]) and bool(candidate[uid]["correct"])
        for uid in uids
    )
    correct_to_wrong = sum(
        bool(reference[uid]["correct"]) and not bool(candidate[uid]["correct"])
        for uid in uids
    )
    result = {
        "accuracy_delta": candidate_metrics["accuracy"] - ref_metrics["accuracy"],
        "correct_delta": candidate_metrics["correct"] - ref_metrics["correct"],
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "exact_mcnemar_two_sided_p": exact_mcnemar_pvalue(
            wrong_to_correct, correct_to_wrong
        ),
        "mean_ttft_delta_ms": statistics.fmean(deltas),
        "mean_ttft_change_percent": (
            candidate_metrics["mean_ttft_ms"] / ref_metrics["mean_ttft_ms"] - 1.0
        )
        * 100.0,
        "median_paired_ttft_delta_ms": statistics.median(deltas),
        "candidate_faster_requests": sum(delta < 0 for delta in deltas),
        "candidate_slower_requests": sum(delta > 0 for delta in deltas),
        "paired_bootstrap_mean_ttft_delta_95ci_ms": bootstrap_mean_delta_ci(
            deltas, bootstrap_samples, seed
        ),
    }
    for field in (
        "selected_tokens",
        "selected_kv_bytes",
        "physical_prefetch_kv_bytes",
        "total_ssd_read_bytes",
    ):
        ref_value = ref_metrics.get(f"mean_{field}")
        candidate_value = candidate_metrics.get(f"mean_{field}")
        if ref_value is not None and candidate_value is not None:
            result[f"{field}_change_percent"] = (
                candidate_value / ref_value - 1.0
            ) * 100.0
    return result


def validate_summary(run_dir: pathlib.Path, records: Mapping[str, Any]) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    expected = int(summary["overall"]["samples"])
    if expected != len(records):
        raise ValueError(
            f"{run_dir}: summary has {expected} samples but records have {len(records)}"
        )
    return summary


def main() -> None:
    args = parse_args()
    if (args.reverse_phase_a_dir is None) != (args.reverse_phase_ab_dir is None):
        raise ValueError(
            "reverse-phase-a-dir and reverse-phase-ab-dir must be provided together"
        )
    run_dirs = {
        "baseline_hyperinfer_block16": args.baseline_dir,
        "phase_a_layer_budget": args.phase_a_dir,
        "phase_ab_period_prefetch": args.phase_ab_dir,
    }
    records = {label: load_records(path) for label, path in run_dirs.items()}
    summaries = {
        label: validate_summary(run_dirs[label], rows)
        for label, rows in records.items()
    }

    uid_sets = {label: set(rows) for label, rows in records.items()}
    first_uids = next(iter(uid_sets.values()))
    if any(uids != first_uids for uids in uid_sets.values()):
        sizes = {label: len(uids) for label, uids in uid_sets.items()}
        raise ValueError(f"run uid sets differ: {sizes}")
    all_uids = sorted(first_uids)

    profile = load_json(args.profile)
    calibration_uids = sorted(
        set(profile.get("calibration", {}).get("request_uids", ())) & first_uids
    )
    held_out_uids = sorted(first_uids - set(calibration_uids))
    if not held_out_uids:
        raise ValueError("calibration exclusion removed every evaluation request")

    baseline = records["baseline_hyperinfer_block16"]
    phase_a = records["phase_a_layer_budget"]
    phase_ab = records["phase_ab_period_prefetch"]
    prediction_mismatches = [
        uid
        for uid in all_uids
        if phase_a[uid]["prediction"] != phase_ab[uid]["prediction"]
    ]
    selection_mismatches = [
        uid
        for uid in all_uids
        if phase_a[uid]["layer_token_selection_sha256"]
        != phase_ab[uid]["layer_token_selection_sha256"]
    ]

    payload = {
        "schema_version": 1,
        "inputs": {
            label: {
                "directory": str(path),
                "summary_sha256": sha256_file(path / "summary.json"),
                "records_sha256": sha256_file(path / "scored_records.jsonl"),
                "runtime_variant": summaries[label]["runtime"]["runtime_variant"],
            }
            for label, path in run_dirs.items()
        },
        "profile": {
            "path": str(args.profile),
            "sha256": sha256_file(args.profile),
            "target_mean_ratio": profile["target_mean_ratio"],
            "layer_ratios": profile["layer_ratios"],
            "calibration_uids_excluded_from_held_out": calibration_uids,
        },
        "all_128": {
            label: metrics(rows, all_uids) for label, rows in records.items()
        },
        "held_out": {
            "uids": len(held_out_uids),
            "methods": {
                label: metrics(rows, held_out_uids)
                for label, rows in records.items()
            },
        },
        "comparisons_all_128": {
            "phase_a_vs_baseline": comparison(
                baseline,
                phase_a,
                all_uids,
                args.bootstrap_samples,
                args.seed,
            ),
            "phase_ab_vs_phase_a": comparison(
                phase_a,
                phase_ab,
                all_uids,
                args.bootstrap_samples,
                args.seed + 1,
            ),
        },
        "phase_ab_integrity": {
            "prediction_mismatches_vs_phase_a": prediction_mismatches,
            "selection_mismatches_vs_phase_a": selection_mismatches,
            "requests_with_period_prefetch_jobs": sum(
                float(row.get("impress_period_prefetch_jobs", 0)) > 0
                for row in phase_ab.values()
            ),
            "total_period_prefetch_jobs": sum(
                float(row.get("impress_period_prefetch_jobs", 0))
                for row in phase_ab.values()
            ),
            "total_period_prefetch_tokens": sum(
                float(row.get("impress_period_prefetch_tokens", 0))
                for row in phase_ab.values()
            ),
        },
    }
    if args.reverse_phase_a_dir is not None:
        reverse_dirs = {
            "phase_a_second": args.reverse_phase_a_dir,
            "phase_ab_first": args.reverse_phase_ab_dir,
        }
        reverse_records = {
            label: load_records(path) for label, path in reverse_dirs.items()
        }
        reverse_summaries = {
            label: validate_summary(reverse_dirs[label], rows)
            for label, rows in reverse_records.items()
        }
        reverse_uids = set(reverse_records["phase_a_second"])
        if reverse_uids != set(reverse_records["phase_ab_first"]):
            raise ValueError("reverse confirmation uid sets differ")
        if not reverse_uids <= first_uids:
            raise ValueError("reverse confirmation is not a subset of formal runs")
        common_uids = sorted(reverse_uids)
        reverse_a = reverse_records["phase_a_second"]
        reverse_ab = reverse_records["phase_ab_first"]
        reverse_prediction_mismatches = [
            uid
            for uid in common_uids
            if reverse_a[uid]["prediction"] != reverse_ab[uid]["prediction"]
        ]
        reverse_selection_mismatches = [
            uid
            for uid in common_uids
            if reverse_a[uid]["layer_token_selection_sha256"]
            != reverse_ab[uid]["layer_token_selection_sha256"]
        ]
        forward_deltas = [
            float(phase_ab[uid]["ttft_ms"]) - float(phase_a[uid]["ttft_ms"])
            for uid in common_uids
        ]
        reverse_deltas = [
            float(reverse_ab[uid]["ttft_ms"]) - float(reverse_a[uid]["ttft_ms"])
            for uid in common_uids
        ]
        balanced_deltas = [
            (forward + reverse_delta) / 2.0
            for forward, reverse_delta in zip(forward_deltas, reverse_deltas)
        ]
        balanced_a_mean = statistics.fmean(
            (
                float(phase_a[uid]["ttft_ms"])
                + float(reverse_a[uid]["ttft_ms"])
            )
            / 2.0
            for uid in common_uids
        )
        balanced_ab_mean = statistics.fmean(
            (
                float(phase_ab[uid]["ttft_ms"])
                + float(reverse_ab[uid]["ttft_ms"])
            )
            / 2.0
            for uid in common_uids
        )
        payload["reverse_confirmation"] = {
            "inputs": {
                label: {
                    "directory": str(path),
                    "summary_sha256": sha256_file(path / "summary.json"),
                    "records_sha256": sha256_file(path / "scored_records.jsonl"),
                    "runtime_variant": reverse_summaries[label]["runtime"][
                        "runtime_variant"
                    ],
                }
                for label, path in reverse_dirs.items()
            },
            "methods": {
                label: metrics(rows, common_uids)
                for label, rows in reverse_records.items()
            },
            "phase_ab_vs_phase_a": comparison(
                reverse_a,
                reverse_ab,
                common_uids,
                args.bootstrap_samples,
                args.seed + 2,
            ),
            "prediction_mismatches": reverse_prediction_mismatches,
            "selection_mismatches": reverse_selection_mismatches,
            "order_balanced_common_subset": {
                "requests": len(common_uids),
                "phase_a_mean_ttft_ms": balanced_a_mean,
                "phase_ab_mean_ttft_ms": balanced_ab_mean,
                "phase_ab_mean_ttft_delta_ms": statistics.fmean(
                    balanced_deltas
                ),
                "phase_ab_mean_ttft_change_percent": (
                    balanced_ab_mean / balanced_a_mean - 1.0
                )
                * 100.0,
                "phase_ab_faster_requests": sum(
                    delta < 0 for delta in balanced_deltas
                ),
                "paired_bootstrap_mean_ttft_delta_95ci_ms": (
                    bootstrap_mean_delta_ci(
                        balanced_deltas,
                        args.bootstrap_samples,
                        args.seed + 3,
                    )
                ),
                "forward_order_mean_delta_ms": statistics.fmean(
                    forward_deltas
                ),
                "reverse_order_mean_delta_ms": statistics.fmean(
                    reverse_deltas
                ),
                "interpretation": (
                    "Exploratory two-sequence order-balanced estimate; "
                    "not a replacement for a randomized repeated formal run."
                ),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
