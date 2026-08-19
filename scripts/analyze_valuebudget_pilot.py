#!/usr/bin/env python3
"""Select a value-ordered prefetch budget using mechanism-first gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in SCALE=RUN_DIRECTORY form; repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-output", required=True, type=Path)
    return parser.parse_args()


def parse_candidates(specs: list[str]) -> list[tuple[float, Path]]:
    parsed = []
    for spec in specs:
        scale_text, separator, directory = spec.partition("=")
        if not separator or not directory:
            raise ValueError(f"invalid candidate specification: {spec!r}")
        scale = float(scale_text)
        if not math.isfinite(scale) or not 0.0 < scale < 1.0:
            raise ValueError(f"candidate scale must be in (0, 1): {scale_text}")
        parsed.append((scale, Path(directory)))
    scales = [scale for scale, _ in parsed]
    if len(scales) != len(set(scales)):
        raise ValueError("candidate scales must be unique")
    return parsed


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


def mean(rows: dict[str, dict[str, Any]], metric: str) -> float:
    return statistics.fmean(float(row.get(metric, 0.0)) for row in rows.values())


def main() -> int:
    args = parse_args()
    candidate_specs = parse_candidates(args.candidate)
    baseline = load_records(args.baseline)
    candidates = {
        f"{scale:g}": (scale, load_records(directory))
        for scale, directory in candidate_specs
    }
    baseline_uids = set(baseline)
    if any(set(rows) != baseline_uids for _, rows in candidates.values()):
        raise ValueError("pilot variants must contain the same request UIDs")

    metrics = (
        "ttft_ms",
        "prefetch_wait_ms",
        "physical_prefetch_kv_bytes",
        "total_ssd_read_bytes",
        "inter_period_hit_tokens",
        "inter_period_missing_tokens",
        "inter_period_unused_tokens",
        "impress_next_prefetch_hit_tokens",
        "impress_next_prefetch_missing_tokens",
        "impress_next_prefetch_unused_tokens",
        "impress_period_prefetch_hit_tokens",
        "impress_period_prefetch_missing_tokens",
        "impress_period_prefetch_unused_tokens",
        "impress_value_ordered_prefetch_jobs",
        "impress_value_prefetch_budget_scale",
    )
    mechanism = {
        "baseline": {metric: mean(baseline, metric) for metric in metrics}
    }
    integrity = {}
    valid = []
    baseline_metrics = mechanism["baseline"]
    for label, (scale, rows) in candidates.items():
        candidate_metrics = {
            metric: mean(rows, metric) for metric in metrics
        }
        mechanism[label] = candidate_metrics
        prediction_mismatches = sum(
            baseline[uid]["prediction"] != rows[uid]["prediction"]
            for uid in baseline
        )
        selection_mismatches = sum(
            baseline[uid]["layer_token_selection_sha256"]
            != rows[uid]["layer_token_selection_sha256"]
            for uid in baseline
        )
        correctness_mismatches = sum(
            bool(baseline[uid]["correct"]) != bool(rows[uid]["correct"])
            for uid in baseline
        )
        flags_ok = all(
            int(row.get("impress_value_ordered_prefetch", -1)) == 1
            and int(row.get("impress_value_prefetch_budget_scaled", -1)) == 1
            and int(row.get("impress_rolling_period_prefetch", -1)) == 0
            and math.isclose(
                float(row.get("impress_value_prefetch_budget_scale", math.nan)),
                scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in rows.values()
        )
        scheduler_failures = sum(
            int(row.get("prefetch_scheduler_total_failed", 0))
            + int(row.get("prefetch_scheduler_total_cancelled", 0))
            for row in rows.values()
        )
        integrity[label] = {
            "prediction_mismatches": prediction_mismatches,
            "selection_hash_mismatches": selection_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "scheduler_failed_or_cancelled_jobs": scheduler_failures,
            "feature_flags_and_scale_correct": flags_ok,
        }
        integrity_ok = (
            prediction_mismatches == 0
            and selection_mismatches == 0
            and correctness_mismatches == 0
            and scheduler_failures == 0
            and flags_ok
        )
        mechanism_ok = (
            candidate_metrics["inter_period_hit_tokens"]
            >= baseline_metrics["inter_period_hit_tokens"]
            and candidate_metrics["inter_period_missing_tokens"]
            <= baseline_metrics["inter_period_missing_tokens"]
            and candidate_metrics["physical_prefetch_kv_bytes"]
            < baseline_metrics["physical_prefetch_kv_bytes"]
            and candidate_metrics["total_ssd_read_bytes"]
            < baseline_metrics["total_ssd_read_bytes"]
            and candidate_metrics["impress_value_ordered_prefetch_jobs"] > 0
        )
        if integrity_ok and mechanism_ok:
            valid.append(label)

    selected = (
        min(
            valid,
            key=lambda label: (
                mechanism[label]["prefetch_wait_ms"],
                mechanism[label]["ttft_ms"],
            ),
        )
        if valid
        else None
    )
    payload = {
        "scope": {
            "execution": "sequential batch-size-1 requests",
            "requests": len(baseline),
            "candidate_scales": [scale for scale, _ in candidate_specs],
        },
        "integrity": integrity,
        "mechanism": mechanism,
        "valid_candidates": valid,
        "selected_scale": float(selected) if selected is not None else None,
        "gate": {
            "requires_unchanged_selection_prediction_and_correctness": True,
            "requires_no_fewer_speculative_hits": True,
            "requires_no_more_speculative_missing": True,
            "requires_lower_prefetch_and_total_ssd_bytes": True,
            "pass": selected is not None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if selected is not None:
        args.selected_output.write_text(selected + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if selected is not None else 4


if __name__ == "__main__":
    raise SystemExit(main())
