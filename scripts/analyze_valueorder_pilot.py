#!/usr/bin/env python3
"""Select a value-ordered prefetch candidate using mechanism-first gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--leader-value", required=True, type=Path)
    parser.add_argument("--rolling-value", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-output", required=True, type=Path)
    return parser.parse_args()


def load_records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["uid"]): row
        for row in (
            json.loads(line)
            for line in (directory / "scored_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }


def mean(rows: dict[str, dict[str, Any]], metric: str) -> float:
    return statistics.fmean(float(row.get(metric, 0.0)) for row in rows.values())


def main() -> int:
    args = parse_args()
    variants = {
        "baseline": load_records(args.baseline),
        "leader_value": load_records(args.leader_value),
        "rolling_value": load_records(args.rolling_value),
    }
    uid_sets = {name: set(rows) for name, rows in variants.items()}
    baseline_uids = uid_sets["baseline"]
    if not baseline_uids or any(uids != baseline_uids for uids in uid_sets.values()):
        raise ValueError("pilot variants must contain the same non-empty request set")

    baseline = variants["baseline"]
    integrity = {}
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
    )
    mechanism = {
        name: {metric: mean(rows, metric) for metric in metrics}
        for name, rows in variants.items()
    }

    valid_candidates = []
    baseline_metrics = mechanism["baseline"]
    for name in ("leader_value", "rolling_value"):
        rows = variants[name]
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
            for row in rows.values()
        ) and all(
            int(row.get("impress_rolling_period_prefetch", -1))
            == (1 if name == "rolling_value" else 0)
            for row in rows.values()
        )
        integrity[name] = {
            "prediction_mismatches": prediction_mismatches,
            "selection_hash_mismatches": selection_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "feature_flags_correct": flags_ok,
        }
        candidate = mechanism[name]
        mechanism_ok = (
            candidate["inter_period_hit_tokens"]
            >= baseline_metrics["inter_period_hit_tokens"]
            and candidate["inter_period_missing_tokens"]
            <= baseline_metrics["inter_period_missing_tokens"]
            and candidate["total_ssd_read_bytes"]
            <= baseline_metrics["total_ssd_read_bytes"] * 1.02
            and candidate["impress_value_ordered_prefetch_jobs"] > 0
        )
        integrity_ok = (
            prediction_mismatches == 0
            and selection_mismatches == 0
            and correctness_mismatches == 0
            and flags_ok
        )
        if integrity_ok and mechanism_ok:
            valid_candidates.append(name)

    selected = (
        min(
            valid_candidates,
            key=lambda name: (
                mechanism[name]["prefetch_wait_ms"],
                mechanism[name]["ttft_ms"],
            ),
        )
        if valid_candidates
        else None
    )
    payload = {
        "scope": {
            "execution": "sequential batch-size-1 requests",
            "requests": len(baseline),
        },
        "integrity": integrity,
        "mechanism": mechanism,
        "valid_candidates": valid_candidates,
        "selected_candidate": selected,
        "gate": {
            "requires_no_selection_or_prediction_change": True,
            "requires_no_fewer_total_speculative_hits": True,
            "requires_no_more_total_speculative_missing": True,
            "maximum_total_ssd_read_increase_percent": 2.0,
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
