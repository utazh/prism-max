#!/usr/bin/env python3
"""Gate rolling-P4 on prediction and physical-prefetch mechanism metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", required=True, type=Path)
    parser.add_argument("--rolling", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    leader = load_records(args.leader)
    rolling = load_records(args.rolling)
    if not leader or set(leader) != set(rolling):
        raise ValueError("pilot variants must contain the same non-empty request set")

    prediction_mismatches = sum(
        leader[uid]["prediction"] != rolling[uid]["prediction"] for uid in leader
    )
    selection_mismatches = sum(
        leader[uid]["layer_token_selection_sha256"]
        != rolling[uid]["layer_token_selection_sha256"]
        for uid in leader
    )
    correctness_mismatches = sum(
        bool(leader[uid]["correct"]) != bool(rolling[uid]["correct"])
        for uid in leader
    )
    metrics = (
        "ttft_ms",
        "impress_period_prediction_mean_jaccard",
        "impress_period_prediction_precision",
        "impress_period_prediction_recall",
        "impress_period_prediction_mean_layer_distance",
        "impress_period_prefetch_hit_tokens",
        "impress_period_prefetch_missing_tokens",
        "impress_period_prefetch_unused_tokens",
        "prefetch_wait_ms",
        "physical_prefetch_kv_bytes",
        "total_ssd_read_bytes",
    )
    mechanism = {
        metric: {
            "leader_mean": mean(leader, metric),
            "rolling_mean": mean(rolling, metric),
        }
        for metric in metrics
    }
    leader_jaccard = mechanism[
        "impress_period_prediction_mean_jaccard"
    ]["leader_mean"]
    rolling_jaccard = mechanism[
        "impress_period_prediction_mean_jaccard"
    ]["rolling_mean"]
    leader_missing = mechanism[
        "impress_period_prefetch_missing_tokens"
    ]["leader_mean"]
    rolling_missing = mechanism[
        "impress_period_prefetch_missing_tokens"
    ]["rolling_mean"]
    leader_unused = mechanism[
        "impress_period_prefetch_unused_tokens"
    ]["leader_mean"]
    rolling_unused = mechanism[
        "impress_period_prefetch_unused_tokens"
    ]["rolling_mean"]

    flags_ok = all(
        int(row.get("impress_rolling_period_prefetch", -1)) == 0
        for row in leader.values()
    ) and all(
        int(row.get("impress_rolling_period_prefetch", -1)) == 1
        for row in rolling.values()
    )
    integrity_ok = (
        prediction_mismatches == 0
        and selection_mismatches == 0
        and correctness_mismatches == 0
        and flags_ok
    )
    mechanism_ok = (
        rolling_jaccard > leader_jaccard
        and rolling_missing <= leader_missing
        and rolling_unused <= leader_unused
    )
    gate_pass = integrity_ok and mechanism_ok
    payload = {
        "scope": {
            "execution": "sequential batch-size-1 requests",
            "requests": len(leader),
        },
        "integrity": {
            "prediction_mismatches": prediction_mismatches,
            "selection_hash_mismatches": selection_mismatches,
            "correctness_mismatches": correctness_mismatches,
            "feature_flags_correct": flags_ok,
        },
        "mechanism": mechanism,
        "gate": {
            "requires_higher_period_prediction_jaccard": True,
            "requires_no_more_period_missing_tokens": True,
            "requires_no_more_period_unused_tokens": True,
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
