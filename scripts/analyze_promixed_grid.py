#!/usr/bin/env python3
"""Strict per-dataset, paired analysis for the ProMixed full grid."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


TASKS = ("sst2", "subj", "trec", "rte")
BUDGETS = ("005", "010", "025", "050")
DISPLAY = {
    "contigkv": "ContiguousKV",
    "ours": "Previous Ours",
    "promixed": "ProMixed",
}


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute P95 of an empty sequence")
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def exact_mcnemar_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_correct, correct_to_wrong)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    try:
        import numpy as np

        source = np.asarray(values, dtype=np.float64)
        generator = np.random.default_rng(seed)
        means: list[float] = []
        for start in range(0, samples, 1000):
            count = min(1000, samples - start)
            indices = generator.integers(
                0,
                len(values),
                size=(count, len(values)),
            )
            means.extend(source[indices].mean(axis=1).tolist())
    except ImportError:
        generator = random.Random(seed)
        means = [
            sum(generator.choice(values) for _ in values) / len(values)
            for _ in range(samples)
        ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def load_run(path: Path, task: str) -> dict[str, Any]:
    summary_path = path / "summary.json"
    records_path = path / "scored_records.jsonl"
    if not summary_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"incomplete run: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary.get("tasks", {})) != {task}:
        raise ValueError(f"{summary_path} is not an independent {task} run")
    if (
        summary.get("runtime", {}).get("accuracy_scoring")
        != "label_continuation_loglikelihood"
    ):
        raise ValueError(f"{summary_path} does not use complete-label scoring")
    records: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        uid = str(row["uid"])
        if uid in records:
            raise ValueError(f"duplicate UID {uid} in {records_path}")
        records[uid] = row
    expected = int(summary["tasks"][task]["samples"])
    if len(records) != expected:
        raise ValueError(f"{records_path} has {len(records)} rows, expected {expected}")
    return {"summary": summary, "records": records}


def absolute_metrics(run: dict[str, Any]) -> dict[str, Any]:
    rows = list(run["records"].values())
    ttfts = [float(row["ttft_ms"]) for row in rows]
    keep_ratios = [
        float(row["effective_mean_keep_ratio"])
        for row in rows
    ]
    ssd_reads = [
        float(row.get("total_ssd_read_bytes", 0.0))
        for row in rows
    ]
    selector_loads = [
        float(row.get("selector_load_ms", 0.0))
        for row in rows
    ]
    return {
        "samples": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "mean_ttft_ms": sum(ttfts) / len(ttfts),
        "p95_ttft_ms": percentile95(ttfts),
        "mean_effective_keep_ratio": sum(keep_ratios) / len(keep_ratios),
        "mean_total_ssd_read_bytes": sum(ssd_reads) / len(ssd_reads),
        "mean_selector_load_ms": sum(selector_loads) / len(selector_loads),
    }


def paired_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    left = baseline["records"]
    right = candidate["records"]
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(
            "paired comparison requires identical UIDs; "
            f"baseline missing {missing_left}, candidate missing {missing_right}"
        )
    uids = sorted(left)
    left_correct = [bool(left[uid]["correct"]) for uid in uids]
    right_correct = [bool(right[uid]["correct"]) for uid in uids]
    left_ttft = [float(left[uid]["ttft_ms"]) for uid in uids]
    right_ttft = [float(right[uid]["ttft_ms"]) for uid in uids]
    deltas = [
        right_value - left_value
        for left_value, right_value in zip(left_ttft, right_ttft)
    ]
    wrong_to_correct = sum(
        not old and new
        for old, new in zip(left_correct, right_correct)
    )
    correct_to_wrong = sum(
        old and not new
        for old, new in zip(left_correct, right_correct)
    )
    left_mean = sum(left_ttft) / len(left_ttft)
    right_mean = sum(right_ttft) / len(right_ttft)
    left_p95 = percentile95(left_ttft)
    right_p95 = percentile95(right_ttft)
    left_ssd = sum(
        float(left[uid].get("total_ssd_read_bytes", 0.0)) for uid in uids
    ) / len(uids)
    right_ssd = sum(
        float(right[uid].get("total_ssd_read_bytes", 0.0)) for uid in uids
    ) / len(uids)
    return {
        "samples": len(uids),
        "baseline_accuracy": sum(left_correct) / len(uids),
        "candidate_accuracy": sum(right_correct) / len(uids),
        "accuracy_delta_pp": (
            sum(right_correct) - sum(left_correct)
        ) / len(uids) * 100.0,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "mcnemar_two_sided_p": exact_mcnemar_p(
            wrong_to_correct,
            correct_to_wrong,
        ),
        "baseline_mean_ttft_ms": left_mean,
        "candidate_mean_ttft_ms": right_mean,
        "mean_ttft_reduction_percent": (
            left_mean - right_mean
        ) / left_mean * 100.0,
        "mean_paired_delta_ms": sum(deltas) / len(deltas),
        "paired_mean_delta_95ci_ms": bootstrap_mean_ci(
            deltas,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "candidate_faster_requests": sum(delta < 0 for delta in deltas),
        "baseline_p95_ttft_ms": left_p95,
        "candidate_p95_ttft_ms": right_p95,
        "p95_ttft_reduction_percent": (
            left_p95 - right_p95
        ) / left_p95 * 100.0,
        "ssd_read_reduction_percent": (
            (left_ssd - right_ssd) / left_ssd * 100.0
            if left_ssd
            else None
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {
        "protocol": {
            "datasets_are_independent": True,
            "pooled_headline_metric": False,
            "accuracy_scoring": "label_continuation_loglikelihood",
            "warmup_samples_per_task": 32,
            "variant": args.variant,
        },
        "tasks": {},
    }
    for task_index, task in enumerate(TASKS):
        task_result: dict[str, Any] = {}
        for budget_index, budget in enumerate(BUDGETS):
            runs = {
                "contigkv": load_run(
                    args.baseline_root / task / f"k{budget}_contigkv",
                    task,
                ),
                "ours": load_run(
                    args.baseline_root / task / f"k{budget}_ours",
                    task,
                ),
                "promixed": load_run(
                    args.promixed_root
                    / task
                    / f"k{budget}_promixed_{args.variant}",
                    task,
                ),
            }
            seed = 42 + task_index * 10 + budget_index
            task_result[budget] = {
                "absolute": {
                    family: absolute_metrics(run)
                    for family, run in runs.items()
                },
                "promixed_vs_contigkv": paired_comparison(
                    runs["contigkv"],
                    runs["promixed"],
                    seed=seed,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                "promixed_vs_previous_ours": paired_comparison(
                    runs["ours"],
                    runs["promixed"],
                    seed=seed + 100,
                    bootstrap_samples=args.bootstrap_samples,
                ),
            }
        output["tasks"][task] = task_result
    return output


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# ProMixed full-grid paired comparison",
        "",
        "Each dataset is a separate process and workload. Every comparison is "
        "paired by UID at the same KV budget; no cross-dataset headline metric "
        "is computed.",
        "",
    ]
    for task, task_result in result["tasks"].items():
        lines.extend(
            [
                f"## {task.upper()}",
                "",
                "| KV | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Observed KV |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for budget, budget_result in task_result.items():
            for family in ("contigkv", "ours", "promixed"):
                metric = budget_result["absolute"][family]
                lines.append(
                    "| {budget:.0f}% | {method} | {samples} | {accuracy:.4f} | "
                    "{mean:.2f} | {p95:.2f} | {keep:.2f}% |".format(
                        budget=int(budget),
                        method=DISPLAY[family],
                        samples=metric["samples"],
                        accuracy=metric["accuracy"],
                        mean=metric["mean_ttft_ms"],
                        p95=metric["p95_ttft_ms"],
                        keep=metric["mean_effective_keep_ratio"] * 100.0,
                    )
                )
        lines.extend(
            [
                "",
                "| KV | ProMixed accuracy delta vs ContiguousKV | Mean TTFT reduction | P95 reduction | W->C / C->W | McNemar p | Faster requests |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for budget, budget_result in task_result.items():
            comparison = budget_result["promixed_vs_contigkv"]
            lines.append(
                "| {budget:.0f}% | {accuracy:+.2f} pp | {mean:+.2f}% | "
                "{p95:+.2f}% | {w2c}/{c2w} | {p:.4g} | {faster}/{samples} |".format(
                    budget=int(budget),
                    accuracy=comparison["accuracy_delta_pp"],
                    mean=comparison["mean_ttft_reduction_percent"],
                    p95=comparison["p95_ttft_reduction_percent"],
                    w2c=comparison["wrong_to_correct"],
                    c2w=comparison["correct_to_wrong"],
                    p=comparison["mcnemar_two_sided_p"],
                    faster=comparison["candidate_faster_requests"],
                    samples=comparison["samples"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--promixed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("k4", "fp16"), default="k4")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args)
    report = render_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
