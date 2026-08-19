#!/usr/bin/env python3
"""Paired, per-dataset comparisons without cross-dataset aggregation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


RUN_NAME = re.compile(r"^k(?P<budget>\d{3})_(?P<family>impress|contigkv|ours)$")
DISPLAY = {
    "impress": "IMPRESS",
    "contigkv": "ContiguousKV",
    "ours": "Ours",
}


def load_exclusions(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(task).lower(): {str(uid) for uid in uids}
        for task, uids in payload.get("exclude_uids_by_task", {}).items()
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def percentile95_nearest_rank(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute P95 of an empty sequence")
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 20_000,
    seed: int = 42,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    try:
        import numpy as np

        source = np.asarray(values, dtype=np.float64)
        generator = np.random.default_rng(seed)
        means = []
        for start in range(0, samples, 1_000):
            count = min(1_000, samples - start)
            indices = generator.integers(
                0,
                len(values),
                size=(count, len(values)),
            )
            means.extend(source[indices].mean(axis=1).tolist())
    except ImportError:
        import random
        import statistics

        generator = random.Random(seed)
        means = [
            statistics.fmean(generator.choice(values) for _ in values)
            for _ in range(samples)
        ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def exact_mcnemar_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_correct, correct_to_wrong)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def load_runs(
    run_root: Path,
    exclusions: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    exclusions = exclusions or {}
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for task_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        task = task_dir.name.lower()
        task_runs: dict[str, dict[str, Any]] = {}
        for run_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            match = RUN_NAME.match(run_dir.name)
            summary_path = run_dir / "summary.json"
            records_path = run_dir / "scored_records.jsonl"
            if match is None or not summary_path.is_file() or not records_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if set(summary.get("tasks", {})) != {task}:
                raise ValueError(f"{summary_path} is not an independent {task} run")
            if (
                summary.get("runtime", {}).get("accuracy_scoring")
                != "label_continuation_loglikelihood"
            ):
                raise ValueError(f"{summary_path} does not use complete label scoring")
            all_records = {
                str(row["uid"]): row
                for row in (
                    json.loads(line)
                    for line in records_path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            }
            excluded = exclusions.get(task, set())
            missing = excluded - set(all_records)
            if missing:
                raise ValueError(
                    f"{records_path} is missing excluded UIDs {sorted(missing)}"
                )
            records = {
                uid: row for uid, row in all_records.items() if uid not in excluded
            }
            expected = int(summary["tasks"][task]["samples"]) - len(excluded)
            if len(records) != expected:
                raise ValueError(f"{records_path} has {len(records)} rows, expected {expected}")
            task_runs[f"{match.group('budget')}_{match.group('family')}"] = {
                "summary": summary,
                "records": records,
            }
        if task_runs:
            runs[task] = task_runs
    return runs


def paired_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    left = baseline["records"]
    right = candidate["records"]
    if set(left) != set(right):
        raise ValueError("paired comparison requires identical request UIDs")
    uids = sorted(left)
    left_ttft = [float(left[uid]["ttft_ms"]) for uid in uids]
    right_ttft = [float(right[uid]["ttft_ms"]) for uid in uids]
    deltas = [right_value - left_value for left_value, right_value in zip(left_ttft, right_ttft)]
    left_correct = [bool(left[uid]["correct"]) for uid in uids]
    right_correct = [bool(right[uid]["correct"]) for uid in uids]
    wrong_to_correct = sum(
        not old and new for old, new in zip(left_correct, right_correct)
    )
    correct_to_wrong = sum(
        old and not new for old, new in zip(left_correct, right_correct)
    )
    left_mean = sum(left_ttft) / len(left_ttft)
    right_mean = sum(right_ttft) / len(right_ttft)
    left_ssd = sum(float(left[uid]["total_ssd_read_bytes"]) for uid in uids) / len(uids)
    right_ssd = sum(float(right[uid]["total_ssd_read_bytes"]) for uid in uids) / len(uids)
    left_p95 = percentile95_nearest_rank(left_ttft)
    right_p95 = percentile95_nearest_rank(right_ttft)
    return {
        "samples": len(uids),
        "baseline_accuracy": sum(left_correct) / len(uids),
        "candidate_accuracy": sum(right_correct) / len(uids),
        "accuracy_delta_pp": (sum(right_correct) - sum(left_correct)) / len(uids) * 100.0,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "mcnemar_two_sided_p": exact_mcnemar_p(wrong_to_correct, correct_to_wrong),
        "baseline_mean_ttft_ms": left_mean,
        "candidate_mean_ttft_ms": right_mean,
        "mean_ttft_reduction_percent": (left_mean - right_mean) / left_mean * 100.0,
        "mean_paired_delta_ms": sum(deltas) / len(deltas),
        "paired_mean_delta_95ci_ms": bootstrap_mean_ci(deltas, seed=seed),
        "candidate_faster_requests": sum(delta < 0 for delta in deltas),
        "baseline_p95_ttft_ms": left_p95,
        "candidate_p95_ttft_ms": right_p95,
        "p95_ttft_reduction_percent": (left_p95 - right_p95) / left_p95 * 100.0,
        "baseline_mean_ssd_read_bytes": left_ssd,
        "candidate_mean_ssd_read_bytes": right_ssd,
        "ssd_read_reduction_percent": (left_ssd - right_ssd) / left_ssd * 100.0,
    }


def analyze(runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for task, task_runs in runs.items():
        task_output: dict[str, Any] = {}
        budgets = sorted(key.split("_", 1)[0] for key in task_runs if key.endswith("_ours"))
        for budget in budgets:
            ours = task_runs[f"{budget}_ours"]
            budget_output: dict[str, Any] = {}
            for index, baseline_family in enumerate(("contigkv", "impress")):
                baseline = task_runs.get(f"{budget}_{baseline_family}")
                if baseline is None:
                    continue
                budget_output[f"ours_vs_{baseline_family}"] = paired_comparison(
                    baseline,
                    ours,
                    seed=42 + index + sum(ord(char) for char in f"{task}{budget}"),
                )
            if budget_output:
                task_output[budget] = budget_output
        if task_output:
            comparisons[task] = task_output
    return comparisons


def render_report(
    comparisons: dict[str, Any],
    exclusions: dict[str, set[str]] | None = None,
) -> str:
    lines = [
        "# Paired dataset-wise comparisons",
        "",
        "Every row is paired by UID within one dataset and one KV budget. No cross-dataset aggregate is computed.",
        "",
    ]
    if exclusions:
        lines.extend(
            [
                "Strict holdout filtering excludes every UID used to calibrate the layer-budget profile.",
                "",
            ]
        )
    for task, task_results in comparisons.items():
        lines.extend([f"## {task.upper()}", ""])
        for budget, budget_results in task_results.items():
            lines.extend(
                [
                    f"### {int(budget)}% KV budget",
                    "",
                    "| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for key, result in budget_results.items():
                baseline = DISPLAY[key.removeprefix("ours_vs_")]
                low, high = result["paired_mean_delta_95ci_ms"]
                lines.append(
                    "| Ours vs {baseline} | {samples} | {accuracy:+.2f} | "
                    "{w2c}/{c2w} | {p:.4g} | {mean:+.2f}% | "
                    "[{low:.2f}, {high:.2f}] | {p95:+.2f}% | {ssd:+.2f}% | "
                    "{faster}/{samples} |".format(
                        baseline=baseline,
                        samples=result["samples"],
                        accuracy=result["accuracy_delta_pp"],
                        w2c=result["wrong_to_correct"],
                        c2w=result["correct_to_wrong"],
                        p=result["mcnemar_two_sided_p"],
                        mean=result["mean_ttft_reduction_percent"],
                        low=low,
                        high=high,
                        p95=result["p95_ttft_reduction_percent"],
                        ssd=result["ssd_read_reduction_percent"],
                        faster=result["candidate_faster_requests"],
                    )
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path)
    args = parser.parse_args()
    exclusions = load_exclusions(args.exclude_uids)
    comparisons = analyze(load_runs(args.run_root, exclusions))
    if not comparisons:
        raise SystemExit("no complete paired comparisons were found")
    report = render_report(comparisons, exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(comparisons, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
