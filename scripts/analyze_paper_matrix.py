#!/usr/bin/env python3
"""Audit and render the Qwen2.5-7B Figure 9-11 paper-count matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METHODS = ("IMPRESS", "ContiguousKV", "Ours")
TASKS = ("sst2", "subj", "trec", "rte")
BUDGETS = (0.05, 0.10, 0.25, 0.50)
COLORS = {"IMPRESS": "#4477AA", "ContiguousKV": "#CC3311", "Ours": "#228833"}


@dataclass(frozen=True)
class Run:
    spec: dict[str, Any]
    path: Path
    summary: dict[str, Any]
    records: dict[str, dict[str, Any]]


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_mean_ci(
    deltas: list[float], *, samples: int = 20_000, seed: int = 42
) -> list[float]:
    generator = random.Random(seed)
    count = len(deltas)
    means = [
        statistics.fmean(generator.choice(deltas) for _ in range(count))
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


def load_run(
    root: Path,
    spec: dict[str, Any],
    expected_samples: int,
    expected_samples_by_task: dict[str, int],
) -> Run:
    path = root / str(spec["label"])
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "scored_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = {str(row["uid"]): row for row in rows}
    if len(rows) != len(records) or len(records) != expected_samples:
        raise ValueError(f"{path} does not contain {expected_samples} unique requests")
    counts = Counter(str(row["task"]) for row in rows)
    if dict(counts) != expected_samples_by_task:
        raise ValueError(
            f"{path} task counts are {dict(counts)}; expected {expected_samples_by_task}"
        )
    if int(summary["overall"]["samples"]) != expected_samples:
        raise ValueError(f"{path} summary sample count is inconsistent")
    return Run(spec=spec, path=path, summary=summary, records=records)


def require_equal(runtime: dict[str, Any], path: Path, expected: dict[str, Any]) -> None:
    for field, value in expected.items():
        actual = runtime.get(field)
        if isinstance(value, float):
            matches = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(
                f"{path} runtime field {field!r} is {actual!r}; expected {value!r}"
            )


def validate_runtime(run: Run, expected_samples: int) -> None:
    runtime = run.summary["runtime"]
    family = str(run.spec["family"])
    budget = float(run.spec["budget"])
    require_equal(
        runtime,
        run.path,
        {
            "model_compute_dtype": "bfloat16",
            "pcache_storage_dtype": "float16",
            "keep_ratio": budget,
            "online_selection": True,
            "period_size": 8,
            "subperiod_size": 4,
            "gpu_cache_mb": 55.0,
            "cpu_cache_mb": 131.0,
            "cache_type": "CKLFU",
            "warmup_passes": 1,
            "warmup_requests": expected_samples,
            "probe_query_heads": [0, 1, 2],
        },
    )
    if family == "contiguouskv":
        expected = {
            "method": "contigkv",
            "chunk_size": 16,
            "selector_kv_head_ids": [0, 1, 2, 3],
            "similarity_alpha": 0.6,
            "impress_selection_block_size": 1,
            "impress_reorder_enabled": False,
            "impress_async_inter_layer_prefetch": False,
            "exact_layer_block_budget": False,
        }
        profile_expected = False
    elif family == "impress":
        expected = {
            "method": "impress",
            "chunk_size": 64,
            "selector_kv_head_ids": [0],
            "similarity_alpha": 0.6,
            "impress_selection_block_size": 1,
            "impress_reorder_enabled": True,
            "impress_async_inter_layer_prefetch": False,
            "exact_layer_block_budget": False,
        }
        profile_expected = False
    elif family == "ours":
        expected = {
            "method": "impress",
            "chunk_size": 16,
            "selector_kv_head_ids": [0],
            "similarity_alpha": 1.0,
            "impress_selection_block_size": 16,
            "impress_reorder_enabled": False,
            "impress_async_inter_layer_prefetch": True,
            "impress_predictive_period_prefetch": True,
            "impress_period_prefetch_size": 4,
            "impress_period_prefetch_budget_scale": 0.25,
            "impress_value_ordered_prefetch": True,
            "impress_value_prefetch_budget_scale": 0.9,
            "exact_layer_block_budget": True,
        }
        profile_expected = True
    else:
        raise ValueError(f"unknown family {family!r}")
    require_equal(runtime, run.path, expected)
    has_profile = runtime.get("layer_budget_profile") is not None
    if has_profile != profile_expected:
        raise ValueError(f"{run.path} has an unexpected layer-budget profile state")

    for record in run.records.values():
        if family == "ours":
            target = record.get("exact_block_budget_target")
            consumed = record.get("exact_block_budget_consumed")
            if not isinstance(target, int) or target <= 0 or consumed != target:
                raise ValueError(f"{run.path} did not enforce its exact block budget")
        elif record.get("exact_block_budget_target") is not None:
            raise ValueError(f"{run.path} unexpectedly reports an exact block budget")


def aggregate(run: Run, task: str | None = None) -> dict[str, Any]:
    rows = [
        row for row in run.records.values() if task is None or row["task"] == task
    ]
    ttfts = [float(row["ttft_ms"]) for row in rows]
    return {
        "label": run.spec["label"],
        "method": run.spec["display"],
        "family": run.spec["family"],
        "budget": float(run.spec["budget"]),
        "task": task or "overall",
        "samples": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": statistics.fmean(bool(row["correct"]) for row in rows),
        "mean_ttft_ms": statistics.fmean(ttfts),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "mean_effective_keep_ratio": statistics.fmean(
            float(row["effective_mean_keep_ratio"]) for row in rows
        ),
        "mean_selected_kv_bytes": statistics.fmean(
            float(row["selected_kv_bytes"]) for row in rows
        ),
        "mean_total_ssd_read_bytes": statistics.fmean(
            float(row["total_ssd_read_bytes"]) for row in rows
        ),
        "mean_prefetch_wait_ms": statistics.fmean(
            float(row["prefetch_wait_ms"]) for row in rows
        ),
    }


def compare(baseline: Run, candidate: Run) -> dict[str, Any]:
    if set(baseline.records) != set(candidate.records):
        raise ValueError("paired comparison requires identical request UIDs")
    uids = sorted(baseline.records)
    baseline_ttft = [float(baseline.records[uid]["ttft_ms"]) for uid in uids]
    candidate_ttft = [float(candidate.records[uid]["ttft_ms"]) for uid in uids]
    deltas = [right - left for left, right in zip(baseline_ttft, candidate_ttft)]
    baseline_correct = [bool(baseline.records[uid]["correct"]) for uid in uids]
    candidate_correct = [bool(candidate.records[uid]["correct"]) for uid in uids]
    wrong_to_correct = sum(
        not left and right for left, right in zip(baseline_correct, candidate_correct)
    )
    correct_to_wrong = sum(
        left and not right for left, right in zip(baseline_correct, candidate_correct)
    )
    left_mean = statistics.fmean(baseline_ttft)
    right_mean = statistics.fmean(candidate_ttft)
    baseline_p95 = percentile(baseline_ttft, 0.95)
    candidate_p95 = percentile(candidate_ttft, 0.95)
    return {
        "budget": float(candidate.spec["budget"]),
        "baseline": baseline.spec["display"],
        "candidate": candidate.spec["display"],
        "samples": len(uids),
        "baseline_accuracy": statistics.fmean(baseline_correct),
        "candidate_accuracy": statistics.fmean(candidate_correct),
        "accuracy_delta_points": (
            statistics.fmean(candidate_correct) - statistics.fmean(baseline_correct)
        ) * 100.0,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "mcnemar_two_sided_p": exact_mcnemar_p(wrong_to_correct, correct_to_wrong),
        "baseline_mean_ttft_ms": left_mean,
        "candidate_mean_ttft_ms": right_mean,
        "mean_ttft_change_percent": (right_mean / left_mean - 1.0) * 100.0,
        "baseline_p95_ttft_ms": baseline_p95,
        "candidate_p95_ttft_ms": candidate_p95,
        "p95_ttft_change_percent": (candidate_p95 / baseline_p95 - 1.0) * 100.0,
        "mean_paired_delta_ms": statistics.fmean(deltas),
        "paired_bootstrap_mean_delta_95ci_ms": bootstrap_mean_ci(deltas),
        "candidate_faster_requests": sum(delta < 0 for delta in deltas),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _task_row(
    rows: list[dict[str, Any]], task: str, budget: float, method: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["task"] == task
        and math.isclose(row["budget"], budget)
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {task}, {budget}, {method}")
    return matches[0]


def _line_grid(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    *,
    field: str,
    ylabel: str,
    filename: str,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), sharex=True)
    for axis, task in zip(axes.flat, TASKS):
        for method in METHODS:
            values = [_task_row(tasks, task, budget, method) for budget in BUDGETS]
            axis.plot(
                [budget * 100 for budget in BUDGETS],
                [float(row[field]) * scale for row in values],
                marker="o",
                linewidth=2,
                label=method,
                color=COLORS[method],
            )
        axis.set_title(task.upper())
        axis.set_xticks([5, 10, 25, 50])
        axis.set_xlabel("KV budget (%)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def make_plots(
    output_dir: Path,
    overall: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _line_grid(
        output_dir,
        tasks,
        field="accuracy",
        ylabel="Accuracy (%)",
        filename="figure9_accuracy_qwen25_7b.png",
        scale=100.0,
    )

    fig, axes = plt.subplots(2, 4, figsize=(14.8, 6.3), sharey=False)
    for row_index, budget in enumerate((0.05, 0.25)):
        for column_index, task in enumerate(TASKS):
            axis = axes[row_index, column_index]
            values = [
                _task_row(tasks, task, budget, method)["mean_ttft_ms"] / 1000.0
                for method in METHODS
            ]
            axis.bar(
                range(len(METHODS)),
                values,
                color=[COLORS[method] for method in METHODS],
                width=0.72,
            )
            axis.set_title(f"{task.upper()} ({budget * 100:.0f}%)")
            axis.set_xticks(range(len(METHODS)), ("IMPRESS", "ContigKV", "Ours"), rotation=18)
            axis.set_ylabel("Mean TTFT (s)")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 10-style mean TTFT - Qwen2.5-7B", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "figure10_mean_ttft_qwen25_7b.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    for axis, task in zip(axes, ("sst2", "rte")):
        values = [
            _task_row(tasks, task, 0.05, method)["p95_ttft_ms"] / 1000.0
            for method in METHODS
        ]
        axis.bar(
            range(len(METHODS)),
            values,
            color=[COLORS[method] for method in METHODS],
            width=0.68,
        )
        axis.set_title(task.upper())
        axis.set_xticks(range(len(METHODS)), ("IMPRESS", "ContigKV", "Ours"), rotation=18)
        axis.set_ylabel("P95 TTFT (s)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 11-style P95 TTFT at 5% - Qwen2.5-7B", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "figure11_p95_qwen25_7b.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    _line_grid(
        output_dir,
        tasks,
        field="mean_ttft_ms",
        ylabel="Mean TTFT (ms)",
        filename="extended_mean_ttft_all_budgets.png",
    )
    _line_grid(
        output_dir,
        tasks,
        field="p95_ttft_ms",
        ylabel="P95 TTFT (ms)",
        filename="extended_p95_all_budgets.png",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    metrics = (
        ("accuracy", "Accuracy", 100.0, "Percent"),
        ("mean_ttft_ms", "Mean TTFT", 1.0, "ms"),
        ("p95_ttft_ms", "P95 TTFT", 1.0, "ms"),
    )
    for axis, (field, title, scale, ylabel) in zip(axes, metrics):
        for method in METHODS:
            points = sorted(
                (row for row in overall if row["method"] == method),
                key=lambda row: row["budget"],
            )
            axis.plot(
                [row["budget"] * 100 for row in points],
                [float(row[field]) * scale for row in points],
                marker="o",
                linewidth=2,
                label=method,
                color=COLORS[method],
            )
        axis.set_title(title)
        axis.set_xticks([5, 10, 25, 50])
        axis.set_xlabel("KV budget (%)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "matrix_overall.png", dpi=200)
    plt.close(fig)


def trend_checks(by_task: list[dict[str, Any]]) -> dict[str, Any]:
    contig_accuracy_wins = 0
    ours_accuracy_not_worse = 0
    for budget in BUDGETS:
        contig = sum(
            _task_row(by_task, task, budget, "ContiguousKV")["correct"]
            for task in TASKS
        )
        impress = sum(
            _task_row(by_task, task, budget, "IMPRESS")["correct"] for task in TASKS
        )
        ours = sum(
            _task_row(by_task, task, budget, "Ours")["correct"] for task in TASKS
        )
        contig_accuracy_wins += contig > impress
        ours_accuracy_not_worse += ours >= contig

    fig10_cells = [(task, budget) for budget in (0.05, 0.25) for task in TASKS]
    contig_faster_fig10 = sum(
        _task_row(by_task, task, budget, "ContiguousKV")["mean_ttft_ms"]
        < _task_row(by_task, task, budget, "IMPRESS")["mean_ttft_ms"]
        for task, budget in fig10_cells
    )
    ours_faster_than_contig_fig10 = sum(
        _task_row(by_task, task, budget, "Ours")["mean_ttft_ms"]
        < _task_row(by_task, task, budget, "ContiguousKV")["mean_ttft_ms"]
        for task, budget in fig10_cells
    )
    fig11_tasks = ("sst2", "rte")
    contig_faster_fig11 = sum(
        _task_row(by_task, task, 0.05, "ContiguousKV")["p95_ttft_ms"]
        < _task_row(by_task, task, 0.05, "IMPRESS")["p95_ttft_ms"]
        for task in fig11_tasks
    )
    ours_faster_than_contig_fig11 = sum(
        _task_row(by_task, task, 0.05, "Ours")["p95_ttft_ms"]
        < _task_row(by_task, task, 0.05, "ContiguousKV")["p95_ttft_ms"]
        for task in fig11_tasks
    )
    return {
        "contiguouskv_accuracy_beats_impress_budgets": contig_accuracy_wins,
        "ours_accuracy_not_worse_than_contiguouskv_budgets": ours_accuracy_not_worse,
        "contiguouskv_mean_ttft_beats_impress_figure10_cells": contig_faster_fig10,
        "ours_mean_ttft_beats_contiguouskv_figure10_cells": ours_faster_than_contig_fig10,
        "contiguouskv_p95_beats_impress_figure11_cells": contig_faster_fig11,
        "ours_p95_beats_contiguouskv_figure11_cells": ours_faster_than_contig_fig11,
        "figure10_cells": len(fig10_cells),
        "figure11_cells": len(fig11_tasks),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    checks = payload["trend_checks"]
    sample_counts = payload["paper_scope"]["reproduction_samples_by_task"]
    sample_total = sum(int(value) for value in sample_counts.values())
    sample_description = ", ".join(
        f"{task.upper()}={count}" for task, count in sample_counts.items()
    )
    lines = [
        "# Qwen2.5-7B Figure 9-11 paper-count matrix",
        "",
        "## Protocol",
        "",
        f"- Four datasets with Table 1 counts: {sample_description} ({sample_total} requests total).",
        f"- Sequential batch-size-1 single-request execution; one complete {sample_total}-request warmup pass before measurement.",
        "- Qwen2.5-7B, BF16 model compute, FP16 KV storage, 55/131 MiB GPU/CPU caches, CKLFU, seed 42 task bundle.",
        "- Methods: reproduced IMPRESS, reproduced ContiguousKV, and the sensitivity-aware HyperInfer-derived method with 16-token blocks.",
        "- All methods use identical request UIDs at 5%, 10%, 25%, and 50% KV budgets.",
        "",
        "## Overall",
        "",
        "| KV budget | Method | Correct | Accuracy | Effective budget | Mean TTFT (ms) | P95 TTFT (ms) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["overall"]:
        lines.append(
            f"| {row['budget'] * 100:.0f}% | {row['method']} | {row['correct']}/{row['samples']} | "
            f"{row['accuracy']:.4f} | {row['mean_effective_keep_ratio'] * 100:.2f}% | "
            f"{row['mean_ttft_ms']:.1f} | {row['p95_ttft_ms']:.1f} |"
        )

    lines.extend((
        "",
        "## Ours Versus Baselines",
        "",
        "Negative TTFT change means Ours is faster. Confidence intervals use paired request-level bootstrap deltas (Ours minus baseline).",
        "",
        "| KV budget | Baseline | Accuracy delta | Mean TTFT change | P95 change | Mean delta 95% CI (ms) | Faster requests |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ))
    for row in payload["comparisons"]:
        ci = row["paired_bootstrap_mean_delta_95ci_ms"]
        lines.append(
            f"| {row['budget'] * 100:.0f}% | {row['baseline']} | "
            f"{row['accuracy_delta_points']:+.2f} pp | {row['mean_ttft_change_percent']:+.2f}% | "
            f"{row['p95_ttft_change_percent']:+.2f}% | [{ci[0]:.1f}, {ci[1]:.1f}] | "
            f"{row['candidate_faster_requests']}/{row['samples']} |"
        )

    lines.extend((
        "",
        "## Paper-Trend Checks",
        "",
        f"- Figure 9 direction: ContiguousKV beats IMPRESS in overall accuracy at {checks['contiguouskv_accuracy_beats_impress_budgets']}/4 budgets.",
        f"- Proposed method has accuracy no worse than ContiguousKV at {checks['ours_accuracy_not_worse_than_contiguouskv_budgets']}/4 budgets.",
        f"- Figure 10 direction: ContiguousKV mean TTFT beats IMPRESS in {checks['contiguouskv_mean_ttft_beats_impress_figure10_cells']}/{checks['figure10_cells']} dataset-budget cells; Ours beats ContiguousKV in {checks['ours_mean_ttft_beats_contiguouskv_figure10_cells']}/{checks['figure10_cells']}.",
        f"- Figure 11 direction: ContiguousKV P95 beats IMPRESS in {checks['contiguouskv_p95_beats_impress_figure11_cells']}/{checks['figure11_cells']} representative datasets; Ours beats ContiguousKV in {checks['ours_p95_beats_contiguouskv_figure11_cells']}/{checks['figure11_cells']}.",
        "",
        "## Per Dataset",
        "",
        "| KV budget | Dataset | Method | Correct | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |",
        "|---:|---|---|---:|---:|---:|---:|",
    ))
    for row in payload["by_task"]:
        lines.append(
            f"| {row['budget'] * 100:.0f}% | {row['task'].upper()} | {row['method']} | "
            f"{row['correct']}/{row['samples']} | {row['accuracy']:.4f} | "
            f"{row['mean_ttft_ms']:.1f} | {row['p95_ttft_ms']:.1f} |"
        )

    lines.extend((
        "",
        "## Scope Relative To The Paper",
        "",
        "- Figure 9 alignment: four datasets and the same 5/10/25/50% budgets; this reproduction currently covers Qwen2.5-7B and three methods rather than three model sizes and four methods.",
        "- Figure 10 alignment: mean TTFT at 5% and 25% on all four datasets. Mean TTFT at 10% and 50% is retained as an extension.",
        "- Figure 11 alignment: P95 TTFT at 5% on SST-2 and RTE. P95 on other datasets and budgets is retained as an extension.",
        "- Absolute latency is not directly comparable: the paper uses A800 80GB, 10/24GB cache limits, and Samsung 990 Pro; this run uses RTX 3090 and 55/131 MiB cache limits on the available server SSD.",
        "- The paper reports average ContiguousKV accuracy gains over IMPRESS of 7.69%, 4.81%, 3.58%, and 1.63% at 5/10/25/50%, plus a 3.85x TTFT reduction at 5%. Those are multi-model aggregates, not Qwen2.5-7B-only targets.",
        "- The 5/10/25% sensitivity profiles reuse and proportionally scale the frozen 50% layer ranking; they were not independently recalibrated per budget.",
    ))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def expected_sample_counts(manifest: dict[str, Any]) -> dict[str, int]:
    if "samples_by_task" in manifest:
        counts = {
            str(task): int(manifest["samples_by_task"][task])
            for task in manifest["tasks"]
        }
    else:
        counts = {
            str(task): int(manifest["samples_per_task"])
            for task in manifest["tasks"]
        }
    if sum(counts.values()) != int(manifest["samples"]):
        raise ValueError(
            f"manifest task counts sum to {sum(counts.values())}; "
            f"expected {manifest['samples']}"
        )
    return counts


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_samples = int(manifest["samples"])
    expected_samples_by_task = expected_sample_counts(manifest)
    runs = [
        load_run(
            args.run_root,
            spec,
            expected_samples,
            expected_samples_by_task,
        )
        for spec in manifest["runs"]
    ]
    for run in runs:
        validate_runtime(run, expected_samples)
    uid_sets = [set(run.records) for run in runs]
    if any(uids != uid_sets[0] for uids in uid_sets[1:]):
        raise ValueError("matrix runs do not use identical request UIDs")

    by_label = {str(run.spec["label"]): run for run in runs}
    overall = [aggregate(run) for run in runs]
    by_task = [
        aggregate(run, task)
        for run in runs
        for task in manifest["tasks"]
    ]
    comparisons = []
    for budget_tag in ("005", "010", "025", "050"):
        ours = by_label[f"k{budget_tag}_ours"]
        comparisons.append(compare(by_label[f"k{budget_tag}_contiguouskv"], ours))
        comparisons.append(compare(by_label[f"k{budget_tag}_impress"], ours))

    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "overall": overall,
        "by_task": by_task,
        "comparisons": comparisons,
        "trend_checks": trend_checks(by_task),
        "paper_scope": {
            "paper_accuracy_budgets": list(BUDGETS),
            "paper_ttft_budgets": [0.05, 0.25],
            "paper_p95_budget": 0.05,
            "paper_p95_tasks": ["sst2", "rte"],
            "paper_prefix_examples_by_task": manifest["paper_prefix_examples_by_task"],
            "paper_evaluation_request_count": manifest.get(
                "paper_evaluation_request_count"
            ),
            "reproduction_samples_by_task": expected_samples_by_task,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fig9_11_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fig9_11_report.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    write_csv(args.output_dir / "fig9_11_overall.csv", overall)
    write_csv(args.output_dir / "fig9_11_by_task.csv", by_task)
    write_csv(args.output_dir / "fig9_11_comparisons.csv", comparisons)
    make_plots(args.output_dir, overall, by_task)
    print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
