"""Compare matched ContiguousKV and IMPRESS paper-client summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPTIONAL_RUNTIME_METRICS = (
    "mean_selected_kv_bytes",
    "mean_physical_prefetch_kv_bytes",
    "mean_read_amplification",
    "mean_prefetch_gpu_source_fraction",
    "mean_prefetch_cpu_source_fraction",
    "mean_prefetch_disk_source_fraction",
    "mean_ssd_prefetch_kv_bytes",
    "mean_total_ssd_read_bytes",
    "mean_selector_key_bytes",
    "mean_selector_gpu_source_bytes",
    "mean_selector_cpu_source_bytes",
    "mean_selector_disk_source_bytes",
    "mean_selector_load_ms",
    "mean_selector_compute_ms",
    "mean_selector_wait_ms",
    "mean_selector_calls",
    "mean_selector_fallbacks",
    "mean_selector_jaccard",
    "mean_selection_reference_jaccard",
    "mean_selection_reference_exact_layer_fraction",
    "mean_speculative_hit_tokens",
    "mean_speculative_missing_tokens",
    "mean_speculative_unused_tokens",
    "mean_impress_prefetch_budget_seconds",
    "mean_prefetch_wait_ms",
    "mean_prefetch_elapsed_ms",
    "mean_cache_update_ms",
)


def _add_runtime_comparison(
    target: dict[str, Any],
    contiguous: dict[str, Any],
    impress: dict[str, Any],
) -> None:
    for metric in OPTIONAL_RUNTIME_METRICS:
        if metric in contiguous and metric in impress:
            target[f"contiguous_{metric}"] = contiguous[metric]
            target[f"impress_{metric}"] = impress[metric]
    physical = "mean_physical_prefetch_kv_bytes"
    if physical in contiguous and physical in impress:
        target["physical_read_reduction_vs_impress"] = (
            float(impress[physical]) / max(float(contiguous[physical]), 1.0)
        )
    ssd = "mean_ssd_prefetch_kv_bytes"
    if ssd in contiguous and ssd in impress:
        target["ssd_read_reduction_vs_impress"] = (
            float(impress[ssd]) / max(float(contiguous[ssd]), 1.0)
        )
    total_ssd = "mean_total_ssd_read_bytes"
    if total_ssd in contiguous and total_ssd in impress:
        target["total_ssd_read_reduction_vs_impress"] = (
            float(impress[total_ssd]) / max(float(contiguous[total_ssd]), 1.0)
        )


def _aggregate_task_metric(summary: dict[str, Any], metric: str) -> float | None:
    values = []
    for task in summary.get("tasks", {}).values():
        if metric not in task:
            continue
        weight = max(1, int(task.get("samples", 1)))
        values.append((float(task[metric]), weight))
    if not values:
        return None
    return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "client" / "summary.json").read_text(encoding="utf-8"))


def compare_summaries(contiguous: dict[str, Any], impress: dict[str, Any]) -> dict[str, Any]:
    tasks = sorted(set(contiguous["tasks"]) & set(impress["tasks"]))
    task_results = {}
    for task in tasks:
        contig = contiguous["tasks"][task]
        baseline = impress["tasks"][task]
        task_results[task] = {
            "contiguous_accuracy": contig["accuracy"],
            "impress_accuracy": baseline["accuracy"],
            "accuracy_delta": contig["accuracy"] - baseline["accuracy"],
            "contiguous_mean_ttft_ms": contig["mean_ttft_ms"],
            "impress_mean_ttft_ms": baseline["mean_ttft_ms"],
            "ttft_speedup_vs_impress": baseline["mean_ttft_ms"] / contig["mean_ttft_ms"],
            "contiguous_p95_ttft_ms": contig["p95_ttft_ms"],
            "impress_p95_ttft_ms": baseline["p95_ttft_ms"],
        }
        _add_runtime_comparison(task_results[task], contig, baseline)
    overall_contig = contiguous["overall"]
    overall_impress = impress["overall"]
    overall = {
        "contiguous_accuracy": overall_contig["accuracy"],
        "impress_accuracy": overall_impress["accuracy"],
        "accuracy_delta": overall_contig["accuracy"] - overall_impress["accuracy"],
        "contiguous_mean_ttft_ms": overall_contig["mean_ttft_ms"],
        "impress_mean_ttft_ms": overall_impress["mean_ttft_ms"],
        "ttft_speedup_vs_impress": overall_impress["mean_ttft_ms"] / overall_contig["mean_ttft_ms"],
        "contiguous_p95_ttft_ms": overall_contig["p95_ttft_ms"],
        "impress_p95_ttft_ms": overall_impress["p95_ttft_ms"],
    }
    aggregate_contig = dict(overall_contig)
    aggregate_impress = dict(overall_impress)
    for metric in OPTIONAL_RUNTIME_METRICS:
        contig_value = _aggregate_task_metric(contiguous, metric)
        impress_value = _aggregate_task_metric(impress, metric)
        if contig_value is not None:
            aggregate_contig[metric] = contig_value
        if impress_value is not None:
            aggregate_impress[metric] = impress_value
    _add_runtime_comparison(overall, aggregate_contig, aggregate_impress)
    return {
        "comparison": "ContiguousKV versus IMPRESS",
        "tasks": task_results,
        "overall": overall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ContiguousKV and IMPRESS paper-client results.")
    parser.add_argument("--contiguous-run", required=True)
    parser.add_argument("--impress-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_summaries(load_summary(args.contiguous_run), load_summary(args.impress_run))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
