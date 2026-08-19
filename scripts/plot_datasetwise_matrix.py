#!/usr/bin/env python3
"""Plot independent, single-model counterparts to ContiguousKV Figures 9-11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ("sst2", "subj", "trec", "rte")
BUDGETS = (5, 10, 25, 50)
TASK_LABELS = {
    "sst2": "SST-2",
    "subj": "SubJ",
    "trec": "TREC",
    "rte": "RTE",
}
METHODS = ("impress", "contigkv", "ours")
METHOD_LABELS = {
    "impress": "IMPRESS",
    "contigkv": "ContiguousKV",
    "ours": "Ours",
}
COLORS = {
    "impress": "#2878B5",
    "contigkv": "#D83A35",
    "ours": "#2E8B57",
}
MARKERS = {"impress": "^", "contigkv": "o", "ours": "s"}


def load_results(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset-wise result must be a JSON object")
    extra = set(payload) - set(TASKS)
    if extra:
        raise ValueError(f"unexpected pooled or unknown result groups: {sorted(extra)}")
    return payload


def override_ours(
    results: dict[str, dict[str, dict[str, Any]]],
    root: Path,
    variant: str,
) -> None:
    """Replace the plotted Ours series with an independently evaluated variant."""
    for task in TASKS:
        for budget in BUDGETS:
            summary_path = (
                root / task / f"k{budget:03d}_promixed_{variant}" / "summary.json"
            )
            if not summary_path.is_file():
                raise ValueError(f"missing override summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            overall = summary["overall"]
            key = f"{budget:03d}_ours"
            current = dict(results[task].get(key, {}))
            for metric in (
                "samples",
                "accuracy",
                "mean_ttft_ms",
                "p95_ttft_ms",
                "mean_effective_keep_ratio",
            ):
                current[metric] = overall[metric]
            results[task][key] = current


def require_metric(
    results: dict[str, dict[str, dict[str, Any]]],
    task: str,
    budget: int,
    method: str,
) -> dict[str, Any]:
    key = f"{budget:03d}_{method}"
    try:
        return results[task][key]
    except KeyError as error:
        raise ValueError(f"missing independent result {task}/{key}") from error


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _save(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _global_legend(fig: Any, *, bars: bool = False, y: float = 0.94) -> None:
    if bars:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=COLORS[method]) for method in METHODS]
    else:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                (0,),
                (0,),
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
            )
            for method in METHODS
        ]
    labels = [METHOD_LABELS[method] for method in METHODS]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.5,
        handletextpad=0.6,
    )


def plot_accuracy(
    results: dict[str, dict[str, dict[str, Any]]], output_dir: Path
) -> None:
    plt = _configure_matplotlib()
    budgets = (5, 10, 25, 50)
    fig, axes_grid = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
    axes = list(axes_grid.flat)
    for axis, task in zip(axes, TASKS):
        for method in METHODS:
            values = [
                100.0
                * float(require_metric(results, task, budget, method)["accuracy"])
                for budget in budgets
            ]
            axis.plot(
                budgets,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                markersize=4.5,
                label=METHOD_LABELS[method],
            )
        axis.set_title(TASK_LABELS[task])
        axis.set_xticks(budgets)
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_xlabel("KV budget (%)")
        axis.set_ylabel("Accuracy (%)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    _global_legend(fig)
    fig.suptitle("Accuracy by independent dataset (Qwen2.5-7B)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    _save(fig, output_dir, "figure9_accuracy_datasetwise")
    plt.close(fig)


def plot_mean_ttft(
    results: dict[str, dict[str, dict[str, Any]]], output_dir: Path
) -> None:
    plt = _configure_matplotlib()
    budgets = (5, 25)
    width = 0.24
    fig, axes_grid = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
    axes = list(axes_grid.flat)
    for axis, task in zip(axes, TASKS):
        for method_index, method in enumerate(METHODS):
            positions = [index + (method_index - 1) * width for index in range(2)]
            values = [
                float(require_metric(results, task, budget, method)["mean_ttft_ms"])
                for budget in budgets
            ]
            axis.bar(
                positions,
                values,
                width=width,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(TASK_LABELS[task])
        axis.set_xticks((0, 1), tuple(f"{budget}%" for budget in budgets))
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_xlabel("KV budget")
        axis.set_ylabel("Mean TTFT (ms)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    _global_legend(fig, bars=True, y=0.91)
    fig.suptitle("Mean TTFT by independent dataset (Qwen2.5-7B)", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    _save(fig, output_dir, "figure10_mean_ttft_datasetwise")
    plt.close(fig)


def plot_p95(
    results: dict[str, dict[str, dict[str, Any]]], output_dir: Path
) -> None:
    plt = _configure_matplotlib()
    tasks = ("sst2", "rte")
    fig, axes_grid = plt.subplots(1, 2, figsize=(7.6, 3.2))
    axes = list(axes_grid.flat)
    for axis, task in zip(axes, tasks):
        values = [
            float(require_metric(results, task, 5, method)["p95_ttft_ms"])
            for method in METHODS
        ]
        axis.bar(
            range(len(METHODS)),
            values,
            color=[COLORS[method] for method in METHODS],
        )
        axis.set_title(TASK_LABELS[task])
        axis.set_xticks(
            range(len(METHODS)),
            tuple(METHOD_LABELS[method] for method in METHODS),
            rotation=12,
        )
        axis.set_ylabel("P95 TTFT (ms)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.suptitle("P95 TTFT at 5% KV budget (Qwen2.5-7B)", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save(fig, output_dir, "figure11_p95_datasetwise")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path)
    parser.add_argument("--ours-variant", choices=("k4", "fp16"), default="k4")
    parser.add_argument("--ours-label", default="Ours")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(args.input)
    if args.ours_root is not None:
        override_ours(results, args.ours_root, args.ours_variant)
    METHOD_LABELS["ours"] = args.ours_label
    plot_accuracy(results, args.output_dir)
    plot_mean_ttft(results, args.output_dir)
    plot_p95(results, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
