#!/usr/bin/env python3
"""Report each paper dataset independently; never pool requests across tasks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


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


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute P95 for an empty filtered run")
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _apply_exclusions(
    metric: dict[str, Any],
    run_dir: Path,
    excluded_uids: set[str],
) -> dict[str, Any]:
    if not excluded_uids:
        return metric
    records_path = run_dir / "scored_records.jsonl"
    if not records_path.is_file():
        raise ValueError(f"{records_path} is required for strict holdout analysis")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != int(metric["samples"]):
        raise ValueError(
            f"{records_path} has {len(records)} rows, expected {metric['samples']}"
        )
    present = {str(row["uid"]) for row in records}
    missing = excluded_uids - present
    if missing:
        raise ValueError(f"{records_path} is missing excluded UIDs {sorted(missing)}")
    retained = [row for row in records if str(row["uid"]) not in excluded_uids]
    if not retained:
        raise ValueError(f"{records_path} has no rows after strict holdout filtering")
    ttfts = [float(row["ttft_ms"]) for row in retained]
    filtered = dict(metric)
    filtered.update(
        {
            "samples": len(retained),
            "accuracy": sum(bool(row["correct"]) for row in retained) / len(retained),
            "mean_ttft_ms": sum(ttfts) / len(ttfts),
            "p95_ttft_ms": _p95(ttfts),
            "mean_selector_calls": sum(
                float(row.get("selector_calls", 0.0)) for row in retained
            )
            / len(retained),
            "mean_effective_keep_ratio": sum(
                float(row["effective_mean_keep_ratio"]) for row in retained
            )
            / len(retained),
            "excluded_calibration_uids": sorted(excluded_uids),
        }
    )
    return filtered


def load_results(
    run_root: Path,
    exclusions: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    exclusions = exclusions or {}
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for task_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        task = task_dir.name.lower()
        task_results: dict[str, dict[str, Any]] = {}
        for run_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            match = RUN_NAME.match(run_dir.name)
            summary_path = run_dir / "summary.json"
            if match is None or not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if set(summary.get("tasks", {})) != {task}:
                raise ValueError(f"{summary_path} is not an independent {task} run")
            accuracy_scoring = summary.get("runtime", {}).get("accuracy_scoring")
            if accuracy_scoring not in {
                "label_first_token_logit",
                "label_continuation_loglikelihood",
            }:
                raise ValueError(f"{summary_path} does not use label-logit scoring")
            metric = dict(summary["tasks"][task])
            metric["accuracy_scoring"] = accuracy_scoring
            metric["runtime_variant"] = summary["runtime"]["runtime_variant"]
            metric["selection_period_size"] = summary["runtime"].get(
                "impress_selection_period_size", 1
            )
            if "mean_effective_keep_ratio" not in metric:
                raise ValueError(
                    f"{summary_path} does not report the observed KV keep ratio"
                )
            metric = _apply_exclusions(
                metric,
                run_dir,
                exclusions.get(task, set()),
            )
            target_keep_ratio = int(match.group("budget")) / 100.0
            metric["target_keep_ratio"] = target_keep_ratio
            metric["observed_budget_delta_pp"] = (
                float(metric["mean_effective_keep_ratio"]) - target_keep_ratio
            ) * 100.0
            task_results[
                f"{match.group('budget')}_{match.group('family')}"
            ] = metric
        if task_results:
            results[task] = task_results
    return results


def _percent_reduction(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def build_report(
    results: dict[str, dict[str, dict[str, Any]]],
    exclusions: dict[str, set[str]] | None = None,
) -> str:
    scoring_modes = {
        metric["accuracy_scoring"]
        for task_results in results.values()
        for metric in task_results.values()
    }
    if len(scoring_modes) != 1:
        raise ValueError("dataset-wise report cannot mix accuracy scoring protocols")
    scoring_mode = next(iter(scoring_modes))
    lines = [
        "# Dataset-wise ContiguousKV comparison",
        "",
        "Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.",
        f"Accuracy scoring: `{scoring_mode}`.",
        "",
    ]
    if exclusions:
        lines.extend(
            [
                "Strict holdout filtering excludes every UID used to calibrate the layer-budget profile.",
                "",
            ]
        )
    for task, task_results in results.items():
        lines.extend(
            [
                f"## {task.upper()}",
                "",
                "| KV budget | Method | N | Accuracy | Observed KV | Delta (pp) | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        budgets = sorted({key.split("_", 1)[0] for key in task_results})
        for budget in budgets:
            for family in ("impress", "contigkv", "ours"):
                metric = task_results.get(f"{budget}_{family}")
                if metric is None:
                    continue
                lines.append(
                    "| {budget:.0f}% | {method} | {samples} | {accuracy:.4f} | "
                    "{observed:.2f}% | {delta:+.2f} | {mean:.2f} | {p95:.2f} | "
                    "{calls:.2f} | {period} |".format(
                        budget=int(budget),
                        method=DISPLAY[family],
                        samples=int(metric["samples"]),
                        accuracy=float(metric["accuracy"]),
                        observed=float(metric["mean_effective_keep_ratio"]) * 100.0,
                        delta=float(metric["observed_budget_delta_pp"]),
                        mean=float(metric["mean_ttft_ms"]),
                        p95=float(metric["p95_ttft_ms"]),
                        calls=float(metric.get("mean_selector_calls", 0.0)),
                        period=int(metric.get("selection_period_size", 1)),
                    )
                )
        lines.extend(
            [
                "",
                "| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |",
                "|---:|---:|---:|---:|",
            ]
        )
        for budget in budgets:
            ours = task_results.get(f"{budget}_ours")
            contig = task_results.get(f"{budget}_contigkv")
            if ours is None or contig is None:
                continue
            lines.append(
                "| {budget:.0f}% | {accuracy:+.2f} | {mean:+.2f}% | {p95:+.2f}% |".format(
                    budget=int(budget),
                    accuracy=(
                        float(ours["accuracy"]) - float(contig["accuracy"])
                    )
                    * 100.0,
                    mean=_percent_reduction(
                        float(contig["mean_ttft_ms"]),
                        float(ours["mean_ttft_ms"]),
                    ),
                    p95=_percent_reduction(
                        float(contig["p95_ttft_ms"]),
                        float(ours["p95_ttft_ms"]),
                    ),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exclusions = load_exclusions(args.exclude_uids)
    results = load_results(args.run_root, exclusions)
    if not results:
        raise SystemExit("no completed dataset-wise runs were found")
    report = build_report(results, exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
