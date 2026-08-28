#!/usr/bin/env python3
"""Validate and summarize the five-method re-prefill experiment grid."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TASKS = ("sst2", "subj", "trec", "rte")
BUDGETS = ("005", "010", "025", "050")
METHODS = ("impress", "contigkv", "promixed", "as_lru", "as_h2o_lru")
DISPLAY_METHODS = {
    "impress": "IMPRESS",
    "contigkv": "ContiguousKV",
    "promixed": "ProMixed",
    "as_lru": "AS+LRU",
    "as_h2o_lru": "AS+H2O+LRU",
}
SUMMARY_METRICS = (
    "accuracy",
    "response_ready_mean_ms",
    "response_ready_p95_ms",
    "logits_ready_mean_ms",
    "logits_ready_p95_ms",
)


@dataclass(frozen=True)
class RunSpec:
    task: str
    budget: str
    method: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.task}/k{self.budget}/{self.method}"


@dataclass(frozen=True)
class LoadedRun:
    spec: RunSpec
    records: dict[str, dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, *, context: str) -> float:
    require(type(value) is not bool, f"{context} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    require(math.isfinite(number), f"{context} must be finite")
    return number


def same_number(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-10, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "cannot compute P95 of an empty sequence")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def normalize_budget(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("k"):
        text = text[1:]
    if text.endswith("%"):
        text = text[:-1]
    require(text.isdigit(), f"invalid budget {value!r}")
    normalized = f"{int(text):03d}"
    require(normalized in BUDGETS, f"unsupported budget {value!r}")
    return normalized


def read_json(path: Path, *, context: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {context}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {context} {path}: {error}") from error
    require(isinstance(value, dict), f"{context} must be a JSON object: {path}")
    return value


def load_manifest(path: Path) -> tuple[list[RunSpec], dict[str, Any]]:
    path = path.resolve()
    payload = read_json(path, context="schedule manifest")
    allowed = {"schema_version", "purpose", "runs"}
    require(
        set(payload) <= allowed,
        f"schedule manifest has unknown keys: {sorted(set(payload) - allowed)}",
    )
    require(payload.get("schema_version") == 1, "manifest schema_version must be 1")
    items = payload.get("runs")
    require(isinstance(items, list) and items, "manifest.runs must be non-empty")

    specs: list[RunSpec] = []
    required = {"task", "budget", "method", "path"}
    for index, item in enumerate(items):
        context = f"manifest.runs[{index}]"
        require(isinstance(item, dict), f"{context} must be an object")
        require(set(item) == required, f"{context} keys must be {sorted(required)}")
        task = str(item["task"]).strip().lower()
        method = str(item["method"]).strip().lower()
        require(task in TASKS, f"{context} has unsupported task {task!r}")
        require(method in METHODS, f"{context} has unsupported method {method!r}")
        budget = normalize_budget(item["budget"])
        path_text = str(item["path"]).strip()
        require(bool(path_text), f"{context}.path must be non-empty")
        run_path = Path(path_text).expanduser()
        if not run_path.is_absolute():
            run_path = path.parent / run_path
        specs.append(
            RunSpec(
                task=task,
                budget=budget,
                method=method,
                path=run_path.resolve(),
            )
        )

    identities = [(spec.task, spec.budget, spec.method) for spec in specs]
    require(len(identities) == len(set(identities)), "duplicate task/budget/method cell")
    expected = {
        (task, budget, method)
        for task in TASKS
        for budget in BUDGETS
        for method in METHODS
    }
    actual = set(identities)
    require(
        actual == expected,
        "manifest does not cover the complete grid; "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )

    require(len(items) == 80, f"manifest must contain 80 actual runs, got {len(items)}")
    for task in TASKS:
        for method in METHODS:
            paths = {
                spec.path for spec in specs if spec.task == task and spec.method == method
            }
            require(
                len(paths) == len(BUDGETS),
                f"{task}/{method} must use four distinct measured output paths",
            )
    require(
        len({spec.path for spec in specs}) == 80,
        "all 80 task/budget/method executions must have distinct output paths",
    )
    specs.sort(
        key=lambda spec: (
            TASKS.index(spec.task),
            BUDGETS.index(spec.budget),
            METHODS.index(spec.method),
        )
    )
    return specs, {
        "path": str(path),
        "purpose": payload.get("purpose"),
        "manifest_entries": len(items),
        "expanded_cells": len(specs),
    }


def read_records(path: Path, *, task: str, context: str) -> dict[str, dict[str, Any]]:
    require(path.is_file(), f"missing {context}: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        require(isinstance(row, dict), f"{path}:{line_number} must be an object")
        uid = row.get("uid")
        require(isinstance(uid, str) and uid, f"{path}:{line_number} has invalid uid")
        require(uid not in records, f"duplicate UID {uid!r} in {path}")
        require(row.get("task") == task, f"{path}:{line_number}.task must be {task!r}")
        require(uid.startswith(f"{task}-"), f"{path}:{line_number} UID does not match task")
        require(type(row.get("correct")) is bool, f"{path}:{line_number}.correct must be bool")
        logits = finite_number(
            row.get("logits_ready_ms"), context=f"{path}:{line_number}.logits_ready_ms"
        )
        response = finite_number(
            row.get("response_ready_ms"), context=f"{path}:{line_number}.response_ready_ms"
        )
        require(logits > 0.0, f"{path}:{line_number}.logits_ready_ms must be positive")
        require(
            response + 1e-9 >= logits,
            f"{path}:{line_number} response-ready precedes logits-ready",
        )
        require("ttft_ms" in row, f"{path}:{line_number} is missing ttft_ms")
        require(
            same_number(row["ttft_ms"], logits),
            f"{path}:{line_number} violates ttft_ms == logits_ready_ms",
        )
        records[uid] = row
    require(bool(records), f"no scored records in {path}")
    return records


def record_metrics(records: Mapping[str, Mapping[str, Any]]) -> dict[str, float | int]:
    rows = list(records.values())
    logits = [float(row["logits_ready_ms"]) for row in rows]
    response = [float(row["response_ready_ms"]) for row in rows]
    return {
        "samples": len(rows),
        "accuracy": statistics.fmean(bool(row["correct"]) for row in rows),
        "response_ready_mean_ms": statistics.fmean(response),
        "response_ready_p95_ms": p95(response),
        "logits_ready_mean_ms": statistics.fmean(logits),
        "logits_ready_p95_ms": p95(logits),
    }


def validate_summary(
    summary: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    spec: RunSpec,
) -> None:
    tasks = summary.get("tasks")
    require(
        isinstance(tasks, dict) and set(tasks) == {spec.task},
        f"{spec.label} summary must contain only task {spec.task!r}",
    )
    expected = record_metrics(records)
    summary_names = {
        "samples": "samples",
        "accuracy": "accuracy",
        "response_ready_mean_ms": "mean_response_ready_ms",
        "response_ready_p95_ms": "p95_response_ready_ms",
        "logits_ready_mean_ms": "mean_logits_ready_ms",
        "logits_ready_p95_ms": "p95_logits_ready_ms",
    }
    scopes = (("task", tasks[spec.task]), ("overall", summary.get("overall")))
    for scope_name, scope in scopes:
        require(
            isinstance(scope, dict),
            f"{spec.label} {scope_name} summary must be an object",
        )
        for metric, summary_key in summary_names.items():
            require(
                summary_key in scope,
                f"{spec.label} {scope_name} is missing {summary_key}",
            )
            require(
                same_number(scope[summary_key], expected[metric]),
                f"{spec.label} {scope_name}.{summary_key} disagrees with records",
            )
    runtime = summary.get("runtime")
    require(isinstance(runtime, dict), f"{spec.label} runtime must be an object")
    require(
        runtime.get("accuracy_scoring") == "label_continuation_loglikelihood",
        f"{spec.label} must use complete-label accuracy scoring",
    )
    require(runtime.get("generation_max_tokens") == 1, f"{spec.label} must be a one-token run")
    require(
        runtime.get("response_ready_metric_valid_for_first_token") is True,
        f"{spec.label} response-ready metric is not valid for first token",
    )
    require(
        runtime.get("response_ready_excludes_accuracy_scoring") is True,
        f"{spec.label} response-ready must exclude benchmark-only scoring",
    )
    if spec.method == "as_lru":
        require(
            same_number(runtime.get("keep_ratio"), 1.0),
            f"{spec.label} AS+LRU runtime.keep_ratio must be full K/V (1.0)",
        )
    else:
        require(
            same_number(runtime.get("keep_ratio"), int(spec.budget) / 100.0),
            f"{spec.label} runtime.keep_ratio does not match k{spec.budget}",
        )


def load_run(spec: RunSpec) -> LoadedRun:
    summary = read_json(spec.path / "summary.json", context=f"{spec.label} summary")
    records = read_records(
        spec.path / "scored_records.jsonl",
        task=spec.task,
        context=f"{spec.label} records",
    )
    validate_summary(summary, records, spec=spec)
    return LoadedRun(spec=spec, records=records)


def validate_budget_semantics(runs: Sequence[LoadedRun]) -> dict[str, Any]:
    """Validate actual K/V residency metadata, independent of shape-only plans."""

    prefix_tokens_by_uid: dict[tuple[str, str], int] = {}
    as_lru_rows = 0
    for run in runs:
        if run.spec.method != "as_lru":
            continue
        for uid, row in run.records.items():
            context = f"{run.spec.label}[{uid}]"
            effective = finite_number(
                row.get("effective_mean_keep_ratio"),
                context=f"{context}.effective_mean_keep_ratio",
            )
            require(
                same_number(effective, 1.0),
                f"{context} AS+LRU must retain full K/V",
            )
            layer_counts = row.get("selected_tokens_by_layer")
            require(
                isinstance(layer_counts, list) and layer_counts,
                f"{context}.selected_tokens_by_layer must be non-empty",
            )
            require(
                all(type(count) is int and count > 0 for count in layer_counts),
                f"{context} has invalid selected token counts",
            )
            require(
                len(set(layer_counts)) == 1,
                f"{context} is not full retention in every layer",
            )
            key = (run.spec.task, uid)
            if key in prefix_tokens_by_uid:
                require(
                    prefix_tokens_by_uid[key] == layer_counts[0],
                    f"{context} prefix length differs across independent AS+LRU runs",
                )
            else:
                prefix_tokens_by_uid[key] = layer_counts[0]
            as_lru_rows += 1

    h2o_rows = 0
    maximum_ceil_error = 0.0
    for run in runs:
        if run.spec.method != "as_h2o_lru":
            continue
        requested = int(run.spec.budget) / 100.0
        for uid, row in run.records.items():
            context = f"{run.spec.label}[{uid}]"
            prefix_tokens = prefix_tokens_by_uid[(run.spec.task, uid)]
            key_ratio = finite_number(
                row.get("as_h2o_full_key_ratio"),
                context=f"{context}.as_h2o_full_key_ratio",
            )
            value_ratio = finite_number(
                row.get("as_h2o_value_keep_ratio"),
                context=f"{context}.as_h2o_value_keep_ratio",
            )
            total_ratio = finite_number(
                row.get("as_h2o_total_logical_payload_ratio"),
                context=f"{context}.as_h2o_total_logical_payload_ratio",
            )
            require(
                same_number(key_ratio, 1.0),
                f"{context} AS+H2O+LRU must retain all keys",
            )
            expected_value = math.ceil(prefix_tokens * requested) / prefix_tokens
            require(
                same_number(value_ratio, expected_value, tolerance=1e-10),
                f"{context} value retention {value_ratio} is not ceil(prefix*k)/prefix "
                f"({expected_value})",
            )
            ceil_error = value_ratio - requested
            require(
                -1e-12 <= ceil_error <= (1.0 / prefix_tokens) + 1e-12,
                f"{context} value-retention rounding exceeds one prefix token",
            )
            require(
                same_number(total_ratio, (1.0 + value_ratio) / 2.0),
                f"{context} logical payload must be (full K + retained V) / 2",
            )
            h2o_rows += 1
            maximum_ceil_error = max(maximum_ceil_error, ceil_error)
    return {
        "as_lru": (
            "four independently timed budget-label runs per task; "
            "effective_mean_keep_ratio=1 and every layer retains prefix_tokens"
        ),
        "as_h2o_lru": (
            "LRU naming follows the paper title/figure legend; key=1, "
            "value=ceil(prefix_tokens*k)/prefix_tokens, total=(1+value)/2"
        ),
        "validated_as_lru_rows": as_lru_rows,
        "validated_h2o_rows": h2o_rows,
        "maximum_h2o_value_ceil_error": maximum_ceil_error,
    }


def macro_metrics(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(cells), "cannot macro-average an empty cell list")
    return {
        "cells": len(cells),
        **{
            metric: statistics.fmean(float(cell[metric]) for cell in cells)
            for metric in SUMMARY_METRICS
        },
    }


def analyze_manifest(path: Path) -> dict[str, Any]:
    specs, input_info = load_manifest(path)
    cache: dict[tuple[str, str, Path], LoadedRun] = {}
    loaded: list[LoadedRun] = []
    for spec in specs:
        cache_key = (spec.task, spec.method, spec.path)
        if cache_key not in cache:
            cache[cache_key] = load_run(spec)
        loaded.append(LoadedRun(spec=spec, records=cache[cache_key].records))

    for task in TASKS:
        task_runs = [run for run in loaded if run.spec.task == task]
        reference = set(task_runs[0].records)
        for run in task_runs[1:]:
            require(
                set(run.records) == reference,
                f"UID set mismatch for {task}: "
                f"{task_runs[0].spec.label} versus {run.spec.label}",
            )

    budget_semantics_validation = validate_budget_semantics(loaded)

    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in loaded:
        cell = record_metrics(run.records)
        cell.update(
            {
                "source_path": str(run.spec.path),
                "requested_budget_label": run.spec.budget,
                "independent_measurement": True,
                "effective_kv_retention_ratio": (
                    1.0 if run.spec.method == "as_lru" else None
                ),
                "budget_semantics": (
                    "full_kv"
                    if run.spec.method == "as_lru"
                    else "full_keys_value_retention"
                    if run.spec.method == "as_h2o_lru"
                    else "kv_retention"
                ),
            }
        )
        cells[(run.spec.task, run.spec.budget, run.spec.method)] = cell

    tasks: dict[str, Any] = {}
    for task in TASKS:
        task_uids = len(next(run.records for run in loaded if run.spec.task == task))
        tasks[task] = {
            "uids": task_uids,
            "budgets": {
                budget: {
                    "methods": {
                        method: cells[(task, budget, method)] for method in METHODS
                    }
                }
                for budget in BUDGETS
            },
        }

    by_method_budget: dict[str, Any] = {}
    overall_by_method: dict[str, Any] = {}
    for method in METHODS:
        by_method_budget[method] = {
            budget: macro_metrics([cells[(task, budget, method)] for task in TASKS])
            for budget in BUDGETS
        }
        overall_by_method[method] = macro_metrics(
            [cells[(task, budget, method)] for task in TASKS for budget in BUDGETS]
        )

    return {
        "schema_version": 1,
        "scope": "re-prefill; complete 4-task x 4-budget x 5-method grid",
        "input": input_info,
        "metric_definitions": {
            "accuracy": "mean exact correctness within each task/budget/method cell",
            "primary_latency": "response_ready_ms (benchmark-only accuracy scoring excluded)",
            "phase_latency": "logits_ready_ms (first-token logits ready)",
            "p95": "nearest-rank P95 within each task/budget/method cell",
            "macro": "unweighted arithmetic mean of cell metrics; requests are not pooled across tasks",
            "as_lru_budget": (
                "each task-budget cell is independently timed with full prefix K/V; "
                "the requested budget is only the comparison-block label"
            ),
            "as_h2o_lru_budget": (
                "k is value retention with full K; total logical (K+V) ratio is "
                "(1+k_actual)/2; LRU naming follows the paper title/figure legend"
            ),
        },
        "as_lru_independent_runs": {
            task: {
                budget: str(
                    next(
                        spec.path
                        for spec in specs
                        if spec.task == task
                        and spec.budget == budget
                        and spec.method == "as_lru"
                    )
                )
                for budget in BUDGETS
            }
            for task in TASKS
        },
        "budget_semantics_validation": budget_semantics_validation,
        "tasks": tasks,
        "by_method_budget": by_method_budget,
        "overall_by_method": overall_by_method,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Five-method Re-prefill Grid",
        "",
        "Primary latency is **response-ready**; logits-ready is retained as a phase metric. Accuracy uses complete-label continuation log-likelihood. Tasks are not sample-pooled.",
        "",
        "AS+LRU uses full prefix K/V in every cell. The 5%, 10%, 25%, and 50% labels identify comparison blocks only; each cell is timed independently and none claims sparse AS+LRU retention.",
        "",
        "For AS+H2O+LRU, `k` is the **value-retention ratio** selected by H2O while K remains full, so total logical `(K+V)/2 = (1+k_actual)/2`. The LRU name follows the baseline title and figure legend in the ContiguousKV paper.",
        "",
        "## Detailed results",
        "",
        "| Dataset | Budget | Method | Samples | Accuracy | Response-ready mean / P95 (ms) | Logits-ready phase mean / P95 (ms) | Run source |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for task in TASKS:
        for budget in BUDGETS:
            methods = result["tasks"][task]["budgets"][budget]["methods"]
            for method in METHODS:
                metrics = methods[method]
                source = (
                    f"independent k{budget} timing; full K/V"
                    if method == "as_lru"
                    else f"independent k{budget} timing"
                )
                lines.append(
                    f"| {task} | {int(budget)}% | {DISPLAY_METHODS[method]} | "
                    f"{metrics['samples']} | {100.0 * metrics['accuracy']:.2f}% | "
                    f"{metrics['response_ready_mean_ms']:.3f} / "
                    f"{metrics['response_ready_p95_ms']:.3f} | "
                    f"{metrics['logits_ready_mean_ms']:.3f} / "
                    f"{metrics['logits_ready_p95_ms']:.3f} | {source} |"
                )

    lines += [
        "",
        "## Macro average by budget",
        "",
        "Each row is the unweighted macro average of four dataset cells.",
        "",
        "| Budget | Method | Cells | Accuracy | Response-ready mean / cell-P95 (ms) | Logits-ready phase mean / cell-P95 (ms) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        for method in METHODS:
            metrics = result["by_method_budget"][method][budget]
            lines.append(
                f"| {int(budget)}% | {DISPLAY_METHODS[method]} | {metrics['cells']} | "
                f"{100.0 * metrics['accuracy']:.2f}% | "
                f"{metrics['response_ready_mean_ms']:.3f} / "
                f"{metrics['response_ready_p95_ms']:.3f} | "
                f"{metrics['logits_ready_mean_ms']:.3f} / "
                f"{metrics['logits_ready_p95_ms']:.3f} |"
            )

    lines += [
        "",
        "## Overall macro average",
        "",
        "Each row is the unweighted macro average of 16 dataset-budget cells.",
        "",
        "| Method | Cells | Accuracy | Response-ready mean / cell-P95 (ms) | Logits-ready phase mean / cell-P95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = result["overall_by_method"][method]
        lines.append(
            f"| {DISPLAY_METHODS[method]} | {metrics['cells']} | "
            f"{100.0 * metrics['accuracy']:.2f}% | "
            f"{metrics['response_ready_mean_ms']:.3f} / "
            f"{metrics['response_ready_p95_ms']:.3f} | "
            f"{metrics['logits_ready_mean_ms']:.3f} / "
            f"{metrics['logits_ready_p95_ms']:.3f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        return output, output.with_suffix(".md")
    if output.suffix.lower() == ".md":
        return output.with_suffix(".json"), output
    return output.with_suffix(".json"), output.with_suffix(".md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output stem, .json, or .md; both JSON and Markdown are written",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_manifest(args.manifest)
        json_path, markdown_path = output_paths(args.output.resolve())
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_markdown(result), encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
