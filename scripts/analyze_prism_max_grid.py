#!/usr/bin/env python3
"""Analyze a matched, task-local Prism-Max re-prefill grid.

The analyzer deliberately reports only accuracy, logits-ready latency, the three
additional requested views (SSD traffic, prefetch stall, accuracy/latency
Pareto), and keep/selected-byte fairness context.  Each UID is averaged over
repeats before any aggregate or paired bootstrap is computed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASKS = ("sst2", "subj", "trec", "rte")
BUDGETS = ("005", "010", "025", "050")
METHODS = ("contigkv", "impress", "promixed")
DISPLAY_METHODS = {
    "contigkv": "ContiguousKV",
    "impress": "IMPRESS",
    "promixed": "ProMixed",
}
IMPRESS_REORDER_SHA256 = (
    "36c5e1ec62187f8916e04e7758e79c9a28cfe1c75cb8999739e6be15287542cf"
)
METHOD_CONTRACTS = {
    "contigkv": {
        "method": "contigkv",
        "promixed_gqa_selection": False,
        "probe_query_heads": [0, 1, 2],
        "selector_kv_head_ids": [0, 1, 2, 3],
        "impress_reorder_enabled": False,
        "impress_reorder_sha256": None,
        "impress_async_inter_layer_prefetch": False,
        "chunk_size": 16,
    },
    "impress": {
        "method": "impress",
        "promixed_gqa_selection": False,
        "probe_query_heads": [0, 1, 2],
        "selector_kv_head_ids": [0],
        "impress_reorder_enabled": True,
        "impress_reorder_sha256": IMPRESS_REORDER_SHA256,
        "impress_async_inter_layer_prefetch": False,
        "chunk_size": 64,
    },
    "promixed": {
        "method": "impress",
        "promixed_gqa_selection": True,
        "probe_query_heads": [0, 7, 14, 21],
        "selector_kv_head_ids": [0, 1, 2, 3],
        "impress_reorder_enabled": False,
        "impress_reorder_sha256": None,
        "impress_async_inter_layer_prefetch": True,
        "chunk_size": 16,
    },
}
MIB = 1024.0 * 1024.0
ROW_METRICS = (
    "logits_ready_ms",
    "prefetch_wait_ms",
    "critical_ssd_read_bytes",
    "selector_disk_source_bytes",
    "total_ssd_read_bytes",
    "effective_mean_keep_ratio",
    "selected_kv_bytes",
)
READY_ORDER = (
    "logits_ready_ms",
    "first_token_ready_ms",
    "latency_ms",
    "response_ready_ms",
    "evaluation_ready_ms",
)
CORE_RUNTIME_KEYS = (
    "backend",
    "accuracy_scoring",
    "generation_max_tokens",
    "online_selection",
    "cache_type",
    "chunk_size",
    "gpu_cache_mb",
    "cpu_cache_mb",
    "model_compute_dtype",
    "pcache_storage_dtype",
    "defer_cache_score_updates",
    "warmup_passes",
    "warmup_samples_per_task",
)
SHARED_RUNTIME_KEYS = (
    "backend",
    "accuracy_scoring",
    "generation_max_tokens",
    "online_selection",
    "cache_type",
    "gpu_cache_mb",
    "cpu_cache_mb",
    "model_compute_dtype",
    "pcache_storage_dtype",
    "defer_cache_score_updates",
    "warmup_passes",
    "warmup_samples_per_task",
    "period_size",
    "subperiod_size",
    "prefetch_time_budget",
    "cache_update_in_ttft",
    "response_ready_metric_valid_for_first_token",
    "response_ready_excludes_accuracy_scoring",
    "evaluation_ready_includes_accuracy_scoring",
    "selector_index_dir",
    "selector_index_bits",
    "selector_index_group_size",
    "selector_index_manifest_sha256",
    "selector_index_preloaded_bytes",
    "similarity_alpha",
)


@dataclass(frozen=True)
class RunSpec:
    task: str
    budget: str
    method: str
    repeat: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.task}/k{self.budget}/{self.method}/{self.repeat}"


@dataclass
class LoadedRun:
    spec: RunSpec
    summary: dict[str, Any]
    records: dict[str, dict[str, Any]]
    raw_count: int
    excluded_uids: list[str]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    require(math.isfinite(number), f"{context} must be finite, got {number!r}")
    return number


def same_number(left: Any, right: Any, *, tolerance: float = 1e-7) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "cannot compute P95 of an empty sequence")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "cannot compute percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    require(samples > 0, "bootstrap samples must be positive")
    source = [float(value) for value in values]
    require(bool(source), "cannot bootstrap an empty sequence")
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choice(source) for _ in source)
        for _ in range(samples)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def normalize_budget(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("k"):
        text = text[1:]
    require(text.isdigit(), f"budget must be one of {BUDGETS}, got {value!r}")
    normalized = f"{int(text):03d}"
    require(normalized in BUDGETS, f"unsupported budget: {value!r}")
    return normalized


def load_json(path: Path, *, context: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {context}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {context} {path}: {error}") from error
    require(isinstance(value, dict), f"{context} must be a JSON object: {path}")
    return value


def parse_run_item(item: Mapping[str, Any], *, base: Path, context: str) -> RunSpec:
    required = {"task", "budget", "method", "repeat", "path"}
    require(set(item) == required, f"{context} keys must be exactly {sorted(required)}")
    task = str(item["task"]).strip().lower()
    method = str(item["method"]).strip().lower()
    repeat = str(item["repeat"]).strip()
    require(task in TASKS, f"{context} has unsupported task {task!r}")
    require(method in METHODS, f"{context} has unsupported method {method!r}")
    require(bool(repeat), f"{context} repeat must be non-empty")
    path_text = str(item["path"]).strip()
    require(bool(path_text), f"{context} path must be non-empty")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    return RunSpec(task, normalize_budget(item["budget"]), method, repeat, path.resolve())


def load_specs(
    *, manifest: Path | None, explicit_runs: Sequence[Sequence[str]] | None
) -> tuple[list[RunSpec], dict[str, Any]]:
    require(bool(manifest) ^ bool(explicit_runs), "choose exactly one of --manifest or --run")
    if manifest is not None:
        manifest = manifest.resolve()
        payload = load_json(manifest, context="schedule manifest")
        allowed = {"schema_version", "purpose", "runs"}
        require(set(payload) <= allowed, f"schedule manifest has unknown keys: {sorted(set(payload) - allowed)}")
        require(payload.get("schema_version") == 1, "schedule manifest schema_version must be 1")
        items = payload.get("runs")
        require(isinstance(items, list) and bool(items), "schedule manifest.runs must be a non-empty list")
        specs = []
        for index, item in enumerate(items):
            require(isinstance(item, dict), f"schedule manifest.runs[{index}] must be an object")
            specs.append(parse_run_item(item, base=manifest.parent, context=f"schedule manifest.runs[{index}]"))
        source = {"kind": "manifest", "path": str(manifest), "purpose": payload.get("purpose")}
    else:
        specs = []
        for index, fields in enumerate(explicit_runs or []):
            item = dict(zip(("task", "budget", "method", "repeat", "path"), fields))
            specs.append(parse_run_item(item, base=Path.cwd(), context=f"--run[{index}]"))
        source = {"kind": "explicit_runs"}
    identities = [(spec.task, spec.budget, spec.method, spec.repeat) for spec in specs]
    require(len(identities) == len(set(identities)), "duplicate task/budget/method/repeat run identity")
    return specs, source


def load_exclusions(path: Path | None) -> tuple[dict[str, set[str]], dict[str, Any]]:
    if path is None:
        return {task: set() for task in TASKS}, {
            "manifest_path": None,
            "purpose": None,
            "requested_by_task": {task: [] for task in TASKS},
        }
    path = path.resolve()
    payload = load_json(path, context="UID exclusion manifest")
    require(payload.get("schema_version") == 1, "UID exclusion manifest schema_version must be 1")
    raw = payload.get("exclude_uids_by_task")
    require(isinstance(raw, dict), "UID exclusion manifest.exclude_uids_by_task must be an object")
    require(set(raw) <= set(TASKS), f"UID exclusion manifest has unsupported tasks: {sorted(set(raw) - set(TASKS))}")
    exclusions = {task: set() for task in TASKS}
    for task, values in raw.items():
        require(isinstance(values, list), f"exclude_uids_by_task.{task} must be a list")
        normalized = [str(value) for value in values]
        require(all(value and value.startswith(f"{task}-") for value in normalized), f"exclude_uids_by_task.{task} contains an invalid UID")
        require(len(normalized) == len(set(normalized)), f"exclude_uids_by_task.{task} contains duplicates")
        exclusions[task].update(normalized)
    return exclusions, {
        "manifest_path": str(path),
        "purpose": payload.get("purpose"),
        "requested_by_task": {task: sorted(exclusions[task]) for task in TASKS},
    }


def read_records(path: Path, *, context: str) -> dict[str, dict[str, Any]]:
    require(path.is_file(), f"missing {context}: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        require(isinstance(row, dict), f"{path}:{line_number} is not an object")
        uid = row.get("uid")
        require(isinstance(uid, str) and bool(uid), f"{path}:{line_number} has invalid uid")
        require(uid not in records, f"duplicate UID {uid!r} in {path}")
        records[uid] = row
    require(bool(records), f"no scored records in {path}")
    return records


def validate_row(row: Mapping[str, Any], *, task: str, context: str) -> None:
    require(row.get("task") == task, f"{context}.task is not {task!r}")
    require(type(row.get("correct")) is bool, f"{context}.correct must be boolean")
    for metric in READY_ORDER:
        require(metric in row, f"{context} is missing {metric}")
        require(finite_number(row[metric], context=f"{context}.{metric}") >= 0.0, f"{context}.{metric} must be non-negative")
    require("ttft_ms" in row, f"{context} is missing ttft_ms")
    require(same_number(row["ttft_ms"], row["logits_ready_ms"]), f"{context} violates ttft_ms == logits_ready_ms")
    require("accuracy_scores_ready_ms" in row, f"{context} is missing accuracy_scores_ready_ms")
    require(same_number(row["accuracy_scores_ready_ms"], row["evaluation_ready_ms"]), f"{context} violates accuracy_scores_ready_ms == evaluation_ready_ms")
    ready = [float(row[name]) for name in READY_ORDER]
    require(all(left <= right + 1e-9 for left, right in zip(ready, ready[1:])), f"{context} has a non-monotonic ready-time boundary")
    for metric in ROW_METRICS:
        require(metric in row, f"{context} is missing {metric}")
        require(finite_number(row[metric], context=f"{context}.{metric}") >= 0.0, f"{context}.{metric} must be non-negative")
    for metric in ("critical_ssd_read_bytes", "selector_disk_source_bytes", "total_ssd_read_bytes", "selected_kv_bytes"):
        require(float(row[metric]).is_integer(), f"{context}.{metric} must be integer-valued")
    expected_total = float(row["critical_ssd_read_bytes"]) + float(row["selector_disk_source_bytes"])
    require(same_number(row["total_ssd_read_bytes"], expected_total), f"{context} violates total SSD bytes == critical + selector")
    require(float(row["selected_kv_bytes"]) > 0.0, f"{context}.selected_kv_bytes must be positive")
    keep = float(row["effective_mean_keep_ratio"])
    require(0.0 < keep <= 1.0, f"{context}.effective_mean_keep_ratio must be in (0, 1]")
    require(float(row["logits_ready_ms"]) > 0.0, f"{context}.logits_ready_ms must be positive")
    require(float(row["prefetch_wait_ms"]) <= float(row["logits_ready_ms"]) + 1e-9, f"{context}.prefetch_wait_ms exceeds logits_ready_ms")


def validate_summary(summary: Mapping[str, Any], raw: Mapping[str, Mapping[str, Any]], *, spec: RunSpec) -> None:
    context = spec.label
    tasks = summary.get("tasks")
    require(isinstance(tasks, dict) and set(tasks) == {spec.task}, f"{context} is not an independent {spec.task} run")
    scopes = [(f"{context}.tasks.{spec.task}", tasks[spec.task]), (f"{context}.overall", summary.get("overall"))]
    expected_accuracy = statistics.fmean(bool(row["correct"]) for row in raw.values())
    logits = [float(row["logits_ready_ms"]) for row in raw.values()]
    for name, scope in scopes:
        require(isinstance(scope, dict), f"{name} must be an object")
        require(scope.get("samples") == len(raw), f"{name}.samples disagrees with records")
        require(same_number(scope.get("accuracy"), expected_accuracy), f"{name}.accuracy disagrees with records")
        expected = {
            "mean_logits_ready_ms": statistics.fmean(logits),
            "p95_logits_ready_ms": p95(logits),
            "mean_ttft_ms": statistics.fmean(logits),
            "p95_ttft_ms": p95(logits),
        }
        for metric, value in expected.items():
            require(same_number(scope.get(metric), value), f"{name}.{metric} disagrees with records")


def validate_method_contract(runtime: Mapping[str, Any], *, spec: RunSpec) -> None:
    for field, expected in METHOD_CONTRACTS[spec.method].items():
        require(
            runtime.get(field) == expected,
            f"{spec.label}.runtime.{field} must be {expected!r}, got {runtime.get(field)!r}",
        )


def load_run(spec: RunSpec, exclusions: set[str]) -> LoadedRun:
    summary = load_json(spec.path / "summary.json", context=f"{spec.label} summary")
    raw = read_records(spec.path / "scored_records.jsonl", context=f"{spec.label} records")
    for uid, row in raw.items():
        validate_row(row, task=spec.task, context=f"{spec.label}[{uid}]")
    validate_summary(summary, raw, spec=spec)
    runtime = summary.get("runtime")
    require(isinstance(runtime, dict), f"{spec.label}.runtime must be an object")
    for key in CORE_RUNTIME_KEYS:
        require(key in runtime, f"{spec.label}.runtime is missing {key}")
    validate_method_contract(runtime, spec=spec)
    require(runtime.get("accuracy_scoring") == "label_continuation_loglikelihood", f"{spec.label} does not use complete-label scoring")
    require(runtime.get("generation_max_tokens") == 1, f"{spec.label} is not a one-token re-prefill run")
    require(runtime.get("online_selection") is True, f"{spec.label} is not online selection")
    require(runtime.get("cache_type") == "CKLFU", f"{spec.label} cache_type must be CKLFU")
    require(runtime.get("defer_cache_score_updates") is False, f"{spec.label} must include cache maintenance in logits-ready")
    require(same_number(runtime.get("keep_ratio"), int(spec.budget) / 100.0), f"{spec.label} keep_ratio does not match k{spec.budget}")
    require(summary.get("model_path") is not None, f"{spec.label} is missing model_path")
    excluded = sorted(set(raw) & exclusions)
    records = {uid: row for uid, row in raw.items() if uid not in exclusions}
    require(bool(records), f"{spec.label} has no records after UID exclusions")
    return LoadedRun(spec, summary, records, len(raw), excluded)


def validate_impress_cells_are_fp16(runs: Sequence[LoadedRun]) -> None:
    cells = {(run.spec.task, run.spec.budget) for run in runs}
    index_keys = (
        "selector_index_dir",
        "selector_index_bits",
        "selector_index_group_size",
        "selector_index_manifest_sha256",
    )
    for cell in cells:
        cell_runs = [
            run
            for run in runs
            if (run.spec.task, run.spec.budget) == cell
        ]
        if not any(run.spec.method == "impress" for run in cell_runs):
            continue
        for run in cell_runs:
            runtime = run.summary["runtime"]
            for key in index_keys:
                require(
                    runtime.get(key) is None,
                    f"{run.spec.label}.runtime.{key} must be None in a cell containing canonical IMPRESS",
                )
            preloaded_bytes = runtime.get("selector_index_preloaded_bytes")
            no_preloaded_index = preloaded_bytes is None or (
                type(preloaded_bytes) in (int, float)
                and preloaded_bytes == 0
            )
            require(
                no_preloaded_index,
                f"{run.spec.label}.runtime.selector_index_preloaded_bytes must be None or numeric zero in a cell containing canonical IMPRESS",
            )


def frozen(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_shared_protocol(runs: Sequence[LoadedRun]) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for task in sorted({run.spec.task for run in runs}):
        task_runs = [run for run in runs if run.spec.task == task]
        first = task_runs[0]
        signature = {
            "model_path": first.summary["model_path"],
            "measurement": first.summary.get("measurement"),
            **{key: first.summary["runtime"].get(key) for key in SHARED_RUNTIME_KEYS},
        }
        for run in task_runs[1:]:
            candidate = {
                "model_path": run.summary["model_path"],
                "measurement": run.summary.get("measurement"),
                **{key: run.summary["runtime"].get(key) for key in SHARED_RUNTIME_KEYS},
            }
            for key, expected in signature.items():
                require(frozen(candidate[key]) == frozen(expected), f"shared protocol mismatch for task {task}, {key}: {first.spec.label}={expected!r}, {run.spec.label}={candidate[key]!r}")
        signatures[task] = signature
    return signatures


def group_runs(runs: Sequence[LoadedRun]) -> dict[tuple[str, str, str], list[LoadedRun]]:
    grouped: dict[tuple[str, str, str], list[LoadedRun]] = {}
    for run in runs:
        grouped.setdefault((run.spec.task, run.spec.budget, run.spec.method), []).append(run)
    for key, family in grouped.items():
        labels = [run.spec.repeat for run in family]
        require(len(labels) == len(set(labels)), f"duplicate repeat labels for {key}")
        family.sort(key=lambda run: run.spec.repeat)
        expected = set(family[0].records)
        for run in family[1:]:
            require(set(run.records) == expected, f"repeats have different post-exclusion UIDs for {key}: {family[0].spec.repeat}={len(expected)}, {run.spec.repeat}={len(run.records)}")
        for uid in expected:
            values = {run.records[uid]["correct"] for run in family}
            require(len(values) == 1, f"{key} UID {uid!r} has nondeterministic correctness across repeats")
    cells = sorted({(task, budget) for task, budget, _method in grouped})
    for task, budget in cells:
        families = {
            method: grouped[(task, budget, method)]
            for method in METHODS
            if (task, budget, method) in grouped
        }
        first_method = next(iter(families))
        expected_labels = {run.spec.repeat for run in families[first_method]}
        for method, family in families.items():
            labels = {run.spec.repeat for run in family}
            require(
                labels == expected_labels,
                f"repeat label sets differ for {task}/k{budget}: {first_method}={sorted(expected_labels)}, {method}={sorted(labels)}",
            )
    return grouped


def aggregate_uids(runs: Sequence[LoadedRun]) -> dict[str, dict[str, float | bool]]:
    uids = sorted(runs[0].records)
    aggregate: dict[str, dict[str, float | bool]] = {}
    for uid in uids:
        row: dict[str, float | bool] = {
            metric: statistics.fmean(float(run.records[uid][metric]) for run in runs)
            for metric in ROW_METRICS
        }
        row["correct"] = bool(runs[0].records[uid]["correct"])
        row["prefetch_stall_ratio"] = statistics.fmean(
            float(run.records[uid]["prefetch_wait_ms"])
            / float(run.records[uid]["logits_ready_ms"])
            for run in runs
        )
        aggregate[uid] = row
    return aggregate


def absolute_metrics(aggregate: Mapping[str, Mapping[str, float | bool]], *, repeats: int) -> dict[str, Any]:
    rows = list(aggregate.values())
    values = lambda metric: [float(row[metric]) for row in rows]
    logits = values("logits_ready_ms")
    return {
        "uids": len(rows),
        "repeats": repeats,
        "accuracy": statistics.fmean(bool(row["correct"]) for row in rows),
        "logits_ready_mean_ms": statistics.fmean(logits),
        "logits_ready_p95_ms": p95(logits),
        "ssd_mib_per_request": {
            "critical": statistics.fmean(values("critical_ssd_read_bytes")) / MIB,
            "selector": statistics.fmean(values("selector_disk_source_bytes")) / MIB,
            "total": statistics.fmean(values("total_ssd_read_bytes")) / MIB,
        },
        "prefetch_stall_ratio": statistics.fmean(values("prefetch_stall_ratio")),
        "fairness_context": {
            "actual_keep_ratio": statistics.fmean(values("effective_mean_keep_ratio")),
            "selected_kv_mib": statistics.fmean(values("selected_kv_bytes")) / MIB,
        },
    }


def paired_comparison(
    candidate: Mapping[str, Mapping[str, float | bool]] | None,
    baseline: Mapping[str, Mapping[str, float | bool]] | None,
    *, baseline_method: str, bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    label = f"promixed_vs_{baseline_method}"
    if candidate is None or baseline is None:
        return {"comparison": label, "available": False, "reason": "missing method in this task/budget cell"}
    candidate_uids = set(candidate)
    baseline_uids = set(baseline)
    if candidate_uids != baseline_uids:
        return {
            "comparison": label,
            "available": False,
            "reason": "post-exclusion UID sets differ; no intersection pooling was performed",
            "candidate_uids": len(candidate_uids),
            "baseline_uids": len(baseline_uids),
            "candidate_only": sorted(candidate_uids - baseline_uids),
            "baseline_only": sorted(baseline_uids - candidate_uids),
        }
    uids = sorted(candidate_uids)
    definitions = {
        "accuracy_delta_pp": (lambda row: 100.0 * float(bool(row["correct"]))),
        "logits_ready_delta_ms": (lambda row: float(row["logits_ready_ms"])),
        "ssd_delta_mib_per_request": (lambda row: float(row["total_ssd_read_bytes"]) / MIB),
        "prefetch_stall_delta_pp": (lambda row: 100.0 * float(row["prefetch_stall_ratio"])),
    }
    result: dict[str, Any] = {
        "comparison": label,
        "available": True,
        "uids": len(uids),
        "aggregation": "per-UID repeat average before paired bootstrap",
        "direction": "ProMixed minus baseline",
    }
    for offset, (name, value) in enumerate(definitions.items()):
        deltas = [value(candidate[uid]) - value(baseline[uid]) for uid in uids]
        result[name] = statistics.fmean(deltas)
        result[f"{name}_paired_bootstrap_95ci"] = bootstrap_mean_ci(deltas, samples=bootstrap_samples, seed=seed + offset)
    return result


def mark_pareto(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    marked = []
    for point in points:
        dominated = any(
            other is not point
            and other["accuracy"] >= point["accuracy"]
            and other["logits_ready_mean_ms"] <= point["logits_ready_mean_ms"]
            and (
                other["accuracy"] > point["accuracy"]
                or other["logits_ready_mean_ms"] < point["logits_ready_mean_ms"]
            )
            for other in points
        )
        marked.append({**point, "pareto_optimal": not dominated})
    return marked


def analyze(
    specs: Sequence[RunSpec], exclusions: Mapping[str, set[str]], *,
    source: Mapping[str, Any], exclusion_info: Mapping[str, Any],
    bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    require(bool(specs), "no run paths supplied")
    runs = [load_run(spec, exclusions[spec.task]) for spec in specs]
    validate_impress_cells_are_fp16(runs)
    protocols = validate_shared_protocol(runs)
    grouped = group_runs(runs)
    aggregates = {key: aggregate_uids(family) for key, family in grouped.items()}
    tasks: dict[str, Any] = {}
    for task in TASKS:
        task_keys = [key for key in grouped if key[0] == task]
        if not task_keys:
            continue
        task_result: dict[str, Any] = {"budgets": {}, "pareto": {"by_method": {}, "joint": []}}
        for budget in BUDGETS:
            if not any(key[1] == budget for key in task_keys):
                continue
            cell: dict[str, Any] = {"methods": {}, "paired": {}}
            for method in METHODS:
                key = (task, budget, method)
                if key in grouped:
                    family = grouped[key]
                    cell["methods"][method] = {
                        **absolute_metrics(aggregates[key], repeats=len(family)),
                        "repeat_labels": [run.spec.repeat for run in family],
                        "run_paths": [str(run.spec.path) for run in family],
                    }
            candidate = aggregates.get((task, budget, "promixed"))
            for offset, baseline in enumerate(("contigkv", "impress")):
                cell["paired"][f"promixed_vs_{baseline}"] = paired_comparison(
                    candidate,
                    aggregates.get((task, budget, baseline)),
                    baseline_method=baseline,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 1000 * TASKS.index(task) + 100 * BUDGETS.index(budget) + 10 * offset,
                )
            task_result["budgets"][budget] = cell
        all_points: list[dict[str, Any]] = []
        for method in METHODS:
            points = []
            for budget, cell in task_result["budgets"].items():
                metrics = cell["methods"].get(method)
                if metrics:
                    point = {
                        "method": method,
                        "budget": budget,
                        "accuracy": metrics["accuracy"],
                        "logits_ready_mean_ms": metrics["logits_ready_mean_ms"],
                    }
                    points.append(point)
                    all_points.append(point)
            task_result["pareto"]["by_method"][method] = mark_pareto(points)
        task_result["pareto"]["joint"] = mark_pareto(all_points)
        tasks[task] = task_result
    observed = {task: sorted({uid for run in runs if run.spec.task == task for uid in run.excluded_uids}) for task in TASKS}
    exclusion_report = {
        **exclusion_info,
        "observed_by_task": observed,
        "requested_count_by_task": {task: len(exclusions[task]) for task in TASKS},
        "observed_count_by_task": {task: len(observed[task]) for task in TASKS},
        "per_run": {
            run.spec.label: {
                "raw_records": run.raw_count,
                "excluded_records": len(run.excluded_uids),
                "analysis_records": len(run.records),
                "excluded_uids": run.excluded_uids,
            }
            for run in runs
        },
        "application_order": "validated raw run, then excluded UIDs, then matched repeats/methods and computed metrics",
    }
    return {
        "schema_version": 1,
        "scope": "re-prefill only; task-local analysis with no cross-task pooling",
        "aggregation": "each UID is averaged over repeats before aggregate and paired metrics",
        "metric_definitions": {
            "ssd_mib_per_request": "(critical_ssd_read_bytes + selector_disk_source_bytes) / 2^20, request-time only",
            "prefetch_stall_ratio": "prefetch_wait_ms / logits_ready_ms within each run, then averaged over repeats per UID",
            "pareto": "non-dominated accuracy (higher) versus mean logits_ready_ms (lower) across budgets",
        },
        "input": dict(source),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "matched UID"},
        "exclusions": exclusion_report,
        "protocol_by_task": protocols,
        "tasks": tasks,
    }


def fmt_ci(value: Any, ci: Sequence[float]) -> str:
    return f"{float(value):+.3f} [{float(ci[0]):+.3f}, {float(ci[1]):+.3f}]"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Prism-Max Re-prefill Grid",
        "",
        "Per-UID repeat averages are used throughout. Tasks are never pooled.",
        "",
        f"Exclusion manifest: `{result['exclusions']['manifest_path']}`",
        "",
    ]
    for task, task_result in result["tasks"].items():
        lines += [f"## {task}", "", "| Budget | Method | UIDs×repeats | Accuracy | Logits mean / P95 (ms) | SSD total (critical + selector) MiB/req | Prefetch stall | Actual keep | Selected MiB |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for budget, cell in task_result["budgets"].items():
            for method in METHODS:
                metrics = cell["methods"].get(method)
                if not metrics:
                    continue
                ssd = metrics["ssd_mib_per_request"]
                fairness = metrics["fairness_context"]
                lines.append(
                    f"| k{budget} | {DISPLAY_METHODS[method]} | {metrics['uids']}×{metrics['repeats']} | {100.0 * metrics['accuracy']:.2f}% | {metrics['logits_ready_mean_ms']:.3f} / {metrics['logits_ready_p95_ms']:.3f} | {ssd['total']:.3f} ({ssd['critical']:.3f} + {ssd['selector']:.3f}) | {100.0 * metrics['prefetch_stall_ratio']:.2f}% | {100.0 * fairness['actual_keep_ratio']:.2f}% | {fairness['selected_kv_mib']:.3f} |"
                )
        lines += ["", "### Paired ProMixed deltas", "", "Positive accuracy is better; negative latency, SSD, and stall are better.", "", "| Budget | Baseline | UIDs | Δ accuracy pp [95% CI] | Δ logits ms [95% CI] | Δ SSD MiB/req [95% CI] | Δ stall pp [95% CI] |", "|---:|---|---:|---:|---:|---:|---:|"]
        for budget, cell in task_result["budgets"].items():
            for baseline in ("contigkv", "impress"):
                paired = cell["paired"][f"promixed_vs_{baseline}"]
                if paired["available"]:
                    lines.append(
                        f"| k{budget} | {DISPLAY_METHODS[baseline]} | {paired['uids']} | {fmt_ci(paired['accuracy_delta_pp'], paired['accuracy_delta_pp_paired_bootstrap_95ci'])} | {fmt_ci(paired['logits_ready_delta_ms'], paired['logits_ready_delta_ms_paired_bootstrap_95ci'])} | {fmt_ci(paired['ssd_delta_mib_per_request'], paired['ssd_delta_mib_per_request_paired_bootstrap_95ci'])} | {fmt_ci(paired['prefetch_stall_delta_pp'], paired['prefetch_stall_delta_pp_paired_bootstrap_95ci'])} |"
                    )
                else:
                    lines.append(f"| k{budget} | {DISPLAY_METHODS[baseline]} | — | unavailable: {paired['reason']} | — | — | — |")
        lines += ["", "### Cross-budget accuracy–latency Pareto", "", "| Method | Pareto-optimal budgets |", "|---|---|"]
        for method in METHODS:
            frontier = [f"k{point['budget']}" for point in task_result["pareto"]["by_method"][method] if point["pareto_optimal"]]
            lines.append(f"| {DISPLAY_METHODS[method]} | {', '.join(frontier) if frontier else '—'} |")
        joint = [f"{DISPLAY_METHODS[point['method']]} k{point['budget']}" for point in task_result["pareto"]["joint"] if point["pareto_optimal"]]
        lines += ["", f"Joint method-budget frontier: {', '.join(joint) if joint else '—'}", ""]
    return "\n".join(lines).rstrip() + "\n"


def output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        return output, output.with_suffix(".md")
    if output.suffix.lower() == ".md":
        return output.with_suffix(".json"), output
    return output.with_suffix(".json"), output.with_suffix(".md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="schema-v1 JSON with runs[{task,budget,method,repeat,path}]")
    source.add_argument("--run", nargs=5, action="append", metavar=("TASK", "BUDGET", "METHOD", "REPEAT", "PATH"), help="explicit run; repeat this option as needed")
    parser.add_argument("--exclude-uids", type=Path, help="optional configs/strict_eval_exclusions.json-schema manifest")
    parser.add_argument("--output", type=Path, required=True, help="output stem, .json, or .md (both JSON and Markdown are written)")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require(args.bootstrap_samples > 0, "--bootstrap-samples must be positive")
        specs, source = load_specs(manifest=args.manifest, explicit_runs=args.run)
        exclusions, exclusion_info = load_exclusions(args.exclude_uids)
        result = analyze(specs, exclusions, source=source, exclusion_info=exclusion_info, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        json_path, markdown_path = output_paths(args.output.resolve())
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(result), encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
