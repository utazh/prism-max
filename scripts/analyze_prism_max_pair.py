#!/usr/bin/env python3
"""Strict ABBA analysis for matched ContiguousKV versus ProMixed runs.

The runner naming contract is::

    <run-root>/<task>/k<budget>_<method>_<backend>_<score-mode>[_adaptive]_rN

Two repeats per method are required.  Per-request ready times are first averaged
over repeats, and the paired bootstrap is then performed over request UIDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIMARY_LATENCY_METRIC = "response_ready_ms"
READY_METRICS = (
    "logits_ready_ms",
    "first_token_ready_ms",
    "latency_ms",
    "response_ready_ms",
    "evaluation_ready_ms",
)
READY_ALIASES = {
    "logits_ready_ms": "ttft_ms",
    "evaluation_ready_ms": "accuracy_scores_ready_ms",
}
SUMMARY_METRICS = {
    "logits_ready_ms": ("mean_logits_ready_ms", "p95_logits_ready_ms"),
    "first_token_ready_ms": (
        "mean_first_token_ready_ms",
        "p95_first_token_ready_ms",
    ),
    "latency_ms": ("mean_latency_ms", "p95_latency_ms"),
    "response_ready_ms": ("mean_response_ready_ms", "p95_response_ready_ms"),
    "evaluation_ready_ms": (
        "mean_evaluation_ready_ms",
        "p95_evaluation_ready_ms",
    ),
}
SHARED_RUNTIME_KEYS = (
    "backend",
    "model_compute_dtype",
    "accuracy_scoring",
    "accuracy_scoring_protocol",
    "generation_max_tokens",
    "pcache_storage_dtype",
    "keep_ratio",
    "chunk_size",
    "period_size",
    "subperiod_size",
    "gpu_cache_mb",
    "cpu_cache_mb",
    "cache_type",
    "prefetch_time_budget",
    "reused_existing_kv_chunks",
    "resumed_partial_kv_chunks",
    "online_selection",
    "registered_store_tasks",
    "selector_kv_head_ids",
    "selector_index_dir",
    "selector_index_bits",
    "selector_index_group_size",
    "selector_index_manifest_sha256",
    "selector_index_preloaded_tasks",
    "selector_index_preloaded_bytes",
    "nominal_cpu_cache_plus_selector_bytes",
    "similarity_alpha",
    "defer_cache_score_updates",
    "cache_update_in_ttft",
    "warmup_passes",
    "warmup_samples_per_task",
    "response_ready_metric_valid_for_first_token",
    "response_ready_excludes_accuracy_scoring",
    "evaluation_ready_includes_accuracy_scoring",
)


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    summary: dict[str, Any]
    records: dict[str, dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {number!r}")
    return number


def same_number(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def percentile95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "cannot compute P95 of an empty sequence")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def interpolated_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    require(samples > 0, "bootstrap samples must be positive")
    source = [float(value) for value in values]
    require(bool(source), "cannot bootstrap an empty sequence")
    try:
        import numpy as np

        array = np.asarray(source, dtype=np.float64)
        generator = np.random.default_rng(seed)
        means: list[float] = []
        for start in range(0, samples, 1000):
            count = min(1000, samples - start)
            indices = generator.integers(
                0,
                len(array),
                size=(count, len(array)),
            )
            means.extend(array[indices].mean(axis=1).tolist())
    except ImportError:
        generator = random.Random(seed)
        means = [
            statistics.fmean(generator.choice(source) for _ in source)
            for _ in range(samples)
        ]
    return [
        interpolated_percentile(means, 0.025),
        interpolated_percentile(means, 0.975),
    ]


def exact_mcnemar_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_correct, correct_to_wrong)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def normalize_budget(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("k"):
        text = text[1:]
    require(text.isdigit(), f"budget must be numeric, got {value!r}")
    budget = int(text)
    require(budget in {5, 10, 25, 50}, f"unsupported budget: {value!r}")
    return f"{budget:03d}"


def output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        return output, output.with_suffix(".md")
    if output.suffix.lower() == ".md":
        return output.with_suffix(".json"), output
    return output.with_suffix(".json"), output.with_suffix(".md")


def auto_run_paths(
    *,
    run_root: Path,
    task: str,
    budget: str,
    backend: str,
    score_mode: str,
    repeats: Sequence[str],
    adaptive_coverage: bool,
) -> tuple[list[Path], list[Path]]:
    suffix = "_adaptive" if adaptive_coverage else ""

    def paths(method: str) -> list[Path]:
        stem = f"k{budget}_{method}_{backend}_{score_mode}{suffix}"
        return [run_root / task / f"{stem}_{repeat}" for repeat in repeats]

    return paths("contigkv"), paths("promixed")


def load_json(path: Path, *, context: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {context}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {context} {path}: {error}") from error
    require(isinstance(value, dict), f"{context} must contain a JSON object: {path}")
    return value


def load_bundle_preexclusions(path: Path | None, *, task: str) -> dict[str, Any]:
    if path is None:
        return {
            "metadata_path": None,
            "metadata_sha256": None,
            "preapplied": False,
            "manifest_path": None,
            "manifest_sha256": None,
            "excluded_uids_by_task": {},
        }
    path = path.resolve()
    metadata_bytes = path.read_bytes()
    payload = load_json(path, context="input bundle metadata")
    strict_filter = payload.get("strict_eval_filter")
    require(
        isinstance(strict_filter, dict)
        and strict_filter.get("schema_version") == 1,
        "input bundle metadata lacks schema-v1 strict_eval_filter provenance",
    )
    raw = strict_filter.get("excluded_uids_by_task")
    require(
        isinstance(raw, dict) and task in raw,
        f"strict_eval_filter lacks exclusions for {task}",
    )
    excluded: dict[str, list[str]] = {}
    for name, values in raw.items():
        require(
            isinstance(name, str) and isinstance(values, list),
            "strict_eval_filter exclusions must map task names to UID lists",
        )
        normalized = [str(uid) for uid in values]
        require(
            all(uid.startswith(f"{name}-") for uid in normalized),
            f"strict_eval_filter has an invalid UID for {name}",
        )
        require(
            len(normalized) == len(set(normalized)),
            f"strict_eval_filter has duplicate UIDs for {name}",
        )
        excluded[name] = normalized
    require(
        f"{task}-0" in excluded[task],
        f"strict input bundle did not pre-exclude {task}-0",
    )
    return {
        "metadata_path": str(path),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "preapplied": True,
        "application_stage": "strict bundle construction before benchmark runs",
        "manifest_path": strict_filter.get("exclusions_manifest"),
        "manifest_sha256": strict_filter.get("exclusions_manifest_sha256"),
        "excluded_uids_by_task": excluded,
    }


def validate_ready_row(
    row: Mapping[str, Any],
    *,
    context: str,
    backend: str,
    score_mode: str,
    selector_preloaded_bytes: int,
) -> None:
    for metric in READY_METRICS:
        require(metric in row, f"{context} is missing {metric}")
        require(
            finite_number(row[metric], context=f"{context}.{metric}") >= 0.0,
            f"{context}.{metric} must be non-negative",
        )
    for metric, alias in READY_ALIASES.items():
        require(alias in row, f"{context} is missing alias {alias}")
        require(
            same_number(row[metric], row[alias]),
            f"{context} violates ready alias {alias} == {metric}",
        )
    logits = float(row["logits_ready_ms"])
    first_token = float(row["first_token_ready_ms"])
    latency = float(row["latency_ms"])
    response = float(row["response_ready_ms"])
    evaluation = float(row["evaluation_ready_ms"])
    require(
        logits <= first_token + 1e-9,
        f"{context} has logits-ready after first-token-ready",
    )
    require(
        first_token <= latency + 1e-9,
        f"{context} has first-token-ready after latency",
    )
    require(
        latency <= response + 1e-9,
        f"{context} has latency after response-ready",
    )
    require(
        response <= evaluation + 1e-9,
        f"{context} has response-ready after evaluation-ready",
    )
    require(
        row.get("accuracy_scoring") == "label_continuation_loglikelihood",
        f"{context} does not use complete-label scoring",
    )
    label_scoring = finite_number(
        row.get("label_scoring_time_ms"),
        context=f"{context}.label_scoring_time_ms",
    )
    require(label_scoring >= 0.0, f"{context}.label_scoring_time_ms is negative")
    require(
        evaluation - response + 1e-6 >= label_scoring,
        f"{context} evaluation-ready does not include label scoring",
    )

    expected_defer = int(score_mode == "defer")
    require(
        int(row.get("cache_update_in_ttft", -1)) == 1 - expected_defer,
        f"{context} has the wrong cache-update TTFT boundary",
    )
    require(
        int(row.get("cache_update_deferred", -1)) == expected_defer,
        f"{context} has the wrong deferred cache-update flag",
    )
    expected_index = int(backend == "k4")
    require(
        int(row.get("selector_index_enabled", -1)) == expected_index,
        f"{context} has the wrong selector-index backend",
    )
    require(
        int(row.get("selector_index_bits", -1)) == 4 * expected_index,
        f"{context} has the wrong selector-index precision",
    )
    require(
        int(row.get("selector_index_preloaded", -1)) == expected_index,
        f"{context} has the wrong selector-index preload state",
    )
    require(
        int(row.get("selector_index_preloaded_bytes", -1))
        == selector_preloaded_bytes,
        f"{context} selector preload bytes disagree with summary.json",
    )
    if backend == "k4":
        require(
            int(row.get("selector_disk_source_bytes", -1)) == 0,
            f"{context} unexpectedly reads the preloaded selector from SSD",
        )


def validate_summary_aggregates(
    summary: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    context: str,
) -> None:
    task_summaries = summary.get("tasks")
    require(
        isinstance(task_summaries, dict) and set(task_summaries) == {task},
        f"{context} is not an independent {task} run",
    )
    scopes = {
        f"{context}.tasks.{task}": task_summaries[task],
        f"{context}.overall": summary.get("overall"),
    }
    for scope_name, scope in scopes.items():
        require(isinstance(scope, dict), f"{scope_name} must be an object")
        require(
            int(scope.get("samples", -1)) == len(records),
            f"{scope_name}.samples disagrees with scored_records.jsonl",
        )
        accuracy = statistics.fmean(bool(row["correct"]) for row in records.values())
        require(
            same_number(scope.get("accuracy"), accuracy),
            f"{scope_name}.accuracy disagrees with scored_records.jsonl",
        )
        for metric, (mean_name, p95_name) in SUMMARY_METRICS.items():
            values = [float(row[metric]) for row in records.values()]
            require(
                same_number(scope.get(mean_name), statistics.fmean(values), tolerance=1e-7),
                f"{scope_name}.{mean_name} disagrees with records",
            )
            require(
                same_number(scope.get(p95_name), percentile95(values), tolerance=1e-7),
                f"{scope_name}.{p95_name} disagrees with records",
            )
        summary_aliases = {
            "mean_ttft_ms": statistics.fmean(
                float(row["ttft_ms"]) for row in records.values()
            ),
            "p95_ttft_ms": percentile95(
                [float(row["ttft_ms"]) for row in records.values()]
            ),
        }
        for alias, expected in summary_aliases.items():
            require(
                same_number(scope.get(alias), expected, tolerance=1e-7),
                f"{scope_name}.{alias} disagrees with records",
            )
        label_times = [float(row["label_scoring_time_ms"]) for row in records.values()]
        require(
            same_number(
                scope.get("mean_label_scoring_time_ms"),
                statistics.fmean(label_times),
                tolerance=1e-7,
            ),
            f"{scope_name}.mean_label_scoring_time_ms disagrees with records",
        )


def load_run(
    path: Path,
    *,
    label: str,
    task: str,
    budget: str,
    backend: str,
    score_mode: str,
    method: str,
    adaptive_coverage: bool,
    expected_samples: int | None,
    expected_warmup_passes: int,
    expected_warmup_samples: int,
    expected_gpu_cache_mb: float,
    expected_cpu_cache_mb: float,
) -> Run:
    path = path.resolve()
    summary = load_json(path / "summary.json", context=f"{label} summary")
    records_path = path / "scored_records.jsonl"
    require(records_path.is_file(), f"missing {label} records: {records_path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {records_path}:{line_number}: {error}"
            ) from error
        require(isinstance(row, dict), f"{records_path}:{line_number} is not an object")
        require("uid" in row, f"{records_path}:{line_number} is missing uid")
        uid = str(row["uid"])
        require(uid not in records, f"duplicate UID {uid!r} in {records_path}")
        require(row.get("task") == task, f"UID {uid!r} in {label} has the wrong task")
        require("answer" in row, f"UID {uid!r} in {label} is missing answer")
        require("prediction" in row, f"UID {uid!r} in {label} is missing prediction")
        require("correct" in row, f"UID {uid!r} in {label} is missing correctness")
        require(
            "layer_token_selection_sha256" in row,
            f"UID {uid!r} in {label} is missing the selection hash",
        )
        records[uid] = row
    require(bool(records), f"no scored records in {records_path}")
    if expected_samples is not None:
        require(
            len(records) == expected_samples,
            f"{label} contains {len(records)} records, expected {expected_samples}",
        )

    runtime = summary.get("runtime")
    require(isinstance(runtime, dict), f"{label} summary.runtime must be an object")
    expected_runtime_method = "contigkv" if method == "contigkv" else "impress"
    require(
        runtime.get("method") == expected_runtime_method,
        f"{label} method is {runtime.get('method')!r}, expected {expected_runtime_method!r}",
    )
    expected_promixed = method == "promixed"
    require(
        runtime.get("promixed_gqa_selection") is expected_promixed,
        f"{label} has the wrong ProMixed feature state",
    )
    policy = runtime.get("promixed_policy")
    if expected_promixed:
        require(isinstance(policy, dict), f"{label} is missing promixed_policy")
        require(
            policy.get("adaptive_coverage") is adaptive_coverage,
            f"{label} has the wrong adaptive-coverage state",
        )
        require(
            runtime.get("exact_layer_block_budget") is True,
            f"{label} must use the exact layer block budget",
        )
        require(
            runtime.get("layer_budget_profile") is not None,
            f"{label} is missing its layer-budget profile",
        )
        require(
            same_number(runtime.get("layer_budget_target_mean_ratio"), int(budget) / 100),
            f"{label} layer-budget ratio does not match k{budget}",
        )
    else:
        require(policy is None, f"{label} unexpectedly has a ProMixed policy")
        require(
            runtime.get("exact_layer_block_budget") is False,
            f"{label} unexpectedly uses a layer-budget profile",
        )

    require(
        summary.get("model_path") is not None,
        f"{label} summary is missing model_path",
    )
    require(
        runtime.get("accuracy_scoring") == "label_continuation_loglikelihood",
        f"{label} does not use complete-label scoring",
    )
    require(runtime.get("generation_max_tokens") == 1, f"{label} is not a one-token run")
    require(runtime.get("online_selection") is True, f"{label} is not online selection")
    require(runtime.get("cache_type") == "CKLFU", f"{label} cache type is not CKLFU")
    require(
        runtime.get("response_ready_metric_valid_for_first_token") is True,
        f"{label} does not mark ready metrics as one-token valid",
    )
    require(
        runtime.get("response_ready_excludes_accuracy_scoring") is True,
        f"{label} response-ready boundary includes accuracy scoring",
    )
    require(
        runtime.get("evaluation_ready_includes_accuracy_scoring") is True,
        f"{label} evaluation-ready boundary excludes accuracy scoring",
    )
    measurement = summary.get("measurement")
    require(isinstance(measurement, str), f"{label} is missing measurement metadata")
    for phrase in (
        "ttft_ms/logits_ready_ms",
        "first-token logits are ready",
        "first_token_ready_ms",
        "cache-score maintenance",
        "first-token ID selection",
        "latency_ms ends when all requested token IDs are ready",
        "response_ready_ms",
        "token decoding and excludes benchmark-only accuracy scoring",
        "evaluation_ready_ms/accuracy_scores_ready_ms",
        "complete-label scoring",
        "one token only when max_tokens=1",
    ):
        require(phrase in measurement, f"{label} measurement metadata is missing {phrase}")

    expected_ratio = int(budget) / 100.0
    require(
        same_number(runtime.get("keep_ratio"), expected_ratio),
        f"{label} keep ratio does not match k{budget}",
    )
    require(
        runtime.get("warmup_passes") == expected_warmup_passes,
        f"{label} warmup_passes does not match the protocol",
    )
    require(
        runtime.get("warmup_samples_per_task") == expected_warmup_samples,
        f"{label} warmup_samples_per_task does not match the protocol",
    )
    expected_warmup_requests = expected_warmup_passes * min(
        len(records), expected_warmup_samples
    )
    require(
        runtime.get("warmup_requests") == expected_warmup_requests,
        f"{label} warmup_requests is inconsistent",
    )
    require(
        same_number(runtime.get("gpu_cache_mb"), expected_gpu_cache_mb),
        f"{label} gpu_cache_mb does not match the protocol",
    )
    require(
        same_number(runtime.get("cpu_cache_mb"), expected_cpu_cache_mb),
        f"{label} cpu_cache_mb does not match the protocol",
    )

    expected_defer = score_mode == "defer"
    require(
        runtime.get("defer_cache_score_updates") is expected_defer,
        f"{label} has the wrong defer-cache-score setting",
    )
    require(
        runtime.get("cache_update_in_ttft") is (not expected_defer),
        f"{label} has the wrong cache-update ready boundary",
    )

    preloaded_bytes = int(runtime.get("selector_index_preloaded_bytes", -1))
    if backend == "k4":
        require(runtime.get("selector_index_dir") is not None, f"{label} is not K4")
        require(runtime.get("selector_index_bits") == 4, f"{label} is not 4-bit")
        require(
            isinstance(runtime.get("selector_index_group_size"), int)
            and runtime["selector_index_group_size"] > 0,
            f"{label} has an invalid selector-index group size",
        )
        manifest_hash = runtime.get("selector_index_manifest_sha256")
        require(
            isinstance(manifest_hash, str) and len(manifest_hash) == 64,
            f"{label} has an invalid selector-index manifest hash",
        )
        require(
            runtime.get("selector_index_preloaded_tasks") == [task],
            f"{label} did not preload exactly the active selector task",
        )
        require(preloaded_bytes > 0, f"{label} reports no preloaded selector bytes")
    else:
        for key in (
            "selector_index_dir",
            "selector_index_bits",
            "selector_index_group_size",
            "selector_index_manifest_sha256",
        ):
            require(runtime.get(key) is None, f"{label} unexpectedly sets {key}")
        require(
            runtime.get("selector_index_preloaded_tasks") == [],
            f"{label} unexpectedly preloads selector tasks",
        )
        require(preloaded_bytes == 0, f"{label} unexpectedly preloads selector bytes")

    for uid, row in records.items():
        validate_ready_row(
            row,
            context=f"{label}[{uid}]",
            backend=backend,
            score_mode=score_mode,
            selector_preloaded_bytes=preloaded_bytes,
        )
    validate_summary_aggregates(summary, records, task=task, context=label)
    return Run(label=label, path=path, summary=summary, records=records)


def frozen_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_shared_protocol(runs: Sequence[Run]) -> dict[str, Any]:
    require(bool(runs), "no runs were supplied")
    first = runs[0]
    signature = {
        "model_path": first.summary["model_path"],
        "measurement": first.summary["measurement"],
        **{
            key: first.summary["runtime"].get(key)
            for key in SHARED_RUNTIME_KEYS
        },
    }
    for run in runs[1:]:
        candidate = {
            "model_path": run.summary["model_path"],
            "measurement": run.summary["measurement"],
            **{
                key: run.summary["runtime"].get(key)
                for key in SHARED_RUNTIME_KEYS
            },
        }
        for key, expected in signature.items():
            require(
                frozen_value(candidate[key]) == frozen_value(expected),
                f"shared protocol mismatch for {key}: "
                f"{first.label}={expected!r}, {run.label}={candidate[key]!r}",
            )
    return signature


def validate_uid_and_repeat_integrity(
    references: Sequence[Run],
    candidates: Sequence[Run],
) -> list[str]:
    all_runs = [*references, *candidates]
    uid_sets = {run.label: set(run.records) for run in all_runs}
    expected = next(iter(uid_sets.values()))
    for label, uids in uid_sets.items():
        require(
            uids == expected,
            "ABBA runs have different UIDs: "
            + json.dumps({name: len(items) for name, items in uid_sets.items()})
            + f"; first mismatch is {label}",
        )
    uids = sorted(expected)
    for uid in uids:
        answers = {frozen_value(run.records[uid]["answer"]) for run in all_runs}
        require(len(answers) == 1, f"UID {uid!r} has different answers across runs")
        for family, runs in (("reference", references), ("candidate", candidates)):
            for field in ("prediction", "correct", "layer_token_selection_sha256"):
                values = {frozen_value(run.records[uid][field]) for run in runs}
                require(
                    len(values) == 1,
                    f"UID {uid!r} has nondeterministic {family} {field} across repeats",
                )
    return uids


def aggregate_metric(
    runs: Sequence[Run],
    uids: Sequence[str],
    metric: str,
) -> list[float]:
    return [
        statistics.fmean(float(run.records[uid][metric]) for run in runs)
        for uid in uids
    ]


def paired_metric_result(
    references: Sequence[Run],
    candidates: Sequence[Run],
    uids: Sequence[str],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    reference = aggregate_metric(references, uids, metric)
    candidate = aggregate_metric(candidates, uids, metric)
    deltas = [new - old for old, new in zip(reference, candidate)]
    reference_mean = statistics.fmean(reference)
    candidate_mean = statistics.fmean(candidate)
    reference_p95 = percentile95(reference)
    candidate_p95 = percentile95(candidate)
    forward_deltas = [
        float(candidates[0].records[uid][metric])
        - float(references[0].records[uid][metric])
        for uid in uids
    ]
    reverse_deltas = [
        float(candidates[-1].records[uid][metric])
        - float(references[-1].records[uid][metric])
        for uid in uids
    ]
    return {
        "aggregation": "mean each UID over repeats before paired analysis",
        "reference_mean_ms": reference_mean,
        "candidate_mean_ms": candidate_mean,
        "mean_reduction_percent": (
            (reference_mean - candidate_mean) / reference_mean * 100.0
            if reference_mean
            else None
        ),
        "reference_p95_ms": reference_p95,
        "candidate_p95_ms": candidate_p95,
        "p95_reduction_percent": (
            (reference_p95 - candidate_p95) / reference_p95 * 100.0
            if reference_p95
            else None
        ),
        "mean_paired_delta_ms": statistics.fmean(deltas),
        "median_paired_delta_ms": statistics.median(deltas),
        "paired_bootstrap_mean_delta_95ci_ms": bootstrap_mean_ci(
            deltas,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "candidate_faster_uids": sum(delta < 0.0 for delta in deltas),
        "tied_uids": sum(delta == 0.0 for delta in deltas),
        "total_uids": len(uids),
        "forward_order_mean_delta_ms": statistics.fmean(forward_deltas),
        "reverse_order_mean_delta_ms": statistics.fmean(reverse_deltas),
        "reference_repeat_means_ms": [
            statistics.fmean(float(run.records[uid][metric]) for uid in uids)
            for run in references
        ],
        "candidate_repeat_means_ms": [
            statistics.fmean(float(run.records[uid][metric]) for uid in uids)
            for run in candidates
        ],
    }


def accuracy_result(
    references: Sequence[Run],
    candidates: Sequence[Run],
    uids: Sequence[str],
) -> dict[str, Any]:
    reference = [bool(references[0].records[uid]["correct"]) for uid in uids]
    candidate = [bool(candidates[0].records[uid]["correct"]) for uid in uids]
    wrong_to_correct = sum(not old and new for old, new in zip(reference, candidate))
    correct_to_wrong = sum(old and not new for old, new in zip(reference, candidate))
    return {
        "reference_accuracy": statistics.fmean(reference),
        "candidate_accuracy": statistics.fmean(candidate),
        "accuracy_delta_pp": (sum(candidate) - sum(reference)) / len(uids) * 100.0,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "mcnemar_two_sided_p": exact_mcnemar_p(wrong_to_correct, correct_to_wrong),
    }


def mechanism_result(runs: Sequence[Run], uids: Sequence[str]) -> dict[str, float]:
    metrics = (
        "effective_mean_keep_ratio",
        "selected_kv_bytes",
        "physical_prefetch_kv_bytes",
        "total_ssd_read_bytes",
        "selector_load_ms",
        "selector_compute_ms",
        "cache_update_ms",
    )
    result: dict[str, float] = {}
    for metric in metrics:
        require(
            all(metric in run.records[uid] for run in runs for uid in uids),
            f"records are missing mechanism metric {metric}",
        )
        result[f"mean_{metric}"] = statistics.fmean(
            float(run.records[uid][metric]) for run in runs for uid in uids
        )
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    budget = normalize_budget(args.budget)
    repeats = list(args.repeats)
    require(len(repeats) == 2, "ABBA analysis requires exactly two repeat labels")
    require(len(set(repeats)) == 2, "repeat labels must be unique")
    explicit = bool(args.reference_run or args.candidate_run)
    if explicit:
        require(
            len(args.reference_run) == 2 and len(args.candidate_run) == 2,
            "explicit ABBA mode requires two --reference-run and two --candidate-run",
        )
        reference_paths = list(args.reference_run)
        candidate_paths = list(args.candidate_run)
    else:
        require(args.run_root is not None, "--run-root is required in automatic mode")
        reference_paths, candidate_paths = auto_run_paths(
            run_root=args.run_root,
            task=args.task,
            budget=budget,
            backend=args.backend,
            score_mode=args.score_mode,
            repeats=repeats,
            adaptive_coverage=args.adaptive_coverage,
        )
    resolved = [path.resolve() for path in [*reference_paths, *candidate_paths]]
    require(len(set(resolved)) == 4, "the four ABBA run directories must be unique")

    common = {
        "task": args.task,
        "budget": budget,
        "backend": args.backend,
        "score_mode": args.score_mode,
        "adaptive_coverage": args.adaptive_coverage,
        "expected_samples": args.expected_samples,
        "expected_warmup_passes": args.expected_warmup_passes,
        "expected_warmup_samples": args.expected_warmup_samples,
        "expected_gpu_cache_mb": args.expected_gpu_cache_mb,
        "expected_cpu_cache_mb": args.expected_cpu_cache_mb,
    }
    references = [
        load_run(path, label=f"reference_{repeat}", method="contigkv", **common)
        for path, repeat in zip(reference_paths, repeats)
    ]
    candidates = [
        load_run(path, label=f"candidate_{repeat}", method="promixed", **common)
        for path, repeat in zip(candidate_paths, repeats)
    ]
    shared = validate_shared_protocol([*references, *candidates])
    uids = validate_uid_and_repeat_integrity(references, candidates)
    ready = {
        metric: paired_metric_result(
            references,
            candidates,
            uids,
            metric,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index * 10_000,
        )
        for index, metric in enumerate(READY_METRICS)
    }
    return {
        "schema_version": 1,
        "protocol": {
            "design": "ABBA",
            "task": args.task,
            "budget": budget,
            "backend": args.backend,
            "score_mode": args.score_mode,
            "adaptive_coverage": args.adaptive_coverage,
            "primary_latency_metric": PRIMARY_LATENCY_METRIC,
            "unique_uids": len(uids),
            "repetitions_per_method": 2,
            "run_order": [
                references[0].label,
                candidates[0].label,
                candidates[1].label,
                references[1].label,
            ],
            "reference_runs": [str(run.path) for run in references],
            "candidate_runs": [str(run.path) for run in candidates],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "shared_runtime_config": shared,
        },
        "integrity": {
            "strict_validation_passed": True,
            "identical_uid_sets": True,
            "repeat_predictions_deterministic": True,
            "repeat_selection_hashes_deterministic": True,
            "ready_aliases_and_boundaries_valid": True,
        },
        "input_bundle_preexclusions": load_bundle_preexclusions(
            args.bundle_metadata,
            task=args.task,
        ),
        "accuracy": accuracy_result(references, candidates, uids),
        "ready_metrics": ready,
        "mechanism": {
            "reference": mechanism_result(references, uids),
            "candidate": mechanism_result(candidates, uids),
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    protocol = result["protocol"]
    accuracy = result["accuracy"]
    lines = [
        "# PRISM-Max matched ABBA comparison",
        "",
        (
            f"Task `{protocol['task']}`, KV budget `{protocol['budget']}`, "
            f"selector backend `{protocol['backend']}`, score mode "
            f"`{protocol['score_mode']}`. Strict protocol validation passed for "
            f"{protocol['unique_uids']} paired UIDs."
        ),
        "",
        "Each ready time is averaged per UID over the two repeats before the paired "
        "bootstrap. Negative paired deltas favor ProMixed; positive reduction "
        "percentages favor ProMixed.",
        "Response-ready is the primary latency; logits-ready is retained as a phase "
        "metric.",
        "",
        "| Boundary | ContiguousKV mean/P95 (ms) | ProMixed mean/P95 (ms) | "
        "Mean/P95 reduction | Paired delta 95% CI (ms) | Faster UIDs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display = {
        "logits_ready_ms": "Logits ready (phase)",
        "first_token_ready_ms": "First token ready",
        "latency_ms": "All requested token IDs ready (latency)",
        "response_ready_ms": "Response ready (primary)",
        "evaluation_ready_ms": "Evaluation ready",
    }
    ordered_metrics = (PRIMARY_LATENCY_METRIC,) + tuple(
        metric for metric in READY_METRICS if metric != PRIMARY_LATENCY_METRIC
    )
    for metric in ordered_metrics:
        row = result["ready_metrics"][metric]
        ci = row["paired_bootstrap_mean_delta_95ci_ms"]
        lines.append(
            f"| {display[metric]} | {row['reference_mean_ms']:.2f}/"
            f"{row['reference_p95_ms']:.2f} | {row['candidate_mean_ms']:.2f}/"
            f"{row['candidate_p95_ms']:.2f} | {row['mean_reduction_percent']:+.2f}%/"
            f"{row['p95_reduction_percent']:+.2f}% | [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{row['candidate_faster_uids']}/{row['total_uids']} |"
        )
    lines.extend(
        [
            "",
            "## Exclusion provenance",
            "",
        ]
    )
    preexclusions = result["input_bundle_preexclusions"]
    if preexclusions["preapplied"]:
        excluded = ", ".join(
            preexclusions["excluded_uids_by_task"].get(protocol["task"], [])
        )
        lines.append(
            "Input-bundle exclusions were pre-applied before all runs: "
            f"`{excluded}`. No analysis-time exclusion manifest was applied."
        )
    else:
        lines.append("Input-bundle pre-exclusions were not declared.")
    lines.extend(
        [
            "",
            "## Accuracy",
            "",
            f"ContiguousKV `{accuracy['reference_accuracy']:.4f}`, ProMixed "
            f"`{accuracy['candidate_accuracy']:.4f}`, delta "
            f"`{accuracy['accuracy_delta_pp']:+.2f} pp`; W→C/C→W "
            f"`{accuracy['wrong_to_correct']}/{accuracy['correct_to_wrong']}`, "
            f"exact McNemar p `{accuracy['mcnemar_two_sided_p']:.4g}`.",
            "",
            "## Run order",
            "",
        ]
    )
    for label in protocol["run_order"]:
        lines.append(f"- `{label}`")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict matched ABBA analysis for PRISM-Max runner outputs."
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--task", required=True, choices=("sst2", "subj", "trec", "rte"))
    parser.add_argument("--budget", required=True)
    parser.add_argument("--backend", required=True, choices=("fp16", "k4"))
    parser.add_argument("--score-mode", required=True, choices=("nodefer", "defer"))
    parser.add_argument("--repeats", nargs="+", default=("r1", "r2"))
    parser.add_argument("--reference-run", action="append", default=[], type=Path)
    parser.add_argument("--candidate-run", action="append", default=[], type=Path)
    parser.add_argument(
        "--adaptive-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-warmup-passes", type=int, default=1)
    parser.add_argument("--expected-warmup-samples", type=int, default=32)
    parser.add_argument("--expected-gpu-cache-mb", type=float, default=55.0)
    parser.add_argument("--expected-cpu-cache-mb", type=float, default=131.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bundle-metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.expected_samples is not None and args.expected_samples <= 0:
        parser.error("--expected-samples must be positive")
    if args.expected_warmup_passes < 0:
        parser.error("--expected-warmup-passes must be non-negative")
    if args.expected_warmup_samples <= 0:
        parser.error("--expected-warmup-samples must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args)
    json_path, markdown_path = output_paths(args.output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result), end="")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


def entrypoint() -> int:
    try:
        return main()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
