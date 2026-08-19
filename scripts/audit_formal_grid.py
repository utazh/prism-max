#!/usr/bin/env python3
"""Audit a formal ContiguousKV/IMPRESS online grid run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RATIO_SPECS = "005:0.05 025:0.25"
STABLE_BASELINE_FIELDS = (
    "answer",
    "prediction",
    "correct",
    "selected_tokens",
    "layer_token_selection_sha256",
    "physical_prefetch_chunks",
    "physical_prefetch_tokens",
    "physical_prefetch_kv_bytes",
    "selector_calls",
    "selector_fallbacks",
    "cache_chunk_size",
)
FATAL_LOG_PATTERNS = (
    re.compile(r"Traceback", re.IGNORECASE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"RuntimeError", re.IGNORECASE),
    re.compile(r"Error:", re.IGNORECASE),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--reorder-manifest", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument("--samples-per-task", type=int, default=32)
    parser.add_argument("--reference-samples-per-task", type=int, default=4)
    parser.add_argument("--gpu-cache-mb", type=float, default=55.0)
    parser.add_argument("--cpu-cache-mb", type=float, default=131.0)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument(
        "--cache-update-in-ttft",
        action="store_true",
        help="Require online CKLFU updates to occur in the measured layer path.",
    )
    parser.add_argument(
        "--ratio-specs",
        default=DEFAULT_RATIO_SPECS,
        help="Space-separated code:ratio entries, for example '005:0.05 025:0.25'.",
    )
    parser.add_argument(
        "--expected-hyperinfer-commit",
        default="91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e",
    )
    parser.add_argument("--expected-reorder-sha256")
    return parser.parse_args()


def parse_ratio_specs(value: str) -> tuple[list[tuple[str, float]], dict[str, tuple[str, float, int]]]:
    ratios: list[tuple[str, float]] = []
    runs: dict[str, tuple[str, float, int]] = {}
    for item in value.split():
        try:
            code, raw_ratio = item.split(":", 1)
            ratio = float(raw_ratio)
        except ValueError as exc:
            raise ValueError(f"invalid ratio spec {item!r}; expected code:ratio") from exc
        if not code.isdigit() or len(code) != 3:
            raise ValueError(f"ratio code must contain exactly three digits: {code!r}")
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        if any(existing_code == code or close(existing_ratio, ratio) for existing_code, existing_ratio in ratios):
            raise ValueError(f"duplicate ratio spec: {item!r}")
        ratios.append((code, ratio))
        runs[f"k{code}_contig"] = ("contigkv", ratio, 16)
        runs[f"k{code}_impress"] = ("impress", ratio, 64)
    if not ratios:
        raise ValueError("at least one ratio spec is required")
    return ratios, runs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def parse_hash_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, file_name = raw_line.split(maxsplit=1)
        entries.append((digest, Path(file_name.strip())))
    return entries


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    return True


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)


def expected_uids(tasks: list[str], samples_per_task: int) -> set[str]:
    return {
        f"{task}-{sample}"
        for task in tasks
        for sample in range(samples_per_task)
    }


def paired_accuracy_stats(
    contig_rows: list[dict[str, Any]],
    impress_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contig = {row["uid"]: bool(row["correct"]) for row in contig_rows}
    impress = {row["uid"]: bool(row["correct"]) for row in impress_rows}
    if set(contig) != set(impress):
        raise ValueError("paired accuracy requires identical request UIDs")
    cells = {
        "both_correct": 0,
        "contig_only_correct": 0,
        "impress_only_correct": 0,
        "both_wrong": 0,
    }
    for uid in contig:
        if contig[uid] and impress[uid]:
            cells["both_correct"] += 1
        elif contig[uid]:
            cells["contig_only_correct"] += 1
        elif impress[uid]:
            cells["impress_only_correct"] += 1
        else:
            cells["both_wrong"] += 1
    left = cells["contig_only_correct"]
    right = cells["impress_only_correct"]
    discordant = left + right
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(left, right) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2 * tail)
    else:
        exact_p = 1.0
    return {
        **cells,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": exact_p,
    }


def validate_hash_manifest(audit: Audit, name: str, path: Path) -> None:
    entries = parse_hash_manifest(path)
    audit.add(f"{name}.entries_present", bool(entries), {"entries": len(entries)})
    for expected, file_path in entries:
        exists = file_path.is_file()
        actual = sha256_file(file_path) if exists else None
        audit.add(
            f"{name}.sha256.{file_path.name}",
            exists and actual == expected,
            {"path": str(file_path), "expected": expected, "actual": actual},
        )


def validate_reorder_manifest(
    audit: Audit,
    manifest_path: Path,
    tasks: list[str],
    scored_uids: set[str],
    expected_digest: str | None,
) -> tuple[str, set[str], dict[str, Any]]:
    digest = sha256_file(manifest_path)
    manifest = load_json(manifest_path)
    audit.add(
        "reorder.sha256_locked",
        expected_digest is None or digest == expected_digest,
        {"expected": expected_digest, "actual": digest},
    )
    audit.add("reorder.method", manifest.get("method") == "impress", manifest.get("method"))
    audit.add("reorder.dtype", manifest.get("model_compute_dtype") == "bfloat16", manifest.get("model_compute_dtype"))
    audit.add("reorder.sample_offset", manifest.get("sample_offset") == 32, manifest.get("sample_offset"))
    audit.add("reorder.samples_per_task", manifest.get("samples_per_task") == 4, manifest.get("samples_per_task"))
    task_data = manifest.get("tasks", {})
    audit.add("reorder.tasks", set(task_data) == set(tasks), sorted(task_data))

    history_uids: set[str] = set()
    permutation_layers = 0
    non_identity_layers = 0
    for task in tasks:
        data = task_data.get(task, {})
        expected_history = [f"{task}-{sample}" for sample in range(32, 36)]
        history = data.get("history_uids", [])
        history_uids.update(history)
        audit.add(f"reorder.{task}.history_uids", history == expected_history, history)
        prefix_tokens = data.get("prefix_tokens")
        layers = data.get("physical_to_logical", [])
        audit.add(f"reorder.{task}.layers", len(layers) == 28, len(layers))
        valid_permutations = True
        for permutation in layers:
            permutation_layers += 1
            if not isinstance(prefix_tokens, int) or sorted(permutation) != list(range(prefix_tokens)):
                valid_permutations = False
                continue
            if permutation != list(range(prefix_tokens)):
                non_identity_layers += 1
        audit.add(
            f"reorder.{task}.permutations",
            valid_permutations,
            {"prefix_tokens": prefix_tokens, "layers": len(layers)},
        )
    audit.add(
        "reorder.non_identity",
        non_identity_layers > 0,
        {"non_identity_layers": non_identity_layers, "total_layers": permutation_layers},
    )
    audit.add(
        "reorder.history_disjoint_from_scored",
        history_uids.isdisjoint(scored_uids),
        {"history_uids": sorted(history_uids), "intersection": sorted(history_uids & scored_uids)},
    )
    return digest, history_uids, {
        "history_uids": sorted(history_uids),
        "permutation_layers": permutation_layers,
        "non_identity_layers": non_identity_layers,
    }


def validate_plan(
    audit: Audit,
    path: Path,
    tag: str,
    method: str,
    keep_ratio: float,
    chunk_size: int,
    tasks: list[str],
    reference_samples_per_task: int,
) -> set[str]:
    plan = load_json(path)
    metadata = plan.get("metadata", {})
    records = metadata.get("records", [])
    uids = {record.get("uid") for record in records}
    expected = expected_uids(tasks, reference_samples_per_task)
    audit.add(f"{tag}.plan.method", metadata.get("method") == method, metadata.get("method"))
    audit.add(f"{tag}.plan.keep_ratio", close(metadata.get("keep_ratio", -1), keep_ratio), metadata.get("keep_ratio"))
    audit.add(f"{tag}.plan.chunk_size", metadata.get("chunk_size") == chunk_size, metadata.get("chunk_size"))
    audit.add(f"{tag}.plan.reference_uids", uids == expected, {"count": len(uids), "unexpected": sorted(uids ^ expected)})
    return uids


def validate_runtime(
    audit: Audit,
    tag: str,
    runtime: dict[str, Any],
    method: str,
    keep_ratio: float,
    chunk_size: int,
    args: argparse.Namespace,
    reorder_digest: str,
    expected_reference_requests: int,
) -> None:
    common = {
        "online_selection": True,
        "model_compute_dtype": "bfloat16",
        "pcache_storage_dtype": "float16",
        "gpu_cache_mb": args.gpu_cache_mb,
        "cpu_cache_mb": args.cpu_cache_mb,
        "warmup_passes": args.warmup_passes,
        "warmup_requests": args.samples_per_task * len(args.task_list),
        "period_size": 8,
        "subperiod_size": 4,
        "cache_update_in_ttft": args.cache_update_in_ttft,
        "reused_existing_kv_chunks": True,
        "resumed_partial_kv_chunks": False,
    }
    for field, expected in common.items():
        actual = runtime.get(field)
        passed = close(actual, expected) if isinstance(expected, float) else actual == expected
        audit.add(f"{tag}.runtime.{field}", passed, {"expected": expected, "actual": actual})
    audit.add(f"{tag}.runtime.method", runtime.get("method") == method, runtime.get("method"))
    audit.add(f"{tag}.runtime.keep_ratio", close(runtime.get("keep_ratio", -1), keep_ratio), runtime.get("keep_ratio"))
    audit.add(f"{tag}.runtime.chunk_size", runtime.get("chunk_size") == chunk_size, runtime.get("chunk_size"))
    audit.add(
        f"{tag}.runtime.selection_reference_requests",
        runtime.get("selection_reference_requests") == expected_reference_requests,
        runtime.get("selection_reference_requests"),
    )
    if method == "contigkv":
        expected = {
            "runtime_variant": "contiguouskv-online-period-prefetch",
            "cache_score_policy": "cumulative-attention-times-frequency",
            "impress_reorder_enabled": False,
            "impress_reorder_manifest": None,
            "impress_reorder_sha256": None,
            "selector_kv_head_ids": [0, 1, 2, 3],
        }
    else:
        expected = {
            "runtime_variant": "paper-impress-sync-reorder",
            "cache_score_policy": "chunk-accesses-and-cumulative-important-tokens",
            "impress_async_inter_layer_prefetch": False,
            "impress_dynamic_prefetch_budget": False,
            "impress_reorder_enabled": True,
            "impress_reorder_sha256": reorder_digest,
            "selector_kv_head_ids": [0],
        }
    for field, expected_value in expected.items():
        actual = runtime.get(field)
        audit.add(
            f"{tag}.runtime.{field}",
            actual == expected_value,
            {"expected": expected_value, "actual": actual},
        )


def validate_records(
    audit: Audit,
    tag: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    method: str,
    chunk_size: int,
    expected: set[str],
    reference_uids: set[str],
    cache_update_in_ttft: bool,
) -> dict[str, Any]:
    uids = [row.get("uid") for row in rows]
    task_counts = Counter(row.get("task") for row in rows)
    audit.add(f"{tag}.records.count", len(rows) == len(expected), len(rows))
    audit.add(f"{tag}.records.unique_uids", len(set(uids)) == len(rows), len(set(uids)))
    audit.add(f"{tag}.records.expected_uids", set(uids) == expected, {"difference": sorted(set(uids) ^ expected)})
    audit.add(f"{tag}.records.task_counts", len(set(task_counts.values())) == 1, dict(task_counts))
    audit.add(f"{tag}.records.finite", all(all_finite(row) for row in rows), "all numeric values finite")
    audit.add(
        f"{tag}.records.selection_hashes",
        all(SHA256_RE.fullmatch(str(row.get("layer_token_selection_sha256", ""))) for row in rows),
        {"records": len(rows), "unique_hashes": len({row.get("layer_token_selection_sha256") for row in rows})},
    )
    audit.add(f"{tag}.records.cache_backend", all(row.get("cache_backend") == "flexgen_pcache" for row in rows), "flexgen_pcache")
    audit.add(f"{tag}.records.disk_type", all(row.get("disk_type") == "KV_Division" for row in rows), "KV_Division")
    audit.add(f"{tag}.records.chunk_size", all(row.get("cache_chunk_size") == chunk_size for row in rows), chunk_size)
    audit.add(f"{tag}.records.ttft_positive", all(row.get("ttft_ms", 0) > 0 for row in rows), "all > 0")
    audit.add(
        f"{tag}.records.cache_update_timing",
        all(bool(row.get("cache_update_in_ttft")) == cache_update_in_ttft for row in rows),
        {"expected_in_ttft": cache_update_in_ttft},
    )
    audit.add(
        f"{tag}.records.cache_score_updates",
        all(row.get("cache_score_updates", 0) > 0 for row in rows),
        "all requests update CKLFU",
    )
    audit.add(
        f"{tag}.records.cache_update_ms",
        all(row.get("cache_update_ms", -1) >= 0 for row in rows),
        "all non-negative",
    )
    audit.add(
        f"{tag}.records.latency_includes_ttft",
        all(row.get("latency_ms", 0) >= row.get("ttft_ms", 0) for row in rows),
        "latency_ms >= ttft_ms",
    )
    audit.add(
        f"{tag}.records.physical_at_least_selected",
        all(row.get("physical_prefetch_kv_bytes", 0) >= row.get("selected_kv_bytes", 0) for row in rows),
        "physical_prefetch_kv_bytes >= selected_kv_bytes",
    )
    audit.add(
        f"{tag}.records.kv_byte_accounting",
        all(
            row.get("selected_kv_bytes") == row.get("selected_tokens") * 2048
            and row.get("physical_prefetch_kv_bytes") == row.get("physical_prefetch_tokens") * 2048
            and row.get("total_ssd_read_bytes")
            == row.get("critical_ssd_read_bytes") + row.get("selector_disk_source_bytes")
            for row in rows
        ),
        "Qwen K+V is 2048 bytes/token; total SSD is critical plus selector",
    )
    reference_count = sum("selection_reference_mean_jaccard" in row for row in rows)
    audit.add(f"{tag}.records.reference_count", reference_count == len(reference_uids), reference_count)
    unknown_uids = set(uids) - reference_uids
    audit.add(
        f"{tag}.records.online_unknown_uids",
        len(unknown_uids) == len(expected) - len(reference_uids),
        {"count": len(unknown_uids)},
    )

    if method == "contigkv":
        audit.add(f"{tag}.records.selector_calls", all(row.get("selector_calls") == 4 for row in rows), "4 Period selectors")
        audit.add(f"{tag}.records.selector_fallbacks", all(row.get("selector_fallbacks") == 0 for row in rows), 0)
        inter_period = {
            "hit_tokens": sum(row.get("inter_period_hit_tokens", 0) for row in rows),
            "missing_tokens": sum(row.get("inter_period_missing_tokens", 0) for row in rows),
            "unused_tokens": sum(row.get("inter_period_unused_tokens", 0) for row in rows),
        }
        elapsed = sum(row.get("prefetch_elapsed_ms", 0) for row in rows)
        waited = sum(row.get("prefetch_wait_ms", 0) for row in rows)
        audit.add(f"{tag}.records.inter_period_prefetch", all(value > 0 for value in inter_period.values()), inter_period)
        audit.add(
            f"{tag}.records.async_overlap",
            elapsed > 0 and 0 < waited < elapsed,
            {"summed_operation_ms": elapsed, "waited_ms": waited, "wait_fraction": waited / elapsed if elapsed else None},
        )
    else:
        audit.add(f"{tag}.records.selector_calls", all(row.get("selector_calls") == 28 for row in rows), "28 layer selectors")
        audit.add(
            f"{tag}.records.sync_wait",
            all(close(row.get("prefetch_wait_ms", -1), row.get("prefetch_elapsed_ms", -2)) for row in rows),
            "prefetch wait equals synchronous read elapsed time",
        )
        audit.add(
            f"{tag}.records.no_async_budget",
            all(close(row.get("impress_mean_prefetch_budget_seconds", -1), 0.0) for row in rows),
            0.0,
        )
        audit.add(
            f"{tag}.records.no_inter_period_prefetch",
            all(
                row.get("inter_period_hit_tokens") == 0
                and row.get("inter_period_missing_tokens") == 0
                and row.get("inter_period_unused_tokens") == 0
                for row in rows
            ),
            "all zero",
        )

    overall = summary.get("overall", {})
    recomputed_accuracy = mean(float(row["correct"]) for row in rows)
    recomputed_ttft = mean(row["ttft_ms"] for row in rows)
    audit.add(f"{tag}.summary.samples", overall.get("samples") == len(rows), overall.get("samples"))
    audit.add(f"{tag}.summary.accuracy", close(overall.get("accuracy", -1), recomputed_accuracy), {"summary": overall.get("accuracy"), "records": recomputed_accuracy})
    audit.add(f"{tag}.summary.mean_ttft", close(overall.get("mean_ttft_ms", -1), recomputed_ttft), {"summary": overall.get("mean_ttft_ms"), "records": recomputed_ttft})
    return {
        "samples": len(rows),
        "accuracy": recomputed_accuracy,
        "mean_ttft_ms": recomputed_ttft,
        "mean_physical_prefetch_kv_bytes": mean(row["physical_prefetch_kv_bytes"] for row in rows),
        "mean_critical_ssd_read_bytes": mean(row["critical_ssd_read_bytes"] for row in rows),
        "mean_total_ssd_read_bytes": mean(row["total_ssd_read_bytes"] for row in rows),
        "selection_hashes": len({row["layer_token_selection_sha256"] for row in rows}),
        "selection_reference_requests": reference_count,
        "online_only_requests": len(unknown_uids),
    }


def validate_baseline(
    audit: Audit,
    baseline_root: Path,
    tag: str,
    rows: list[dict[str, Any]],
    tasks: list[str],
    reference_samples_per_task: int,
) -> None:
    baseline_rows = load_jsonl(baseline_root / tag / "scored_records.jsonl")
    expected = expected_uids(tasks, reference_samples_per_task)
    current = {row["uid"]: row for row in rows if row["uid"] in expected}
    baseline = {row["uid"]: row for row in baseline_rows}
    audit.add(f"{tag}.baseline.uid_match", set(current) == set(baseline) == expected, {"current": len(current), "baseline": len(baseline)})
    differences: list[dict[str, Any]] = []
    for uid in sorted(expected):
        if uid not in current or uid not in baseline:
            continue
        for field in STABLE_BASELINE_FIELDS:
            if current[uid].get(field) != baseline[uid].get(field):
                differences.append(
                    {
                        "uid": uid,
                        "field": field,
                        "baseline": baseline[uid].get(field),
                        "expanded": current[uid].get(field),
                    }
                )
    audit.add(
        f"{tag}.baseline.stable_behavior",
        not differences,
        {"compared_values": len(expected) * len(STABLE_BASELINE_FIELDS), "differences": differences[:20]},
    )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Formal Grid Audit",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        f"Checks: {report['passed_checks']}/{report['check_count']} passed.",
        "",
        "| KV budget | ContiguousKV TTFT | IMPRESS TTFT | Speedup | Physical read reduction | Critical SSD reduction | Total SSD reduction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget, stats in report["comparisons"].items():
        lines.append(
            f"| {budget} | {stats['contig_mean_ttft_ms']:.2f} ms | "
            f"{stats['impress_mean_ttft_ms']:.2f} ms | {stats['speedup']:.2f}x | "
            f"{stats['physical_read_reduction']:.2f}x | {stats['critical_ssd_reduction']:.2f}x | "
            f"{stats['total_ssd_reduction']:.2f}x |"
        )
    lines.extend(["", "Failed checks:"])
    failures = [check for check in report["checks"] if not check["passed"]]
    lines.extend(f"- `{check['name']}`: {check['detail']}" for check in failures)
    if not failures:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.task_list = [task.strip() for task in args.tasks.split(",") if task.strip()]
    ratio_items, run_specs = parse_ratio_specs(args.ratio_specs)
    grid_root = args.grid_root.resolve()
    baseline_root = args.baseline_root.resolve()
    output = (args.output or grid_root / "audit.json").resolve()
    markdown_output = (args.markdown_output or grid_root / "audit.md").resolve()
    audit = Audit()

    required_dirs = set(run_specs)
    actual_dirs = {path.name for path in grid_root.iterdir() if path.is_dir()}
    audit.add("grid.run_directories", actual_dirs == required_dirs, {"expected": sorted(required_dirs), "actual": sorted(actual_dirs)})
    for name in ("summary.json", "summary.md", "reproduction_source_sha256.txt", "reproduction_input_sha256.txt", "hyperinfer_commit.txt"):
        audit.add(f"grid.file.{name}", (grid_root / name).is_file(), str(grid_root / name))

    commit = (grid_root / "hyperinfer_commit.txt").read_text(encoding="utf-8").strip()
    audit.add("grid.hyperinfer_commit", commit == args.expected_hyperinfer_commit, {"expected": args.expected_hyperinfer_commit, "actual": commit})
    validate_hash_manifest(audit, "source_manifest", grid_root / "reproduction_source_sha256.txt")
    validate_hash_manifest(audit, "input_manifest", grid_root / "reproduction_input_sha256.txt")

    expected_all = expected_uids(args.task_list, args.samples_per_task)
    reorder_digest, _, reorder_stats = validate_reorder_manifest(
        audit,
        args.reorder_manifest.resolve(),
        args.task_list,
        expected_all,
        args.expected_reorder_sha256,
    )

    logs = sorted(grid_root.rglob("*.log"))
    expected_log_names = {
        *(f"{tag}.log" for tag in run_specs),
        *(f"k{code}_comparison.log" for code, _ in ratio_items),
    }
    fatal_hits: list[dict[str, Any]] = []
    for log in logs:
        for line_number, line in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if any(pattern.search(line) for pattern in FATAL_LOG_PATTERNS):
                fatal_hits.append({"file": str(log), "line": line_number, "text": line[:500]})
    actual_root_log_names = {log.name for log in grid_root.glob("*.log")}
    audit.add(
        "grid.logs.expected",
        expected_log_names <= actual_root_log_names,
        {
            "expected": sorted(expected_log_names),
            "actual": sorted(actual_root_log_names),
        },
    )
    audit.add("grid.logs.no_fatal_errors", not fatal_hits, fatal_hits)

    run_stats: dict[str, dict[str, Any]] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for tag, (method, keep_ratio, chunk_size) in run_specs.items():
        run_root = grid_root / tag
        summary = load_json(run_root / "summary.json")
        rows = load_jsonl(run_root / "scored_records.jsonl")
        all_rows[tag] = rows
        reference_uids = validate_plan(
            audit,
            args.plan_dir / f"{tag.rsplit('_', 1)[0]}_{method if method == 'impress' else 'contig'}.json",
            tag,
            method,
            keep_ratio,
            chunk_size,
            args.task_list,
            args.reference_samples_per_task,
        )
        validate_runtime(
            audit,
            tag,
            summary.get("runtime", {}),
            method,
            keep_ratio,
            chunk_size,
            args,
            reorder_digest,
            len(reference_uids),
        )
        run_stats[tag] = validate_records(
            audit,
            tag,
            rows,
            summary,
            method,
            chunk_size,
            expected_all,
            reference_uids,
            args.cache_update_in_ttft,
        )
        validate_baseline(
            audit,
            baseline_root,
            tag,
            rows,
            args.task_list,
            args.reference_samples_per_task,
        )

    comparisons: dict[str, dict[str, float]] = {}
    for code, ratio in ratio_items:
        contig = run_stats[f"k{code}_contig"]
        impress = run_stats[f"k{code}_impress"]
        comparisons[f"{ratio * 100:g}%"] = {
            "contig_accuracy": contig["accuracy"],
            "impress_accuracy": impress["accuracy"],
            "contig_mean_ttft_ms": contig["mean_ttft_ms"],
            "impress_mean_ttft_ms": impress["mean_ttft_ms"],
            "speedup": impress["mean_ttft_ms"] / contig["mean_ttft_ms"],
            "physical_read_reduction": impress["mean_physical_prefetch_kv_bytes"] / contig["mean_physical_prefetch_kv_bytes"],
            "critical_ssd_reduction": impress["mean_critical_ssd_read_bytes"] / contig["mean_critical_ssd_read_bytes"],
            "total_ssd_reduction": impress["mean_total_ssd_read_bytes"] / contig["mean_total_ssd_read_bytes"],
            "paired_accuracy": paired_accuracy_stats(
                all_rows[f"k{code}_contig"],
                all_rows[f"k{code}_impress"],
            ),
        }

    passed_checks = sum(check["passed"] for check in audit.checks)
    report = {
        "schema_version": 1,
        "status": "pass" if audit.passed else "fail",
        "grid_root": str(grid_root),
        "baseline_root": str(baseline_root),
        "reorder_manifest": str(args.reorder_manifest.resolve()),
        "reorder_sha256": reorder_digest,
        "hyperinfer_commit": commit,
        "samples_per_task": args.samples_per_task,
        "ratio_specs": [{"code": code, "ratio": ratio} for code, ratio in ratio_items],
        "scored_records": sum(stats["samples"] for stats in run_stats.values()),
        "check_count": len(audit.checks),
        "passed_checks": passed_checks,
        "failed_checks": len(audit.checks) - passed_checks,
        "run_statistics": run_stats,
        "comparisons": comparisons,
        "reorder_statistics": reorder_stats,
        "checks": audit.checks,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(markdown_output, report)
    print(json.dumps({key: report[key] for key in ("status", "check_count", "passed_checks", "failed_checks", "scored_records", "comparisons")}, indent=2, sort_keys=True))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
