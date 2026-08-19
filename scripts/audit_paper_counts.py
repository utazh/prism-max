#!/usr/bin/env python3
"""Perform strict post-run checks for the 410-request paper-count matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {"sst2": 100, "subj": 110, "trec": 120, "rte": 80}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(
    *,
    root: Path,
    manifest_path: Path,
    bundle_dir: Path,
    reorder_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    reorder_bytes = reorder_path.read_bytes()
    reorder = json.loads(reorder_bytes)
    expected_total = sum(EXPECTED_COUNTS.values())

    if manifest["samples_by_task"] != EXPECTED_COUNTS or manifest["samples"] != expected_total:
        raise ValueError("manifest does not specify the 100/110/120/80 paper counts")
    if bundle_metadata["evaluation_requests_by_task"] != EXPECTED_COUNTS:
        raise ValueError("bundle metadata has incorrect evaluation counts")
    if bundle_metadata["excluded_uids_manifest_sha256"] != hashlib.sha256(reorder_bytes).hexdigest():
        raise ValueError("bundle exclusion manifest does not match the active IMPRESS reorder")

    bundle_uids: set[str] = set()
    excluded_uids: set[str] = set()
    for task, count in EXPECTED_COUNTS.items():
        rows = _load_jsonl(bundle_dir / f"{task}.jsonl")
        task_uids = {str(row["uid"]) for row in rows}
        task_excluded = {str(uid) for uid in reorder["tasks"][task]["history_uids"]}
        if len(rows) != count or len(task_uids) != count:
            raise ValueError(f"{task} bundle does not contain {count} unique requests")
        if task_uids.intersection(task_excluded):
            raise ValueError(f"{task} evaluation requests overlap IMPRESS calibration")
        if bundle_uids.intersection(task_uids):
            raise ValueError("request UIDs overlap across tasks")
        bundle_uids.update(task_uids)
        excluded_uids.update(task_excluded)

    run_audits: dict[str, Any] = {}
    common_uids: set[str] | None = None
    for spec in manifest["runs"]:
        label = str(spec["label"])
        family = str(spec["family"])
        run_dir = root / label
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows = _load_jsonl(run_dir / "scored_records.jsonl")
        uids = {str(row["uid"]) for row in rows}
        counts = Counter(str(row["task"]) for row in rows)
        if len(rows) != expected_total or len(uids) != expected_total:
            raise ValueError(f"{label} does not contain 410 unique requests")
        if dict(counts) != EXPECTED_COUNTS or uids != bundle_uids:
            raise ValueError(f"{label} request population differs from the bundle")
        if common_uids is None:
            common_uids = uids
        elif uids != common_uids:
            raise ValueError(f"{label} cannot be paired with the other runs")
        if int(summary["overall"]["samples"]) != expected_total:
            raise ValueError(f"{label} summary has an incorrect request count")
        if any(
            not math.isfinite(float(row["ttft_ms"])) or float(row["ttft_ms"]) <= 0
            for row in rows
        ):
            raise ValueError(f"{label} contains an invalid TTFT")

        exact_budget_mismatches = 0
        prefetch_submitted = 0
        prefetch_completed = 0
        prefetch_failed = 0
        prefetch_cancelled = 0
        period_prefetch_jobs = 0
        if family == "ours":
            for row in rows:
                exact_budget_mismatches += int(
                    row.get("exact_block_budget_target")
                    != row.get("exact_block_budget_consumed")
                )
                prefetch_submitted += int(row["prefetch_scheduler_total_submitted"])
                prefetch_completed += int(row["prefetch_scheduler_total_completed"])
                prefetch_failed += int(row["prefetch_scheduler_total_failed"])
                prefetch_cancelled += int(row["prefetch_scheduler_total_cancelled"])
                period_prefetch_jobs += int(row["impress_period_prefetch_jobs"])
            if exact_budget_mismatches:
                raise ValueError(f"{label} violated its exact layer-block budget")
            if prefetch_failed or prefetch_cancelled or prefetch_completed != prefetch_submitted:
                raise ValueError(f"{label} has incomplete asynchronous prefetch work")
            if period_prefetch_jobs <= 0:
                raise ValueError(f"{label} did not execute period prefetch jobs")

        run_audits[label] = {
            "family": family,
            "samples": len(rows),
            "counts_by_task": dict(counts),
            "accuracy": float(summary["overall"]["accuracy"]),
            "mean_ttft_ms": float(summary["overall"]["mean_ttft_ms"]),
            "p95_ttft_ms": float(summary["overall"]["p95_ttft_ms"]),
            "effective_keep_ratio": float(summary["overall"]["mean_effective_keep_ratio"]),
            "exact_budget_mismatches": exact_budget_mismatches,
            "prefetch_submitted": prefetch_submitted,
            "prefetch_completed": prefetch_completed,
            "prefetch_failed": prefetch_failed,
            "prefetch_cancelled": prefetch_cancelled,
            "period_prefetch_jobs": period_prefetch_jobs,
        }

    environment_files = sorted((root / "environment").glob("k*.txt"))
    if len(environment_files) != 24:
        raise ValueError(f"expected 24 run-boundary snapshots, found {len(environment_files)}")
    for path in environment_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or not lines[1].startswith("3, NVIDIA GeForce RTX 3090"):
            raise ValueError(f"unexpected GPU snapshot in {path}")

    return {
        "schema_version": 1,
        "status": "passed",
        "expected_samples": expected_total,
        "expected_counts_by_task": EXPECTED_COUNTS,
        "common_request_uids": len(common_uids or ()),
        "excluded_calibration_uids": len(excluded_uids),
        "calibration_overlap": len(bundle_uids.intersection(excluded_uids)),
        "run_boundary_gpu_snapshots": len(environment_files),
        "runs": run_audits,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Paper-count matrix audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"- Requests: {payload['common_request_uids']} paired UIDs with counts {payload['expected_counts_by_task']}.",
        f"- IMPRESS calibration overlap: {payload['calibration_overlap']} of {payload['excluded_calibration_uids']} excluded UIDs.",
        f"- GPU boundary snapshots: {payload['run_boundary_gpu_snapshots']} on GPU 3.",
        "",
        "| Run | Samples | Exact-budget mismatches | Prefetch submitted/completed | Failed | Cancelled |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in payload["runs"].items():
        lines.append(
            f"| {label} | {row['samples']} | {row['exact_budget_mismatches']} | "
            f"{row['prefetch_submitted']}/{row['prefetch_completed']} | "
            f"{row['prefetch_failed']} | {row['prefetch_cancelled']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--reorder-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        root=args.run_root,
        manifest_path=args.manifest,
        bundle_dir=args.bundle_dir,
        reorder_path=args.reorder_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_count_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "paper_count_audit.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
