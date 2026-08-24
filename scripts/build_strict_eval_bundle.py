#!/usr/bin/env python3
"""Build a byte-preserving evaluation bundle with strict UID exclusions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
TASK_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(
    path: Path,
    *,
    context: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {context} {path}: {error}") from error
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object: {path}")
    return value, payload


def _load_exclusions(
    path: Path,
) -> tuple[dict[str, tuple[str, ...]], bytes]:
    manifest, payload = _read_json_object(path, context="exclusions manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "exclusions manifest schema_version must be "
            f"{SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    tasks = manifest.get("exclude_uids_by_task")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(
            "exclusions manifest exclude_uids_by_task must be a non-empty object"
        )

    excluded_by_task: dict[str, tuple[str, ...]] = {}
    for raw_task, raw_uids in tasks.items():
        task = str(raw_task)
        if not TASK_NAME.fullmatch(task):
            raise ValueError(f"invalid task name in exclusions manifest: {task!r}")
        if not isinstance(raw_uids, list):
            raise ValueError(f"exclusions for {task!r} must be a list")
        if any(not isinstance(uid, str) or not uid for uid in raw_uids):
            raise ValueError(f"exclusions for {task!r} must be non-empty strings")
        if len(set(raw_uids)) != len(raw_uids):
            raise ValueError(f"exclusions for {task!r} contain duplicate UIDs")
        excluded_by_task[task] = tuple(raw_uids)
    return excluded_by_task, payload


def _metadata_count(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _validate_source_metadata(
    metadata: Mapping[str, Any],
    *,
    exclusion_tasks: set[str],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    task_metadata = metadata.get("tasks")
    evaluation_counts = metadata.get("evaluation_requests_by_task")
    if not isinstance(task_metadata, dict) or not task_metadata:
        raise ValueError("source metadata.tasks must be a non-empty object")
    if not isinstance(evaluation_counts, dict):
        raise ValueError("source metadata.evaluation_requests_by_task must be an object")

    tasks = tuple(str(task) for task in task_metadata)
    if any(not TASK_NAME.fullmatch(task) for task in tasks):
        raise ValueError("source metadata contains an invalid task name")
    source_tasks = set(tasks)
    if source_tasks != exclusion_tasks:
        missing = sorted(source_tasks - exclusion_tasks)
        unexpected = sorted(exclusion_tasks - source_tasks)
        raise ValueError(
            "exclusions manifest tasks must exactly match source tasks; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if set(str(task) for task in evaluation_counts) != source_tasks:
        raise ValueError(
            "source evaluation_requests_by_task keys must exactly match metadata.tasks"
        )
    return task_metadata, evaluation_counts, tasks


def _filter_task_jsonl(
    path: Path,
    *,
    task: str,
    excluded_uids: Sequence[str],
    globally_seen_uids: dict[str, str],
) -> tuple[bytes, int, tuple[str, ...], str]:
    try:
        source_payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read source task JSONL {path}: {error}") from error

    excluded = set(excluded_uids)
    found_excluded: list[str] = []
    seen: set[str] = set()
    retained: list[bytes] = []
    shared_prefix: Any = None
    prefix_initialized = False
    source_rows = 0
    for line_number, raw_line in enumerate(source_payload.splitlines(keepends=True), 1):
        if not raw_line.strip():
            raise ValueError(f"blank line in {path}:{line_number}")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"row in {path}:{line_number} must be a JSON object")
        uid_value = row.get("uid")
        if not isinstance(uid_value, str) or not uid_value:
            raise ValueError(f"row in {path}:{line_number} has a missing or invalid UID")
        uid = uid_value
        if uid in seen:
            raise ValueError(f"duplicate UID {uid!r} in task {task!r}")
        previous_task = globally_seen_uids.get(uid)
        if previous_task is not None:
            raise ValueError(
                f"duplicate UID {uid!r} across tasks {previous_task!r} and {task!r}"
            )
        seen.add(uid)
        globally_seen_uids[uid] = task
        source_rows += 1

        if "prefix_text" not in row:
            raise ValueError(f"row {uid!r} in task {task!r} is missing prefix_text")
        if not prefix_initialized:
            shared_prefix = row["prefix_text"]
            prefix_initialized = True
        elif row["prefix_text"] != shared_prefix:
            raise ValueError(f"task {task!r} does not preserve one shared prefix")

        if uid in excluded:
            found_excluded.append(uid)
        else:
            # Preserve every retained row exactly; do not parse and reserialize it.
            retained.append(raw_line)

    if source_rows == 0:
        raise ValueError(f"source task {task!r} has no rows")
    missing = sorted(excluded - set(found_excluded))
    if missing:
        raise ValueError(f"task {task!r} is missing configured exclusion UIDs {missing}")
    if not retained:
        raise ValueError(f"task {task!r} has no rows after strict exclusions")
    return b"".join(retained), source_rows, tuple(found_excluded), _sha256(source_payload)


def build_strict_eval_bundle(
    *,
    source_bundle: str | Path,
    output_bundle: str | Path,
    exclusions_manifest: str | Path,
) -> dict[str, Any]:
    """Filter a bundle without reserializing any retained task record."""

    source = Path(source_bundle).resolve()
    output = Path(output_bundle).resolve()
    exclusions_path = Path(exclusions_manifest).resolve()
    if source == output:
        raise ValueError("source and output bundles must differ")
    if not source.is_dir():
        raise ValueError(f"source bundle is not a directory: {source}")
    if output.exists():
        raise ValueError(f"output bundle already exists: {output}")

    metadata, metadata_payload = _read_json_object(
        source / "metadata.json",
        context="source metadata",
    )
    excluded_by_task, exclusions_payload = _load_exclusions(exclusions_path)
    task_metadata, evaluation_counts, tasks = _validate_source_metadata(
        metadata,
        exclusion_tasks=set(excluded_by_task),
    )

    filtered_payloads: dict[str, bytes] = {}
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    output_counts: dict[str, int] = {}
    actual_exclusions: dict[str, list[str]] = {}
    globally_seen_uids: dict[str, str] = {}
    for task in tasks:
        task_entry = task_metadata[task]
        if not isinstance(task_entry, dict):
            raise ValueError(f"source metadata.tasks.{task} must be an object")
        filtered, source_count, found, source_hash = _filter_task_jsonl(
            source / f"{task}.jsonl",
            task=task,
            excluded_uids=excluded_by_task[task],
            globally_seen_uids=globally_seen_uids,
        )
        declared_counts = {
            "evaluation_requests_by_task": _metadata_count(
                evaluation_counts.get(task),
                context=f"source evaluation_requests_by_task.{task}",
            ),
            "tasks.records": _metadata_count(
                task_entry.get("records"),
                context=f"source tasks.{task}.records",
            ),
            "tasks.evaluation_requests": _metadata_count(
                task_entry.get("evaluation_requests"),
                context=f"source tasks.{task}.evaluation_requests",
            ),
        }
        if any(count != source_count for count in declared_counts.values()):
            raise ValueError(
                f"source metadata counts for {task!r} disagree with its "
                f"{source_count} JSONL rows: {declared_counts}"
            )
        filtered_payloads[task] = filtered
        source_hashes[task] = source_hash
        output_hashes[task] = _sha256(filtered)
        source_counts[task] = source_count
        output_counts[task] = source_count - len(found)
        actual_exclusions[task] = list(found)

    output_metadata = copy.deepcopy(metadata)
    for task in tasks:
        output_metadata["evaluation_requests_by_task"][task] = output_counts[task]
        output_metadata["tasks"][task]["records"] = output_counts[task]
        output_metadata["tasks"][task]["evaluation_requests"] = output_counts[task]
        output_metadata["tasks"][task]["strict_source_records"] = source_counts[task]
        output_metadata["tasks"][task]["strict_excluded_uids"] = actual_exclusions[task]
    output_metadata["strict_eval_filter"] = {
        "schema_version": SCHEMA_VERSION,
        "source_bundle": str(source),
        "source_metadata_sha256": _sha256(metadata_payload),
        "source_task_jsonl_sha256": source_hashes,
        "output_task_jsonl_sha256": output_hashes,
        "exclusions_manifest": str(exclusions_path),
        "exclusions_manifest_sha256": _sha256(exclusions_payload),
        "excluded_uids_by_task": actual_exclusions,
        "source_records_by_task": source_counts,
        "evaluation_records_by_task": output_counts,
        "retained_row_serialization": "copied byte-for-byte from source JSONL",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent))
    try:
        for task in tasks:
            (staging / f"{task}.jsonl").write_bytes(filtered_payloads[task])
        (staging / "metadata.json").write_text(
            json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise ValueError(f"output bundle appeared during build: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter calibrated UIDs from an evaluation bundle."
    )
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--output-bundle", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_strict_eval_bundle(
        source_bundle=args.source_bundle,
        output_bundle=args.output_bundle,
        exclusions_manifest=args.exclusions,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def entrypoint() -> int:
    try:
        return main()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
