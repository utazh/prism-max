#!/usr/bin/env python3
"""Build the four paper task bundles with Table 1 evaluation counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contiguous_fuxian.paper_tasks import ensure_paper_datasets, write_paper_task_bundles


PAPER_EVALUATION_COUNTS = {
    "sst2": 100,
    "subj": 110,
    "trec": 120,
    "rte": 80,
}


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _text_sha256(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def build_bundles(
    *,
    data_root: Path,
    output_dir: Path,
    tokenizer: str,
    reference_bundle: Path | None,
    excluded_uids_manifest: Path | None,
    seed: int,
    skip_download: bool,
) -> dict[str, dict[str, object]]:
    if not skip_download:
        ensure_paper_datasets(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_by_task: dict[str, set[str]] = {
        task: set() for task in PAPER_EVALUATION_COUNTS
    }
    exclusion_manifest_sha256 = None
    if excluded_uids_manifest is not None:
        exclusion_payload = json.loads(
            excluded_uids_manifest.read_text(encoding="utf-8")
        )
        exclusion_manifest_sha256 = hashlib.sha256(
            excluded_uids_manifest.read_bytes()
        ).hexdigest()
        for task in PAPER_EVALUATION_COUNTS:
            excluded_by_task[task] = {
                str(uid)
                for uid in exclusion_payload["tasks"][task].get("history_uids", [])
            }

    merged_metadata: dict[str, dict[str, object]] = {}
    all_uids: set[str] = set()
    for task, count in PAPER_EVALUATION_COUNTS.items():
        excluded_uids = excluded_by_task[task]
        metadata = write_paper_task_bundles(
            output_dir,
            data_root,
            [task],
            seed=seed,
            eval_samples=count + len(excluded_uids),
            tokenizer_path=tokenizer,
            qwen_chat_format=True,
        )[task]
        path = output_dir / f"{task}.jsonl"
        rows = [
            row
            for row in _records(path)
            if str(row["uid"]) not in excluded_uids
        ][:count]
        if len(rows) != count:
            raise ValueError(f"{task} contains {len(rows)} records; expected {count}")
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        task_uids = {str(row["uid"]) for row in rows}
        if len(task_uids) != count or all_uids.intersection(task_uids):
            raise ValueError(f"{task} contains duplicate request UIDs")
        all_uids.update(task_uids)

        prefix_hash = _text_sha256(rows[0]["prefix_text"])
        if any(_text_sha256(row["prefix_text"]) != prefix_hash for row in rows):
            raise ValueError(f"{task} does not use one stable shared prefix")

        reference_records = 0
        if reference_bundle is not None:
            reference_rows = _records(reference_bundle / f"{task}.jsonl")
            reference_records = len(reference_rows)
            if not reference_rows:
                raise ValueError(f"reference bundle for {task} is empty")
            if _text_sha256(reference_rows[0]["prefix_text"]) != prefix_hash:
                raise ValueError(f"{task} prefix differs from the validated 32-request bundle")
            for index, reference in enumerate(reference_rows):
                if rows[index] != reference:
                    raise ValueError(
                        f"{task} record {index} differs from the validated 32-request bundle"
                    )

        merged_metadata[task] = {
            **metadata,
            "records": count,
            "evaluation_requests": count,
            "prefix_text_sha256": prefix_hash,
            "validated_reference_records": reference_records,
            "excluded_calibration_uids": sorted(excluded_uids),
        }

    expected_total = sum(PAPER_EVALUATION_COUNTS.values())
    if len(all_uids) != expected_total:
        raise ValueError(f"bundle contains {len(all_uids)} UIDs; expected {expected_total}")
    payload = {
        "schema_version": 2,
        "seed": seed,
        "total_evaluation_requests": expected_total,
        "evaluation_requests_by_task": PAPER_EVALUATION_COUNTS,
        "excluded_uids_manifest": (
            str(excluded_uids_manifest) if excluded_uids_manifest is not None else None
        ),
        "excluded_uids_manifest_sha256": exclusion_manifest_sha256,
        "tasks": merged_metadata,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--reference-bundle", type=Path)
    parser.add_argument("--excluded-uids-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_bundles(
        data_root=args.data_root,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
        reference_bundle=args.reference_bundle,
        excluded_uids_manifest=args.excluded_uids_manifest,
        seed=args.seed,
        skip_download=args.skip_download,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
