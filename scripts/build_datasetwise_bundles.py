#!/usr/bin/env python3
"""Build full, independently evaluated paper-task bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from contiguous_fuxian.paper_tasks import (
    PAPER_TASKS,
    ensure_paper_datasets,
    load_task_rows,
    write_paper_task_bundles,
)


TASKS = ("sst2", "subj", "trec", "rte")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def build_datasetwise_bundles(
    *,
    data_root: Path,
    output_dir: Path,
    tokenizer: str,
    reference_bundle: Path | None,
    excluded_uids_manifest: Path | None,
    seed: int,
    skip_download: bool,
) -> dict[str, Any]:
    if not skip_download:
        ensure_paper_datasets(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_counts = {
        task: len(load_task_rows(task, data_root, seed)[1]) for task in TASKS
    }
    generated = write_paper_task_bundles(
        output_dir,
        data_root,
        TASKS,
        seed=seed,
        eval_samples=max(evaluation_counts.values()),
        tokenizer_path=tokenizer,
        qwen_chat_format=True,
    )

    excluded_by_task = {task: set() for task in TASKS}
    exclusion_sha256 = None
    if excluded_uids_manifest is not None:
        exclusion_bytes = excluded_uids_manifest.read_bytes()
        exclusion_sha256 = hashlib.sha256(exclusion_bytes).hexdigest()
        exclusion = json.loads(exclusion_bytes)
        for task in TASKS:
            excluded_by_task[task] = {
                str(uid)
                for uid in exclusion["tasks"][task].get("history_uids", ())
            }

    task_metadata: dict[str, Any] = {}
    for task in TASKS:
        path = output_dir / f"{task}.jsonl"
        all_rows = _read_jsonl(path)
        if len(all_rows) != evaluation_counts[task]:
            raise ValueError(
                f"{task} generated {len(all_rows)} rows; "
                f"expected {evaluation_counts[task]}"
            )
        rows = [
            {**row, "label_continuation_prefix": ""}
            for row in all_rows
            if str(row["uid"]) not in excluded_by_task[task]
        ]
        if not rows:
            raise ValueError(f"{task} has no evaluation rows after exclusions")
        if len({str(row["uid"]) for row in rows}) != len(rows):
            raise ValueError(f"{task} contains duplicate evaluation UIDs")
        prefix_sha256 = _sha256_text(rows[0]["prefix_text"])
        if any(_sha256_text(row["prefix_text"]) != prefix_sha256 for row in rows):
            raise ValueError(f"{task} does not use one shared prefix")

        reference_records = 0
        if reference_bundle is not None:
            reference_rows = _read_jsonl(reference_bundle / f"{task}.jsonl")
            reference_records = len(reference_rows)
            if not reference_rows:
                raise ValueError(f"the {task} reference bundle is empty")
            if _sha256_text(reference_rows[0]["prefix_text"]) != prefix_sha256:
                raise ValueError(f"the {task} shared prefix changed")
            comparable = {
                str(row["uid"]): row for row in rows
            }
            for reference in reference_rows:
                current = comparable.get(str(reference["uid"]))
                if current is None:
                    raise ValueError(
                        f"reference UID {reference['uid']} is absent from {task}"
                    )
                expected = {**reference, "label_continuation_prefix": ""}
                if current != expected:
                    raise ValueError(
                        f"reference UID {reference['uid']} changed in {task}"
                    )

        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        task_metadata[task] = {
            **generated[task],
            "records": len(rows),
            "prefix_fewshot_examples": PAPER_TASKS[task].fewshot_examples,
            "source_evaluation_rows": evaluation_counts[task],
            "excluded_calibration_uids": sorted(excluded_by_task[task]),
            "evaluation_requests": len(rows),
            "prefix_text_sha256": prefix_sha256,
            "validated_reference_records": reference_records,
            "accuracy_protocol": "label_continuation_loglikelihood",
            "label_continuation_prefix": "",
        }

    payload = {
        "schema_version": 3,
        "seed": seed,
        "evaluation_mode": "each dataset is an independent workload",
        "pooled_headline_metrics_allowed": False,
        "prefix_fewshot_examples_by_task": {
            task: PAPER_TASKS[task].fewshot_examples for task in TASKS
        },
        "evaluation_requests_by_task": {
            task: task_metadata[task]["evaluation_requests"] for task in TASKS
        },
        "excluded_uids_manifest": (
            str(excluded_uids_manifest)
            if excluded_uids_manifest is not None
            else None
        ),
        "excluded_uids_manifest_sha256": exclusion_sha256,
        "tasks": task_metadata,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
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
    payload = build_datasetwise_bundles(
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
