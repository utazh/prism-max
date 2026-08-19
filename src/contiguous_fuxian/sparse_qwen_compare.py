"""Compare direct-output sparse Qwen ContiguousKV and IMPRESS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paper_compare import compare_summaries


def load_sparse_summary(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records_path = root / "scored_records.jsonl"
    if not records_path.is_file():
        return summary
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for task, metrics in summary.get("tasks", {}).items():
        task_rows = [row for row in records if row.get("task") == task]
        if not task_rows:
            continue
        if all("critical_ssd_read_bytes" in row for row in task_rows):
            metrics["mean_ssd_prefetch_kv_bytes"] = sum(
                float(row["critical_ssd_read_bytes"]) for row in task_rows
            ) / len(task_rows)
        elif all(
            "physical_prefetch_kv_bytes" in row and "prefetch_disk_source_fraction" in row
            for row in task_rows
        ):
            metrics["mean_ssd_prefetch_kv_bytes"] = sum(
                float(row["physical_prefetch_kv_bytes"])
                * float(row["prefetch_disk_source_fraction"])
                for row in task_rows
            ) / len(task_rows)
        if all("total_ssd_read_bytes" in row for row in task_rows):
            metrics["mean_total_ssd_read_bytes"] = sum(
                float(row["total_ssd_read_bytes"]) for row in task_rows
            ) / len(task_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare sparse Qwen ContiguousKV and IMPRESS summaries.")
    parser.add_argument("--contiguous-run", required=True)
    parser.add_argument("--impress-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_summaries(
        load_sparse_summary(args.contiguous_run), load_sparse_summary(args.impress_run)
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
