"""Re-score completed sparse-Qwen records with template-aware label matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .paper_client import percentile95, prediction_is_correct


def rebuild_summary(records: list[dict[str, Any]], prior: dict[str, Any]) -> dict[str, Any]:
    tasks = list(prior.get("tasks", {}))
    task_summaries = {}
    for task in tasks:
        task_rows = [row for row in records if row["task"] == task]
        ttfts = [float(row["ttft_ms"]) for row in task_rows]
        task_summaries[task] = {
            "samples": len(task_rows),
            "accuracy": sum(bool(row["correct"]) for row in task_rows) / max(1, len(task_rows)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
            "mean_selected_kv_bytes": sum(float(row["selected_kv_bytes"]) for row in task_rows)
            / max(1, len(task_rows)),
        }
    ttfts = [float(row["ttft_ms"]) for row in records]
    return {
        **{key: value for key, value in prior.items() if key not in {"tasks", "overall"}},
        "tasks": task_summaries,
        "overall": {
            "samples": len(records),
            "accuracy": sum(bool(row["correct"]) for row in records) / max(1, len(records)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
        },
    }


def rescore_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    records_path = root / "scored_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    for row in records:
        row["correct"] = prediction_is_correct(str(row["prediction"]), str(row["answer"]))
    prior = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary = rebuild_summary(records, prior)
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score sparse Qwen output records.")
    parser.add_argument("--run-dir", action="append", required=True)
    args = parser.parse_args()
    outputs = {str(path): rescore_run(path)["overall"] for path in args.run_dir}
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
