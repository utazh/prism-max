"""Average forward/reverse PRISM-Gao screening runs into one compact table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_run(summary_path: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    if len(tasks) != 1:
        raise ValueError(f"expected one task in {summary_path}")
    task, summary = next(iter(tasks.items()))
    records_path = summary_path.parent / "scored_records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return task, summary, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    for path in sorted(args.root.rglob("summary.json")):
        task, summary, records = _load_run(path)
        name = path.parent.name
        for suffix in ("_forward", "_reverse"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        groups[(task, name)].append((summary, records))

    rows = []
    for (task, name), runs in sorted(groups.items()):
        summaries = [row[0] for row in runs]
        records = [record for _, rows_ in runs for record in rows_]

        def summary_mean(field: str, default: float = 0.0) -> float:
            return sum(float(row.get(field, default)) for row in summaries) / len(summaries)

        def record_mean(field: str, default: float = 0.0) -> float:
            return sum(float(row.get(field, default)) for row in records) / max(1, len(records))

        rows.append(
            {
                "task": task,
                "run": name,
                "orders": len(runs),
                "samples_total": len(records),
                "accuracy": summary_mean("accuracy"),
                "response_ready_ms": summary_mean(
                    "mean_response_ready_ms", summary_mean("mean_ttft_ms")
                ),
                "selector_calls": summary_mean("mean_selector_calls"),
                "mean_period": summary_mean("mean_promixed_period"),
                "payload_byte_ratio": record_mean("prism_gao_payload_byte_ratio"),
                "payload_read_ms": record_mean("prism_gao_payload_read_ms"),
                "host_wait_ms": record_mean("prism_gao_host_prefetch_wait_ms"),
                "materialize_ms": record_mean("prism_gao_materialize_ms"),
                "pread_calls": record_mean("prism_gao_payload_pread_calls"),
            }
        )

    text = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
