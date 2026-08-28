import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_five_method_grid.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_five_method_grid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


def percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def make_rows(
    task,
    *,
    latency,
    both_correct,
    method="contigkv",
    budget="010",
    prefix_tokens=101,
):
    requested = int(budget) / 100.0
    effective = 1.0 if method == "as_lru" else requested
    metadata = {
        "effective_mean_keep_ratio": effective,
        "selected_tokens_by_layer": [
            prefix_tokens
            if method == "as_lru"
            else max(1, math.ceil(prefix_tokens * requested))
        ]
        * 2,
        "layer_keep_ratios": [requested, requested],
    }
    if method == "as_h2o_lru":
        value_ratio = math.ceil(prefix_tokens * requested) / prefix_tokens
        metadata.update(
            {
                "effective_mean_keep_ratio": value_ratio,
                "as_h2o_full_key_ratio": 1.0,
                "as_h2o_value_keep_ratio": value_ratio,
                "as_h2o_total_logical_payload_ratio": (1.0 + value_ratio) / 2.0,
            }
        )
    rows = [
        {
            "uid": f"{task}-0",
            "task": task,
            "correct": True,
            "ttft_ms": latency,
            "logits_ready_ms": latency,
            "response_ready_ms": latency + 0.5,
        },
        {
            "uid": f"{task}-1",
            "task": task,
            "correct": both_correct,
            "ttft_ms": latency + 10.0,
            "logits_ready_ms": latency + 10.0,
            "response_ready_ms": latency + 10.5,
        },
    ]
    for row in rows:
        row.update(metadata)
    return rows


def write_run(path, *, task, budget, rows, keep_ratio=None):
    path.mkdir(parents=True)
    logits = [row["logits_ready_ms"] for row in rows]
    response = [row["response_ready_ms"] for row in rows]
    aggregate = {
        "samples": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_logits_ready_ms": sum(logits) / len(logits),
        "p95_logits_ready_ms": percentile95(logits),
        "mean_response_ready_ms": sum(response) / len(response),
        "p95_response_ready_ms": percentile95(response),
    }
    runtime = {
        "accuracy_scoring": "label_continuation_loglikelihood",
        "generation_max_tokens": 1,
        "response_ready_metric_valid_for_first_token": True,
        "response_ready_excludes_accuracy_scoring": True,
        "keep_ratio": (
            keep_ratio if keep_ratio is not None else int(budget) / 100.0
        ),
    }
    summary = {
        "runtime": runtime,
        "tasks": {task: aggregate},
        "overall": aggregate,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "scored_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def build_grid(root):
    runs = []
    correct_methods = {"promixed", "as_lru", "as_h2o_lru"}
    for task_index, task in enumerate(grid.TASKS):
        as_path = root / "runs" / task / "as_lru_full"
        write_run(
            as_path,
            task=task,
            budget="050",
            rows=make_rows(
                task,
                latency=180.0 + task_index,
                both_correct=True,
                method="as_lru",
                budget="050",
            ),
            # Shape-only plan metadata is not the actual AS+LRU retention budget.
            keep_ratio=0.05,
        )
        runs.append(
            {
                "task": task,
                "budget": "full",
                "method": "as_lru",
                "path": str(as_path.relative_to(root)),
            }
        )
        for method_index, method in enumerate(grid.METHODS):
            if method == "as_lru":
                continue
            for budget in grid.BUDGETS:
                path = root / "runs" / task / f"{method}_k{budget}"
                latency = (
                    100.0
                    + 20.0 * method_index
                    + int(budget) / 10.0
                    + task_index
                )
                write_run(
                    path,
                    task=task,
                    budget=budget,
                    rows=make_rows(
                        task,
                        latency=latency,
                        both_correct=method in correct_methods,
                        method=method,
                        budget=budget,
                    ),
                )
                runs.append(
                    {
                        "task": task,
                        "budget": budget,
                        "method": method,
                        "path": str(path.relative_to(root)),
                    }
                )
    manifest = root / "schedule_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "five-method test grid",
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    return manifest


class FiveMethodGridAnalysisTest(unittest.TestCase):
    def test_full_grid_cli_and_as_lru_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_grid(root)
            output = root / "report.json"

            self.assertEqual(
                grid.main(["--manifest", str(manifest), "--output", str(output)]),
                0,
            )

            result = json.loads(output.read_text(encoding="utf-8"))
            markdown = output.with_suffix(".md").read_text(encoding="utf-8")

        self.assertEqual(result["input"]["manifest_entries"], 68)
        self.assertEqual(result["input"]["expanded_cells"], 80)
        self.assertEqual(result["overall_by_method"]["as_lru"]["cells"], 16)
        self.assertEqual(
            result["budget_semantics_validation"]["validated_h2o_rows"], 32
        )
        self.assertAlmostEqual(
            result["overall_by_method"]["as_lru"]["accuracy"], 1.0
        )
        for budget in grid.BUDGETS:
            cell = result["tasks"]["sst2"]["budgets"][budget]["methods"]["as_lru"]
            self.assertTrue(cell["reused_full_run"])
            self.assertEqual(cell["source_budget"], "full")
            self.assertEqual(cell["budget_semantics"], "full_kv")
            self.assertAlmostEqual(cell["response_ready_mean_ms"], 185.5)
        self.assertIn("full (reused)", markdown)
        self.assertIn("value/KV-retention ratio", markdown)
        self.assertIn("Primary latency is **response-ready**", markdown)
        self.assertIn("Overall macro average", markdown)

    def test_four_as_lru_entries_may_reference_one_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_grid(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            expanded = []
            for item in payload["runs"]:
                if item["method"] == "as_lru":
                    expanded.extend(
                        [{**item, "budget": budget} for budget in grid.BUDGETS]
                    )
                else:
                    expanded.append(item)
            payload["runs"] = expanded
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            specs, info = grid.load_manifest(manifest)

        self.assertEqual(info["manifest_entries"], 80)
        as_specs = [spec for spec in specs if spec.method == "as_lru"]
        self.assertEqual(len(as_specs), 16)
        self.assertTrue(all(spec.reused_full_run for spec in as_specs))

    def test_duplicate_uid_and_wrong_retention_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "as_h2o"
            rows = make_rows(
                "trec",
                latency=100.0,
                both_correct=True,
                method="as_h2o_lru",
                budget="010",
            )
            write_run(
                path,
                task="trec",
                budget="010",
                rows=rows,
                keep_ratio=0.25,
            )
            spec = grid.RunSpec(
                task="trec",
                budget="010",
                method="as_h2o_lru",
                path=path,
                source_budget="010",
            )
            with self.assertRaisesRegex(ValueError, "keep_ratio does not match"):
                grid.load_run(spec)

            semantic_runs = []
            for task in grid.TASKS:
                full_rows = make_rows(
                    task,
                    latency=100.0,
                    both_correct=True,
                    method="as_lru",
                    budget="050",
                )
                semantic_runs.append(
                    grid.LoadedRun(
                        spec=grid.RunSpec(
                            task,
                            "005",
                            "as_lru",
                            root / task,
                            "full",
                            True,
                        ),
                        records={row["uid"]: row for row in full_rows},
                    )
                )
            bad_h2o_rows = make_rows(
                "trec",
                latency=100.0,
                both_correct=True,
                method="as_h2o_lru",
                budget="010",
            )
            bad_h2o_rows[0]["as_h2o_total_logical_payload_ratio"] += 0.01
            semantic_runs.append(
                grid.LoadedRun(
                    spec=grid.RunSpec(
                        "trec",
                        "010",
                        "as_h2o_lru",
                        root / "bad_h2o",
                        "010",
                    ),
                    records={row["uid"]: row for row in bad_h2o_rows},
                )
            )
            with self.assertRaisesRegex(ValueError, "logical payload"):
                grid.validate_budget_semantics(semantic_runs)

            write_run(
                root / "duplicate",
                task="trec",
                budget="010",
                rows=rows,
            )
            records_path = root / "duplicate" / "scored_records.jsonl"
            records_path.write_text(
                json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate UID"):
                grid.read_records(
                    records_path, task="trec", context="duplicate test records"
                )


if __name__ == "__main__":
    unittest.main()
