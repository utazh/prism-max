import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "analyze_datasetwise_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_datasetwise_matrix", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _summary(task, *, accuracy, mean, p95, period=1):
    return {
        "tasks": {
            task: {
                "samples": 10,
                "accuracy": accuracy,
                "mean_ttft_ms": mean,
                "p95_ttft_ms": p95,
                "mean_selector_calls": 4.0,
                "mean_effective_keep_ratio": 0.05,
            }
        },
        "runtime": {
            "accuracy_scoring": "label_first_token_logit",
            "runtime_variant": "test",
            "impress_selection_period_size": period,
        },
    }


class DatasetwiseAnalysisTest(unittest.TestCase):
    def test_report_keeps_datasets_separate_and_computes_ours_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("sst2", "rte"):
                for name, summary in (
                    (
                        "k005_contigkv",
                        _summary(task, accuracy=0.8, mean=100.0, p95=120.0),
                    ),
                    (
                        "k005_ours",
                        _summary(
                            task,
                            accuracy=0.9,
                            mean=90.0,
                            p95=108.0,
                            period=8,
                        ),
                    ),
                ):
                    run = root / task / name
                    run.mkdir(parents=True)
                    (run / "summary.json").write_text(
                        json.dumps(summary), encoding="utf-8"
                    )

            results = MODULE.load_results(root)
            report = MODULE.build_report(results)

        self.assertEqual(set(results), {"sst2", "rte"})
        self.assertIn("## SST2", report)
        self.assertIn("## RTE", report)
        self.assertIn("| 5% | Ours | 10 | 0.9000 | 5.00% | +0.00 |", report)
        self.assertIn("| 5% | +10.00 | +10.00% | +10.00% |", report)
        self.assertNotIn("Overall", report)
        self.assertNotIn("pooled accuracy", report.lower())

    def test_loader_rejects_a_summary_containing_multiple_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "sst2" / "k005_ours"
            run.mkdir(parents=True)
            summary = _summary("sst2", accuracy=1.0, mean=1.0, p95=1.0)
            summary["tasks"]["rte"] = summary["tasks"]["sst2"]
            (run / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "not an independent"):
                MODULE.load_results(root)

    def test_strict_holdout_recomputes_metrics_from_retained_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "sst2" / "k005_ours"
            run.mkdir(parents=True)
            summary = _summary("sst2", accuracy=0.5, mean=150.0, p95=200.0)
            summary["tasks"]["sst2"]["samples"] = 2
            (run / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            rows = [
                {
                    "uid": "sst2-0",
                    "correct": False,
                    "ttft_ms": 200.0,
                    "selector_calls": 4,
                    "effective_mean_keep_ratio": 0.04,
                },
                {
                    "uid": "sst2-1",
                    "correct": True,
                    "ttft_ms": 100.0,
                    "selector_calls": 2,
                    "effective_mean_keep_ratio": 0.06,
                },
            ]
            (run / "scored_records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            results = MODULE.load_results(root, {"sst2": {"sst2-0"}})
            metric = results["sst2"]["005_ours"]

        self.assertEqual(metric["samples"], 1)
        self.assertEqual(metric["accuracy"], 1.0)
        self.assertEqual(metric["mean_ttft_ms"], 100.0)
        self.assertEqual(metric["p95_ttft_ms"], 100.0)
        self.assertEqual(metric["mean_selector_calls"], 2.0)
        self.assertEqual(metric["mean_effective_keep_ratio"], 0.06)
        self.assertAlmostEqual(metric["observed_budget_delta_pp"], 1.0)


if __name__ == "__main__":
    unittest.main()
