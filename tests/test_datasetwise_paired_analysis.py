import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "analyze_datasetwise_paired.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_datasetwise_paired", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_run(root: Path, name: str, ttfts, correct) -> None:
    run = root / "sst2" / name
    run.mkdir(parents=True)
    rows = [
        {
            "uid": f"sst2-{index}",
            "task": "sst2",
            "ttft_ms": ttft,
            "correct": is_correct,
            "total_ssd_read_bytes": 100.0 if "contigkv" in name else 50.0,
        }
        for index, (ttft, is_correct) in enumerate(zip(ttfts, correct))
    ]
    summary = {
        "tasks": {"sst2": {"samples": len(rows)}},
        "runtime": {
            "accuracy_scoring": "label_continuation_loglikelihood",
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "scored_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class DatasetwisePairedAnalysisTest(unittest.TestCase):
    def test_analysis_stays_paired_and_dataset_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "k005_contigkv", [100.0, 120.0], [False, True])
            write_run(root, "k005_ours", [80.0, 90.0], [True, True])

            comparisons = MODULE.analyze(MODULE.load_runs(root))
            result = comparisons["sst2"]["005"]["ours_vs_contigkv"]
            report = MODULE.render_report(comparisons)

        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["wrong_to_correct"], 1)
        self.assertEqual(result["correct_to_wrong"], 0)
        self.assertEqual(result["candidate_faster_requests"], 2)
        self.assertAlmostEqual(result["accuracy_delta_pp"], 50.0)
        self.assertEqual(MODULE.percentile95_nearest_rank([1, 2, 3, 4]), 4.0)
        self.assertIn("## SST2", report)
        self.assertNotIn("Overall", report)

    def test_strict_holdout_excludes_calibration_uid_before_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "k005_contigkv", [100.0, 120.0], [True, False])
            write_run(root, "k005_ours", [80.0, 90.0], [False, True])

            runs = MODULE.load_runs(root, {"sst2": {"sst2-0"}})
            result = MODULE.analyze(runs)["sst2"]["005"]["ours_vs_contigkv"]

        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["wrong_to_correct"], 1)
        self.assertEqual(result["correct_to_wrong"], 0)
        self.assertEqual(result["candidate_faster_requests"], 1)


if __name__ == "__main__":
    unittest.main()
