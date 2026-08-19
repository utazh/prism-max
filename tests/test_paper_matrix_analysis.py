import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_paper_matrix.py"
SPEC = importlib.util.spec_from_file_location("analyze_paper_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PaperMatrixAnalysisTest(unittest.TestCase):
    def test_percentile_interpolates_and_mcnemar_is_exact(self):
        self.assertEqual(MODULE.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(MODULE.exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(MODULE.exact_mcnemar_p(3, 0), 0.25)

    def test_paired_comparison_uses_candidate_minus_baseline(self):
        def make_run(display, values, correct):
            path = Path(tempfile.gettempdir()) / display
            records = {
                f"x-{index}": {
                    "uid": f"x-{index}",
                    "ttft_ms": value,
                    "correct": outcome,
                }
                for index, (value, outcome) in enumerate(zip(values, correct))
            }
            return MODULE.Run(
                spec={"display": display, "budget": 0.25},
                path=path,
                summary={},
                records=records,
            )

        baseline = make_run("A", [100.0, 200.0], [False, True])
        candidate = make_run("B", [80.0, 180.0], [True, True])
        result = MODULE.compare(baseline, candidate)

        self.assertEqual(result["mean_paired_delta_ms"], -20.0)
        self.assertEqual(result["accuracy_delta_points"], 50.0)
        self.assertEqual(result["wrong_to_correct"], 1)
        self.assertEqual(result["correct_to_wrong"], 0)

    def test_expected_sample_counts_accepts_paper_task_counts(self):
        manifest = {
            "samples": 410,
            "tasks": ["sst2", "subj", "trec", "rte"],
            "samples_by_task": {
                "sst2": 100,
                "subj": 110,
                "trec": 120,
                "rte": 80,
            },
        }

        self.assertEqual(
            MODULE.expected_sample_counts(manifest),
            manifest["samples_by_task"],
        )


if __name__ == "__main__":
    unittest.main()
