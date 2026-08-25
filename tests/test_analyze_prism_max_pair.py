import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_prism_max_pair.py"
SPEC = importlib.util.spec_from_file_location("analyze_prism_max_pair", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pair
SPEC.loader.exec_module(pair)


class PrismMaxPairAnalysisTest(unittest.TestCase):
    def test_bundle_uid0_preexclusion_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.json"
            path.write_text(
                json.dumps(
                    {
                        "strict_eval_filter": {
                            "schema_version": 1,
                            "exclusions_manifest": "/bundle/strict.json",
                            "exclusions_manifest_sha256": "a" * 64,
                            "excluded_uids_by_task": {
                                "subj": ["subj-0"],
                                "trec": ["trec-0"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = pair.load_bundle_preexclusions(path, task="trec")

        self.assertTrue(result["preapplied"])
        self.assertEqual(result["excluded_uids_by_task"]["trec"], ["trec-0"])
        self.assertEqual(result["manifest_path"], "/bundle/strict.json")
        self.assertEqual(len(result["metadata_sha256"]), 64)

    def test_response_ready_is_primary_and_logits_ready_is_phase(self):
        provenance = {
            "preapplied": True,
            "excluded_uids_by_task": {"trec": ["trec-0"]},
        }
        result = {
            "protocol": {
                "task": "trec",
                "budget": "010",
                "backend": "k4",
                "score_mode": "nodefer",
                "unique_uids": 1,
                "run_order": [
                    "reference_r1",
                    "candidate_r1",
                    "candidate_r2",
                    "reference_r2",
                ],
            },
            "accuracy": {
                "reference_accuracy": 0.5,
                "candidate_accuracy": 0.6,
                "accuracy_delta_pp": 10.0,
                "wrong_to_correct": 1,
                "correct_to_wrong": 0,
                "mcnemar_two_sided_p": 1.0,
            },
            "input_bundle_preexclusions": provenance,
            "ready_metrics": {},
        }
        for index, metric in enumerate(pair.READY_METRICS):
            result["ready_metrics"][metric] = {
                "reference_mean_ms": 10.0 + index,
                "reference_p95_ms": 11.0 + index,
                "candidate_mean_ms": 9.0 + index,
                "candidate_p95_ms": 10.0 + index,
                "mean_reduction_percent": 10.0,
                "p95_reduction_percent": 9.0,
                "paired_bootstrap_mean_delta_95ci_ms": [-2.0, -1.0],
                "candidate_faster_uids": 1,
                "total_uids": 1,
            }

        markdown = pair.render_markdown(result)

        self.assertLess(
            markdown.index("Response ready (primary)"),
            markdown.index("Logits ready (phase)"),
        )
        self.assertIn("No analysis-time exclusion manifest", markdown)


if __name__ == "__main__":
    unittest.main()
