import unittest

from contiguous_fuxian.sparse_qwen_rescore import rebuild_summary


class SparseQwenRescoreTest(unittest.TestCase):
    def test_rebuild_summary_preserves_measurement_and_updates_metrics(self):
        records = [
            {
                "task": "trec",
                "answer": "HUM",
                "prediction": "Type: HUM",
                "correct": True,
                "ttft_ms": 10,
                "selected_kv_bytes": 4,
            },
            {
                "task": "trec",
                "answer": "LOC",
                "prediction": "DESC",
                "correct": False,
                "ttft_ms": 20,
                "selected_kv_bytes": 6,
            },
        ]
        summary = rebuild_summary(
            records,
            {"tasks": {"trec": {}}, "overall": {}, "measurement": "test"},
        )

        self.assertEqual(summary["tasks"]["trec"]["accuracy"], 0.5)
        self.assertEqual(summary["overall"]["mean_ttft_ms"], 15.0)
        self.assertEqual(summary["measurement"], "test")


if __name__ == "__main__":
    unittest.main()
