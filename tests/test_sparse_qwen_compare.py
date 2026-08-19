import json
import tempfile
import unittest
from pathlib import Path

from contiguous_fuxian.sparse_qwen_compare import load_sparse_summary


class SparseQwenCompareTest(unittest.TestCase):
    def test_loads_direct_summary_from_sparse_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "summary.json").write_text('{"tasks": {}, "overall": {}}', encoding="utf-8")

            self.assertEqual(load_sparse_summary(path), {"tasks": {}, "overall": {}})

    def test_derives_mean_ssd_reads_from_scored_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "summary.json").write_text(
                '{"tasks": {"rte": {"samples": 2}}, "overall": {}}', encoding="utf-8"
            )
            rows = [
                {"task": "rte", "physical_prefetch_kv_bytes": 100.0, "prefetch_disk_source_fraction": 0.25, "total_ssd_read_bytes": 45.0},
                {"task": "rte", "physical_prefetch_kv_bytes": 200.0, "prefetch_disk_source_fraction": 0.5, "total_ssd_read_bytes": 140.0},
            ]
            (path / "scored_records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            summary = load_sparse_summary(path)

            self.assertEqual(summary["tasks"]["rte"]["mean_ssd_prefetch_kv_bytes"], 62.5)
            self.assertEqual(summary["tasks"]["rte"]["mean_total_ssd_read_bytes"], 92.5)


if __name__ == "__main__":
    unittest.main()
