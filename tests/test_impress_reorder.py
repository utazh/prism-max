import json
import tempfile
import unittest
from pathlib import Path

from contiguous_fuxian.impress_reorder import (
    importance_order,
    invert_permutation,
    load_task_reorder,
)
from contiguous_fuxian.sparse_qwen_reprefill import PrefixStoreInfo


class ImpressReorderTest(unittest.TestCase):
    def test_importance_order_is_descending_and_stable(self):
        self.assertEqual(importance_order([0.2, 0.9, 0.9, 0.1]), [1, 2, 0, 3])

    def test_importance_order_rejects_nonfinite_or_negative_scores(self):
        for scores in ([], [float("nan")], [-0.1]):
            with self.assertRaises(ValueError):
                importance_order(scores)

    def test_invert_permutation_round_trips_logical_positions(self):
        physical_to_logical = [2, 0, 3, 1]
        logical_to_physical = invert_permutation(physical_to_logical)

        self.assertEqual(logical_to_physical, [1, 3, 0, 2])
        self.assertEqual(
            [physical_to_logical[position] for position in logical_to_physical],
            [0, 1, 2, 3],
        )

    def test_invert_permutation_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            invert_permutation([0, 0, 2])

    def test_manifest_validates_store_geometry_and_builds_inverse(self):
        info = PrefixStoreInfo("rte", 4, 4, 128, 2, "token-hash")
        payload = {
            "schema_version": 1,
            "method": "impress",
            "tasks": {
                "rte": {
                    "prefix_tokens": 4,
                    "layers": 2,
                    "token_hash": "token-hash",
                    "physical_to_logical": [[2, 0, 3, 1], [1, 3, 0, 2]],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reorder.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            physical, logical = load_task_reorder(path, task="rte", info=info)

        self.assertEqual(physical[0], [2, 0, 3, 1])
        self.assertEqual(logical[0], [1, 3, 0, 2])
        self.assertEqual(logical[1], [2, 0, 3, 1])

    def test_manifest_rejects_wrong_token_hash(self):
        info = PrefixStoreInfo("rte", 2, 4, 128, 1, "expected")
        payload = {
            "schema_version": 1,
            "method": "impress",
            "tasks": {
                "rte": {
                    "prefix_tokens": 2,
                    "layers": 1,
                    "token_hash": "wrong",
                    "physical_to_logical": [[0, 1]],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reorder.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "token_hash"):
                load_task_reorder(path, task="rte", info=info)


if __name__ == "__main__":
    unittest.main()
