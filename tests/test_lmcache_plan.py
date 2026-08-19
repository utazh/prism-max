import unittest

from contiguous_fuxian.lmcache_plan import convert_probe_to_lmcache_plan, policy_request_id


class LMCachePlanTest(unittest.TestCase):
    def test_policy_request_id_matches_runtime_convention(self):
        self.assertEqual(policy_request_id("contiguous-2"), "cmpl-contiguous-2-score")

    def test_probe_grid_converts_to_layer_and_chunk_runtime_plans(self):
        probe = {
            "model_path": "/data1/llm/Qwen/Qwen2.5-7B-Instruct",
            "contiguous_chunk_size": 16,
            "keep_ratio": 0.05,
            "period_size": 8,
            "subperiod_size": 4,
            "records": [
                {
                    "prompt_index": 7,
                    "prefix_tokens": 64,
                    "layer_plan": [["keep", "drop", "drop"], ["drop", "keep", "drop"]],
                }
            ],
        }

        plan = convert_probe_to_lmcache_plan(probe, uid_prefix="paper")

        request_id = "cmpl-paper-7-score"
        self.assertEqual(
            plan["layer_request_prefixes"][request_id],
            [["int8", "drop", "drop"], ["drop", "int8", "drop"]],
        )
        self.assertEqual(plan["request_prefixes"][request_id], ["base", "base", "drop"])
        self.assertEqual(plan["metadata"]["runtime_tier_counts"], {"int8": 2, "drop": 4})

    def test_no_drop_variant_maps_unselected_chunks_to_int4(self):
        probe = {"records": [{"layer_plan": [["keep", "drop"]]}]}

        plan = convert_probe_to_lmcache_plan(probe, unselected_tier="int4")

        self.assertEqual(
            plan["layer_request_prefixes"]["cmpl-contiguous-0-score"],
            [["int8", "int4"]],
        )
        self.assertEqual(plan["request_prefixes"]["cmpl-contiguous-0-score"], ["base", "base"])

    def test_rejects_non_rectangular_plan(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            convert_probe_to_lmcache_plan(
                {"records": [{"layer_plan": [["keep"], ["keep", "drop"]]}]}
            )


if __name__ == "__main__":
    unittest.main()
