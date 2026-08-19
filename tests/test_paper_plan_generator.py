import unittest

from contiguous_fuxian.paper_plan_generator import (
    _head_prefix_scores,
    build_lmcache_payload,
    build_sparse_layer_plan,
    impress_probe_token_selection,
    impress_probe_token_selection_with_ranking,
    impress_similarity_threshold,
    tokens_to_chunks,
)


class PaperPlanGeneratorTest(unittest.TestCase):
    def test_head_prefix_scores_rejects_non_finite_attention(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local unit-test runtime")

        attention = torch.ones((1, 2, 2, 4), dtype=torch.float32)
        attention[0, 0, 1, 0] = float("nan")

        with self.assertRaisesRegex(RuntimeError, "layer 0.*non-finite"):
            _head_prefix_scores([attention], prefix_len=3, query_len=1)

    def test_impress_uses_probe_consensus_when_heads_are_similar(self):
        selected, used_probe, similarity = impress_probe_token_selection(
            [[0.9, 0.8, 0.1], [0.8, 0.9, 0.2]],
            keep_tokens=2,
            probe_heads=[0, 1],
            similarity_alpha=0.6,
        )

        self.assertTrue(used_probe)
        self.assertEqual(selected, {0, 1})
        self.assertEqual(similarity, 1.0)

    def test_impress_falls_back_to_full_prefix_when_probe_sets_diverge(self):
        selected, used_probe, _ = impress_probe_token_selection(
            [[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]],
            keep_tokens=1,
            probe_heads=[0, 1],
            similarity_alpha=0.6,
        )

        self.assertFalse(used_probe)
        self.assertEqual(selected, {0, 1, 2})

    def test_impress_prefetch_ranking_does_not_change_selected_set(self):
        selected, priority, used_probe, similarity = (
            impress_probe_token_selection_with_ranking(
                [[0.8, 0.1, 0.9], [0.7, 0.0, 1.0]],
                keep_tokens=2,
                probe_heads=[0, 1],
                similarity_alpha=0.6,
            )
        )

        self.assertTrue(used_probe)
        self.assertEqual(similarity, 1.0)
        self.assertEqual(selected, {0, 2})
        self.assertEqual(priority, [2, 0])

    def test_impress_similarity_threshold_uses_retention_ratio(self):
        self.assertAlmostEqual(
            impress_similarity_threshold(total_tokens=100, keep_tokens=5, alpha=0.6),
            (5 / 195) ** 0.6,
        )

    def test_impress_uses_probe_when_similarity_equals_threshold(self):
        selected, used_probe, similarity = impress_probe_token_selection(
            [[1.0], [1.0]],
            keep_tokens=1,
            probe_heads=[0, 1],
            similarity_alpha=0.6,
        )

        self.assertTrue(used_probe)
        self.assertEqual(selected, {0})
        self.assertEqual(similarity, 1.0)

    def test_tokens_to_chunks_models_64_token_impress_io_blocks(self):
        self.assertEqual(tokens_to_chunks({1, 63, 64, 127, 128}, 64, 129), {0, 1, 2})

    def test_runtime_payload_exposes_layer_plan_and_active_chunk_projection(self):
        plan = build_sparse_layer_plan([{0}, {1}], num_chunks=3)
        payload = build_lmcache_payload("contigkv", {"cmpl-x-score": plan}, [], {})

        self.assertEqual(payload["request_prefixes"]["cmpl-x-score"], ["base", "base", "drop"])
        self.assertEqual(payload["metadata"]["runtime_tier_counts"], {"int8": 2, "drop": 4})

    def test_impress_payload_preserves_useful_token_selections(self):
        plan = build_sparse_layer_plan([{0}], num_chunks=2)
        payload = build_lmcache_payload(
            "impress",
            {"cmpl-x-score": plan},
            [],
            {},
            layer_token_selections={"cmpl-x-score": [[1, 3]]},
        )

        self.assertEqual(payload["layer_token_selections"]["cmpl-x-score"], [[1, 3]])

    def test_contiguous_payload_preserves_attention_scores_for_cache_manager(self):
        plan = build_sparse_layer_plan([{0}], num_chunks=2)
        payload = build_lmcache_payload(
            "contigkv",
            {"cmpl-x-score": plan},
            [],
            {},
            layer_chunk_attention_scores={"cmpl-x-score": [[0.75, 0.25]]},
        )

        self.assertEqual(
            payload["layer_chunk_attention_scores"]["cmpl-x-score"],
            [[0.75, 0.25]],
        )

    def test_layer_plan_can_match_layer_specific_warm_codecs(self):
        plan = build_sparse_layer_plan(
            [{0}, {1}],
            num_chunks=3,
            selected_tiers=("int8", "int4"),
        )

        self.assertEqual(plan, [["int8", "drop", "drop"], ["drop", "int4", "drop"]])


if __name__ == "__main__":
    unittest.main()
