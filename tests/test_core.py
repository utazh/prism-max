import math
import unittest

from contiguous_fuxian.core import (
    AttentionGuidedCache,
    build_layer_chunk_plan,
    chunk_bounds,
    contiguous_chunk_scores,
    inter_period_prefetch_delta,
    jaccard_similarity,
    select_period_chunks,
    select_top_chunks,
)


class ContiguousKVCoreTest(unittest.TestCase):
    def test_chunk_bounds_cover_prefix_with_short_tail(self):
        self.assertEqual(chunk_bounds(length=10, chunk_size=4), [(0, 4), (4, 8), (8, 10)])


    def test_contiguous_chunk_scores_sum_token_scores_inside_each_chunk(self):
        scores = [0.5, 1.5, 10.0, 2.0, 3.0]

        self.assertEqual(contiguous_chunk_scores(scores, chunk_size=2), [2.0, 12.0, 3.0])


    def test_select_top_chunks_keeps_high_scores_and_recent_tail(self):
        got = select_top_chunks(
            scores=[0.1, 0.9, 0.2, 0.05, 0.4],
            keep_chunks=2,
            recent_keep_chunks=1,
        )

        self.assertEqual(got, {1, 4})


    def test_select_period_chunks_reuses_first_layer_indices_inside_period(self):
        layer_scores = [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.2],
            [0.3, 0.4, 0.9],
            [0.7, 0.2, 0.1],
        ]

        got = select_period_chunks(layer_scores, keep_chunks=1, period_size=2)

        self.assertEqual(got, [{1}, {1}, {2}, {2}])


    def test_inter_period_prefetch_delta_reuses_previous_period_and_loads_missing(self):
        prefetched, missing = inter_period_prefetch_delta(
            previous_period_chunks={1, 2, 5},
            current_period_chunks={2, 3, 5},
        )

        self.assertEqual(prefetched, {2, 5})
        self.assertEqual(missing, {3})


    def test_build_layer_chunk_plan_drops_unimportant_chunks_per_layer(self):
        layer_chunks = [{0, 2}, {2}]

        self.assertEqual(
            build_layer_chunk_plan(num_chunks=4, layer_chunks=layer_chunks),
            [
                ["keep", "drop", "keep", "drop"],
                ["drop", "drop", "keep", "drop"],
            ],
        )


    def test_attention_guided_cache_evicts_lowest_importance_frequency_product(self):
        cache = AttentionGuidedCache(capacity=2)
        cache.touch(chunk_id=("l0", 0), attention_score=0.5)
        cache.touch(chunk_id=("l0", 1), attention_score=0.2)
        cache.touch(chunk_id=("l0", 0), attention_score=0.5)
        evicted = cache.touch(chunk_id=("l0", 2), attention_score=0.4)

        self.assertEqual(evicted, [("l0", 1)])
        self.assertEqual(set(cache.resident_chunks()), {("l0", 0), ("l0", 2)})
        self.assertTrue(math.isclose(cache.score(("l0", 0)), 2.0))


    def test_jaccard_similarity_handles_empty_sets_and_overlap(self):
        self.assertEqual(jaccard_similarity(set(), set()), 1.0)
        self.assertAlmostEqual(jaccard_similarity({1, 2}, {2, 3}), 1 / 3)


if __name__ == "__main__":
    unittest.main()
