import unittest

import torch

from contiguous_fuxian.as_baselines import (
    scatter_h2o_selected_values,
    select_h2o_gqa_value_positions,
)


class H2OGQASelectionTest(unittest.TestCase):
    def test_sums_attention_mass_within_each_gqa_group(self):
        scores = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ]
        )

        selected = select_h2o_gqa_value_positions(
            scores,
            num_query_heads=4,
            num_kv_heads=2,
            keep_ratio=0.5,
        )

        self.assertEqual(selected.dtype, torch.long)
        self.assertTrue(torch.equal(selected, torch.tensor([[1, 0], [2, 3]])))

    def test_uses_ceiling_for_the_per_head_budget(self):
        selected = select_h2o_gqa_value_positions(
            torch.arange(10, dtype=torch.float32).reshape(2, 5),
            num_query_heads=2,
            num_kv_heads=2,
            keep_ratio=0.21,
        )

        self.assertEqual(tuple(selected.shape), (2, 2))

    def test_ties_use_ascending_token_ids_deterministically(self):
        scores = torch.ones((4, 6), dtype=torch.float32)
        calls = [
            select_h2o_gqa_value_positions(
                scores,
                num_query_heads=4,
                num_kv_heads=2,
                keep_ratio=0.5,
            )
            for _ in range(2)
        ]

        expected = torch.tensor([[0, 1, 2], [0, 1, 2]])
        self.assertTrue(all(torch.equal(result, expected) for result in calls))

    def test_rejects_invalid_shape_counts_ratio_and_score_values(self):
        valid = torch.ones((4, 5), dtype=torch.float32)
        cases = (
            (valid.reshape(1, 4, 5), 4, 2, 0.5),
            (valid.to(torch.int64), 4, 2, 0.5),
            (valid, 3, 2, 0.5),
            (valid, 4, 3, 0.5),
            (valid, 4, 2, 0.0),
            (valid, 4, 2, 1.01),
            (valid, 4, 2, float("nan")),
        )
        for scores, query_heads, kv_heads, ratio in cases:
            with self.subTest(
                shape=tuple(scores.shape),
                query_heads=query_heads,
                kv_heads=kv_heads,
                ratio=ratio,
            ), self.assertRaises(ValueError):
                select_h2o_gqa_value_positions(
                    scores,
                    num_query_heads=query_heads,
                    num_kv_heads=kv_heads,
                    keep_ratio=ratio,
                )

        bad_scores = (
            torch.tensor([[1.0, float("nan")], [1.0, 1.0]]),
            torch.tensor([[1.0, -0.1], [1.0, 1.0]]),
            torch.empty((2, 0), dtype=torch.float32),
        )
        for scores in bad_scores:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_h2o_gqa_value_positions(
                    scores,
                    num_query_heads=2,
                    num_kv_heads=1,
                    keep_ratio=0.5,
                )

        with self.assertRaises(TypeError):
            select_h2o_gqa_value_positions(
                [[1.0, 2.0]],
                num_query_heads=1,
                num_kv_heads=1,
                keep_ratio=0.5,
            )


class H2OValueScatterTest(unittest.TestCase):
    def test_scatters_per_head_and_leaves_unselected_values_zero(self):
        full_keys = torch.ones((5, 2, 3), dtype=torch.float32)
        positions = torch.tensor([[4, 1], [4, 3]], dtype=torch.long)
        values = torch.tensor(
            [
                [[4.0, 4.1, 4.2], [40.0, 40.1, 40.2]],
                [[1.0, 1.1, 1.2], [30.0, 30.1, 30.2]],
            ],
            dtype=torch.float32,
        )

        scattered = scatter_h2o_selected_values(full_keys, values, positions)

        expected = torch.zeros_like(full_keys)
        expected[4, 0] = values[0, 0]
        expected[1, 0] = values[1, 0]
        expected[4, 1] = values[0, 1]
        expected[3, 1] = values[1, 1]
        self.assertTrue(torch.equal(scattered, expected))
        self.assertTrue(torch.equal(full_keys, torch.ones_like(full_keys)))

    def test_supports_an_empty_selection(self):
        full_keys = torch.ones((3, 2, 4), dtype=torch.float16)
        scattered = scatter_h2o_selected_values(
            full_keys,
            torch.empty((0, 2, 4), dtype=torch.float16),
            torch.empty((2, 0), dtype=torch.long),
        )

        self.assertTrue(torch.equal(scattered, torch.zeros_like(full_keys)))

    def test_rejects_duplicate_and_out_of_range_positions(self):
        full_keys = torch.zeros((4, 2, 3), dtype=torch.float32)
        values = torch.zeros((2, 2, 3), dtype=torch.float32)
        invalid = (
            torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            torch.tensor([[-1, 0], [1, 2]], dtype=torch.long),
            torch.tensor([[0, 4], [1, 2]], dtype=torch.long),
        )
        for positions in invalid:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                scatter_h2o_selected_values(full_keys, values, positions)

    def test_rejects_shape_and_dtype_mismatches(self):
        full_keys = torch.zeros((4, 2, 3), dtype=torch.float32)
        values = torch.zeros((2, 2, 3), dtype=torch.float32)
        positions = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        cases = (
            (full_keys.reshape(4, 6), values, positions),
            (full_keys, values.reshape(4, 3), positions),
            (full_keys, torch.zeros((3, 2, 3)), positions),
            (full_keys, torch.zeros((2, 3, 3)), positions),
            (full_keys, torch.zeros((2, 2, 4)), positions),
            (full_keys, values.to(torch.float16), positions),
            (full_keys, values, positions.to(torch.int32)),
            (full_keys, values, positions[:1]),
        )
        for keys, selected, selected_positions in cases:
            with self.subTest(
                key_shape=tuple(keys.shape),
                value_shape=tuple(selected.shape),
                position_shape=tuple(selected_positions.shape),
            ), self.assertRaises(ValueError):
                scatter_h2o_selected_values(keys, selected, selected_positions)


if __name__ == "__main__":
    unittest.main()
