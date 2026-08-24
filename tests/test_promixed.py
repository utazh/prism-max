import unittest

from contiguous_fuxian.promixed import select_promixed_gqa_blocks


class PromixedSelectionTest(unittest.TestCase):
    def test_coverage_reserves_a_block_for_each_gqa_group(self):
        decision = select_promixed_gqa_blocks(
            [
                [9.0, 0.0, 0.0, 1.0, 0.5, 0.1, 0.0, 0.0],
                [0.0, 8.0, 0.0, 1.0, 0.5, 0.1, 0.0, 0.0],
                [0.0, 0.0, 7.0, 1.0, 0.5, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 6.0, 0.5, 0.1, 0.0, 0.0],
            ],
            keep_blocks=4,
            coverage_fraction=1.0,
        )

        self.assertEqual(decision.selected_blocks, (0, 1, 2, 3))
        self.assertEqual(set(decision.priority_blocks), {0, 1, 2, 3})
        self.assertLess(decision.agreement, 1.0)

    def test_global_fill_uses_scores_instead_of_block_id_ties(self):
        decision = select_promixed_gqa_blocks(
            [
                [0.0, 0.0, 0.0, 1.0, 9.0, 8.0],
                [0.0, 0.0, 0.0, 1.0, 9.0, 7.0],
                [0.0, 0.0, 0.0, 1.0, 9.0, 6.0],
                [0.0, 0.0, 0.0, 1.0, 9.0, 5.0],
            ],
            keep_blocks=2,
            coverage_fraction=0.0,
        )

        self.assertEqual(decision.selected_blocks, (4, 5))

    def test_period_shortens_monotonically_with_disagreement(self):
        agreed = select_promixed_gqa_blocks(
            [
                [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
                [9.1, 8.1, 7.1, 0.0, 0.0, 0.0],
                [8.9, 7.9, 6.9, 0.0, 0.0, 0.0],
                [9.2, 8.2, 7.2, 0.0, 0.0, 0.0],
            ],
            keep_blocks=2,
        )
        disagreed = select_promixed_gqa_blocks(
            [
                [9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 9.0, 8.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 8.0],
            ],
            keep_blocks=2,
        )

        self.assertLessEqual(disagreed.period, agreed.period)
        self.assertGreater(disagreed.uncertainty, agreed.uncertainty)

    def test_high_sensitivity_never_lengthens_period(self):
        rows = [
            [9.0, 8.0, 7.0, 6.0],
            [9.0, 8.0, 7.0, 6.0],
            [9.0, 8.0, 7.0, 6.0],
            [9.0, 8.0, 7.0, 6.0],
        ]
        normal = select_promixed_gqa_blocks(
            rows,
            keep_blocks=2,
            sensitivity_risk=0.0,
        )
        sensitive = select_promixed_gqa_blocks(
            rows,
            keep_blocks=2,
            sensitivity_risk=1.0,
        )

        self.assertLessEqual(sensitive.period, normal.period)
        self.assertGreaterEqual(sensitive.uncertainty, normal.uncertainty)

    def test_adaptive_coverage_relaxes_only_low_uncertainty_requests(self):
        score_rows = [
            [100.0, 80.0, 70.0, 1.0, 1.0, 90.0, 1.0],
            [1.0, 80.0, 70.0, 1.0, 100.0, 1.0, 90.0],
        ]
        fixed = select_promixed_gqa_blocks(
            score_rows,
            keep_blocks=4,
            coverage_fraction=1.0,
            adaptive_coverage=False,
        )
        low_uncertainty = select_promixed_gqa_blocks(
            score_rows,
            keep_blocks=4,
            coverage_fraction=1.0,
            adaptive_coverage=True,
        )
        high_uncertainty = select_promixed_gqa_blocks(
            [
                [9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 9.0, 8.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 8.0],
            ],
            keep_blocks=2,
            coverage_fraction=0.5,
            adaptive_coverage=True,
        )

        self.assertEqual(fixed.selected_blocks, (0, 4, 5, 6))
        self.assertEqual(low_uncertainty.selected_blocks, (0, 1, 2, 4))
        self.assertEqual(low_uncertainty.uncertainty, 0.5)
        self.assertEqual(low_uncertainty.period, 8)
        self.assertEqual(fixed.effective_coverage_fraction, 1.0)
        self.assertEqual(low_uncertainty.effective_coverage_fraction, 0.0)
        self.assertGreaterEqual(high_uncertainty.uncertainty, 0.68)
        self.assertEqual(high_uncertainty.effective_coverage_fraction, 0.5)

    def test_rejects_utility_weights_that_do_not_form_a_convex_mix(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            select_promixed_gqa_blocks(
                [[1.0, 0.0], [1.0, 0.0]],
                keep_blocks=1,
                utility_max_weight=0.7,
                utility_mean_weight=0.7,
                utility_vote_weight=0.1,
            )

    def test_rejects_invalid_period_threshold_order(self):
        with self.assertRaisesRegex(ValueError, "p4 <= p2 <= p1"):
            select_promixed_gqa_blocks(
                [[1.0, 0.0], [1.0, 0.0]],
                keep_blocks=1,
                p1_threshold=0.5,
                p2_threshold=0.6,
                p4_threshold=0.4,
            )


if __name__ == "__main__":
    unittest.main()
