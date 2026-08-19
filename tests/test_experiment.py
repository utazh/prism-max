import unittest

from contiguous_fuxian.experiment import (
    make_synthetic_layer_scores,
    run_synthetic_reproduction,
)


class ContiguousKVExperimentTest(unittest.TestCase):
    def test_synthetic_layer_scores_are_deterministic(self):
        left = make_synthetic_layer_scores(
            num_layers=4,
            num_chunks=8,
            hot_chunks=2,
            period_size=2,
            seed=7,
        )
        right = make_synthetic_layer_scores(
            num_layers=4,
            num_chunks=8,
            hot_chunks=2,
            period_size=2,
            seed=7,
        )

        self.assertEqual(left, right)

    def test_synthetic_reproduction_reports_contiguous_speedup(self):
        got = run_synthetic_reproduction(
            prefix_tokens=1024,
            num_layers=8,
            contiguous_chunk_size=16,
            impress_chunk_size=64,
            keep_ratio=0.25,
            period_size=8,
            subperiod_size=4,
            seed=11,
        )

        self.assertEqual(got["config"]["period_size"], 8)
        self.assertEqual(got["config"]["subperiod_size"], 4)
        self.assertGreater(got["metrics"]["speedup_vs_impress"], 1.0)
        self.assertIn("attention_cache", got)
        self.assertEqual(got["attention_cache"]["policy"], "S=I*F")
        self.assertGreater(got["attention_cache"]["resident_count"], 0)
        self.assertLess(
            got["contiguous"]["read_amplification"],
            got["impress"]["read_amplification"],
        )
        self.assertGreaterEqual(got["metrics"]["mean_adjacent_period_jaccard"], 0.0)


if __name__ == "__main__":
    unittest.main()
