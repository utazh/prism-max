import unittest

from contiguous_fuxian.simulator import (
    KVShape,
    LayerRequest,
    compare_impress_contiguous,
    estimate_read_amplification,
    simulate_contiguous_prefetch,
)


class ContiguousKVSimulatorTest(unittest.TestCase):
    def test_read_amplification_compares_physical_and_useful_tokens(self):
        got = estimate_read_amplification(
            selected_token_indices=[1, 2, 63, 64],
            physical_chunk_size=64,
        )

        self.assertEqual(got.useful_tokens, 4)
        self.assertEqual(got.physical_tokens_read, 128)
        self.assertAlmostEqual(got.amplification, 32.0)

    def test_contiguous_prefetch_overlaps_intra_period_io_after_first_subperiod(self):
        requests = [
            LayerRequest(layer=0, chunks={0, 1}),
            LayerRequest(layer=1, chunks={0, 1}),
            LayerRequest(layer=2, chunks={2, 3}),
            LayerRequest(layer=3, chunks={2, 3}),
        ]

        got = simulate_contiguous_prefetch(
            requests=requests,
            period_size=2,
            subperiod_size=1,
            chunk_load_ms=10.0,
            compute_ms=30.0,
        )

        self.assertLess(got.total_ms, got.unoptimized_ms)
        self.assertEqual(got.periods, 2)
        self.assertEqual(got.compute_ms, 120.0)
        self.assertEqual(got.blocking_io_ms, 40.0)
        self.assertEqual(got.overlapped_io_ms, 40.0)
        self.assertEqual(got.total_ms, 160.0)
        self.assertEqual(got.unoptimized_ms, 200.0)

    def test_compare_impress_contiguous_reports_lower_contiguous_io_tokens(self):
        selected = [
            LayerRequest(layer=0, chunks={0, 1}),
            LayerRequest(layer=1, chunks={0, 1}),
        ]
        shape = KVShape(num_layers=2, prefix_tokens=128, contiguous_chunk_size=16)

        got = compare_impress_contiguous(
            selected,
            shape=shape,
            impress_chunk_size=64,
            budget_ratio=0.25,
            chunk_load_ms=1.0,
            compute_ms=1.0,
        )

        self.assertLess(got["contiguous"]["physical_tokens_read"], got["impress"]["physical_tokens_read"])
        self.assertLess(got["contiguous"]["read_amplification"], got["impress"]["read_amplification"])


if __name__ == "__main__":
    unittest.main()
