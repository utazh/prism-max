import unittest

from contiguous_fuxian.lmcache_partial_prefix import contiguous_prefix_hits


class LMCachePartialPrefixTest(unittest.TestCase):
    def test_counts_complete_leading_chunks(self):
        self.assertEqual(contiguous_prefix_hits([[True], [True, True], [False], [True]]), 2)

    def test_dropped_all_layer_chunk_is_a_valid_sparse_hit(self):
        self.assertEqual(contiguous_prefix_hits([[], [True], []]), 3)


if __name__ == "__main__":
    unittest.main()
