import unittest

from contiguous_fuxian.sparse_qwen_reprefill import contiguous_spans, selected_chunk_indices


class SparseQwenRePrefillTest(unittest.TestCase):
    def test_selected_chunk_indices_excludes_only_drop_cells(self):
        self.assertEqual(selected_chunk_indices(["drop", "int4", "int8", "drop"]), [1, 2])

    def test_contiguous_spans_coalesces_and_sorts_indices(self):
        self.assertEqual(contiguous_spans([7, 1, 2, 4, 2, 5]), [(1, 3), (4, 6), (7, 8)])


if __name__ == "__main__":
    unittest.main()
