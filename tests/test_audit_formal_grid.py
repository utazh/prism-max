import unittest

from contiguous_fuxian.scripts.audit_formal_grid import (
    all_finite,
    expected_uids,
    paired_accuracy_stats,
    parse_ratio_specs,
)


class FormalGridAuditTest(unittest.TestCase):
    def test_ratio_specs_build_both_methods_for_each_budget(self):
        ratios, runs = parse_ratio_specs(
            "005:0.05 010:0.10 025:0.25 050:0.50"
        )

        self.assertEqual(
            ratios,
            [("005", 0.05), ("010", 0.10), ("025", 0.25), ("050", 0.50)],
        )
        self.assertEqual(runs["k005_contig"], ("contigkv", 0.05, 16))
        self.assertEqual(runs["k050_impress"], ("impress", 0.50, 64))
        self.assertEqual(len(runs), 8)

    def test_ratio_specs_reject_invalid_or_duplicate_entries(self):
        for value in (
            "",
            "5:0.05",
            "005=0.05",
            "005:0",
            "005:1.1",
            "005:0.05 005:0.10",
            "005:0.05 010:0.05",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_ratio_specs(value)

    def test_uid_and_finite_helpers_cover_formal_record_shapes(self):
        self.assertEqual(
            expected_uids(["sst2", "rte"], 2),
            {"sst2-0", "sst2-1", "rte-0", "rte-1"},
        )
        self.assertTrue(all_finite({"metrics": [1, 2.5, None], "uid": "rte-0"}))
        self.assertFalse(all_finite({"metrics": [1, float("nan")]}))

    def test_paired_accuracy_reports_discordance_and_exact_test(self):
        contig = [
            {"uid": "x-0", "correct": True},
            {"uid": "x-1", "correct": True},
            {"uid": "x-2", "correct": False},
            {"uid": "x-3", "correct": False},
        ]
        impress = [
            {"uid": "x-0", "correct": True},
            {"uid": "x-1", "correct": False},
            {"uid": "x-2", "correct": True},
            {"uid": "x-3", "correct": False},
        ]

        result = paired_accuracy_stats(contig, impress)

        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["contig_only_correct"], 1)
        self.assertEqual(result["impress_only_correct"], 1)
        self.assertEqual(result["both_wrong"], 1)
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
