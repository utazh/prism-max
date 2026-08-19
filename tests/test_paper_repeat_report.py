import unittest

from contiguous_fuxian.paper_repeat_report import build_repeat_report, render_markdown


def grid(*rows):
    return {
        "comparison": "ContiguousKV versus IMPRESS",
        "budgets": [
            {
                "keep_ratio": ratio,
                "overall": {
                    "contiguous_accuracy": 1.0,
                    "impress_accuracy": 1.0,
                    "contiguous_mean_ttft_ms": contig,
                    "impress_mean_ttft_ms": impress,
                    "ttft_speedup_vs_impress": impress / contig,
                },
            }
            for ratio, contig, impress in rows
        ],
    }


class PaperRepeatReportTest(unittest.TestCase):
    def test_aggregates_only_common_budgets_by_median(self):
        report = build_repeat_report(
            [
                grid((0.05, 10.0, 30.0), (0.25, 20.0, 30.0)),
                grid((0.05, 12.0, 24.0)),
                grid((0.05, 11.0, 44.0), (0.25, 21.0, 42.0)),
            ]
        )

        self.assertEqual(len(report["budgets"]), 1)
        budget = report["budgets"][0]
        self.assertEqual(budget["keep_percent"], 5.0)
        self.assertEqual(budget["contiguous_mean_ttft_ms"]["median"], 11.0)
        self.assertEqual(budget["impress_mean_ttft_ms"]["median"], 30.0)
        self.assertEqual(budget["paired_speedup"]["median"], 3.0)
        self.assertIn("5%", render_markdown(report))

    def test_rejects_grids_without_common_budget(self):
        with self.assertRaisesRegex(ValueError, "no common"):
            build_repeat_report([grid((0.05, 1.0, 2.0)), grid((0.25, 1.0, 2.0))])

    def test_aggregates_ssd_read_reduction_when_available(self):
        first = grid((0.05, 10.0, 30.0))
        second = grid((0.05, 12.0, 24.0))
        first["budgets"][0]["overall"]["ssd_read_reduction_vs_impress"] = 16.0
        second["budgets"][0]["overall"]["ssd_read_reduction_vs_impress"] = 20.0

        report = build_repeat_report([first, second])

        self.assertEqual(report["budgets"][0]["ssd_read_reduction"]["median"], 18.0)
        self.assertIn("Median SSD-read reduction", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
