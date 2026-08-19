import unittest

from contiguous_fuxian.paper_trend_report import (
    build_paper_trend_report,
    render_markdown,
)


def budget(percent, speedup, contiguous_accuracy, impress_accuracy, ssd_reduction):
    return {
        "keep_percent": percent,
        "overall": {
            "ttft_speedup_vs_impress": speedup,
            "contiguous_accuracy": contiguous_accuracy,
            "impress_accuracy": impress_accuracy,
            "ssd_read_reduction_vs_impress": ssd_reduction,
        },
    }


class PaperTrendReportTest(unittest.TestCase):
    def test_reports_latency_match_and_accuracy_mismatch_separately(self):
        grid = {
            "comparison": "ContiguousKV versus IMPRESS",
            "budgets": [
                budget(25, 2.8, 0.74, 0.82, 4.9),
                budget(5, 8.2, 0.63, 0.82, 27.4),
            ],
        }

        report = build_paper_trend_report(grid)

        self.assertEqual(report["assessment"]["latency_direction"], "match")
        self.assertEqual(report["assessment"]["budget_scaling_direction"], "match")
        self.assertEqual(report["assessment"]["accuracy_direction"], "mismatch")
        self.assertEqual(
            report["trend_checks"]["mean_measured_critical_kv_ssd_reduction"],
            16.15,
        )
        self.assertIn("| 5% | 8.20x | -19.00 | +7.69 | 27.40x |", render_markdown(report))

    def test_rejects_wrong_comparison_or_duplicate_budget(self):
        with self.assertRaisesRegex(ValueError, "not a ContiguousKV"):
            build_paper_trend_report({"comparison": "other", "budgets": [{}]})
        with self.assertRaisesRegex(ValueError, "duplicate budget"):
            build_paper_trend_report(
                {
                    "comparison": "ContiguousKV versus IMPRESS",
                    "budgets": [
                        budget(5, 2.0, 0.8, 0.7, 3.0),
                        budget(5, 2.0, 0.8, 0.7, 3.0),
                    ],
                }
            )

    def test_nonpaper_fifty_percent_reversal_does_not_change_latency_claim(self):
        grid = {
            "comparison": "ContiguousKV versus IMPRESS",
            "budgets": [
                budget(5, 3.8, 0.8, 0.7, 10.0),
                budget(25, 1.7, 0.8, 0.7, 4.0),
                budget(50, 0.8, 0.8, 0.8, 2.0),
            ],
        }

        report = build_paper_trend_report(grid)

        self.assertEqual(report["assessment"]["latency_direction"], "match")
        self.assertFalse(
            report["trend_checks"]["contiguous_faster_at_every_measured_budget"]
        )
        self.assertTrue(
            report["trend_checks"][
                "contiguous_faster_at_paper_reported_latency_budgets"
            ]
        )


if __name__ == "__main__":
    unittest.main()
