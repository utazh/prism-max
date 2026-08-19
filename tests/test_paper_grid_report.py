import unittest
from pathlib import Path

from contiguous_fuxian.paper_grid_report import (
    build_grid_report,
    parse_comparison_spec,
    render_markdown,
)


def comparison(contiguous_ttft, impress_ttft):
    return {
        "comparison": "ContiguousKV versus IMPRESS",
        "overall": {
            "contiguous_accuracy": 0.5,
            "impress_accuracy": 0.5,
            "contiguous_mean_ttft_ms": contiguous_ttft,
            "impress_mean_ttft_ms": impress_ttft,
            "ttft_speedup_vs_impress": impress_ttft / contiguous_ttft,
        },
        "tasks": {
            "trec": {
                "contiguous_accuracy": 0.25,
                "impress_accuracy": 0.5,
                "contiguous_mean_ttft_ms": contiguous_ttft,
                "impress_mean_ttft_ms": impress_ttft,
                "ttft_speedup_vs_impress": impress_ttft / contiguous_ttft,
            }
        },
    }


class PaperGridReportTest(unittest.TestCase):
    def test_parse_comparison_spec(self):
        ratio, path = parse_comparison_spec("0.05=results/k005.json")

        self.assertEqual(ratio, 0.05)
        self.assertEqual(path, Path("results/k005.json"))

    def test_report_sorts_budgets_and_renders_only_matched_methods(self):
        report = build_grid_report({0.5: comparison(100.0, 80.0), 0.05: comparison(50.0, 100.0)})

        self.assertEqual([item["keep_percent"] for item in report["budgets"]], [5.0, 50.0])
        markdown = render_markdown(report)
        self.assertIn("| 5% | 0.5000 | 0.5000 | 50.00 | 100.00 | 2.00x |", markdown)
        self.assertNotIn("full", markdown.lower())

    def test_report_includes_physical_read_reduction_when_available(self):
        payload = comparison(50.0, 100.0)
        payload["overall"]["physical_read_reduction_vs_impress"] = 8.0
        payload["tasks"]["trec"]["physical_read_reduction_vs_impress"] = 7.5

        markdown = render_markdown(build_grid_report({0.05: payload}))

        self.assertIn("Physical-block reduction", markdown)
        self.assertIn("| 5% | 0.5000 | 0.5000 | 50.00 | 100.00 | 2.00x | 8.00x |", markdown)

    def test_report_includes_ssd_read_reduction_when_available(self):
        payload = comparison(50.0, 100.0)
        payload["overall"]["ssd_read_reduction_vs_impress"] = 16.0
        payload["tasks"]["trec"]["ssd_read_reduction_vs_impress"] = 15.0

        markdown = render_markdown(build_grid_report({0.05: payload}))

        self.assertIn("SSD-read reduction", markdown)
        self.assertIn("| 5% | 0.5000 | 0.5000 | 50.00 | 100.00 | 2.00x | 16.00x |", markdown)

    def test_report_includes_total_ssd_read_reduction_when_available(self):
        payload = comparison(50.0, 100.0)
        payload["overall"]["total_ssd_read_reduction_vs_impress"] = 4.0
        payload["tasks"]["trec"]["total_ssd_read_reduction_vs_impress"] = 3.5

        markdown = render_markdown(build_grid_report({0.05: payload}))

        self.assertIn("Total SSD-read reduction", markdown)
        self.assertIn("| 5% | 0.5000 | 0.5000 | 50.00 | 100.00 | 2.00x | 4.00x |", markdown)


if __name__ == "__main__":
    unittest.main()
