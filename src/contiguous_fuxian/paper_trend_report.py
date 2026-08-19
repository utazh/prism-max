"""Compare measured ContiguousKV/IMPRESS results with paper-level trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


PAPER_REFERENCE = {
    "five_percent_ttft_speedup_vs_impress": 3.85,
    "accuracy_gain_percentage_points_vs_impress": {
        5: 7.69,
        10: 4.81,
        25: 3.58,
        50: 1.63,
    },
    "mean_critical_kv_ssd_reduction_vs_impress": 16.33,
}


def _budget_key(value: Any) -> int:
    percent = float(value)
    rounded = round(percent)
    if abs(percent - rounded) > 1e-9 or rounded <= 0:
        raise ValueError(f"budget percent must be a positive integer, got {value!r}")
    return int(rounded)


def build_paper_trend_report(grid: Mapping[str, Any]) -> dict[str, Any]:
    """Build an explicit trend audit from a paper-grid report."""

    if grid.get("comparison") != "ContiguousKV versus IMPRESS":
        raise ValueError("grid is not a ContiguousKV versus IMPRESS comparison")
    raw_budgets = grid.get("budgets")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError("grid must contain at least one budget")

    rows = []
    seen: set[int] = set()
    for item in raw_budgets:
        budget = _budget_key(item["keep_percent"])
        if budget in seen:
            raise ValueError(f"duplicate budget {budget}%")
        seen.add(budget)
        overall = item["overall"]
        speedup = float(overall["ttft_speedup_vs_impress"])
        accuracy_delta = 100.0 * (
            float(overall["contiguous_accuracy"])
            - float(overall["impress_accuracy"])
        )
        row = {
            "keep_percent": budget,
            "ttft_speedup_vs_impress": speedup,
            "accuracy_delta_percentage_points": accuracy_delta,
            "paper_accuracy_gain_percentage_points": PAPER_REFERENCE[
                "accuracy_gain_percentage_points_vs_impress"
            ].get(budget),
        }
        if "ssd_read_reduction_vs_impress" in overall:
            row["critical_kv_ssd_reduction_vs_impress"] = float(
                overall["ssd_read_reduction_vs_impress"]
            )
        rows.append(row)
    rows.sort(key=lambda row: row["keep_percent"])
    by_budget = {row["keep_percent"]: row for row in rows}

    all_measured_speedup_direction = all(
        row["ttft_speedup_vs_impress"] > 1.0 for row in rows
    )
    paper_latency_rows = [by_budget[budget] for budget in (5, 25) if budget in by_budget]
    paper_speedup_direction = (
        all(row["ttft_speedup_vs_impress"] > 1.0 for row in paper_latency_rows)
        if paper_latency_rows
        else None
    )
    budget_order = None
    if 5 in by_budget and 25 in by_budget:
        budget_order = (
            by_budget[5]["ttft_speedup_vs_impress"]
            > by_budget[25]["ttft_speedup_vs_impress"]
        )
    accuracy_direction = all(
        row["accuracy_delta_percentage_points"] > 0.0 for row in rows
    )
    io_rows = [
        row["critical_kv_ssd_reduction_vs_impress"]
        for row in rows
        if "critical_kv_ssd_reduction_vs_impress" in row
    ]
    io_direction = all(value > 1.0 for value in io_rows) if io_rows else None

    five_percent_scale = None
    if 5 in by_budget:
        five_percent_scale = (
            by_budget[5]["ttft_speedup_vs_impress"]
            / PAPER_REFERENCE["five_percent_ttft_speedup_vs_impress"]
        )

    return {
        "comparison": "ContiguousKV versus IMPRESS",
        "paper_reference": {
            **PAPER_REFERENCE,
            "scope_note": (
                "Paper values are averages across its reported datasets/models; "
                "the reproduction uses the locally available Qwen2.5-7B-Instruct."
            ),
        },
        "measured_budgets": rows,
        "trend_checks": {
            "contiguous_faster_at_every_measured_budget": (
                all_measured_speedup_direction
            ),
            "contiguous_faster_at_paper_reported_latency_budgets": (
                paper_speedup_direction
            ),
            "five_percent_speedup_exceeds_twenty_five_percent": budget_order,
            "contiguous_accuracy_exceeds_impress_at_every_measured_budget": (
                accuracy_direction
            ),
            "critical_kv_ssd_reads_lower_at_every_measured_budget": io_direction,
            "five_percent_speedup_over_paper_reference": five_percent_scale,
            "mean_measured_critical_kv_ssd_reduction": (
                mean(io_rows) if io_rows else None
            ),
        },
        "assessment": {
            "latency_direction": (
                "match"
                if paper_speedup_direction
                else "mismatch"
                if paper_speedup_direction is False
                else "unavailable"
            ),
            "budget_scaling_direction": (
                "match" if budget_order else "mismatch" if budget_order is False else "unavailable"
            ),
            "accuracy_direction": "match" if accuracy_direction else "mismatch",
            "critical_kv_ssd_direction": (
                "match" if io_direction else "mismatch" if io_direction is False else "unavailable"
            ),
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the trend audit as a compact paper-comparison table."""

    lines = [
        "# Paper Trend Audit",
        "",
        "| KV budget | Measured TTFT speedup | Accuracy delta (pp) | Paper accuracy gain (pp) | Critical-KV SSD reduction |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["measured_budgets"]:
        paper_accuracy = row["paper_accuracy_gain_percentage_points"]
        ssd = row.get("critical_kv_ssd_reduction_vs_impress")
        lines.append(
            "| {budget}% | {speedup:.2f}x | {accuracy:+.2f} | {paper_accuracy} | {ssd} |".format(
                budget=row["keep_percent"],
                speedup=row["ttft_speedup_vs_impress"],
                accuracy=row["accuracy_delta_percentage_points"],
                paper_accuracy=(
                    f"{paper_accuracy:+.2f}" if paper_accuracy is not None else "n/a"
                ),
                ssd=f"{ssd:.2f}x" if ssd is not None else "n/a",
            )
        )
    lines.extend(["", "## Assessment", ""])
    for name, value in report["assessment"].items():
        lines.append(f"- `{name}`: {value}")
    lines.extend(
        [
            "",
            "The paper reports 3.85x TTFT speedup over IMPRESS at 5% and a smaller advantage at 25%. "
            "Its 16.33x critical-KV SSD reduction is an average over the paper's reported configurations, "
            "so it is retained as a reference rather than an exact equality target.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-report", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args(argv)

    grid = json.loads(args.grid_report.read_text(encoding="utf-8"))
    report = build_paper_trend_report(grid)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["assessment"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
