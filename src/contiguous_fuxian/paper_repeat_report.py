"""Aggregate repeated ContiguousKV versus IMPRESS grid measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paper_grid_report import EXPECTED_COMPARISON


def load_grid_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("comparison") != EXPECTED_COMPARISON:
        raise ValueError(f"{path} is not a {EXPECTED_COMPARISON} grid")
    if not isinstance(payload.get("budgets"), list) or not payload["budgets"]:
        raise ValueError(f"{path} has no budget measurements")
    return payload


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    return {
        "values": normalized,
        "median": statistics.median(normalized),
        "min": min(normalized),
        "max": max(normalized),
    }


def build_repeat_report(grids: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate budgets present in every supplied repeat."""

    if not grids:
        raise ValueError("at least one grid summary is required")
    indexed = []
    for grid in grids:
        rows = {float(row["keep_ratio"]): row for row in grid["budgets"]}
        if len(rows) != len(grid["budgets"]):
            raise ValueError("a grid contains duplicate keep ratios")
        indexed.append(rows)
    common_ratios = set(indexed[0])
    for rows in indexed[1:]:
        common_ratios &= set(rows)
    if not common_ratios:
        raise ValueError("grid summaries have no common KV budget")

    budgets = []
    for ratio in sorted(common_ratios):
        rows = [grid[ratio]["overall"] for grid in indexed]
        contig_ttft = [float(row["contiguous_mean_ttft_ms"]) for row in rows]
        impress_ttft = [float(row["impress_mean_ttft_ms"]) for row in rows]
        paired_speedup = [float(row["ttft_speedup_vs_impress"]) for row in rows]
        contig_distribution = _distribution(contig_ttft)
        impress_distribution = _distribution(impress_ttft)
        budget = {
                "keep_ratio": ratio,
                "keep_percent": ratio * 100,
                "repeats": len(rows),
                "contiguous_accuracy_values": [float(row["contiguous_accuracy"]) for row in rows],
                "impress_accuracy_values": [float(row["impress_accuracy"]) for row in rows],
                "contiguous_mean_ttft_ms": contig_distribution,
                "impress_mean_ttft_ms": impress_distribution,
                "paired_speedup": _distribution(paired_speedup),
                "ratio_of_median_ttfts": (
                    impress_distribution["median"] / contig_distribution["median"]
                ),
            }
        for output_name, source_name in (
            ("physical_read_reduction", "physical_read_reduction_vs_impress"),
            ("ssd_read_reduction", "ssd_read_reduction_vs_impress"),
            ("total_ssd_read_reduction", "total_ssd_read_reduction_vs_impress"),
        ):
            if all(source_name in row for row in rows):
                budget[output_name] = _distribution([float(row[source_name]) for row in rows])
        budgets.append(budget)
    return {
        "comparison": EXPECTED_COMPARISON,
        "aggregation": "median and range of matched run-level mean TTFT measurements",
        "input_repeats": len(grids),
        "budgets": budgets,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    include_ssd_io = all("ssd_read_reduction" in budget for budget in report["budgets"])
    include_total_ssd_io = all(
        "total_ssd_read_reduction" in budget for budget in report["budgets"]
    )
    header = (
        "| KV budget | Repeats | ContiguousKV median TTFT (range) | "
        "IMPRESS median TTFT (range) | Median paired speedup |"
    )
    separator = "| ---: | ---: | ---: | ---: | ---: |"
    if include_ssd_io:
        header += " Median SSD-read reduction (critical KV) |"
        separator += " ---: |"
    if include_total_ssd_io:
        header += " Median total SSD-read reduction |"
        separator += " ---: |"
    lines = [
        "# Repeated ContiguousKV vs IMPRESS",
        "",
        header,
        separator,
    ]
    for budget in report["budgets"]:
        contig = budget["contiguous_mean_ttft_ms"]
        impress = budget["impress_mean_ttft_ms"]
        speedup = budget["paired_speedup"]
        row = (
            "| {percent:.0f}% | {repeats} | {cm:.2f} ms ({cmin:.2f}-{cmax:.2f}) | "
            "{im:.2f} ms ({imin:.2f}-{imax:.2f}) | {speed:.2f}x |".format(
                percent=budget["keep_percent"],
                repeats=budget["repeats"],
                cm=contig["median"],
                cmin=contig["min"],
                cmax=contig["max"],
                im=impress["median"],
                imin=impress["min"],
                imax=impress["max"],
                speed=speedup["median"],
            )
        )
        if include_ssd_io:
            row += " {reduction:.2f}x |".format(
                reduction=budget["ssd_read_reduction"]["median"]
            )
        if include_total_ssd_io:
            row += " {reduction:.2f}x |".format(
                reduction=budget["total_ssd_read_reduction"]["median"]
            )
        lines.append(row)
    lines.extend(
        [
            "",
            "Speedup is paired within each repeat as IMPRESS mean TTFT divided by ContiguousKV mean TTFT, then aggregated by median.",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate repeated matched grid summaries.")
    parser.add_argument("--grid-summary", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args(argv)
    report = build_repeat_report([load_grid_summary(path) for path in args.grid_summary])
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"budgets": len(report["budgets"]), "repeats": report["input_repeats"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
