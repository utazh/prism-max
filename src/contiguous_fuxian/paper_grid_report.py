"""Aggregate only the paper-aligned ContiguousKV versus IMPRESS results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_COMPARISON = "ContiguousKV versus IMPRESS"


def parse_comparison_spec(value: str) -> tuple[float, Path]:
    """Parse a CLI item in the form ``ratio=/absolute/or/relative/path.json``."""

    ratio_text, separator, path_text = value.partition("=")
    if not separator or not ratio_text or not path_text:
        raise ValueError(f"comparison must be RATIO=PATH, got {value!r}")
    ratio = float(ratio_text)
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio_text!r}")
    return ratio, Path(path_text)


def load_comparison(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("comparison") != EXPECTED_COMPARISON:
        raise ValueError(f"{path} is not a {EXPECTED_COMPARISON} comparison")
    if not isinstance(payload.get("overall"), Mapping) or not isinstance(payload.get("tasks"), Mapping):
        raise ValueError(f"{path} is missing overall or task metrics")
    return payload


def build_grid_report(comparisons: Mapping[float, Mapping[str, Any]]) -> dict[str, Any]:
    """Return a stable, budget-sorted view of matched method comparisons."""

    budgets = []
    for ratio in sorted(comparisons):
        payload = comparisons[ratio]
        tasks = {
            task: payload["tasks"][task]
            for task in sorted(payload["tasks"])
        }
        budgets.append(
            {
                "keep_ratio": ratio,
                "keep_percent": ratio * 100,
                "overall": payload["overall"],
                "tasks": tasks,
            }
        )
    return {
        "comparison": EXPECTED_COMPARISON,
        "speedup_definition": "IMPRESS mean TTFT divided by ContiguousKV mean TTFT",
        "budgets": budgets,
    }


def _format_metric(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise, standalone result table without unrelated baselines."""

    include_physical_io = all(
        "physical_read_reduction_vs_impress" in budget["overall"]
        for budget in report["budgets"]
    )
    include_ssd_io = all(
        "ssd_read_reduction_vs_impress" in budget["overall"]
        for budget in report["budgets"]
    )
    include_total_ssd_io = all(
        "total_ssd_read_reduction_vs_impress" in budget["overall"]
        for budget in report["budgets"]
    )
    overall_header = (
        "| KV budget | ContiguousKV accuracy | IMPRESS accuracy | "
        "ContiguousKV mean TTFT (ms) | IMPRESS mean TTFT (ms) | Speedup |"
    )
    overall_separator = "| ---: | ---: | ---: | ---: | ---: | ---: |"
    if include_physical_io:
        overall_header += " Physical-block reduction |"
        overall_separator += " ---: |"
    if include_ssd_io:
        overall_header += " Critical-KV SSD-read reduction |"
        overall_separator += " ---: |"
    if include_total_ssd_io:
        overall_header += " Total SSD-read reduction |"
        overall_separator += " ---: |"
    lines = [
        "# ContiguousKV vs IMPRESS",
        "",
        "Only the two methods compared in the paper are included. "
        "Speedup is IMPRESS mean TTFT divided by ContiguousKV mean TTFT; values below 1.00 mean ContiguousKV was slower in this runtime.",
        "",
        overall_header,
        overall_separator,
    ]
    for budget in report["budgets"]:
        overall = budget["overall"]
        row = (
            "| {budget:.0f}% | {contig_acc} | {impress_acc} | {contig_ttft} | "
            "{impress_ttft} | {speedup}x |"
        ).format(
                budget=budget["keep_percent"],
                contig_acc=_format_metric(overall["contiguous_accuracy"], 4),
                impress_acc=_format_metric(overall["impress_accuracy"], 4),
                contig_ttft=_format_metric(overall["contiguous_mean_ttft_ms"]),
                impress_ttft=_format_metric(overall["impress_mean_ttft_ms"]),
                speedup=_format_metric(overall["ttft_speedup_vs_impress"]),
            )
        if include_physical_io:
            row += " {reduction}x |".format(
                reduction=_format_metric(overall["physical_read_reduction_vs_impress"])
            )
        if include_ssd_io:
            row += " {reduction}x |".format(
                reduction=_format_metric(overall["ssd_read_reduction_vs_impress"])
            )
        if include_total_ssd_io:
            row += " {reduction}x |".format(
                reduction=_format_metric(overall["total_ssd_read_reduction_vs_impress"])
            )
        lines.append(row)
    lines.extend(["", "## Per-task Results", ""])
    for budget in report["budgets"]:
        lines.extend(
            [
                f"### {budget['keep_percent']:.0f}% KV Budget",
                "",
                "| Task | ContiguousKV accuracy | IMPRESS accuracy | ContiguousKV mean TTFT (ms) | IMPRESS mean TTFT (ms) | Speedup |"
                + (" Physical-block reduction |" if include_physical_io else "")
                + (" Critical-KV SSD-read reduction |" if include_ssd_io else "")
                + (" Total SSD-read reduction |" if include_total_ssd_io else ""),
                "| --- | ---: | ---: | ---: | ---: | ---: |"
                + (" ---: |" if include_physical_io else "")
                + (" ---: |" if include_ssd_io else "")
                + (" ---: |" if include_total_ssd_io else ""),
            ]
        )
        for task, metrics in budget["tasks"].items():
            row = (
                "| {task} | {contig_acc} | {impress_acc} | {contig_ttft} | "
                "{impress_ttft} | {speedup}x |"
            ).format(
                    task=task,
                    contig_acc=_format_metric(metrics["contiguous_accuracy"], 4),
                    impress_acc=_format_metric(metrics["impress_accuracy"], 4),
                    contig_ttft=_format_metric(metrics["contiguous_mean_ttft_ms"]),
                    impress_ttft=_format_metric(metrics["impress_mean_ttft_ms"]),
                    speedup=_format_metric(metrics["ttft_speedup_vs_impress"]),
                )
            if include_physical_io:
                row += " {reduction}x |".format(
                    reduction=_format_metric(metrics["physical_read_reduction_vs_impress"])
                )
            if include_ssd_io:
                row += " {reduction}x |".format(
                    reduction=_format_metric(metrics["ssd_read_reduction_vs_impress"])
                )
            if include_total_ssd_io:
                row += " {reduction}x |".format(
                    reduction=_format_metric(metrics["total_ssd_read_reduction_vs_impress"])
                )
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate ContiguousKV versus IMPRESS comparison JSON files.")
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        metavar="RATIO=PATH",
        help="One comparison JSON for a KV keep ratio; may be repeated.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args(argv)

    comparisons: dict[float, Mapping[str, Any]] = {}
    for spec in args.comparison:
        ratio, path = parse_comparison_spec(spec)
        if ratio in comparisons:
            raise ValueError(f"duplicate ratio: {ratio}")
        comparisons[ratio] = load_comparison(path)
    report = build_grid_report(comparisons)
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"budgets": len(report["budgets"]), "output_json": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
