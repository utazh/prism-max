
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


VALID_PERIODS = (1, 2, 4, 8)


def budget_key(value: float | str) -> str:
    return f"{float(value):.3f}"


@dataclass(frozen=True)
class PeriodMeasurement:
    task: str
    budget: float
    period: int
    correct: int
    total: int
    mean_ttft_ms: float

    @property
    def accuracy(self) -> float:
        if self.total <= 0:
            raise ValueError("total must be positive")
        return self.correct / self.total


@dataclass(frozen=True)
class ReuseCellProfile:
    base_period: int
    guard_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.base_period not in VALID_PERIODS:
            raise ValueError(f"invalid base period: {self.base_period}")
        if not 0.0 <= self.guard_threshold <= 1.01:
            raise ValueError("guard_threshold must be in [0, 1.01]")


def choose_period(
    rows: Sequence[PeriodMeasurement],
    *,
    quality_slack_examples: int = 1,
    latency_tie_fraction: float = 0.015,
) -> int:
    """
    Pick the fastest fixed period whose calibration accuracy is within
    `quality_slack_examples` of the best fixed period.

    If several eligible periods are within `latency_tie_fraction` of the
    fastest, prefer higher calibration accuracy, then the longer period.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    totals = {row.total for row in rows}
    if len(totals) != 1:
        raise ValueError("all periods for a cell must use the same sample count")
    periods = {row.period for row in rows}
    if not periods.issubset(set(VALID_PERIODS)):
        raise ValueError(f"unexpected period(s): {sorted(periods)}")

    best_correct = max(row.correct for row in rows)
    eligible = [
        row for row in rows
        if row.correct >= best_correct - int(quality_slack_examples)
    ]
    min_latency = min(row.mean_ttft_ms for row in eligible)
    near_fastest = [
        row for row in eligible
        if row.mean_ttft_ms <= min_latency * (1.0 + float(latency_tie_fraction))
    ]
    winner = max(
        near_fastest,
        key=lambda row: (row.correct, row.period, -row.mean_ttft_ms),
    )
    return winner.period


def build_profile(
    rows: Iterable[PeriodMeasurement],
    *,
    quality_slack_examples: int = 1,
    latency_tie_fraction: float = 0.015,
    guard_threshold: float = 0.95,
) -> Dict[str, object]:
    grouped: MutableMapping[Tuple[str, str], List[PeriodMeasurement]] = {}
    for row in rows:
        key = (row.task.lower(), budget_key(row.budget))
        grouped.setdefault(key, []).append(row)

    cells: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for (task, budget), cell_rows in sorted(grouped.items()):
        base = choose_period(
            cell_rows,
            quality_slack_examples=quality_slack_examples,
            latency_tie_fraction=latency_tie_fraction,
        )
        cells.setdefault(task, {})[budget] = asdict(
            ReuseCellProfile(
                base_period=base,
                guard_threshold=guard_threshold,
            )
        )

    return {
        "version": 1,
        "selection_rule": {
            "quality_slack_examples": int(quality_slack_examples),
            "latency_tie_fraction": float(latency_tie_fraction),
            "description": (
                "Fastest fixed period within the calibration quality slack; "
                "runtime uncertainty can only shorten that period."
            ),
        },
        "cells": cells,
    }


class ProfiledAdaptiveReuse:
    """
    Workload-profiled base period plus a conservative online safety guard.

    The task/budget profile must be built on a calibration split, never on the
    held-out evaluation examples.
    """

    def __init__(self, profile: Mapping[str, object]) -> None:
        self.profile = profile

    @classmethod
    def from_json(cls, path: str | Path) -> "ProfiledAdaptiveReuse":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def get_cell(self, task: str, budget: float) -> ReuseCellProfile:
        cells = self.profile.get("cells")
        if not isinstance(cells, Mapping):
            raise KeyError("profile has no 'cells' mapping")
        task_map = cells.get(task.lower())
        if not isinstance(task_map, Mapping):
            raise KeyError(f"task {task!r} is absent from reuse profile")
        raw = task_map.get(budget_key(budget))
        if not isinstance(raw, Mapping):
            raise KeyError(
                f"task/budget {task!r}/{budget_key(budget)} is absent from reuse profile"
            )
        return ReuseCellProfile(
            base_period=int(raw["base_period"]),
            guard_threshold=float(raw.get("guard_threshold", 0.95)),
        )

    def choose(
        self,
        *,
        task: str,
        budget: float,
        uncertainty: float,
    ) -> int:
        cell = self.get_cell(task, budget)
        base = cell.base_period
        # A deliberately rare safety downgrade.  It preserves the calibrated
        # result-oriented base period for normal requests.
        if float(uncertainty) >= cell.guard_threshold:
            return max(1, base // 2)
        return base


def _load_csv(path: Path) -> List[PeriodMeasurement]:
    out: List[PeriodMeasurement] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append(
                PeriodMeasurement(
                    task=row["task"],
                    budget=float(row["budget"]),
                    period=int(row["period"]),
                    correct=int(row["correct"]),
                    total=int(row["total"]),
                    mean_ttft_ms=float(row["mean_ttft_ms"]),
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--quality-slack-examples", type=int, default=1)
    parser.add_argument("--latency-tie-fraction", type=float, default=0.015)
    parser.add_argument("--guard-threshold", type=float, default=0.95)
    args = parser.parse_args()

    profile = build_profile(
        _load_csv(args.input_csv),
        quality_slack_examples=args.quality_slack_examples,
        latency_tie_fraction=args.latency_tie_fraction,
        guard_threshold=args.guard_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
