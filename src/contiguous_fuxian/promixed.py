"""GQA-aware, uncertainty-gated block selection for ProMixed Re-Prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PromixedSelectionDecision:
    """A fixed-budget selection plus the safe cross-layer reuse horizon."""

    selected_blocks: tuple[int, ...]
    priority_blocks: tuple[int, ...]
    agreement: float
    boundary_margin: float
    uncertainty: float
    period: int


def _validate_score_rows(
    block_scores: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not block_scores or not block_scores[0]:
        raise ValueError("ProMixed selection requires non-empty GQA score rows")
    block_count = len(block_scores[0])
    normalized_rows = []
    for row in block_scores:
        if len(row) != block_count:
            raise ValueError("all GQA score rows must cover the same blocks")
        normalized = tuple(float(value) for value in row)
        if any(not math.isfinite(value) or value < 0 for value in normalized):
            raise ValueError("GQA block scores must be finite and non-negative")
        normalized_rows.append(normalized)
    return tuple(normalized_rows)


def _mean_pairwise_jaccard(sets: Sequence[set[int]]) -> float:
    if len(sets) <= 1:
        return 1.0
    values = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            values.append(
                len(sets[left] & sets[right]) / max(1, len(union))
            )
    return sum(values) / len(values)


def _boundary_margin(
    rows: Sequence[Sequence[float]],
    rankings: Sequence[Sequence[int]],
    keep_blocks: int,
) -> float:
    margins = []
    block_count = len(rows[0])
    if keep_blocks >= block_count:
        return 1.0
    for row, ranking in zip(rows, rankings):
        kept_score = float(row[int(ranking[keep_blocks - 1])])
        next_score = float(row[int(ranking[keep_blocks])])
        margins.append(
            max(0.0, kept_score - next_score)
            / max(abs(kept_score), 1e-12)
        )
    return sum(margins) / len(margins)


def _adaptive_period(
    uncertainty: float,
    *,
    max_period: int,
    p1_threshold: float,
    p2_threshold: float,
    p4_threshold: float,
) -> int:
    if max_period not in {1, 2, 4, 8}:
        raise ValueError("ProMixed max period must be one of 1, 2, 4, or 8")
    if not 0 <= p4_threshold <= p2_threshold <= p1_threshold <= 1:
        raise ValueError(
            "ProMixed uncertainty thresholds must satisfy "
            "0 <= p4 <= p2 <= p1 <= 1"
        )
    if uncertainty >= p1_threshold:
        requested = 1
    elif uncertainty >= p2_threshold:
        requested = 2
    elif uncertainty >= p4_threshold:
        requested = 4
    else:
        requested = 8
    return min(requested, max_period)


def select_promixed_gqa_blocks(
    block_scores: Sequence[Sequence[float]],
    *,
    keep_blocks: int,
    coverage_fraction: float = 0.5,
    max_period: int = 8,
    margin_reference: float = 0.05,
    agreement_weight: float = 0.75,
    sensitivity_risk: float = 0.0,
    sensitivity_weight: float = 0.1,
    p1_threshold: float = 0.90,
    p2_threshold: float = 0.82,
    p4_threshold: float = 0.68,
) -> PromixedSelectionDecision:
    """Select exact-budget blocks without averaging away a minority GQA group.

    A fraction of the budget is filled round-robin from each physical GQA
    group's ranking. The remainder uses a scale-normalized max/mean fusion.
    Cross-group Jaccard, top-K boundary margin, and offline layer risk jointly
    choose a P1/P2/P4/P8 reuse horizon.
    """

    rows = _validate_score_rows(block_scores)
    block_count = len(rows[0])
    head_count = len(rows)
    keep_blocks = int(keep_blocks)
    if not 1 <= keep_blocks <= block_count:
        raise ValueError(
            f"ProMixed keep_blocks must be in [1, {block_count}]"
        )
    if not 0 <= coverage_fraction <= 1:
        raise ValueError("ProMixed coverage fraction must be in [0, 1]")
    if not math.isfinite(margin_reference) or margin_reference <= 0:
        raise ValueError("ProMixed margin reference must be positive")
    if not 0 <= agreement_weight <= 1:
        raise ValueError("ProMixed agreement weight must be in [0, 1]")
    if not 0 <= sensitivity_risk <= 1:
        raise ValueError("ProMixed sensitivity risk must be in [0, 1]")
    if not 0 <= sensitivity_weight <= 1:
        raise ValueError("ProMixed sensitivity weight must be in [0, 1]")

    rankings = tuple(
        tuple(
            sorted(
                range(block_count),
                key=lambda block: (-float(row[block]), block),
            )
        )
        for row in rows
    )
    probe_sets = [set(ranking[:keep_blocks]) for ranking in rankings]
    agreement = _mean_pairwise_jaccard(probe_sets)
    boundary_margin = _boundary_margin(rows, rankings, keep_blocks)
    margin_confidence = min(1.0, boundary_margin / margin_reference)
    uncertainty = (
        agreement_weight * (1.0 - agreement)
        + (1.0 - agreement_weight) * (1.0 - margin_confidence)
        + sensitivity_weight * sensitivity_risk
    )
    uncertainty = min(1.0, max(0.0, uncertainty))
    period = _adaptive_period(
        uncertainty,
        max_period=max_period,
        p1_threshold=p1_threshold,
        p2_threshold=p2_threshold,
        p4_threshold=p4_threshold,
    )

    totals = [max(sum(row), 1e-12) for row in rows]
    normalized = [
        [float(value) / total for value in row]
        for row, total in zip(rows, totals)
    ]
    votes = [
        sum(block in selected for selected in probe_sets)
        for block in range(block_count)
    ]
    score_scale = max(max(row) for row in normalized)
    utilities = []
    for block in range(block_count):
        group_values = [row[block] for row in normalized]
        utilities.append(
            0.55 * max(group_values)
            + 0.35 * (sum(group_values) / head_count)
            + 0.10 * score_scale * (votes[block] / head_count)
        )

    coverage_target = min(
        keep_blocks,
        max(
            min(head_count, keep_blocks),
            math.ceil(keep_blocks * coverage_fraction),
        ),
    )
    selected: set[int] = set()
    depth = 0
    while len(selected) < coverage_target and depth < block_count:
        for ranking in rankings:
            selected.add(int(ranking[depth]))
            if len(selected) >= coverage_target:
                break
        depth += 1

    global_ranking = sorted(
        range(block_count),
        key=lambda block: (-utilities[block], block),
    )
    for block in global_ranking:
        if len(selected) >= keep_blocks:
            break
        selected.add(block)

    priority = tuple(
        sorted(selected, key=lambda block: (-utilities[block], block))
    )
    return PromixedSelectionDecision(
        selected_blocks=tuple(sorted(selected)),
        priority_blocks=priority,
        agreement=agreement,
        boundary_margin=boundary_margin,
        uncertainty=uncertainty,
        period=period,
    )
