"""GQA-aware, uncertainty-gated block selection for ProMixed Re-Prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class PromixedSelectionDecision:
    """A fixed-budget selection plus the heuristic or explicitly fixed reuse horizon."""

    selected_blocks: tuple[int, ...]
    priority_blocks: tuple[int, ...]
    agreement: float
    boundary_margin: float
    uncertainty: float
    period: int
    effective_coverage_fraction: float



# Explicit run-scoped options used by audited_reprefill. The historical runner
# is unchanged outside this context; no process-global function monkey-patch.
_RUN_OPTIONS: ContextVar[dict | None] = ContextVar("promixed_run_options", default=None)
_TRACE: ContextVar[list | None] = ContextVar("promixed_run_trace", default=None)


@contextmanager
def selection_experiment(*, fixed_period: int | None = 8,
                         strategy: str = "legacy", coverage_enabled: bool = True,
                         trace: list | None = None, preload_anchors_only: bool = False):
    if fixed_period is not None and (isinstance(fixed_period, bool)
            or not isinstance(fixed_period, Integral) or fixed_period not in {1, 2, 4, 8}):
        raise ValueError("fixed_period must be None, 1, 2, 4, or 8")
    if strategy not in {"legacy", "balanced", "mean", "normalized_mean", "max"}:
        raise ValueError("unknown selection strategy")
    if strategy != "legacy" and fixed_period is None:
        raise ValueError("experimental selectors require a fixed period")
    if not isinstance(coverage_enabled, bool):
        raise ValueError("coverage_enabled must be boolean")
    options_token = _RUN_OPTIONS.set(dict(fixed_period=fixed_period,
        strategy=strategy, coverage_enabled=coverage_enabled,
        preload_anchors_only=preload_anchors_only))
    trace_token = _TRACE.set(trace)
    try:
        yield
    finally:
        _TRACE.reset(trace_token)
        _RUN_OPTIONS.reset(options_token)


def current_selection_experiment() -> dict | None:
    options = _RUN_OPTIONS.get()
    return dict(options) if options is not None else None


def _observe(decision, rows, keep_blocks, strategy, coverage_blocks):
    trace = _TRACE.get()
    if trace is not None:
        from .group_coverage import coverage_of
        retained, relative = coverage_of(rows, decision.selected_blocks, keep_blocks=keep_blocks)
        trace.append({"invocation": len(trace), "keep_blocks": keep_blocks,
            "strategy": strategy, "period": decision.period,
            "selected_blocks": list(decision.selected_blocks),
            "group_retained_mass": retained, "group_relative_to_topk": relative,
            "reserved_coverage_blocks": coverage_blocks, "score_rows": rows})
    return decision


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
    adaptive_coverage: bool = False,
    utility_max_weight: float = 0.55,
    utility_mean_weight: float = 0.35,
    utility_vote_weight: float = 0.10,
    fixed_period: int | None = None,
    coverage_enabled: bool = True,
    selection_strategy: str = "legacy",
) -> PromixedSelectionDecision:
    """Select exact-budget blocks without averaging away a minority GQA group.

    A fraction of the budget is filled round-robin from each physical GQA
    group's ranking. The remainder uses a scale-normalized max/mean/vote fusion.
    Cross-group Jaccard, top-K boundary margin, and offline layer risk jointly
    choose a P1/P2/P4/P8 reuse horizon.
    """

    rows = _validate_score_rows(block_scores)
    block_count = len(rows[0])
    head_count = len(rows)
    options = _RUN_OPTIONS.get()
    if options is not None:
        fixed_period = options["fixed_period"]
        coverage_enabled = options["coverage_enabled"]
        selection_strategy = options["strategy"]
    if isinstance(keep_blocks, bool) or not isinstance(keep_blocks, Integral):
        raise ValueError("ProMixed keep_blocks must be an integer")
    if fixed_period is not None and (isinstance(fixed_period, bool)
            or not isinstance(fixed_period, Integral) or fixed_period not in {1, 2, 4, 8}
            or fixed_period > max_period):
        raise ValueError("fixed_period must be 1, 2, 4, or 8 and <= max_period")
    if selection_strategy not in {"legacy", "balanced", "mean", "normalized_mean", "max"}:
        raise ValueError("unknown selection strategy")
    if selection_strategy != "legacy" and fixed_period is None:
        raise ValueError("experimental selectors require a fixed period")
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
    utility_weights = (
        float(utility_max_weight),
        float(utility_mean_weight),
        float(utility_vote_weight),
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in utility_weights):
        raise ValueError("ProMixed utility weights must be finite and non-negative")
    if not math.isclose(
        sum(utility_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("ProMixed utility weights must sum to 1")

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

    if fixed_period is not None:
        period = int(fixed_period)
    if selection_strategy != "legacy":
        from .group_coverage import select_group_blocks
        result = select_group_blocks(rows, keep_blocks=keep_blocks, strategy=selection_strategy)
        decision = PromixedSelectionDecision(
            selected_blocks=tuple(sorted(result.priority_blocks)),
            priority_blocks=result.priority_blocks, agreement=agreement,
            boundary_margin=boundary_margin, uncertainty=uncertainty,
            period=period, effective_coverage_fraction=0.0)
        return _observe(decision, rows, keep_blocks, selection_strategy, 0)

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
            utility_weights[0] * max(group_values)
            + utility_weights[1] * (sum(group_values) / head_count)
            + utility_weights[2] * score_scale * (votes[block] / head_count)
        )

    effective_coverage_fraction = (
        0.0
        if adaptive_coverage and uncertainty < p4_threshold
        else coverage_fraction
    )
    coverage_target = min(
        keep_blocks,
        max(
            min(head_count, keep_blocks),
            math.ceil(keep_blocks * effective_coverage_fraction),
        ),
    )
    # Preserve historical coverage_fraction=0 semantics. A SEPARATE explicit
    # switch is needed for a genuine no-round-robin-coverage ablation.
    if not coverage_enabled:
        coverage_target = 0
        effective_coverage_fraction = 0.0
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
    decision = PromixedSelectionDecision(
        selected_blocks=tuple(sorted(selected)),
        priority_blocks=priority,
        agreement=agreement,
        boundary_margin=boundary_margin,
        uncertainty=uncertainty,
        period=period,
        effective_coverage_fraction=effective_coverage_fraction,
    )
    return _observe(decision, rows, keep_blocks, selection_strategy, coverage_target)
