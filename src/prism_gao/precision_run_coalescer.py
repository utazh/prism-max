"""Minimal 16/8/drop precision planner for PRISM.

The planner does *not* physically reorder KV blocks.

1. Use ``priority_blocks`` only to decide which already-selected blocks receive
   FP16 versus INT8.
2. Write those labels back by original block id.
3. In original physical order, promote short INT8 runs to FP16 when INT8 is not
   expected to amortize its read/dequantization overhead.
4. Never change a ``drop`` block, so the sparse selection/budget is unchanged.

The same run-coalescing helper can later be reused for 16/8/4 by applying it
first to INT4 -> INT8 and then, optionally, INT8 -> FP16.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


FP16 = "fp16"
INT8 = "int8"
DROP = "drop"
_VALID_TIERS = frozenset({FP16, INT8, DROP})


@dataclass(frozen=True)
class TierRun:
    """One half-open run ``[start, end)`` in original block order."""

    tier: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TierReadGroup:
    """Blocks of one precision and their destination slots after gathering."""

    blocks: tuple[int, ...]
    destination_slots: tuple[int, ...]


@dataclass(frozen=True)
class CoalescedPrecisionPlan:
    """Initial and coalesced plans plus small audit statistics."""

    initial_tiers: tuple[str, ...]
    tiers: tuple[str, ...]
    promoted_int8_blocks: tuple[int, ...]
    int8_runs_before: int
    int8_runs_after: int

    @property
    def selected_blocks(self) -> tuple[int, ...]:
        return tuple(i for i, tier in enumerate(self.tiers) if tier != DROP)

    @property
    def fp16_blocks(self) -> tuple[int, ...]:
        return tuple(i for i, tier in enumerate(self.tiers) if tier == FP16)

    @property
    def int8_blocks(self) -> tuple[int, ...]:
        return tuple(i for i, tier in enumerate(self.tiers) if tier == INT8)

    @property
    def dropped_blocks(self) -> tuple[int, ...]:
        return tuple(i for i, tier in enumerate(self.tiers) if tier == DROP)


def _normalize_unique_blocks(
    values: Iterable[int], *, total_blocks: int, name: str
) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate block ids")
    if any(block < 0 or block >= total_blocks for block in normalized):
        raise ValueError(f"{name} contains a block outside [0, {total_blocks})")
    return normalized


def encode_runs(tiers: Sequence[str]) -> tuple[TierRun, ...]:
    """Run-length encode a tier plan in original block order."""

    normalized = tuple(str(tier).lower() for tier in tiers)
    if not normalized:
        return ()
    invalid = set(normalized) - _VALID_TIERS
    if invalid:
        raise ValueError(f"unsupported precision tiers: {sorted(invalid)}")

    runs: list[TierRun] = []
    start = 0
    current = normalized[0]
    for index in range(1, len(normalized)):
        if normalized[index] != current:
            runs.append(TierRun(current, start, index))
            start = index
            current = normalized[index]
    runs.append(TierRun(current, start, len(normalized)))
    return tuple(runs)


def assign_16_8_drop(
    *,
    total_blocks: int,
    selected_blocks: Sequence[int],
    priority_blocks: Sequence[int],
    fp16_fraction: float,
) -> tuple[str, ...]:
    """Assign FP16/INT8 to selected blocks without changing physical order.

    ``priority_blocks`` is an importance ranking only. The returned tuple is
    indexed by the original block id and therefore performs no IMPRESS-style
    physical reordering.
    """

    total_blocks = int(total_blocks)
    if total_blocks <= 0:
        raise ValueError("total_blocks must be positive")
    fraction = float(fp16_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fp16_fraction must be finite and in [0, 1]")

    selected = _normalize_unique_blocks(
        selected_blocks, total_blocks=total_blocks, name="selected_blocks"
    )
    priority = _normalize_unique_blocks(
        priority_blocks, total_blocks=total_blocks, name="priority_blocks"
    )
    if not selected:
        raise ValueError("at least one block must be selected")
    if set(priority) != set(selected):
        raise ValueError("priority_blocks must be a permutation of selected_blocks")

    if fraction == 0.0:
        fp16_count = 0
    else:
        fp16_count = min(len(priority), max(1, math.ceil(len(priority) * fraction)))
    fp16_blocks = set(priority[:fp16_count])
    selected_set = set(selected)

    return tuple(
        FP16
        if block in fp16_blocks
        else INT8
        if block in selected_set
        else DROP
        for block in range(total_blocks)
    )


def promote_short_int8_runs(
    tiers: Sequence[str], *, min_int8_run_blocks: int
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Promote unprofitable short INT8 runs to FP16.

    A run is measured only in original physical block order. ``drop`` is never
    changed, so this function cannot expand the selected set or violate the KV
    retention budget.
    """

    threshold = int(min_int8_run_blocks)
    if threshold <= 0:
        raise ValueError("min_int8_run_blocks must be positive")

    output = [str(tier).lower() for tier in tiers]
    promoted: list[int] = []
    for run in encode_runs(output):
        if run.tier == INT8 and run.length < threshold:
            for block in range(run.start, run.end):
                output[block] = FP16
                promoted.append(block)
    return tuple(output), tuple(promoted)


def build_coalesced_16_8_drop_plan(
    *,
    total_blocks: int,
    selected_blocks: Sequence[int],
    priority_blocks: Sequence[int],
    fp16_fraction: float,
    min_int8_run_blocks: int,
) -> CoalescedPrecisionPlan:
    """Build the minimal cost-aware 16/8/drop plan from the user's sketch."""

    initial = assign_16_8_drop(
        total_blocks=total_blocks,
        selected_blocks=selected_blocks,
        priority_blocks=priority_blocks,
        fp16_fraction=fp16_fraction,
    )
    final, promoted = promote_short_int8_runs(
        initial, min_int8_run_blocks=min_int8_run_blocks
    )
    before = sum(run.tier == INT8 for run in encode_runs(initial))
    after = sum(run.tier == INT8 for run in encode_runs(final))
    if tuple(i for i, tier in enumerate(initial) if tier != DROP) != tuple(
        i for i, tier in enumerate(final) if tier != DROP
    ):
        raise AssertionError("coalescing unexpectedly changed the sparse selection")
    return CoalescedPrecisionPlan(
        initial_tiers=initial,
        tiers=final,
        promoted_int8_blocks=promoted,
        int8_runs_before=before,
        int8_runs_after=after,
    )


def build_read_groups(tiers: Sequence[str]) -> Mapping[str, TierReadGroup]:
    """Group reads by precision while preserving original-order output slots.

    The storage layer may issue one FP16 request and one INT8 request. A fused
    materialization kernel then writes both groups into ``destination_slots``
    so the final K/V tensor follows original selected-block order.
    """

    normalized = tuple(str(tier).lower() for tier in tiers)
    encode_runs(normalized)  # validation
    selected = [block for block, tier in enumerate(normalized) if tier != DROP]
    destination = {block: slot for slot, block in enumerate(selected)}
    result: dict[str, TierReadGroup] = {}
    for tier in (FP16, INT8):
        blocks = tuple(block for block, label in enumerate(normalized) if label == tier)
        result[tier] = TierReadGroup(
            blocks=blocks,
            destination_slots=tuple(destination[block] for block in blocks),
        )
    return result


def choose_min_profitable_int8_run(
    measurements: Sequence[tuple[int, float, float]], *, margin_ratio: float = 0.05
) -> int | None:
    """Choose the first measured run length where INT8 clearly beats FP16.

    Each row is ``(run_blocks, fp16_ms, int8_ms)``.  With the default 5% safety
    margin, INT8 is considered profitable when ``int8_ms <= 0.95 * fp16_ms``.
    The returned value can be passed directly as ``min_int8_run_blocks``.
    """

    margin = float(margin_ratio)
    if not math.isfinite(margin) or not 0.0 <= margin < 1.0:
        raise ValueError("margin_ratio must be finite and in [0, 1)")
    normalized = sorted(
        (int(blocks), float(fp16_ms), float(int8_ms))
        for blocks, fp16_ms, int8_ms in measurements
    )
    for blocks, fp16_ms, int8_ms in normalized:
        if blocks <= 0 or fp16_ms <= 0 or int8_ms <= 0:
            raise ValueError("benchmark blocks and times must be positive")
        if int8_ms <= fp16_ms * (1.0 - margin):
            return blocks
    return None


def _parse_blocks(text: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in text.split(",") if piece.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a coalesced PRISM 16/8/drop plan")
    parser.add_argument("--total-blocks", type=int, required=True)
    parser.add_argument("--selected", required=True, help="comma-separated selected block ids")
    parser.add_argument("--priority", required=True, help="importance order of selected blocks")
    parser.add_argument("--fp16-fraction", type=float, default=0.25)
    parser.add_argument("--min-int8-run-blocks", type=int, default=4)
    args = parser.parse_args()

    plan = build_coalesced_16_8_drop_plan(
        total_blocks=args.total_blocks,
        selected_blocks=_parse_blocks(args.selected),
        priority_blocks=_parse_blocks(args.priority),
        fp16_fraction=args.fp16_fraction,
        min_int8_run_blocks=args.min_int8_run_blocks,
    )
    groups = build_read_groups(plan.tiers)
    print(
        json.dumps(
            {
                "initial_tiers": plan.initial_tiers,
                "tiers": plan.tiers,
                "promoted_int8_blocks": plan.promoted_int8_blocks,
                "int8_runs_before": plan.int8_runs_before,
                "int8_runs_after": plan.int8_runs_after,
                "fp16_blocks": groups[FP16].blocks,
                "fp16_destination_slots": groups[FP16].destination_slots,
                "int8_blocks": groups[INT8].blocks,
                "int8_destination_slots": groups[INT8].destination_slots,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
