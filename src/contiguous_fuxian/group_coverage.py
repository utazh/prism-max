"""Experimental shared-chunk selection; no claim of measured model improvement.

Every selected block stores ALL physical KV heads, as in the current Pcache.
This is not a ragged per-head cache allocator. The greedy objective is
sum_g sqrt(sum_{c in S} score[g,c] / oracle_topk_mass[g]).
The oracle here is only a score reference, not an answer-quality oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GroupCoverageSelection:
    priority_blocks: tuple[int, ...]
    retained_mass: tuple[float, ...]
    relative_to_topk: tuple[float, ...]


def _prepare(scores: Sequence[Sequence[float]], keep_blocks: int):
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 2 or not all(a.shape):
        raise ValueError("scores must have nonempty [group, block] shape")
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("scores must be finite and non-negative")
    if isinstance(keep_blocks, bool) or not isinstance(keep_blocks, Integral):
        raise ValueError("keep_blocks must be an integer")
    if not 1 <= keep_blocks <= a.shape[1]:
        raise ValueError("keep_blocks is outside the block count")
    # Row rescaling avoids overflow without changing any relative coverage.
    peak = a.max(axis=1, keepdims=True)
    scaled = np.divide(a, peak, out=np.zeros_like(a), where=peak > 0)
    totals = scaled.sum(axis=1, keepdims=True)
    probability = np.divide(scaled, totals, out=np.zeros_like(a), where=totals > 0)
    oracle = np.partition(probability, a.shape[1] - keep_blocks, axis=1)[:, -keep_blocks:].sum(axis=1)
    relative = np.divide(probability, oracle[:, None], out=np.zeros_like(a), where=oracle[:, None] > 0)
    return a, probability, relative


def coverage_of(scores, selected: Sequence[int], *, keep_blocks: int):
    _, probability, relative = _prepare(scores, keep_blocks)
    ids = tuple(selected)
    if len(ids) != keep_blocks or len(set(ids)) != len(ids):
        raise ValueError("selection must contain exactly keep_blocks unique IDs")
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < probability.shape[1] for i in ids):
        raise ValueError("selected block ID is outside the score matrix")
    return (
        tuple(float(x) for x in probability[:, ids].sum(axis=1)),
        tuple(float(x) for x in relative[:, ids].sum(axis=1)),
    )


def select_group_blocks(scores, *, keep_blocks: int, strategy: str = "balanced") -> GroupCoverageSelection:
    """CPU NumPy reference. Exact cardinality, stable low-ID tie-breaking.

    balanced has O(groups * blocks * keep_blocks) arithmetic. Benchmark this
    overhead in the real request before considering it a speed optimization.
    mean/normalized_mean/max are controls, not new methods.
    """
    a, probability, relative = _prepare(scores, keep_blocks)
    if strategy == "balanced":
        covered = np.zeros(a.shape[0], dtype=np.float64)
        available = np.ones(a.shape[1], dtype=bool)
        chosen: list[int] = []
        for _ in range(keep_blocks):
            gains = (np.sqrt(covered[:, None] + relative) - np.sqrt(covered)[:, None]).sum(axis=0)
            gains[~available] = -np.inf
            block = int(np.argmax(gains))
            chosen.append(block)
            available[block] = False
            covered += relative[:, block]
    else:
        if strategy == "mean":
            # Global rescaling preserves the RAW head mean ranking.
            peak = float(a.max())
            utility = (a / peak).mean(axis=0) if peak else np.zeros(a.shape[1])
        elif strategy == "normalized_mean":
            utility = probability.mean(axis=0)
        elif strategy == "max":
            utility = probability.max(axis=0)
        else:
            raise ValueError("strategy must be balanced, mean, normalized_mean, or max")
        chosen = np.argsort(-utility, kind="stable")[:keep_blocks].tolist()
    retained, ratios = coverage_of(scores, chosen, keep_blocks=keep_blocks)
    return GroupCoverageSelection(tuple(chosen), retained, ratios)
