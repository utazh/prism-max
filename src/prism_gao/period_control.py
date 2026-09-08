"""Small, opt-in controls for ProMixed cross-layer reuse.

Policies:
- ``original``: keep the selector's original P1/P2/P4/P8 decision.
- ``budget_v2``: preserve the original decision in the middle budget range,
  cap long reuse at P4 for high retention, and allow P8 more readily for
  low retention.  Block selection/ranking is never changed.

Environment variables are used so the existing runner CLI does not need to be
modified.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any, Callable, Sequence

from .profiled_reuse import ProfiledAdaptiveReuse

_VALID_PERIODS = (1, 2, 4, 8)
_PROFILE_CACHE: dict[str, ProfiledAdaptiveReuse] = {}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else float(default)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _global_keep_ratio(args: Sequence[Any], kwargs: dict[str, Any]) -> float:
    """Use the run-level ratio when supplied, otherwise derive the local ratio."""

    raw = os.environ.get("PRISM_GAO_GLOBAL_KEEP_RATIO", "").strip()
    if raw:
        ratio = float(raw)
    else:
        block_scores = kwargs.get("block_scores")
        if block_scores is None and args:
            block_scores = args[0]
        if not block_scores or not block_scores[0]:
            raise ValueError(
                "budget_v2 needs PRISM_GAO_GLOBAL_KEEP_RATIO or non-empty block_scores"
            )
        ratio = int(kwargs["keep_blocks"]) / len(block_scores[0])
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("keep ratio must be finite and in (0, 1]")
    return ratio


def _low_budget_period(uncertainty: float, *, max_period: int, kwargs: dict[str, Any]) -> int:
    """Remap only the P4/P8 boundary; keep conservative P1/P2 guards."""

    p1 = float(kwargs.get("p1_threshold", 0.90))
    p2 = float(kwargs.get("p2_threshold", 0.82))
    p4 = _env_float("PRISM_GAO_LOW_BUDGET_P4_THRESHOLD", 0.76)
    if not 0.0 <= p4 <= p2 <= p1 <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= low_budget_p4 <= p2 <= p1 <= 1"
        )
    if uncertainty >= p1:
        requested = 1
    elif uncertainty >= p2:
        requested = 2
    elif uncertainty >= p4:
        requested = 4
    else:
        requested = 8
    return min(requested, max_period)


def wrap_selector(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``select_promixed_gqa_blocks`` without changing selected blocks."""

    def select(*args: Any, **kwargs: Any) -> Any:
        decision = original(*args, **kwargs)

        fixed_raw = os.environ.get("PRISM_GAO_FIXED_PERIOD", "0").strip()
        fixed = int(fixed_raw or "0")
        max_period = int(kwargs.get("max_period", 8))
        if max_period not in _VALID_PERIODS:
            raise ValueError("max_period must be one of 1, 2, 4, or 8")
        if fixed:
            if fixed not in _VALID_PERIODS:
                raise ValueError(
                    "PRISM_GAO_FIXED_PERIOD must be 0, 1, 2, 4, or 8"
                )
            return replace(decision, period=min(fixed, max_period))

        policy = os.environ.get("PRISM_GAO_PERIOD_POLICY", "original").strip().lower()
        if policy in {"", "original", "v1"}:
            return decision
        if policy == "profiled":
            path = os.environ.get("PRISM_GAO_REUSE_PROFILE", "").strip()
            if not path:
                raise ValueError("profiled policy requires PRISM_GAO_REUSE_PROFILE")
            reuse_policy = _PROFILE_CACHE.get(path)
            if reuse_policy is None:
                reuse_policy = ProfiledAdaptiveReuse.from_json(path)
                _PROFILE_CACHE[path] = reuse_policy
            task = os.environ.get("PRISM_GAO_TASK", os.environ.get("TASK", "")).strip().lower()
            if not task:
                raise ValueError("profiled policy requires PRISM_GAO_TASK or TASK")
            period = reuse_policy.choose(
                task=task,
                budget=_global_keep_ratio(args, kwargs),
                uncertainty=float(decision.uncertainty),
            )
            period = min(int(period), max_period)
            return decision if period == int(decision.period) else replace(decision, period=period)
        if policy != "budget_v2":
            raise ValueError(
                "PRISM_GAO_PERIOD_POLICY must be original, budget_v2, or profiled"
            )

        ratio = _global_keep_ratio(args, kwargs)
        low_budget_max = _env_float("PRISM_GAO_LOW_BUDGET_MAX", 0.125)
        high_budget_min = _env_float("PRISM_GAO_HIGH_BUDGET_MIN", 0.40)
        if not 0.0 < low_budget_max < high_budget_min <= 1.0:
            raise ValueError(
                "budget thresholds must satisfy 0 < low_budget_max < high_budget_min <= 1"
            )

        if ratio >= high_budget_min:
            # Dense retention has a weak K/K+1 boundary; never trust P8.  P1/P2
            # decisions are kept, so this is still uncertainty-adaptive.
            period = min(int(decision.period), 4, max_period)
        elif ratio <= low_budget_max:
            # Sparse retention benefits most from amortizing the selector.  Make
            # P8 easier while retaining the original high-uncertainty P1/P2 guards.
            period = _low_budget_period(
                float(decision.uncertainty), max_period=max_period, kwargs=kwargs
            )
        else:
            period = min(int(decision.period), max_period)

        return decision if period == int(decision.period) else replace(decision, period=period)

    return select
