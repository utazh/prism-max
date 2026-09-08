"""Fixed-period controls for ProMixed without changing its block ranking."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable


def wrap_selector(original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a selector that optionally overrides only decision.period."""

    def select(*args: Any, **kwargs: Any) -> Any:
        decision = original(*args, **kwargs)
        raw = os.environ.get("PRISM_GAO_FIXED_PERIOD", "0").strip()
        fixed_period = int(raw or "0")
        if fixed_period == 0:
            return decision
        if fixed_period not in (1, 4, 8):
            raise ValueError(
                "PRISM_GAO_FIXED_PERIOD must be 0, 1, 4, or 8"
            )
        max_period = int(kwargs.get("max_period", fixed_period))
        if fixed_period > max_period:
            raise ValueError(
                f"fixed period {fixed_period} exceeds max period {max_period}"
            )
        return replace(decision, period=fixed_period)

    return select
