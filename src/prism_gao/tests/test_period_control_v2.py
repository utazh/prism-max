from dataclasses import dataclass

import pytest

from prism_gao.period_control import wrap_selector


@dataclass(frozen=True)
class Decision:
    selected_blocks: tuple[int, ...] = (1, 3)
    priority_blocks: tuple[int, ...] = (3, 1)
    uncertainty: float = 0.70
    period: int = 4


def _original(*args, **kwargs):
    return Decision(
        uncertainty=float(kwargs.pop("test_uncertainty", 0.70)),
        period=int(kwargs.pop("test_period", 4)),
    )


def _call(monkeypatch, *, ratio, uncertainty, period, policy="budget_v2"):
    monkeypatch.setenv("PRISM_GAO_PERIOD_POLICY", policy)
    monkeypatch.setenv("PRISM_GAO_GLOBAL_KEEP_RATIO", str(ratio))
    monkeypatch.delenv("PRISM_GAO_FIXED_PERIOD", raising=False)
    wrapped = wrap_selector(_original)
    return wrapped(
        [[1.0] * 100] * 4,
        keep_blocks=max(1, int(ratio * 100)),
        max_period=8,
        p1_threshold=0.90,
        p2_threshold=0.82,
        p4_threshold=0.68,
        test_uncertainty=uncertainty,
        test_period=period,
    )


def test_high_budget_caps_only_long_period(monkeypatch):
    result = _call(monkeypatch, ratio=0.50, uncertainty=0.60, period=8)
    assert result.period == 4
    assert result.selected_blocks == (1, 3)
    assert result.priority_blocks == (3, 1)


def test_high_budget_preserves_p2(monkeypatch):
    result = _call(monkeypatch, ratio=0.50, uncertainty=0.85, period=2)
    assert result.period == 2


def test_low_budget_uses_relaxed_p4_boundary(monkeypatch):
    monkeypatch.setenv("PRISM_GAO_LOW_BUDGET_P4_THRESHOLD", "0.76")
    assert _call(monkeypatch, ratio=0.10, uncertainty=0.72, period=4).period == 8
    assert _call(monkeypatch, ratio=0.10, uncertainty=0.78, period=4).period == 4
    assert _call(monkeypatch, ratio=0.10, uncertainty=0.85, period=2).period == 2


def test_middle_budget_is_unchanged(monkeypatch):
    assert _call(monkeypatch, ratio=0.25, uncertainty=0.70, period=8).period == 8


def test_fixed_period_still_wins(monkeypatch):
    monkeypatch.setenv("PRISM_GAO_PERIOD_POLICY", "budget_v2")
    monkeypatch.setenv("PRISM_GAO_FIXED_PERIOD", "4")
    wrapped = wrap_selector(_original)
    result = wrapped(
        [[1.0] * 10] * 4,
        keep_blocks=1,
        max_period=2,
        test_uncertainty=0.1,
        test_period=8,
    )
    assert result.period == 2
