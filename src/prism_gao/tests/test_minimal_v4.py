
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prism_gao.profiled_reuse import (
    PeriodMeasurement,
    ProfiledAdaptiveReuse,
    build_profile,
    choose_period,
)
from prism_gao.async_mixed_pipeline import MixedCostCalibration, MixedPrecisionGate
from prism_gao.selector_resident_cache import PackedSelectorLayer, SelectorResidentCache


def test_profile_chooses_fast_period_within_one_example() -> None:
    rows = [
        PeriodMeasurement("trec", 0.1, 1, 44, 64, 340.0),
        PeriodMeasurement("trec", 0.1, 4, 44, 64, 285.0),
        PeriodMeasurement("trec", 0.1, 8, 43, 64, 240.0),
    ]
    assert choose_period(rows, quality_slack_examples=1) == 8
    assert choose_period(rows, quality_slack_examples=0) == 4


def test_profile_guard_only_shortens_period() -> None:
    rows = [
        PeriodMeasurement("subj", 0.25, 1, 52, 64, 500.0),
        PeriodMeasurement("subj", 0.25, 4, 52, 64, 450.0),
        PeriodMeasurement("subj", 0.25, 8, 40, 64, 420.0),
    ]
    profile = build_profile(rows, guard_threshold=0.95)
    policy = ProfiledAdaptiveReuse(profile)
    assert policy.choose(task="subj", budget=0.25, uncertainty=0.50) == 4
    assert policy.choose(task="subj", budget=0.25, uncertainty=0.96) == 2


def test_cost_gate_rejects_hidden_io() -> None:
    gate = MixedPrecisionGate(
        MixedCostCalibration(
            io_gib_per_s=3.0,
            dequant_giga_elements_per_s=100.0,
            exposed_io_fraction=0.0,
            minimum_gain_ms=0.01,
        )
    )
    assert not gate.keep_mixed(
        fp16_bytes=1_000_000,
        mixed_bytes=500_000,
        int8_elements=500_000,
        int8_runs=1,
    )


def test_cost_gate_accepts_io_bound_layer() -> None:
    gate = MixedPrecisionGate(
        MixedCostCalibration(
            io_gib_per_s=0.5,
            dequant_giga_elements_per_s=500.0,
            launch_overhead_us=3.0,
            exposed_io_fraction=1.0,
            minimum_gain_ms=0.01,
        )
    )
    assert gate.keep_mixed(
        fp16_bytes=8_000_000,
        mixed_bytes=4_000_000,
        int8_elements=4_000_000,
        int8_runs=1,
    )


def test_cpu_selector_cache_without_pinning() -> None:
    def load(layer: int):
        return PackedSelectorLayer(
            codes=torch.full((4,), layer, dtype=torch.uint8),
            scales=torch.ones((2,), dtype=torch.float16),
            meta={"layer": layer},
        )

    cache = SelectorResidentCache(
        load,
        device="cpu",
        max_gpu_bytes=0,
        pin_fallback=False,
    )
    assert cache.preload([0, 1]) == "cpu"
    assert cache.get(1).meta["layer"] == 1
    assert cache.total_bytes > 0
