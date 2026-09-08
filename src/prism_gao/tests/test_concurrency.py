from __future__ import annotations

import math

import pytest

from prism_gao.concurrency_io import aligned_range
from prism_gao.benchmark_concurrency import make_plans, percentile
from prism_gao.precision_run_coalescer import DROP, FP16, INT8


@pytest.mark.parametrize("offset,size,expected", [
    (0, 16384, (0, 16384, 0)),
    (513, 512, (512, 1024, 1)),
    (190821, 1555, (190464, 2048, 357)),
])
def test_direct_alignment(offset, size, expected):
    assert aligned_range(offset, size) == expected


@pytest.mark.parametrize("offset,size", [(-1, 10), (0, -1)])
def test_reject_bad_alignment(offset, size):
    with pytest.raises(ValueError):
        aligned_range(offset, size)


def test_replay_freezes_selection_and_never_fills_drop():
    trace = {"prefix_tokens": 16 * 10 - 3, "block_size": 16,
             "layers": [{"layer": 0,
                         "selected_blocks": [0, 1, 2, 4, 6, 7, 8, 9],
                         "priority_blocks": [2, 4, 0, 1, 6, 7, 8, 9]}]}
    fp16 = make_plans(trace, "fp16")[0][1]
    naive = make_plans(trace, "naive")[0][1]
    coalesced = make_plans(trace, "coalesced", min_run=3)[0][1]
    assert fp16.selected_blocks == naive.selected_blocks == coalesced.selected_blocks
    assert fp16.dropped_blocks == naive.dropped_blocks == coalesced.dropped_blocks
    assert set(naive.fp16_blocks) <= set(coalesced.fp16_blocks)
    assert set(coalesced.int8_blocks) <= set(naive.int8_blocks)
    assert coalesced.tiers[0:3] == (FP16, FP16, FP16)
    assert coalesced.tiers[3] == DROP
    assert coalesced.tiers[6:10] == (INT8,) * 4


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert math.isclose(percentile([1, 2, 3, 4], .95), 3.85)
