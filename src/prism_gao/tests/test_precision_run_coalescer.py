from prism_gao.precision_run_coalescer import (
    DROP,
    FP16,
    INT8,
    assign_16_8_drop,
    build_coalesced_16_8_drop_plan,
    build_read_groups,
    choose_min_profitable_int8_run,
)


def test_assignment_uses_importance_but_keeps_original_order():
    # Importance order is [6, 1, 4, 0], but the returned plan is indexed by
    # original block id: 0,1,2,...,7.
    plan = assign_16_8_drop(
        total_blocks=8,
        selected_blocks=[0, 1, 4, 6],
        priority_blocks=[6, 1, 4, 0],
        fp16_fraction=0.5,
    )
    assert plan == (INT8, FP16, DROP, DROP, INT8, DROP, FP16, DROP)


def test_short_int8_islands_are_promoted_and_drop_is_unchanged():
    result = build_coalesced_16_8_drop_plan(
        total_blocks=12,
        selected_blocks=[0, 1, 2, 4, 7, 8, 9, 10],
        priority_blocks=[0, 2, 7, 9, 1, 4, 8, 10],
        fp16_fraction=0.5,
        min_int8_run_blocks=2,
    )
    assert result.initial_tiers == (
        FP16, INT8, FP16, DROP, INT8, DROP, DROP, FP16, INT8, FP16, INT8, DROP
    )
    assert result.tiers == (
        FP16, FP16, FP16, DROP, FP16, DROP, DROP, FP16, FP16, FP16, FP16, DROP
    )
    assert result.selected_blocks == (0, 1, 2, 4, 7, 8, 9, 10)
    assert result.dropped_blocks == (3, 5, 6, 11)
    assert result.int8_runs_before == 4
    assert result.int8_runs_after == 0


def test_long_int8_run_remains_int8():
    result = build_coalesced_16_8_drop_plan(
        total_blocks=8,
        selected_blocks=[0, 1, 2, 3, 4, 5],
        priority_blocks=[0, 5, 1, 2, 3, 4],
        fp16_fraction=2 / 6,
        min_int8_run_blocks=3,
    )
    # Initial: fp16 at 0 and 5, a contiguous INT8 run 1..4 of length 4.
    assert result.tiers == (FP16, INT8, INT8, INT8, INT8, FP16, DROP, DROP)
    assert result.promoted_int8_blocks == ()


def test_read_groups_restore_original_selected_order():
    groups = build_read_groups((INT8, DROP, FP16, INT8, DROP, FP16))
    # Selected original order is blocks [0, 2, 3, 5].
    assert groups[FP16].blocks == (2, 5)
    assert groups[FP16].destination_slots == (1, 3)
    assert groups[INT8].blocks == (0, 3)
    assert groups[INT8].destination_slots == (0, 2)


def test_choose_break_even_run():
    assert choose_min_profitable_int8_run(
        [(1, 0.10, 0.15), (2, 0.18, 0.19), (4, 0.35, 0.30)],
        margin_ratio=0.05,
    ) == 4
    assert choose_min_profitable_int8_run(
        [(1, 0.10, 0.15), (2, 0.18, 0.19)], margin_ratio=0.05
    ) is None
