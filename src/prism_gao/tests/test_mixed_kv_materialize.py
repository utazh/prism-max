import pytest
import torch

from prism_gao.mixed_kv_materialize import (
    mixed_materialize,
    reference_mixed_materialize,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("tiers", [
    [0, 1, 1, 0, 1],
    [0, 0],
    [1, 1, 1],
])
def test_mixed_materialize_matches_reference(tiers):
    torch.manual_seed(11)
    device = torch.device("cuda")
    block_elements = 1024
    group_size = 32
    n16 = tiers.count(0)
    n8 = tiers.count(1)
    fp16 = torch.randn(
        n16, block_elements, device=device, dtype=torch.float16
    )
    int8 = torch.randint(
        -127, 128, (n8, block_elements), device=device, dtype=torch.int8
    )
    scales = torch.rand(
        n8,
        block_elements // group_size,
        device=device,
        dtype=torch.float16,
    )
    next_slot = {0: 0, 1: 0}
    sources = []
    for tier in tiers:
        sources.append(next_slot[tier])
        next_slot[tier] += 1
    tier_tensor = torch.tensor(tiers, device=device, dtype=torch.int32)
    source_tensor = torch.tensor(sources, device=device, dtype=torch.int32)

    actual = mixed_materialize(
        fp16_blocks=fp16,
        int8_blocks=int8,
        int8_scales=scales,
        tier_by_selected_slot=tier_tensor,
        source_slot_by_selected_slot=source_tensor,
        group_size=group_size,
    )
    expected = reference_mixed_materialize(
        fp16_blocks=fp16,
        int8_blocks=int8,
        int8_scales=scales,
        tier_by_selected_slot=tier_tensor,
        source_slot_by_selected_slot=source_tensor,
        group_size=group_size,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
