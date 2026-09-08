import pytest
import torch

from prism_gao.int4_selector import (
    dequantize_int4_keys,
    int4_qk_logits,
    reference_int4_qk_logits,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _inputs(dtype=torch.bfloat16):
    torch.manual_seed(7)
    tokens, heads, head_dim, group_size = 67, 4, 128, 32
    signed = torch.randint(
        -7, 8, (tokens, heads, head_dim), device="cuda", dtype=torch.int16
    )
    encoded = signed.to(torch.int8).to(torch.uint8) & 0x0F
    codes = (
        encoded[..., 0::2] | (encoded[..., 1::2] << 4)
    ).contiguous()
    scales = torch.rand(
        tokens,
        heads,
        head_dim // group_size,
        device="cuda",
        dtype=torch.float16,
    )
    query = torch.randn(
        heads, 23, head_dim, device="cuda", dtype=dtype
    )
    return query, codes, scales, group_size


def test_fused_dequant_is_bitwise_equal_to_torch_path():
    _, codes, scales, group_size = _inputs()
    tokens, heads, packed_dim = codes.shape
    head_dim = packed_dim * 2
    low = (codes & 0x0F).to(torch.int16)
    high = (codes >> 4).to(torch.int16)
    unpacked = torch.stack((low, high), dim=-1).reshape(
        tokens, heads, head_dim
    )
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    expected = (
        unpacked.reshape(
            tokens, heads, head_dim // group_size, group_size
        ).to(torch.float16)
        * scales.unsqueeze(-1)
    ).reshape(tokens, heads, head_dim)

    actual = dequantize_int4_keys(
        codes, scales, group_size=group_size
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype,atol", [
    (torch.float16, 0.0625),
    (torch.bfloat16, 0.5),
])
def test_direct_qk_matches_reference_with_expected_tensorcore_tolerance(
    dtype, atol
):
    query, codes, scales, group_size = _inputs(dtype)
    actual = int4_qk_logits(
        query, codes, scales, group_size=group_size, scaling=128 ** -0.5
    )
    expected = reference_int4_qk_logits(
        query, codes, scales, group_size=group_size, scaling=128 ** -0.5
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, atol=atol, rtol=0.01)
