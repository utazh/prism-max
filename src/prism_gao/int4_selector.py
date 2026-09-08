"""Reference Triton kernel for ProMixed's packed signed-INT4 selector index.

This is an implementation skeleton for Codex. It is intentionally isolated from
project plumbing. The target operation is:

    query[h, q, d] @ dequant(key[token, h, d]).T

where key codes are packed two signed 4-bit values per uint8 and use one FP16
scale per token/head/group. The kernel fuses unpack + scale + QK, so it does not
materialize a full FP16/BF16 key tensor.

Expected current geometry: Qwen2.5-7B, 4 probe heads, head_dim=128,
group_size=32, CUDA/Ampere. Validate and benchmark on the target RTX 3090.
"""

from __future__ import annotations

from typing import Final

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("Triton is required for the fused INT4 selector kernel") from exc


_SUPPORTED_QUERY_DTYPES: Final = (torch.float16, torch.bfloat16)


@triton.jit
def _int4_qk_kernel(
    q_ptr,
    code_ptr,
    scale_ptr,
    out_ptr,
    q_tokens,
    prefix_tokens,
    head_dim,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_ct,
    stride_ch,
    stride_cd,
    stride_st,
    stride_sh,
    stride_sg,
    stride_oh,
    stride_oq,
    stride_ot,
    scaling,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUERY_IS_BF16: tl.constexpr,
):
    """One program computes [BLOCK_M query tokens, BLOCK_N prefix tokens]."""

    head = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_mask = (offs_m[:, None] < q_tokens) & (offs_k[None, :] < head_dim)
    q = tl.load(
        q_ptr
        + head * stride_qh
        + offs_m[:, None] * stride_qq
        + offs_k[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )

    # Build a [K, N] key tile directly from packed uint8 codes. A byte is
    # loaded twice (once for each nibble); this keeps v1 simple. Optimize only
    # after the end-to-end benchmark proves this path matters.
    byte_k = offs_k // 2
    code_mask = (offs_k[:, None] < head_dim) & (offs_n[None, :] < prefix_tokens)
    packed = tl.load(
        code_ptr
        + offs_n[None, :] * stride_ct
        + head * stride_ch
        + byte_k[:, None] * stride_cd,
        mask=code_mask,
        other=0,
    ).to(tl.int32)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble = tl.where((offs_k[:, None] & 1) == 0, low, high)
    signed = tl.where(nibble >= 8, nibble - 16, nibble)

    group_id = offs_k // GROUP_SIZE
    scale = tl.load(
        scale_ptr
        + offs_n[None, :] * stride_st
        + head * stride_sh
        + group_id[:, None] * stride_sg,
        mask=code_mask,
        other=0.0,
    )
    key = signed.to(tl.float32) * scale.to(tl.float32)
    if QUERY_IS_BF16:
        key = key.to(tl.bfloat16)
    else:
        key = key.to(tl.float16)

    # q: [M, K], key: [K, N]. Accumulate in FP32, store in query dtype.
    acc = tl.dot(q, key, out_dtype=tl.float32) * scaling
    out_mask = (offs_m[:, None] < q_tokens) & (offs_n[None, :] < prefix_tokens)
    tl.store(
        out_ptr
        + head * stride_oh
        + offs_m[:, None] * stride_oq
        + offs_n[None, :] * stride_ot,
        acc,
        mask=out_mask,
    )


@torch.inference_mode()
def int4_qk_logits(
    query: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    scaling: float,
) -> torch.Tensor:
    """Return prefix logits with shape [probe_head, query_token, prefix_token].

    Shapes:
      query:  [H, Q, D], FP16/BF16 CUDA, contiguous
      codes:  [T, H, D/2], uint8 CUDA, contiguous
      scales: [T, H, D/group_size], FP16 CUDA, contiguous
    """

    if not (query.is_cuda and codes.is_cuda and scales.is_cuda):
        raise ValueError("query, codes, and scales must be CUDA tensors")
    if query.dtype not in _SUPPORTED_QUERY_DTYPES:
        raise TypeError(f"unsupported query dtype: {query.dtype}")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise TypeError("codes must be uint8 and scales must be float16")
    if query.ndim != 3 or codes.ndim != 3 or scales.ndim != 3:
        raise ValueError("expected query/codes/scales to be rank-3")

    query = query.contiguous()
    codes = codes.contiguous()
    scales = scales.contiguous()

    heads, q_tokens, head_dim = map(int, query.shape)
    prefix_tokens, code_heads, packed_dim = map(int, codes.shape)
    scale_tokens, scale_heads, group_count = map(int, scales.shape)
    if code_heads != heads or scale_heads != heads:
        raise ValueError("probe-head dimensions do not match")
    if scale_tokens != prefix_tokens:
        raise ValueError("code and scale token dimensions do not match")
    if head_dim % 2 or packed_dim * 2 != head_dim:
        raise ValueError("packed INT4 width does not match query head_dim")
    if group_size <= 0 or head_dim % group_size:
        raise ValueError("group_size must divide head_dim")
    if group_count != head_dim // group_size:
        raise ValueError("scale group count does not match group_size")

    # Initial tuning point for RTX 3090 / D=128. Add autotune only after this
    # simple version is correct and useful end to end.
    block_m = 16
    block_n = 32
    block_k = triton.next_power_of_2(head_dim)
    output = torch.empty(
        (heads, q_tokens, prefix_tokens),
        device=query.device,
        dtype=query.dtype,
    )
    grid = (
        heads,
        triton.cdiv(q_tokens, block_m),
        triton.cdiv(prefix_tokens, block_n),
    )
    _int4_qk_kernel[grid](
        query,
        codes,
        scales,
        output,
        q_tokens=q_tokens,
        prefix_tokens=prefix_tokens,
        head_dim=head_dim,
        stride_qh=query.stride(0),
        stride_qq=query.stride(1),
        stride_qd=query.stride(2),
        stride_ct=codes.stride(0),
        stride_ch=codes.stride(1),
        stride_cd=codes.stride(2),
        stride_st=scales.stride(0),
        stride_sh=scales.stride(1),
        stride_sg=scales.stride(2),
        stride_oh=output.stride(0),
        stride_oq=output.stride(1),
        stride_ot=output.stride(2),
        scaling=float(scaling),
        GROUP_SIZE=int(group_size),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        QUERY_IS_BF16=query.dtype == torch.bfloat16,
        num_warps=4,
        num_stages=2,
    )
    return output


def reference_int4_qk_logits(
    query: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    scaling: float,
) -> torch.Tensor:
    """Slow PyTorch reference matching the project's current dequant path."""

    prefix_tokens, heads, packed_dim = codes.shape
    head_dim = packed_dim * 2
    low = (codes & 0x0F).to(torch.int16)
    high = (codes >> 4).to(torch.int16)
    unpacked = torch.stack((low, high), dim=-1).reshape(
        prefix_tokens, heads, head_dim
    )
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    grouped = unpacked.reshape(
        prefix_tokens,
        heads,
        head_dim // group_size,
        group_size,
    )
    keys = (grouped.to(torch.float16) * scales.unsqueeze(-1)).reshape(
        prefix_tokens, heads, head_dim
    )
    keys = keys.to(query.dtype)
    return torch.einsum("hqd,thd->hqt", query, keys) * float(scaling)
@triton.jit
def _int4_dequant_kernel(
    code_ptr,
    scale_ptr,
    out_ptr,
    total_values,
    heads,
    head_dim,
    packed_dim,
    groups_per_head,
    GROUP_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Fused unpack/scale kernel that materializes one FP16 key tensor."""

    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offsets < total_values
    dim = offsets % head_dim
    token_head = offsets // head_dim
    head = token_head % heads
    token = token_head // heads

    packed = tl.load(
        code_ptr
        + token * heads * packed_dim
        + head * packed_dim
        + dim // 2,
        mask=valid,
        other=0,
    ).to(tl.int32)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble = tl.where((dim & 1) == 0, low, high)
    signed = tl.where(nibble >= 8, nibble - 16, nibble)

    scale = tl.load(
        scale_ptr
        + token * heads * groups_per_head
        + head * groups_per_head
        + dim // GROUP_SIZE,
        mask=valid,
        other=0.0,
    )
    value = signed.to(tl.float32) * scale.to(tl.float32)
    tl.store(out_ptr + offsets, value, mask=valid)


@torch.inference_mode()
def dequantize_int4_keys(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize packed keys with one launch and return canonical FP16 keys."""

    if not (codes.is_cuda and scales.is_cuda):
        raise ValueError("codes and scales must be CUDA tensors")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise TypeError("codes must be uint8 and scales must be float16")
    if codes.ndim != 3 or scales.ndim != 3:
        raise ValueError("expected rank-3 codes and scales")

    codes = codes.contiguous()
    scales = scales.contiguous()
    prefix_tokens, heads, packed_dim = map(int, codes.shape)
    head_dim = packed_dim * 2
    scale_tokens, scale_heads, groups_per_head = map(int, scales.shape)
    group_size = int(group_size)
    if scale_tokens != prefix_tokens or scale_heads != heads:
        raise ValueError("code and scale token/head dimensions do not match")
    if group_size <= 0 or head_dim % group_size:
        raise ValueError("group_size must divide head_dim")
    if groups_per_head != head_dim // group_size:
        raise ValueError("scale group count does not match group_size")

    expected_shape = (prefix_tokens, heads, head_dim)
    if out is None:
        output = torch.empty(
            expected_shape,
            device=codes.device,
            dtype=torch.float16,
        )
    else:
        if (
            tuple(out.shape) != expected_shape
            or out.device != codes.device
            or out.dtype != torch.float16
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous FP16 with the canonical key shape/device")
        output = out
    total_values = output.numel()
    tile = 256
    grid = (triton.cdiv(total_values, tile),)
    _int4_dequant_kernel[grid](
        codes,
        scales,
        output,
        total_values,
        heads,
        head_dim,
        packed_dim,
        groups_per_head,
        GROUP_SIZE=group_size,
        TILE=tile,
        num_warps=4,
    )
    return output
