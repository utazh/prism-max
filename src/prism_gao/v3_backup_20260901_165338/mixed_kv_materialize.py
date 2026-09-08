"""Minimal fused materialization kernel for FP16/INT8 KV blocks.

This is deliberately *not* a full mixed-precision attention kernel.  It solves
only the first implementation problem:

- read all FP16 blocks as one compact group;
- read all INT8 blocks/scales as one compact group;
- in one GPU launch, copy FP16 blocks and dequantize INT8 blocks into one
  contiguous output ordered by original selected-block id.

Call it once for K and once for V.  The normal Qwen attention path can remain
unchanged.  Adapt only ``scale_index`` if the existing INT8 store uses a
quantization layout other than symmetric contiguous groups.
"""

from __future__ import annotations

from typing import Final

import torch

try:  # Keep the file importable on development machines without Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - machine dependent
    triton = None
    tl = None


TIER_FP16: Final[int] = 0
TIER_INT8: Final[int] = 1


if triton is not None:

    @triton.jit
    def _mixed_materialize_kernel(
        fp16_ptr,
        int8_ptr,
        scale_ptr,
        tier_ptr,
        source_slot_ptr,
        output_ptr,
        block_elements,
        groups_per_block,
        GROUP_SIZE: tl.constexpr,
        TILE: tl.constexpr,
    ):
        selected_slot = tl.program_id(0)
        tile_id = tl.program_id(1)
        offsets = tile_id * TILE + tl.arange(0, TILE)
        valid = offsets < block_elements

        tier = tl.load(tier_ptr + selected_slot).to(tl.int32)
        source_slot = tl.load(source_slot_ptr + selected_slot).to(tl.int64)
        is_fp16 = tier == 0
        is_int8 = tier == 1

        fp16_value = tl.load(
            fp16_ptr + source_slot * block_elements + offsets,
            mask=valid & is_fp16,
            other=0.0,
        ).to(tl.float32)

        int8_value = tl.load(
            int8_ptr + source_slot * block_elements + offsets,
            mask=valid & is_int8,
            other=0,
        ).to(tl.float32)
        group_offset = offsets // GROUP_SIZE
        scale = tl.load(
            scale_ptr
            + source_slot * groups_per_block
            + group_offset,
            mask=valid & is_int8,
            other=0.0,
        ).to(tl.float32)

        value = tl.where(is_fp16, fp16_value, int8_value * scale)
        tl.store(
            output_ptr + selected_slot * block_elements + offsets,
            value,
            mask=valid,
        )


@torch.inference_mode()
def mixed_materialize(
    *,
    fp16_blocks: torch.Tensor,
    int8_blocks: torch.Tensor,
    int8_scales: torch.Tensor,
    tier_by_selected_slot: torch.Tensor,
    source_slot_by_selected_slot: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Materialize selected blocks into one original-order FP16/BF16 buffer.

    Args:
        fp16_blocks:
            ``[N16, E]`` compact FP16 blocks.
        int8_blocks:
            ``[N8, E]`` compact signed INT8 blocks.
        int8_scales:
            ``[N8, E/group_size]`` FP16 scales.
        tier_by_selected_slot:
            ``[N]`` int32/uint8 values: 0=FP16, 1=INT8.  Slot order is the
            original selected-block order.
        source_slot_by_selected_slot:
            ``[N]`` index into ``fp16_blocks`` or ``int8_blocks`` according to
            the tier for that selected slot.
        group_size:
            Number of contiguous elements sharing one INT8 scale.

    Returns:
        ``[N, E]`` in ``output_dtype``. Reshape to
        ``[selected_blocks, block_tokens, kv_heads, head_dim]`` afterwards.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    tensors = (
        fp16_blocks,
        int8_blocks,
        int8_scales,
        tier_by_selected_slot,
        source_slot_by_selected_slot,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if fp16_blocks.ndim != 2 or int8_blocks.ndim != 2 or int8_scales.ndim != 2:
        raise ValueError("block tensors must be rank-2 compact matrices")
    if tier_by_selected_slot.ndim != 1 or source_slot_by_selected_slot.ndim != 1:
        raise ValueError("tier and source-slot metadata must be rank-1")
    if fp16_blocks.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("fp16_blocks must be FP16 or BF16")
    if int8_blocks.dtype != torch.int8:
        raise TypeError("int8_blocks must use torch.int8")
    if int8_scales.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("int8_scales must be FP16 or BF16")
    if output_dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("output_dtype must be FP16 or BF16")

    selected_blocks = int(tier_by_selected_slot.numel())
    if int(source_slot_by_selected_slot.numel()) != selected_blocks:
        raise ValueError("tier and source-slot metadata lengths differ")
    if selected_blocks <= 0:
        raise ValueError("at least one selected block is required")

    block_elements = int(fp16_blocks.shape[1] if fp16_blocks.shape[0] else int8_blocks.shape[1])
    if block_elements <= 0:
        raise ValueError("block size must be positive")
    if fp16_blocks.shape[0] and int(fp16_blocks.shape[1]) != block_elements:
        raise ValueError("FP16 block width mismatch")
    if int8_blocks.shape[0] and int(int8_blocks.shape[1]) != block_elements:
        raise ValueError("INT8 block width mismatch")
    group_size = int(group_size)
    if group_size <= 0 or block_elements % group_size:
        raise ValueError("group_size must divide flattened block elements")
    groups_per_block = block_elements // group_size
    if int8_blocks.shape[0] and tuple(int8_scales.shape) != (
        int(int8_blocks.shape[0]),
        groups_per_block,
    ):
        raise ValueError("INT8 scale geometry mismatch")

    # Triton still needs valid pointers when one tier is absent. These one-item
    # dummies are never read because the corresponding mask is false.
    if fp16_blocks.shape[0] == 0:
        fp16_blocks = torch.empty((1, block_elements), device=int8_blocks.device, dtype=output_dtype)
    if int8_blocks.shape[0] == 0:
        int8_blocks = torch.empty((1, block_elements), device=fp16_blocks.device, dtype=torch.int8)
        int8_scales = torch.empty((1, groups_per_block), device=fp16_blocks.device, dtype=torch.float16)

    tier = tier_by_selected_slot.to(device=fp16_blocks.device, dtype=torch.int32).contiguous()
    source_slot = source_slot_by_selected_slot.to(
        device=fp16_blocks.device, dtype=torch.int32
    ).contiguous()
    if bool(torch.any((tier != TIER_FP16) & (tier != TIER_INT8))):
        raise ValueError("tier metadata must contain only 0 (FP16) or 1 (INT8)")

    output = torch.empty(
        (selected_blocks, block_elements),
        device=fp16_blocks.device,
        dtype=output_dtype,
    )
    tile = 256
    grid = (selected_blocks, triton.cdiv(block_elements, tile))
    _mixed_materialize_kernel[grid](
        fp16_blocks.contiguous(),
        int8_blocks.contiguous(),
        int8_scales.contiguous(),
        tier,
        source_slot,
        output,
        block_elements,
        groups_per_block,
        GROUP_SIZE=group_size,
        TILE=tile,
        num_warps=4,
    )
    return output


def reference_mixed_materialize(
    *,
    fp16_blocks: torch.Tensor,
    int8_blocks: torch.Tensor,
    int8_scales: torch.Tensor,
    tier_by_selected_slot: torch.Tensor,
    source_slot_by_selected_slot: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Small PyTorch reference used for correctness tests."""

    selected = int(tier_by_selected_slot.numel())
    block_elements = int(fp16_blocks.shape[1] if fp16_blocks.shape[0] else int8_blocks.shape[1])
    groups = block_elements // int(group_size)
    output = torch.empty(
        (selected, block_elements),
        device=tier_by_selected_slot.device,
        dtype=output_dtype,
    )
    for destination in range(selected):
        tier = int(tier_by_selected_slot[destination])
        source = int(source_slot_by_selected_slot[destination])
        if tier == TIER_FP16:
            output[destination] = fp16_blocks[source].to(output_dtype)
        elif tier == TIER_INT8:
            values = int8_blocks[source].reshape(groups, group_size).float()
            scales = int8_scales[source].float().unsqueeze(-1)
            output[destination] = (values * scales).reshape(-1).to(output_dtype)
        else:
            raise ValueError(f"unsupported tier code {tier}")
    return output
