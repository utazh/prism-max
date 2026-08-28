"""Pure tensor utilities for the AttentionStore/H2O comparison baseline.

The helpers in this module intentionally contain no ProMixed policy or cache
logic. They only implement GQA-aware H2O selection and compact K/V gathering,
so the baseline can be integrated without coupling it to the proposed method.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


__all__ = [
    "gather_h2o_selected_kv",
    "select_h2o_gqa_value_positions",
]


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _retention_ratio(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("keep_ratio must be a finite number in (0, 1]")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("keep_ratio must be a finite number in (0, 1]")
    return ratio


def select_h2o_gqa_value_positions(
    head_scores: torch.Tensor,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    keep_ratio: float,
) -> torch.Tensor:
    """Select a fixed number of value positions independently per KV head.

    Query heads are mapped to their physical KV head using the standard
    contiguous GQA layout. Attention mass is summed across all query heads
    sharing a KV head, then ``ceil(prefix_tokens * keep_ratio)`` positions are
    retained for every KV head. Rows are ordered by decreasing score; exact
    ties retain the lower token id first.

    Args:
        head_scores: Per-query-head attention mass with shape
            ``[num_query_heads, prefix_tokens]``.
        num_query_heads: Number of logical query heads.
        num_kv_heads: Number of physical key/value heads.
        keep_ratio: Per-head prefix retention ratio in ``(0, 1]``.

    Returns:
        A ``torch.long`` tensor with shape ``[num_kv_heads, keep_tokens]``.
    """

    if not isinstance(head_scores, torch.Tensor):
        raise TypeError("head_scores must be a torch.Tensor")
    if head_scores.ndim != 2:
        raise ValueError(
            "head_scores must have shape [num_query_heads, prefix_tokens]"
        )
    if not torch.is_floating_point(head_scores):
        raise ValueError("head_scores must have a floating-point dtype")

    query_heads = _positive_int(num_query_heads, "num_query_heads")
    kv_heads = _positive_int(num_kv_heads, "num_kv_heads")
    if query_heads % kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if int(head_scores.shape[0]) != query_heads:
        raise ValueError(
            "head_scores first dimension must equal num_query_heads"
        )
    prefix_tokens = int(head_scores.shape[1])
    if prefix_tokens <= 0:
        raise ValueError("head_scores must cover at least one prefix token")
    if not bool(torch.isfinite(head_scores).all()):
        raise ValueError("head_scores must be finite")
    if bool(torch.any(head_scores < 0)):
        raise ValueError("head_scores must be non-negative attention mass")

    ratio = _retention_ratio(keep_ratio)
    keep_tokens = min(prefix_tokens, math.ceil(prefix_tokens * ratio))
    query_heads_per_kv = query_heads // kv_heads
    grouped_scores = head_scores.reshape(
        kv_heads,
        query_heads_per_kv,
        prefix_tokens,
    ).sum(dim=1)

    # Stable sorting preserves the original ascending token order for exact
    # score ties, giving the explicit (score desc, token id asc) ordering.
    rankings = torch.argsort(
        grouped_scores,
        dim=1,
        descending=True,
        stable=True,
    )
    return rankings[:, :keep_tokens].to(dtype=torch.long)


def gather_h2o_selected_kv(
    full_keys: torch.Tensor,
    selected_values: torch.Tensor,
    selected_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather compact per-head K/V tensors for true sparse attention.

    Full keys are used only by the online selector. For attention, each KV head
    gathers the keys at ``selected_positions[head]`` and pairs them with the
    already-loaded selected values. Different heads may choose different token
    positions because the common compact sequence axis represents rank slots,
    not shared token ids. Positions must remain unique within each head.
    """

    if not isinstance(full_keys, torch.Tensor):
        raise TypeError("full_keys must be a torch.Tensor")
    if not isinstance(selected_values, torch.Tensor):
        raise TypeError("selected_values must be a torch.Tensor")
    if not isinstance(selected_positions, torch.Tensor):
        raise TypeError("selected_positions must be a torch.Tensor")
    if full_keys.ndim != 3:
        raise ValueError(
            "full_keys must have shape [prefix_tokens, num_kv_heads, head_dim]"
        )
    prefix_tokens, kv_heads, head_dim = map(int, full_keys.shape)
    if prefix_tokens <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("full_keys dimensions must all be positive")
    if selected_positions.ndim != 2:
        raise ValueError(
            "selected_positions must have shape [num_kv_heads, keep_tokens]"
        )
    if selected_positions.dtype != torch.long:
        raise ValueError("selected_positions must have torch.long dtype")
    keep_tokens = int(selected_positions.shape[1])
    if tuple(selected_positions.shape) != (kv_heads, keep_tokens):
        raise ValueError(
            "selected_positions first dimension must match the KV-head count"
        )
    if selected_values.ndim != 3 or tuple(selected_values.shape) != (
        keep_tokens,
        kv_heads,
        head_dim,
    ):
        raise ValueError(
            "selected_values must have shape "
            "[keep_tokens, num_kv_heads, head_dim]"
        )
    if selected_values.device != full_keys.device:
        raise ValueError("selected_values and full_keys must use the same device")
    if selected_positions.device != full_keys.device:
        raise ValueError(
            "selected_positions and full_keys must use the same device"
        )
    if selected_values.dtype != full_keys.dtype:
        raise ValueError("selected_values and full_keys must use the same dtype")
    if keep_tokens > prefix_tokens:
        raise ValueError("keep_tokens cannot exceed prefix_tokens")

    if selected_positions.numel():
        if bool(torch.any(selected_positions < 0)) or bool(
            torch.any(selected_positions >= prefix_tokens)
        ):
            raise ValueError("selected_positions contains an out-of-range token")
        ordered_positions = torch.sort(selected_positions, dim=1).values
        if keep_tokens > 1 and bool(
            torch.any(ordered_positions[:, 1:] == ordered_positions[:, :-1])
        ):
            raise ValueError(
                "selected_positions must not contain duplicates within a KV head"
            )

    head_ids = torch.arange(
        kv_heads,
        dtype=torch.long,
        device=full_keys.device,
    ).unsqueeze(1).expand(kv_heads, keep_tokens)
    selected_keys = full_keys[selected_positions, head_ids, :]
    selected_keys = selected_keys.permute(1, 0, 2).contiguous()
    compact_values = selected_values.contiguous()
    if selected_keys.shape != compact_values.shape:
        raise RuntimeError("compact H2O key/value shapes diverged")
    return selected_keys, compact_values
