"""FlexGen-backed ContiguousKV and IMPRESS runner for Qwen2.5-7B.

Both methods use the same Pcache-backed storage hierarchy and Qwen attention
path. ContiguousKV uses 16-token aligned chunks and two-level prefetching. The
paper IMPRESS baseline uses 64-token reordered chunks and synchronous loading;
HyperInfer's later asynchronous extension is an explicit optional mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .as_baselines import select_h2o_gqa_value_positions
from .core import contiguous_chunk_scores, select_top_chunks
from .flexgen_pcache import (
    FlexGenPcacheConfig,
    FlexGenPcacheStore,
    expected_chunk_count,
    selected_tokens_for_plan,
)
from .lmcache_plan import policy_request_id
from .impress_reorder import reorder_manifest_sha256
from .layer_budget import load_layer_budget_profile, profile_to_json_dict
from .paper_client import (
    continuation_token_ids,
    percentile95,
    predict_from_label_logits,
    predict_from_label_token_logprobs,
    prediction_is_correct,
)
from .paper_plan_generator import (
    impress_similarity_threshold,
    impress_probe_token_selection_with_ranking,
    mean_pairwise_jaccard,
)
from .promixed import (
    PromixedSelectionDecision,
    select_promixed_gqa_blocks,
)
from .sparse_qwen_reprefill import (
    _layer_causal_mask,
    _load_bundle_records,
    _load_layer_plan,
    read_store_info,
)


def runtime_variant(
    *,
    method: str,
    online_selection: bool,
    impress_async_prefetch: bool,
    impress_reorder_enabled: bool,
    impress_selection_block_size: int = 1,
    layer_budget_profile_enabled: bool = False,
    exact_layer_block_budget: bool = False,
    impress_period_prefetch_size: int = 1,
    impress_period_prefetch_budget_scale: float = 1.0,
    impress_priority_prefetch: bool = False,
    impress_deferred_compute_timing: bool = False,
    impress_rolling_period_prefetch: bool = False,
    impress_value_ordered_prefetch: bool = False,
    impress_value_prefetch_budget_scale: float = 1.0,
    impress_selection_period_size: int = 1,
    impress_known_period_prefetch: bool = False,
    promixed_gqa_selection: bool = False,
    defer_cache_score_updates: bool = False,
    selector_index_bits: int = 0,
) -> str:
    if not online_selection:
        return "offline-plan"
    if method == "as_lru":
        return "attentionstore-full-kv-lru-c64"
    if method == "as_h2o_lru":
        return "attentionstore-h2o-full-k-selector-compact-kv-lru-c64"
    if method == "contigkv":
        return "contiguouskv-online-period-prefetch"
    if method != "impress":
        raise ValueError(f"unsupported online runtime method: {method}")
    if impress_selection_block_size > 1:
        mode = (
            "hyperinfer-async-contiguous-blocks"
            if impress_async_prefetch
            else "impress-sync-contiguous-blocks"
        )
        variant = f"{mode}-c{impress_selection_block_size}"
    elif impress_async_prefetch:
        variant = (
            "hyperinfer-async-reorder"
            if impress_reorder_enabled
            else "hyperinfer-async-no-reorder"
        )
    else:
        variant = (
            "paper-impress-sync-reorder"
            if impress_reorder_enabled
            else "impress-sync-no-reorder-ablation"
        )
    if layer_budget_profile_enabled:
        variant += "+layer-budget"
    if exact_layer_block_budget:
        variant += "+exact-total-block-budget"
    if impress_period_prefetch_size > 1:
        variant += f"+predictive-period-p{impress_period_prefetch_size}"
        if not math.isclose(
            impress_period_prefetch_budget_scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            variant += f"-budget-s{impress_period_prefetch_budget_scale:g}"
    if impress_priority_prefetch:
        variant += "+priority-prefetch"
    if impress_deferred_compute_timing:
        variant += "+deferred-compute-timing"
    if impress_rolling_period_prefetch:
        variant += "+rolling-period-source"
    if impress_value_ordered_prefetch:
        variant += "+value-ordered-prefetch"
        if impress_value_prefetch_budget_scale != 1.0:
            variant += (
                f"+value-budget-s{impress_value_prefetch_budget_scale:g}"
            )
    if promixed_gqa_selection:
        variant += f"+promixed-gqa-adaptive-p{impress_selection_period_size}"
    elif impress_selection_period_size > 1:
        variant += f"+periodic-selection-p{impress_selection_period_size}"
    if selector_index_bits:
        variant += f"+k{selector_index_bits}-selector-index"
    if impress_known_period_prefetch:
        variant += "+known-period-prefetch"
    if defer_cache_score_updates:
        variant += "+post-ttft-cache-score"
    return variant


def validate_impress_block_mode(
    *,
    method: str,
    online_selection: bool,
    block_size: int,
    physical_chunk_size: int,
    reorder_enabled: bool,
) -> None:
    """Reject ambiguous combinations for importance-selected contiguous blocks."""

    if block_size <= 0:
        raise ValueError("IMPRESS selection block size must be positive")
    if block_size == 1:
        return
    if method != "impress" or not online_selection:
        raise ValueError(
            "IMPRESS contiguous-block selection requires online IMPRESS"
        )
    if block_size != physical_chunk_size:
        raise ValueError(
            "IMPRESS selection block size must match the physical Pcache chunk size"
        )
    if reorder_enabled:
        raise ValueError(
            "IMPRESS contiguous-block selection cannot use token-level reordering"
        )


def resolve_store_tasks(
    tasks: Sequence[str],
    store_tasks: Sequence[str] | None,
) -> tuple[str, ...]:
    """Keep persisted numeric prefix IDs stable for independent task runs."""

    evaluated = tuple(str(task).strip().lower() for task in tasks)
    registered = tuple(
        str(task).strip().lower()
        for task in (store_tasks if store_tasks is not None else evaluated)
    )
    if not evaluated or any(not task for task in evaluated):
        raise ValueError("one or more evaluation tasks are required")
    if not registered or any(not task for task in registered):
        raise ValueError("one or more store tasks are required")
    if len(set(registered)) != len(registered):
        raise ValueError("store tasks must be unique and ordered")
    missing = sorted(set(evaluated) - set(registered))
    if missing:
        raise ValueError(
            "evaluation tasks are absent from the store task order: "
            + ", ".join(missing)
        )
    return registered


def allocate_exact_layer_blocks(
    layer_ratios: Sequence[float],
    *,
    blocks_per_layer: int,
    target_ratio: float,
) -> tuple[int, ...]:
    """Allocate a fixed global block budget while preserving layer priorities."""

    ratios = tuple(float(ratio) for ratio in layer_ratios)
    if not ratios:
        raise ValueError("exact block allocation requires at least one layer")
    if blocks_per_layer <= 0:
        raise ValueError("blocks per layer must be positive")
    if not math.isfinite(target_ratio) or not 0 < target_ratio <= 1:
        raise ValueError("target ratio must be finite and in (0, 1]")
    if any(not math.isfinite(ratio) or not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("layer ratios must be finite and in (0, 1]")

    quotas = [ratio * blocks_per_layer for ratio in ratios]
    tie_bits = max(1, (len(ratios) - 1).bit_length())
    tie_order = sorted(
        range(len(ratios)),
        key=lambda layer: int(f"{layer:0{tie_bits}b}"[::-1], 2),
    )
    tie_rank = {layer: rank for rank, layer in enumerate(tie_order)}
    counts = [
        min(blocks_per_layer, max(1, math.floor(quota)))
        for quota in quotas
    ]
    target_blocks = math.floor(
        target_ratio * blocks_per_layer * len(ratios) + 0.5
    )
    target_blocks = min(
        blocks_per_layer * len(ratios),
        max(len(ratios), target_blocks),
    )

    while sum(counts) < target_blocks:
        candidates = [
            layer for layer, count in enumerate(counts) if count < blocks_per_layer
        ]
        if not candidates:
            raise RuntimeError("exact block allocation cannot reach its target")
        layer = min(
            candidates,
            key=lambda index: (
                -(quotas[index] - math.floor(quotas[index])),
                -ratios[index],
                tie_rank[index],
            ),
        )
        counts[layer] += 1
        quotas[layer] = math.floor(quotas[layer])

    while sum(counts) > target_blocks:
        candidates = [layer for layer, count in enumerate(counts) if count > 1]
        if not candidates:
            raise RuntimeError("exact block allocation cannot reduce to its target")
        layer = min(
            candidates,
            key=lambda index: (
                quotas[index] - math.floor(quotas[index]),
                ratios[index],
                tie_rank[index],
            ),
        )
        counts[layer] -= 1
        quotas[layer] = math.ceil(quotas[layer])

    return tuple(counts)


def layer_token_selection_sha256(selections: Sequence[Sequence[int]]) -> str:
    canonical = [[int(token) for token in layer] for layer in selections]
    encoded = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_layer_token_selections(
    plan_path: str | Path,
    uid: str,
) -> list[list[int]] | None:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    selections = payload.get("layer_token_selections", {}).get(policy_request_id(uid))
    if selections is None:
        return None
    if not isinstance(selections, list) or not all(isinstance(row, list) for row in selections):
        raise ValueError(f"invalid layer token selections for {uid} in {plan_path}")
    return [[int(token_id) for token_id in row] for row in selections]


def _load_layer_chunk_scores(
    plan_path: str | Path,
    uid: str,
) -> list[list[float]] | None:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    scores = payload.get("layer_chunk_attention_scores", {}).get(policy_request_id(uid))
    if scores is None:
        return None
    if not isinstance(scores, list) or not all(isinstance(row, list) for row in scores):
        raise ValueError(f"invalid layer chunk attention scores for {uid} in {plan_path}")
    normalized = [[float(score) for score in row] for row in scores]
    for layer, row in enumerate(normalized):
        for chunk, score in enumerate(row):
            if not math.isfinite(score) or score < 0:
                raise ValueError(
                    f"invalid attention score for {uid} layer {layer} chunk {chunk}: {score}"
                )
    return normalized


def _load_plan_metadata(plan_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"plan metadata is missing or invalid in {plan_path}")
    return metadata


def _load_optional_layer_plan(
    plan_path: str | Path,
    uid: str,
) -> list[list[str]] | None:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    plan = payload.get("layer_request_prefixes", {}).get(policy_request_id(uid))
    if plan is None:
        return None
    if not isinstance(plan, list) or not plan or not all(isinstance(row, list) for row in plan):
        raise ValueError(f"invalid layer plan for {uid} in {plan_path}")
    return [[str(tier) for tier in row] for row in plan]


def build_online_layer_plan(
    *,
    layers: int,
    prefix_tokens: int,
    chunk_size: int,
) -> list[list[str]]:
    """Build shape-only Pcache scaffolding for request-time online selection."""

    if layers <= 0:
        raise ValueError("online layer plan requires at least one model layer")
    chunks = expected_chunk_count(prefix_tokens=prefix_tokens, chunk_size=chunk_size)
    if chunks <= 0:
        raise ValueError("online layer plan requires a non-empty prefix")
    return [["online"] * chunks for _ in range(layers)]


def validate_plan_keep_ratio(
    metadata: dict[str, Any],
    expected_keep_ratio: float | None = None,
) -> float:
    """Validate the plan's fractional KV budget and an optional CLI expectation."""

    if "keep_ratio" not in metadata:
        raise ValueError("plan metadata must declare keep_ratio")
    keep_ratio = float(metadata["keep_ratio"])
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"plan keep_ratio must be in (0, 1], got {keep_ratio}")
    if expected_keep_ratio is not None:
        expected = float(expected_keep_ratio)
        if not 0 < expected <= 1:
            raise ValueError(f"expected keep ratio must be in (0, 1], got {expected}")
        if not math.isclose(keep_ratio, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"plan keep_ratio {keep_ratio} does not match expected ratio {expected}"
            )
    return keep_ratio


def process_peak_rss_bytes() -> int | None:
    """Read Linux's process high-water RSS without adding a dependency."""

    try:
        for line in Path("/proc/self/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def parse_head_ids(value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated list of attention head IDs."""

    try:
        heads = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"head IDs must be comma-separated integers, got {value!r}") from exc
    if not heads or any(head < 0 for head in heads):
        raise ValueError("head IDs must contain one or more non-negative integers")
    return heads


def validate_online_selector_mapping(
    model_config: Any,
    *,
    method: str,
    selector_kv_head_ids: Sequence[int],
    probe_query_heads: Sequence[int],
) -> None:
    """Validate the physical selector heads against Qwen's GQA mapping."""

    query_heads = int(model_config.num_attention_heads)
    kv_heads = int(model_config.num_key_value_heads)
    if query_heads % kv_heads:
        raise ValueError("Qwen attention heads are not evenly grouped over KV heads")
    groups = query_heads // kv_heads
    selector_ids = tuple(int(head) for head in selector_kv_head_ids)
    probes = tuple(int(head) for head in probe_query_heads)
    if method in {"contigkv", "as_h2o_lru"}:
        expected = tuple(range(kv_heads))
        if selector_ids != expected:
            label = "ContiguousKV" if method == "contigkv" else "AS+H2O+LRU"
            raise ValueError(
                f"online {label} must persist every Qwen KV head in order; "
                f"expected {expected}, got {selector_ids}"
            )
        return
    if method == "as_lru":
        return
    if method != "impress":
        raise ValueError(f"unsupported online selector method: {method}")
    if any(head >= query_heads for head in probes):
        raise ValueError(f"probe query heads must be below {query_heads}")
    expected = tuple(head // groups for head in probes)
    unique_expected = tuple(dict.fromkeys(expected))
    if selector_ids not in {expected, unique_expected}:
        raise ValueError(
            "online IMPRESS selector KV heads must match the Qwen GQA mapping; "
            f"expected one-to-one {expected} or deduplicated {unique_expected} "
            f"for probe heads {probes}, got {selector_ids}"
        )


def selector_key_slots(
    query_head_ids: Sequence[int],
    *,
    num_query_heads: int,
    num_kv_heads: int,
    selector_kv_head_ids: Sequence[int],
) -> tuple[int, ...]:
    """Map probe query heads to loaded key slots, reusing shared GQA heads."""

    if num_query_heads <= 0 or num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("query heads must be evenly grouped over positive KV heads")
    query_ids = tuple(int(head) for head in query_head_ids)
    loaded_ids = tuple(int(head) for head in selector_kv_head_ids)
    if not query_ids or not loaded_ids:
        raise ValueError("probe query heads and selector KV heads must be non-empty")
    if any(not 0 <= head < num_query_heads for head in query_ids):
        raise ValueError("probe query head is outside the model")
    if any(not 0 <= head < num_kv_heads for head in loaded_ids):
        raise ValueError("selector KV head is outside the model")
    groups = num_query_heads // num_kv_heads
    required_ids = tuple(head // groups for head in query_ids)
    missing = sorted(set(required_ids) - set(loaded_ids))
    if missing:
        raise ValueError(f"selector keys do not contain required KV heads {missing}")
    return tuple(loaded_ids.index(head) for head in required_ids)


def impress_contiguous_block_selection(
    head_scores: Sequence[Sequence[float]],
    *,
    keep_ratio: float,
    block_size: int,
    similarity_alpha: float,
) -> tuple[list[int], bool, float]:
    """Apply IMPRESS probe voting to aligned block scores and expand to tokens."""

    selected, _, used_probe, similarity = (
        impress_contiguous_block_selection_with_ranking(
            head_scores,
            keep_ratio=keep_ratio,
            block_size=block_size,
            similarity_alpha=similarity_alpha,
        )
    )
    return selected, used_probe, similarity


@dataclass(frozen=True)
class PreparedImpressBlockScores:
    block_scores: tuple[tuple[float, ...], ...]
    ranked_blocks: tuple[tuple[int, ...], ...]
    mean_scores: tuple[float, ...]
    block_size: int
    prefix_tokens: int


def prepare_impress_block_scores(
    block_scores: Sequence[Sequence[float]],
    *,
    block_size: int,
    prefix_tokens: int,
) -> PreparedImpressBlockScores:
    """Validate already-aggregated block scores and rank them once."""

    if not block_scores or not block_scores[0]:
        raise ValueError("IMPRESS block selection requires non-empty block scores")
    if block_size <= 0:
        raise ValueError("IMPRESS selection block size must be positive")
    if prefix_tokens <= 0:
        raise ValueError("IMPRESS prefix token count must be positive")
    normalized = tuple(
        tuple(float(value) for value in row) for row in block_scores
    )
    block_count = math.ceil(prefix_tokens / block_size)
    if any(len(row) != block_count for row in normalized):
        raise ValueError(
            "IMPRESS block score rows do not match the prefix geometry"
        )
    ranked_blocks = tuple(
        tuple(
            sorted(
                range(block_count),
                key=lambda block: (-float(row[block]), block),
            )
        )
        for row in normalized
    )
    mean_scores = tuple(
        sum(float(row[block]) for row in normalized) / len(normalized)
        for block in range(block_count)
    )
    return PreparedImpressBlockScores(
        block_scores=normalized,
        ranked_blocks=ranked_blocks,
        mean_scores=mean_scores,
        block_size=block_size,
        prefix_tokens=prefix_tokens,
    )


def prepare_impress_contiguous_block_scores(
    head_scores: Sequence[Sequence[float]],
    *,
    block_size: int,
) -> PreparedImpressBlockScores:
    """Aggregate and rank one Period leader's token scores exactly once."""

    if not head_scores or not head_scores[0]:
        raise ValueError("IMPRESS block selection requires non-empty head scores")
    if block_size <= 0:
        raise ValueError("IMPRESS selection block size must be positive")
    prefix_tokens = len(head_scores[0])
    if any(len(row) != prefix_tokens for row in head_scores):
        raise ValueError("IMPRESS head score rows must have the same token count")
    block_scores = tuple(
        tuple(contiguous_chunk_scores(row, block_size, prefix_tokens))
        for row in head_scores
    )
    return prepare_impress_block_scores(
        block_scores,
        block_size=block_size,
        prefix_tokens=prefix_tokens,
    )


def select_prepared_impress_blocks(
    prepared: PreparedImpressBlockScores,
    *,
    keep_ratio: float,
    similarity_alpha: float,
    keep_blocks: int | None = None,
    fallback_keep_blocks_limit: int | None = None,
) -> tuple[list[int], list[int], bool, float]:
    """Apply unchanged IMPRESS voting to cached block scores and rankings."""

    if not 0 < keep_ratio <= 1:
        raise ValueError("IMPRESS block keep ratio must be in (0, 1]")
    block_count = len(prepared.mean_scores)
    if keep_blocks is None:
        resolved_keep_blocks = max(1, math.ceil(block_count * keep_ratio))
    else:
        resolved_keep_blocks = int(keep_blocks)
        if (
            resolved_keep_blocks != keep_blocks
            or not 1 <= resolved_keep_blocks <= block_count
        ):
            raise ValueError(
                f"exact IMPRESS keep blocks must be in [1, {block_count}]"
            )
    if fallback_keep_blocks_limit is not None and not (
        1 <= int(fallback_keep_blocks_limit) <= block_count
    ):
        raise ValueError(
            f"fallback keep block limit must be in [1, {block_count}]"
        )

    probe_sets = [
        set(ranking[:resolved_keep_blocks])
        for ranking in prepared.ranked_blocks
    ]
    similarity = mean_pairwise_jaccard(probe_sets)
    threshold = impress_similarity_threshold(
        block_count,
        resolved_keep_blocks,
        similarity_alpha,
    )
    if similarity < threshold:
        selected_blocks = set(range(block_count))
        priority_blocks = sorted(
            selected_blocks,
            key=lambda block: (-prepared.mean_scores[block], block),
        )
        used_probe = False
    else:
        counts: dict[int, int] = {}
        for selected in probe_sets:
            for block in selected:
                counts[block] = counts.get(block, 0) + 1
        ranked = sorted(counts, key=lambda block: (-counts[block], block))
        selected_blocks = set(ranked[:resolved_keep_blocks])
        priority_blocks = sorted(
            selected_blocks,
            key=lambda block: (
                -counts[block],
                -prepared.mean_scores[block],
                block,
            ),
        )
        used_probe = True
    if (
        not used_probe
        and fallback_keep_blocks_limit is not None
        and len(selected_blocks) > int(fallback_keep_blocks_limit)
    ):
        priority_blocks = priority_blocks[: int(fallback_keep_blocks_limit)]
        selected_blocks = set(priority_blocks)

    selected_tokens = [
        token
        for block in sorted(selected_blocks)
        for token in range(
            block * prepared.block_size,
            min(
                prepared.prefix_tokens,
                (block + 1) * prepared.block_size,
            ),
        )
    ]
    priority_tokens = [
        token
        for block in priority_blocks
        for token in range(
            block * prepared.block_size,
            min(
                prepared.prefix_tokens,
                (block + 1) * prepared.block_size,
            ),
        )
    ]
    return selected_tokens, priority_tokens, used_probe, similarity


def impress_contiguous_block_selection_with_ranking(
    head_scores: Sequence[Sequence[float]],
    *,
    keep_ratio: float,
    block_size: int,
    similarity_alpha: float,
    keep_blocks: int | None = None,
    fallback_keep_blocks_limit: int | None = None,
) -> tuple[list[int], list[int], bool, float]:
    """Return the same selected blocks plus an importance-ranked I/O order."""

    prepared = prepare_impress_contiguous_block_scores(
        head_scores,
        block_size=block_size,
    )
    return select_prepared_impress_blocks(
        prepared,
        keep_ratio=keep_ratio,
        similarity_alpha=similarity_alpha,
        keep_blocks=keep_blocks,
        fallback_keep_blocks_limit=fallback_keep_blocks_limit,
    )


def layer_selection_agreement(
    actual: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
) -> tuple[float, float]:
    """Return mean layer Jaccard and the fraction of exact layer selections."""

    if len(actual) != len(reference) or not actual:
        raise ValueError("actual and reference selections must have the same nonzero layer count")
    similarities = []
    exact = 0
    for actual_row, reference_row in zip(actual, reference):
        actual_set = set(int(token) for token in actual_row)
        reference_set = set(int(token) for token in reference_row)
        union = actual_set | reference_set
        similarities.append(len(actual_set & reference_set) / max(1, len(union)))
        exact += actual_set == reference_set
    return sum(similarities) / len(similarities), exact / len(similarities)


def _cache_layout(token_major: Any, *, dtype, device):
    """Convert Pcache [token, kv_head, dim] output to DynamicCache layout."""

    return token_major.to(device=device, dtype=dtype).permute(1, 0, 2).unsqueeze(0).contiguous()


def _copy_dynamic_cache(
    cache_data: Sequence[tuple[Any, Any]],
    model_config: Any,
) -> Any:
    """Copy a layer-variable DynamicCache for one candidate continuation."""

    from transformers.cache_utils import DynamicCache

    return DynamicCache(
        ddp_cache_data=(
            (keys, values)
            for keys, values in cache_data
        ),
        config=model_config,
    )


def qwen_online_prefix_head_scores(
    *,
    decoder_layer: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    selector_keys: Any,
    score_block_size: int = 1,
    query_heads: Sequence[int] | None,
    selector_kv_head_ids: Sequence[int] | None = None,
) -> Any:
    """Compute RoPE/GQA-correct prefix attention for online selection.

    Block mode reduces scores on GPU before the device-to-host transfer.
    """

    if score_block_size <= 0:
        raise ValueError("online selector score block size must be positive")
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    attention = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query_states = attention.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    suffix_keys = attention.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, suffix_keys = apply_rotary_pos_emb(query_states, suffix_keys, cos, sin)
    suffix_keys = repeat_kv(suffix_keys, attention.num_key_value_groups)

    prefix_keys = selector_keys.to(device=query_states.device, dtype=query_states.dtype)
    prefix_keys = prefix_keys.permute(1, 0, 2).unsqueeze(0).contiguous()
    if query_heads is None:
        if int(prefix_keys.shape[1]) != int(attention.config.num_key_value_heads):
            raise ValueError(
                "ContiguousKV online selector must load every Qwen key/value head"
            )
        prefix_keys = repeat_kv(prefix_keys, attention.num_key_value_groups)
    else:
        heads = torch.tensor(list(query_heads), dtype=torch.long, device=query_states.device)
        if bool(torch.any(heads >= query_states.shape[1])):
            raise ValueError("probe query head is outside the Qwen attention head range")
        if selector_kv_head_ids is None:
            if int(prefix_keys.shape[1]) != int(heads.numel()):
                raise ValueError(
                    "IMPRESS selector keys must correspond one-to-one with probe heads"
                )
        else:
            loaded_ids = tuple(int(head) for head in selector_kv_head_ids)
            if int(prefix_keys.shape[1]) != len(loaded_ids):
                raise ValueError("selector key tensor does not match its physical KV head IDs")
            slots = selector_key_slots(
                query_heads,
                num_query_heads=int(attention.config.num_attention_heads),
                num_kv_heads=int(attention.config.num_key_value_heads),
                selector_kv_head_ids=loaded_ids,
            )
            prefix_keys = prefix_keys.index_select(
                1,
                torch.tensor(slots, dtype=torch.long, device=prefix_keys.device),
            )
        query_states = query_states.index_select(1, heads)
        suffix_keys = suffix_keys.index_select(1, heads)

    prefix_tokens = int(prefix_keys.shape[2])
    query_tokens = int(query_states.shape[2])
    full_keys = torch.cat((prefix_keys, suffix_keys), dim=2)
    logits = torch.matmul(query_states, full_keys.transpose(2, 3)) * float(attention.scaling)
    suffix_future = torch.triu(
        torch.ones((query_tokens, query_tokens), dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    logits[..., prefix_tokens:] = logits[..., prefix_tokens:].masked_fill(
        suffix_future.view(1, 1, query_tokens, query_tokens),
        torch.finfo(logits.dtype).min,
    )
    probabilities = torch.softmax(logits.float(), dim=-1)
    scores = probabilities[..., :prefix_tokens].sum(dim=2)[0]
    if score_block_size > 1:
        padding = (-prefix_tokens) % score_block_size
        if padding:
            scores = torch.nn.functional.pad(scores, (0, padding))
        scores = scores.reshape(
            scores.shape[0], -1, score_block_size
        ).sum(dim=-1)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("online Qwen selector produced non-finite prefix attention")
    return scores.detach().cpu()


def configure_online_layer_selection(
    *,
    decoder_layer: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    layer_index: int,
    loader: Any,
    period_size: int,
    impress_selection_block_size: int = 1,
    impress_selection_period_size: int = 1,
) -> int:
    """Load selector keys, calculate critical indices, and update the loader."""

    import torch

    configured_period = period_size
    selector_keys = loader.load_selector_keys(layer_index)
    if loader.method == "as_h2o_lru":
        score_started = torch.cuda.Event(enable_timing=True)
        score_finished = torch.cuda.Event(enable_timing=True)
        score_started.record()
        head_scores = qwen_online_prefix_head_scores(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys,
            query_heads=None,
        )
        score_finished.record()
        score_finished.synchronize()
        score_path_ms = score_started.elapsed_time(score_finished)
        decision_started = time.perf_counter()
        attention_config = decoder_layer.self_attn.config
        positions = select_h2o_gqa_value_positions(
            head_scores,
            num_query_heads=int(attention_config.num_attention_heads),
            num_kv_heads=int(attention_config.num_key_value_heads),
            keep_ratio=float(loader.keep_ratio_for_layer(layer_index)),
        )
        loader.configure_as_h2o_layer(
            layer=layer_index,
            positions=positions,
        )
        loader.record_selector_compute(
            score_path_ms + (time.perf_counter() - decision_started) * 1000
        )
        configured_period = 1
    elif loader.method == "contigkv":
        score_started = torch.cuda.Event(enable_timing=True)
        score_finished = torch.cuda.Event(enable_timing=True)
        score_started.record()
        head_scores = qwen_online_prefix_head_scores(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys,
            query_heads=None,
        )
        score_finished.record()
        score_finished.synchronize()
        score_path_ms = score_started.elapsed_time(score_finished)
        decision_started = time.perf_counter()
        token_scores = head_scores.mean(dim=0).tolist()
        chunk_scores = contiguous_chunk_scores(
            token_scores,
            loader.chunk_size,
            loader.prefix_tokens,
        )
        keep_chunks = max(1, math.ceil(len(chunk_scores) * float(loader.keep_ratio)))
        selected_chunks = sorted(select_top_chunks(chunk_scores, keep_chunks))
        loader.configure_contiguous_period(
            period_start=layer_index,
            selected_chunks=selected_chunks,
            chunk_scores=chunk_scores,
            period_size=period_size,
        )
        loader.record_selector_compute(
            score_path_ms + (time.perf_counter() - decision_started) * 1000
        )
    else:
        score_started = torch.cuda.Event(enable_timing=True)
        score_finished = torch.cuda.Event(enable_timing=True)
        score_started.record()
        head_scores = qwen_online_prefix_head_scores(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys,
            query_heads=loader.probe_query_heads,
            selector_kv_head_ids=loader.selector_kv_head_ids,
            score_block_size=impress_selection_block_size,
        )
        score_finished.record()
        score_finished.synchronize()
        score_path_ms = score_started.elapsed_time(score_finished)
        decision_started = time.perf_counter()
        score_rows = head_scores.tolist()
        prepared_block_scores = (
            prepare_impress_block_scores(
                score_rows,
                block_size=impress_selection_block_size,
                prefix_tokens=loader.prefix_tokens,
            )
            if impress_selection_block_size > 1
            else None
        )
        promixed_policy = loader.promixed_policy
        decisions_by_keep_blocks: dict[int, PromixedSelectionDecision] = {}
        if promixed_policy is not None:
            if prepared_block_scores is None:
                raise RuntimeError(
                    "ProMixed GQA selection requires contiguous block scores"
                )
            base_ratio = max(float(loader.keep_ratio), 1e-12)
            window_end = min(
                loader.layers,
                layer_index + impress_selection_period_size,
            )
            window_risk = max(
                min(
                    1.0,
                    max(
                        0.0,
                        float(loader.keep_ratio_for_layer(target)) / base_ratio
                        - 1.0,
                    ),
                )
                for target in range(layer_index, window_end)
            )
            leader_keep_blocks = loader.keep_blocks_for_layer(layer_index)
            leader_decision = select_promixed_gqa_blocks(
                prepared_block_scores.block_scores,
                keep_blocks=leader_keep_blocks,
                max_period=impress_selection_period_size,
                sensitivity_risk=window_risk,
                **promixed_policy,
            )
            adaptive_coverage = bool(
                promixed_policy.get("adaptive_coverage", False)
            )
            if not adaptive_coverage:
                decisions_by_keep_blocks[leader_keep_blocks] = leader_decision
            configured_period = leader_decision.period
            loader.record_promixed_decision(
                agreement=leader_decision.agreement,
                boundary_margin=leader_decision.boundary_margin,
                uncertainty=leader_decision.uncertainty,
                period=leader_decision.period,
            )
        else:
            configured_period = impress_selection_period_size
        target_layers = range(
            layer_index,
            min(loader.layers, layer_index + configured_period),
        )
        similarities = []
        used_probe_for_all = True
        for target_layer in target_layers:
            keep_ratio = float(loader.keep_ratio_for_layer(target_layer))
            if impress_selection_block_size > 1:
                if prepared_block_scores is None:
                    raise RuntimeError("IMPRESS block scores were not prepared")
                keep_blocks = loader.keep_blocks_for_layer(target_layer)
                if promixed_policy is not None:
                    decision = (
                        None if adaptive_coverage
                        else decisions_by_keep_blocks.get(keep_blocks)
                    )
                    if decision is None:
                        target_risk = min(
                            1.0,
                            max(0.0, keep_ratio / base_ratio - 1.0),
                        )
                        decision = select_promixed_gqa_blocks(
                            prepared_block_scores.block_scores,
                            keep_blocks=keep_blocks,
                            max_period=configured_period,
                            sensitivity_risk=target_risk,
                            **promixed_policy,
                        )
                        if not adaptive_coverage:
                            decisions_by_keep_blocks[keep_blocks] = decision
                    selected = [
                        token
                        for block in decision.selected_blocks
                        for token in range(
                            block * prepared_block_scores.block_size,
                            min(
                                prepared_block_scores.prefix_tokens,
                                (block + 1)
                                * prepared_block_scores.block_size,
                            ),
                        )
                    ]
                    priority = [
                        token
                        for block in decision.priority_blocks
                        for token in range(
                            block * prepared_block_scores.block_size,
                            min(
                                prepared_block_scores.prefix_tokens,
                                (block + 1)
                                * prepared_block_scores.block_size,
                            ),
                        )
                    ]
                    used_probe = True
                    similarity = decision.agreement
                else:
                    fallback_keep_blocks_limit = loader.max_blocks_for_layer(
                        target_layer
                    )
                    selected, priority, used_probe, similarity = (
                        select_prepared_impress_blocks(
                            prepared_block_scores,
                            keep_ratio=keep_ratio,
                            similarity_alpha=loader.similarity_alpha,
                            keep_blocks=keep_blocks,
                            fallback_keep_blocks_limit=(
                                fallback_keep_blocks_limit
                            ),
                        )
                    )
            else:
                keep_tokens = max(
                    1, math.ceil(loader.prefix_tokens * keep_ratio)
                )
                selected, priority, used_probe, similarity = (
                    impress_probe_token_selection_with_ranking(
                        score_rows,
                        keep_tokens,
                        tuple(range(len(loader.probe_query_heads))),
                        loader.similarity_alpha,
                    )
                )
            loader.configure_impress_layer(
                layer=target_layer,
                selected_tokens=sorted(selected),
                prefetch_priority_tokens=priority,
            )
            similarities.append(similarity)
            used_probe_for_all = used_probe_for_all and used_probe
        loader.schedule_impress_missing(layer_index)
        loader.record_selector_compute(
            score_path_ms
            + (time.perf_counter() - decision_started) * 1000,
            similarity=sum(similarities) / len(similarities),
            fallback=not used_probe_for_all,
        )
    del selector_keys
    return configured_period


def flexgen_sparse_decoder_logits(
    *,
    model,
    input_ids,
    cache,
    position_start: int,
    loader,
    period_size: int,
    subperiod_size: int,
    impress_selection_block_size: int = 1,
    impress_period_prefetch_size: int = 1,
    impress_selection_period_size: int = 1,
    impress_known_period_prefetch: bool = False,
):
    """Run Qwen with subperiod-gated FlexGen cache prefetching."""

    import torch

    if period_size <= 0:
        raise ValueError("period_size must be positive")
    if not 0 < subperiod_size <= period_size:
        raise ValueError("subperiod_size must be in [1, period_size]")
    if impress_selection_period_size <= 0:
        raise ValueError("impress_selection_period_size must be positive")

    core = model.model
    hidden_states = core.embed_tokens(input_ids)
    query_tokens = int(input_ids.shape[1])
    cache_position = torch.arange(
        position_start, position_start + query_tokens, device=input_ids.device
    )
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden_states, position_ids)
    first_prefill = cache.get_seq_length(0) == 0
    next_impress_selection_layer = 0

    for layer_index, decoder_layer in enumerate(core.layers[: core.config.num_hidden_layers]):
        if first_prefill:
            within_period = layer_index % period_size
            if (
                loader.online_selection
                and loader.method == "impress"
                and loader.impress_async_prefetch
            ):
                loader.resolve_impress_speculation(layer_index)
            promixed_selection = (
                loader.method == "impress"
                and getattr(loader, "promixed_policy", None) is not None
            )
            impress_selection_leader = (
                loader.method == "impress"
                and (
                    layer_index == next_impress_selection_layer
                    if promixed_selection
                    else layer_index % impress_selection_period_size == 0
                )
            )
            configured_impress_period = impress_selection_period_size
            if loader.online_selection and (
                loader.method == "as_h2o_lru"
                or impress_selection_leader
                or (loader.method == "contigkv" and within_period == 0)
            ):
                configured_impress_period = configure_online_layer_selection(
                    decoder_layer=decoder_layer,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    layer_index=layer_index,
                    loader=loader,
                    period_size=period_size,
                    impress_selection_block_size=impress_selection_block_size,
                    impress_selection_period_size=impress_selection_period_size,
                )
                if promixed_selection:
                    next_impress_selection_layer = (
                        layer_index + configured_impress_period
                    )
                if (
                    loader.method == "impress"
                    and impress_known_period_prefetch
                    and configured_impress_period > 1
                ):
                    loader.schedule_range(
                        layer_index,
                        min(
                            loader.layers,
                            layer_index + configured_impress_period,
                        ),
                    )
            elif (
                loader.online_selection
                and loader.method == "impress"
                and loader.impress_async_prefetch
                and impress_selection_period_size > 1
            ):
                loader.schedule_impress_missing(layer_index)
            if (
                loader.online_selection
                and loader.method == "impress"
                and loader.impress_async_prefetch
                and loader.impress_deferred_compute_timing
            ):
                loader.resolve_deferred_impress_compute()
            if loader.method == "contigkv":
                if within_period == 0:
                    if layer_index > 0:
                        loader.schedule_inter_period_missing(layer_index, period_size)
                    loader.prime_period(layer_index, subperiod_size, period_size)
                elif within_period == subperiod_size:
                    current_period_start = layer_index - subperiod_size
                    loader.schedule_inter_period(
                        previous_period_start=current_period_start,
                        target_period_start=current_period_start + period_size,
                        period_size=period_size,
                    )

            key, value = loader.resolve(layer_index)
            cache.layers[layer_index].update(
                _cache_layout(key, dtype=hidden_states.dtype, device=hidden_states.device),
                _cache_layout(value, dtype=hidden_states.dtype, device=hidden_states.device),
            )
            if (
                loader.online_selection
                and loader.method == "impress"
                and loader.impress_async_prefetch
            ):
                loader.schedule_impress_next(layer_index)
                if promixed_selection and impress_selection_leader:
                    loader.prefetch_selector_keys(
                        next_impress_selection_layer
                    )
                elif impress_selection_period_size == 1:
                    loader.prefetch_selector_keys(layer_index + 1)
                elif layer_index % impress_selection_period_size == 0:
                    loader.prefetch_selector_keys(
                        layer_index + impress_selection_period_size
                    )

        attention_mask = _layer_causal_mask(
            query_tokens=query_tokens,
            past_tokens=cache.get_seq_length(layer_index),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        measure_impress_compute = (
            first_prefill
            and loader.online_selection
            and loader.method == "impress"
            and loader.impress_async_prefetch
        )
        if measure_impress_compute:
            compute_started = torch.cuda.Event(enable_timing=True)
            compute_finished = torch.cuda.Event(enable_timing=True)
            compute_started.record()
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if measure_impress_compute:
            compute_finished.record()
            if loader.impress_deferred_compute_timing:
                loader.defer_impress_layer_compute(
                    compute_started,
                    compute_finished,
                )
            else:
                compute_finished.synchronize()
                loader.record_impress_layer_compute(
                    compute_started.elapsed_time(compute_finished)
                )
            loader.schedule_impress_period(
                layer_index,
                impress_period_prefetch_size,
            )
    return model.lm_head(core.norm(hidden_states))


def greedy_flexgen_completion(
    *,
    model,
    tokenizer,
    query_token_ids: Sequence[int],
    prefix_tokens: int,
    loader,
    max_tokens: int,
    period_size: int,
    subperiod_size: int,
    impress_selection_block_size: int = 1,
    impress_period_prefetch_size: int = 1,
    impress_selection_period_size: int = 1,
    impress_known_period_prefetch: bool = False,
    first_token_logits_out: list[Any] | None = None,
    label_continuations: Mapping[str, Sequence[int]] | None = None,
    label_token_logprobs_out: dict[str, list[float]] | None = None,
    label_scoring_time_ms_out: list[float] | None = None,
    first_token_ready_time_ms_out: list[float] | None = None,
    response_ready_time_ms_out: list[float] | None = None,
    evaluation_ready_time_ms_out: list[float] | None = None,
) -> tuple[str, float, float, dict[str, Any]]:
    """Generate a short completion while timing the FlexGen-backed TTFT."""

    import torch
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device
    if label_token_logprobs_out is not None and label_continuations is None:
        raise ValueError("label continuations are required for continuation scoring")
    normalized_continuations = (
        {
            str(label): tuple(int(token) for token in tokens)
            for label, tokens in label_continuations.items()
        }
        if label_continuations is not None
        else None
    )
    if normalized_continuations is not None and (
        not normalized_continuations
        or any(not tokens for tokens in normalized_continuations.values())
    ):
        raise ValueError("every label continuation must contain at least one token")
    query = torch.tensor([list(query_token_ids)], dtype=torch.long, device=device)
    cache = DynamicCache(config=model.config)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = flexgen_sparse_decoder_logits(
            model=model,
            input_ids=query,
            cache=cache,
            position_start=prefix_tokens,
            loader=loader,
            period_size=period_size,
            subperiod_size=subperiod_size,
            impress_selection_block_size=impress_selection_block_size,
            impress_period_prefetch_size=impress_period_prefetch_size,
            impress_selection_period_size=impress_selection_period_size,
            impress_known_period_prefetch=impress_known_period_prefetch,
        )
    torch.cuda.synchronize()
    first_token_time = time.perf_counter()
    flush_cache_scores = getattr(loader, "flush_deferred_cache_score_updates", None)
    if flush_cache_scores is not None:
        flush_cache_scores()
    if loader.impress_deferred_compute_timing:
        loader.resolve_deferred_impress_compute()
    query_last_logits = logits[0, -1].float()
    generated = [int(query_last_logits.argmax().item())]
    if first_token_ready_time_ms_out is not None:
        first_token_ready_time_ms_out.append(
            (time.perf_counter() - start) * 1000
        )
    label_cache_data = (
        tuple((layer.keys, layer.values) for layer in cache.layers)
        if label_token_logprobs_out is not None and max_tokens > 1
        else None
    )
    for step in range(1, max_tokens):
        if tokenizer.eos_token_id is not None and generated[-1] == tokenizer.eos_token_id:
            break
        next_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = flexgen_sparse_decoder_logits(
                model=model,
                input_ids=next_input,
                cache=cache,
                position_start=prefix_tokens + len(query_token_ids) + step - 1,
                loader=loader,
                period_size=period_size,
                subperiod_size=subperiod_size,
                impress_selection_block_size=impress_selection_block_size,
                impress_period_prefetch_size=impress_period_prefetch_size,
                impress_selection_period_size=impress_selection_period_size,
                impress_known_period_prefetch=impress_known_period_prefetch,
            )
        generated.append(int(logits[0, -1].argmax().item()))
    torch.cuda.synchronize()
    end = time.perf_counter()
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    if response_ready_time_ms_out is not None:
        response_ready_time_ms_out.append(
            (time.perf_counter() - start) * 1000
        )
    if first_token_logits_out is not None:
        first_token_logits_out.append(query_last_logits.cpu())
    if label_token_logprobs_out is not None and label_cache_data is None:
        label_cache_data = tuple(
            (layer.keys, layer.values) for layer in cache.layers
        )
    if label_token_logprobs_out is not None:
        if normalized_continuations is None or label_cache_data is None:
            raise RuntimeError("continuation scoring state was not initialized")
        scoring_started = time.perf_counter()
        first_log_probs = query_last_logits.log_softmax(dim=-1)
        with torch.inference_mode():
            for label, token_ids in normalized_continuations.items():
                token_logprobs = [first_log_probs[token_ids[0]]]
                if len(token_ids) > 1:
                    candidate_cache = _copy_dynamic_cache(
                        label_cache_data,
                        model.config,
                    )
                    for token_index, (input_token, target_token) in enumerate(
                        zip(token_ids, token_ids[1:])
                    ):
                        candidate_input = torch.tensor(
                            [[input_token]],
                            dtype=torch.long,
                            device=device,
                        )
                        candidate_logits = flexgen_sparse_decoder_logits(
                            model=model,
                            input_ids=candidate_input,
                            cache=candidate_cache,
                            position_start=(
                                prefix_tokens
                                + len(query_token_ids)
                                + token_index
                            ),
                            loader=loader,
                            period_size=period_size,
                            subperiod_size=subperiod_size,
                            impress_selection_block_size=(
                                impress_selection_block_size
                            ),
                            impress_period_prefetch_size=(
                                impress_period_prefetch_size
                            ),
                            impress_selection_period_size=(
                                impress_selection_period_size
                            ),
                            impress_known_period_prefetch=(
                                impress_known_period_prefetch
                            ),
                        )
                        token_logprobs.append(
                            candidate_logits[0, -1]
                            .float()
                            .log_softmax(dim=-1)[target_token]
                        )
                    del candidate_cache
                label_token_logprobs_out[label] = [
                    float(value)
                    for value in torch.stack(token_logprobs).cpu().tolist()
                ]
        torch.cuda.synchronize()
        if label_scoring_time_ms_out is not None:
            label_scoring_time_ms_out.append(
                (time.perf_counter() - scoring_started) * 1000
            )
    if evaluation_ready_time_ms_out is not None:
        evaluation_ready_time_ms_out.append(
            (time.perf_counter() - start) * 1000
        )
    metrics = loader.metrics()
    loader.close()
    del cache
    return (
        completion,
        (first_token_time - start) * 1000,
        (end - start) * 1000,
        metrics,
    )


def _load_model_and_tokenizer(model_path: str, *, dtype: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    return model, tokenizer


def run_flexgen_reprefill(
    *,
    model_path: str,
    bundle_dir: str | Path,
    store_root: str | Path,
    plan_path: str | Path,
    tasks: Sequence[str],
    store_tasks: Sequence[str] | None = None,
    samples_per_task: int,
    output_dir: str | Path,
    max_tokens: int,
    accuracy_scoring: str = "generation",
    device: str,
    dtype: str,
    flexgen_config: FlexGenPcacheConfig,
    period_size: int,
    subperiod_size: int,
    expected_keep_ratio: float | None = None,
    warmup_passes: int = 0,
    warmup_samples_per_task: int | None = None,
    online_selection: bool = False,
    probe_query_heads: Sequence[int] = (0, 1, 2),
    similarity_alpha: float = 0.6,
    impress_async_prefetch: bool = False,
    impress_selection_block_size: int = 1,
    layer_budget_profile: str | Path | None = None,
    exact_layer_block_budget: bool = False,
    impress_period_prefetch_size: int = 1,
    impress_period_prefetch_budget_scale: float = 1.0,
    impress_priority_prefetch: bool = False,
    impress_deferred_compute_timing: bool = False,
    impress_rolling_period_prefetch: bool = False,
    impress_value_ordered_prefetch: bool = False,
    impress_value_prefetch_budget_scale: float = 1.0,
    impress_selection_period_size: int = 1,
    impress_known_period_prefetch: bool = False,
    promixed_gqa_selection: bool = False,
    promixed_coverage_fraction: float = 0.5,
    promixed_margin_reference: float = 0.05,
    promixed_agreement_weight: float = 0.75,
    promixed_sensitivity_weight: float = 0.1,
    promixed_p1_threshold: float = 0.90,
    promixed_p2_threshold: float = 0.82,
    promixed_p4_threshold: float = 0.68,
    promixed_adaptive_coverage: bool = False,
    promixed_utility_max_weight: float = 0.55,
    promixed_utility_mean_weight: float = 0.35,
    promixed_utility_vote_weight: float = 0.10,
    defer_cache_score_updates: bool = False,
    as_baseline_mode: str = "none",
) -> dict[str, Any]:
    """Run a matched method plan on Qwen through the shared FlexGen cache path."""

    import torch

    if warmup_passes < 0:
        raise ValueError("warmup_passes must be non-negative")
    if warmup_samples_per_task is not None and warmup_samples_per_task <= 0:
        raise ValueError("warmup_samples_per_task must be positive")
    if accuracy_scoring not in {
        "generation",
        "label_first_token_logit",
        "label_continuation_loglikelihood",
    }:
        raise ValueError(
            "accuracy_scoring must be generation, label_first_token_logit, "
            "or label_continuation_loglikelihood"
        )
    registered_store_tasks = resolve_store_tasks(tasks, store_tasks)
    rows = _load_bundle_records(bundle_dir, tasks, samples_per_task)
    if online_selection:
        plan_metadata = _load_plan_metadata(plan_path)
    else:
        _, plan_metadata = _load_layer_plan(plan_path, str(rows[0]["uid"]))
    plan_keep_ratio = validate_plan_keep_ratio(plan_metadata, expected_keep_ratio)
    chunk_size = int(plan_metadata.get("chunk_size", 0))
    if chunk_size != flexgen_config.chunk_size:
        raise ValueError(
            f"plan chunk size {chunk_size} does not match Pcache chunk size {flexgen_config.chunk_size}"
        )
    plan_method = str(plan_metadata.get("method", ""))
    if plan_method not in {"contigkv", "impress"}:
        raise ValueError(f"plan metadata has unsupported method {plan_method!r}")
    if as_baseline_mode not in {"none", "as_lru", "as_h2o_lru"}:
        raise ValueError(
            "as_baseline_mode must be none, as_lru, or as_h2o_lru"
        )
    if as_baseline_mode != "none":
        if not online_selection:
            raise ValueError("AttentionStore baselines require online selection")
        if plan_method != "impress":
            raise ValueError(
                "AttentionStore baselines require a chunk-64 IMPRESS shape plan"
            )
        if flexgen_config.chunk_size != 64:
            raise ValueError("AttentionStore baselines require 64-token chunks")
        if flexgen_config.cache_type != "LRU":
            raise ValueError("AttentionStore baselines require the original LRU policy")
        if flexgen_config.selector_index_dir is not None:
            raise ValueError("AttentionStore baselines cannot use a selector index")
        if flexgen_config.impress_reorder_path is not None:
            raise ValueError("AttentionStore baselines cannot use IMPRESS reordering")
        if (
            impress_async_prefetch
            or impress_selection_block_size != 1
            or impress_selection_period_size != 1
            or layer_budget_profile is not None
            or exact_layer_block_budget
            or promixed_gqa_selection
            or defer_cache_score_updates
        ):
            raise ValueError(
                "AttentionStore baselines cannot use IMPRESS/ProMixed extensions"
            )
        method = as_baseline_mode
    else:
        method = plan_method
    if flexgen_config.impress_reorder_path is not None and method != "impress":
        raise ValueError("an IMPRESS reorder manifest cannot be used for ContiguousKV")
    if impress_async_prefetch and method != "impress":
        raise ValueError("IMPRESS asynchronous prefetch cannot be used for ContiguousKV")
    if layer_budget_profile is not None and (
        not online_selection or method != "impress"
    ):
        raise ValueError(
            "layer budget profiles require online IMPRESS/HyperInfer selection"
        )
    if exact_layer_block_budget and (
        layer_budget_profile is None
        or not online_selection
        or method != "impress"
        or impress_selection_block_size <= 1
    ):
        raise ValueError(
            "exact layer block budgets require an online IMPRESS layer profile "
            "and contiguous block selection"
        )
    if impress_period_prefetch_size <= 0:
        raise ValueError("IMPRESS predictive period size must be positive")
    if impress_period_prefetch_size > 1 and (
        not online_selection or method != "impress" or not impress_async_prefetch
    ):
        raise ValueError(
            "IMPRESS predictive periods require asynchronous online IMPRESS"
        )
    if (
        not math.isfinite(float(impress_period_prefetch_budget_scale))
        or not 0 < float(impress_period_prefetch_budget_scale) <= 1
    ):
        raise ValueError(
            "IMPRESS predictive-period budget scale must be finite and in (0, 1]"
        )
    if impress_priority_prefetch and (
        not online_selection or method != "impress" or not impress_async_prefetch
    ):
        raise ValueError(
            "IMPRESS priority prefetch requires asynchronous online IMPRESS"
        )
    if impress_deferred_compute_timing and (
        not online_selection or method != "impress" or not impress_async_prefetch
    ):
        raise ValueError(
            "IMPRESS deferred compute timing requires asynchronous online IMPRESS"
        )
    if impress_rolling_period_prefetch and (
        not online_selection
        or method != "impress"
        or not impress_async_prefetch
        or impress_period_prefetch_size <= 1
    ):
        raise ValueError(
            "IMPRESS rolling Period prefetch requires predictive asynchronous "
            "online IMPRESS"
        )
    if impress_value_ordered_prefetch and (
        not online_selection or method != "impress" or not impress_async_prefetch
    ):
        raise ValueError(
            "IMPRESS value-ordered prefetch requires asynchronous online IMPRESS"
        )
    value_budget_scale = float(impress_value_prefetch_budget_scale)
    if (
        not math.isfinite(value_budget_scale)
        or value_budget_scale <= 0.0
        or value_budget_scale > 1.0
    ):
        raise ValueError(
            "IMPRESS value prefetch budget scale must be in (0, 1]"
        )
    if value_budget_scale != 1.0 and not impress_value_ordered_prefetch:
        raise ValueError(
            "IMPRESS value prefetch budget scaling requires value-ordered "
            "prefetch"
        )
    if impress_selection_period_size <= 0:
        raise ValueError("IMPRESS selection Period size must be positive")
    if impress_selection_period_size > 1 and (
        not online_selection
        or method != "impress"
        or impress_selection_block_size <= 1
    ):
        raise ValueError(
            "periodic importance selection requires online IMPRESS/HyperInfer "
            "with contiguous block selection"
        )
    promixed_policy: dict[str, float] | None = None
    if promixed_gqa_selection:
        if (
            not online_selection
            or method != "impress"
            or impress_selection_block_size <= 1
        ):
            raise ValueError(
                "ProMixed GQA selection requires online IMPRESS with "
                "contiguous block selection"
            )
        if impress_selection_period_size not in {1, 2, 4, 8}:
            raise ValueError("ProMixed max Period must be one of 1, 2, 4, or 8")
        policy_values = (
            promixed_coverage_fraction,
            promixed_margin_reference,
            promixed_agreement_weight,
            promixed_sensitivity_weight,
            promixed_p1_threshold,
            promixed_p2_threshold,
            promixed_p4_threshold,
            promixed_utility_max_weight,
            promixed_utility_mean_weight,
            promixed_utility_vote_weight,
        )
        if any(not math.isfinite(float(value)) for value in policy_values):
            raise ValueError("ProMixed policy values must be finite")
        if not 0 <= promixed_coverage_fraction <= 1:
            raise ValueError("ProMixed coverage fraction must be in [0, 1]")
        if promixed_margin_reference <= 0:
            raise ValueError("ProMixed margin reference must be positive")
        if not 0 <= promixed_agreement_weight <= 1:
            raise ValueError("ProMixed agreement weight must be in [0, 1]")
        if not 0 <= promixed_sensitivity_weight <= 1:
            raise ValueError("ProMixed sensitivity weight must be in [0, 1]")
        if not 0 <= promixed_p4_threshold <= promixed_p2_threshold <= promixed_p1_threshold <= 1:
            raise ValueError("ProMixed thresholds must satisfy 0 <= P4 <= P2 <= P1 <= 1")
        utility_weights = (
            float(promixed_utility_max_weight),
            float(promixed_utility_mean_weight),
            float(promixed_utility_vote_weight),
        )
        if any(weight < 0 for weight in utility_weights):
            raise ValueError("ProMixed utility weights must be non-negative")
        if not math.isclose(
            sum(utility_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("ProMixed utility weights must sum to 1")
        promixed_policy = {
            "coverage_fraction": float(promixed_coverage_fraction),
            "margin_reference": float(promixed_margin_reference),
            "agreement_weight": float(promixed_agreement_weight),
            "sensitivity_weight": float(promixed_sensitivity_weight),
            "p1_threshold": float(promixed_p1_threshold),
            "p2_threshold": float(promixed_p2_threshold),
            "p4_threshold": float(promixed_p4_threshold),
            "adaptive_coverage": bool(promixed_adaptive_coverage),
            "utility_max_weight": utility_weights[0],
            "utility_mean_weight": utility_weights[1],
            "utility_vote_weight": utility_weights[2],
        }
    if defer_cache_score_updates and (
        not online_selection or flexgen_config.cache_type != "CKLFU"
    ):
        raise ValueError("deferred cache-score updates require online CKLFU")
    if impress_known_period_prefetch and (
        impress_selection_period_size <= 1
        or not online_selection
        or method != "impress"
        or not impress_async_prefetch
    ):
        raise ValueError(
            "known Period prefetch requires periodic asynchronous online IMPRESS"
        )
    validate_impress_block_mode(
        method=method,
        online_selection=online_selection,
        block_size=impress_selection_block_size,
        physical_chunk_size=flexgen_config.chunk_size,
        reorder_enabled=flexgen_config.impress_reorder_path is not None,
    )
    if (
        not online_selection
        and method == "impress"
        and _load_layer_token_selections(plan_path, str(rows[0]["uid"])) is None
    ):
        raise ValueError(
            "IMPRESS plans must include useful token selections; regenerate them with paper_plan_generator"
        )
    if (
        not online_selection
        and method == "contigkv"
        and flexgen_config.cache_type == "CKLFU"
        and _load_layer_chunk_scores(plan_path, str(rows[0]["uid"])) is None
    ):
        raise ValueError(
            "CKLFU ContiguousKV runs require attention scores; regenerate the plan"
        )
    plan_period = int(plan_metadata.get("period_size", period_size))
    plan_subperiod = int(plan_metadata.get("subperiod_size", subperiod_size))
    if method == "contigkv" and (plan_period != period_size or plan_subperiod != subperiod_size):
        raise ValueError("CLI period configuration must match the plan metadata")

    model, tokenizer = _load_model_and_tokenizer(model_path, dtype=dtype, device=device)
    budget_profile = None
    if layer_budget_profile is not None:
        budget_profile = load_layer_budget_profile(
            layer_budget_profile,
            expected_layers=int(model.config.num_hidden_layers),
            expected_model=str(model_path),
        )
        if not math.isclose(
            budget_profile.target_mean_ratio,
            plan_keep_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "layer budget target mean "
                f"{budget_profile.target_mean_ratio} does not match plan keep ratio "
                f"{plan_keep_ratio}"
            )
    if online_selection:
        validate_online_selector_mapping(
            model.config,
            method=method,
            selector_kv_head_ids=flexgen_config.selector_kv_head_ids,
            probe_query_heads=probe_query_heads,
        )
    if promixed_gqa_selection:
        query_heads = int(model.config.num_attention_heads)
        kv_heads = int(model.config.num_key_value_heads)
        groups = query_heads // kv_heads
        mapped_groups = tuple(int(head) // groups for head in probe_query_heads)
        expected_groups = tuple(range(kv_heads))
        if (
            len(mapped_groups) != kv_heads
            or tuple(sorted(mapped_groups)) != expected_groups
        ):
            raise ValueError(
                "ProMixed requires exactly one probe query head per physical GQA group"
            )
    store = FlexGenPcacheStore(flexgen_config)
    active_tasks = frozenset(str(task) for task in tasks)
    for task in registered_store_tasks:
        store.add_task(
            store_root=store_root,
            task=task,
            preload_selector_index=task in active_tasks,
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored: list[dict[str, Any]] = []

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        task = str(row["task"])
        info = read_store_info(store_root, task)
        uid = str(row["uid"])
        reference_tokens: list[list[int]] | None
        if online_selection:
            layer_plan = build_online_layer_plan(
                layers=info.layers,
                prefix_tokens=info.prefix_tokens,
                chunk_size=flexgen_config.chunk_size,
            )
            token_selections = None
            chunk_scores = None
            if method in {"as_lru", "as_h2o_lru"}:
                # The IMPRESS plan is shape/budget metadata only for these
                # baselines, never a selection reference.
                reference_tokens = None
            else:
                reference_plan = _load_optional_layer_plan(plan_path, uid)
                if reference_plan is None:
                    reference_tokens = None
                else:
                    reference_tokens = selected_tokens_for_plan(
                        reference_plan,
                        chunk_size=flexgen_config.chunk_size,
                        prefix_tokens=info.prefix_tokens,
                        layer_token_selections=_load_layer_token_selections(plan_path, uid),
                    )
        else:
            layer_plan, _ = _load_layer_plan(plan_path, uid)
            token_selections = _load_layer_token_selections(plan_path, uid)
            chunk_scores = _load_layer_chunk_scores(plan_path, uid)
            reference_tokens = selected_tokens_for_plan(
                layer_plan,
                chunk_size=flexgen_config.chunk_size,
                prefix_tokens=info.prefix_tokens,
                layer_token_selections=token_selections,
            )
        loader = store.new_layer_loader(
            task=task,
            layer_plan=layer_plan,
            method=method,
            layer_token_selections=token_selections,
            online_selection=online_selection,
            keep_ratio=plan_keep_ratio,
            layer_keep_ratios=(
                budget_profile.layer_ratios if budget_profile is not None else None
            ),
            layer_keep_blocks=(
                allocate_exact_layer_blocks(
                    budget_profile.layer_ratios,
                    blocks_per_layer=math.ceil(
                        info.prefix_tokens / impress_selection_block_size
                    ),
                    target_ratio=plan_keep_ratio,
                )
                if exact_layer_block_budget and budget_profile is not None
                else None
            ),
            probe_query_heads=probe_query_heads,
            similarity_alpha=similarity_alpha,
            impress_async_prefetch=impress_async_prefetch,
            impress_period_prefetch_size=impress_period_prefetch_size,
            impress_period_prefetch_budget_scale=(
                impress_period_prefetch_budget_scale
            ),
            impress_priority_prefetch=impress_priority_prefetch,
            impress_deferred_compute_timing=impress_deferred_compute_timing,
            impress_rolling_period_prefetch=impress_rolling_period_prefetch,
            impress_value_ordered_prefetch=impress_value_ordered_prefetch,
            impress_value_prefetch_budget_scale=value_budget_scale,
            promixed_policy=promixed_policy,
            defer_cache_score_updates=defer_cache_score_updates,
        )
        if method == "as_lru":
            loader.configure_as_full_retention()
        query_ids = tokenizer(str(row["query_text"]), add_special_tokens=False).input_ids
        labels = tuple(str(label) for label in row["labels"])
        label_continuation_prefix = str(
            row.get("label_continuation_prefix", "")
        )
        complete_label_ids = (
            continuation_token_ids(
                tokenizer,
                labels,
                continuation_prefix=label_continuation_prefix,
            )
            if accuracy_scoring == "label_continuation_loglikelihood"
            else None
        )
        first_token_logits: list[Any] | None = (
            [] if accuracy_scoring != "generation" else None
        )
        label_token_logprobs: dict[str, list[float]] | None = (
            {}
            if accuracy_scoring == "label_continuation_loglikelihood"
            else None
        )
        label_scoring_times: list[float] | None = (
            []
            if accuracy_scoring == "label_continuation_loglikelihood"
            else None
        )
        first_token_ready_times: list[float] = []
        response_ready_times: list[float] = []
        evaluation_ready_times: list[float] = []
        generation_prediction, ttft_ms, latency_ms, metrics = greedy_flexgen_completion(
            model=model,
            tokenizer=tokenizer,
            query_token_ids=query_ids,
            prefix_tokens=info.prefix_tokens,
            loader=loader,
            max_tokens=max_tokens,
            period_size=period_size,
            subperiod_size=subperiod_size,
            impress_selection_block_size=impress_selection_block_size,
            impress_period_prefetch_size=impress_period_prefetch_size,
            impress_selection_period_size=impress_selection_period_size,
            impress_known_period_prefetch=impress_known_period_prefetch,
            first_token_logits_out=first_token_logits,
            label_continuations=complete_label_ids,
            label_token_logprobs_out=label_token_logprobs,
            label_scoring_time_ms_out=label_scoring_times,
            first_token_ready_time_ms_out=first_token_ready_times,
            response_ready_time_ms_out=response_ready_times,
            evaluation_ready_time_ms_out=evaluation_ready_times,
        )
        if (
            len(first_token_ready_times) != 1
            or len(response_ready_times) != 1
            or len(evaluation_ready_times) != 1
        ):
            raise RuntimeError("request timing boundaries were not recorded")
        label_first_token_scores = None
        label_first_token_prediction = None
        label_continuation_scores = None
        label_ids = None
        if first_token_logits is not None:
            if len(first_token_logits) != 1:
                raise RuntimeError("the measured forward did not expose one logits row")
            if accuracy_scoring == "label_first_token_logit":
                (
                    prediction,
                    label_first_token_scores,
                    label_ids,
                ) = predict_from_label_logits(
                    first_token_logits[0],
                    tokenizer,
                    labels,
                    continuation_prefix=label_continuation_prefix,
                )
                label_first_token_prediction = prediction
            else:
                if complete_label_ids is None or label_token_logprobs is None:
                    raise RuntimeError("continuation label scores were not captured")
                label_ids = complete_label_ids
                label_first_token_scores = {
                    label: float(first_token_logits[0][ids[0]].item())
                    for label, ids in label_ids.items()
                }
                label_first_token_prediction = max(
                    label_first_token_scores,
                    key=label_first_token_scores.__getitem__,
                )
                (
                    prediction,
                    label_continuation_scores,
                ) = predict_from_label_token_logprobs(label_token_logprobs)
        else:
            prediction = generation_prediction
        actual_selected_tokens = loader.selected_tokens_by_layer()
        if reference_tokens is None:
            selection_jaccard = selection_exact_fraction = None
        else:
            selection_jaccard, selection_exact_fraction = layer_selection_agreement(
                actual_selected_tokens, reference_tokens
            )
        if online_selection:
            score_updates = int(metrics["cache_score_updates"])
            cache_update_ms = float(metrics["cache_update_ms"])
        else:
            cache_update_start = time.perf_counter()
            score_updates = 0
            if chunk_scores is not None:
                score_updates = store.update_attention_scores(
                    task=task,
                    layer_plan=layer_plan,
                    layer_chunk_scores=chunk_scores,
                )
            elif method == "impress" and token_selections is not None:
                score_updates = store.update_impress_scores(
                    task=task,
                    layer_token_selections=token_selections,
                )
            cache_update_ms = (time.perf_counter() - cache_update_start) * 1000
        result = {
            "uid": row["uid"],
            "task": task,
            "answer": row["answer"],
            "prediction": prediction,
            "generation_prediction": generation_prediction,
            "accuracy_scoring": accuracy_scoring,
            "correct": prediction_is_correct(prediction, str(row["answer"])),
            "ttft_ms": ttft_ms,
            "logits_ready_ms": ttft_ms,
            "latency_ms": latency_ms,
            "first_token_ready_ms": first_token_ready_times[0],
            "response_ready_ms": response_ready_times[0],
            "evaluation_ready_ms": evaluation_ready_times[0],
            "accuracy_scores_ready_ms": evaluation_ready_times[0],
            "cache_score_updates": score_updates,
            "cache_update_ms": cache_update_ms,
            "layer_token_selection_sha256": layer_token_selection_sha256(
                actual_selected_tokens
            ),
            **metrics,
        }
        if label_first_token_scores is not None and label_ids is not None:
            result["label_first_token_logits"] = label_first_token_scores
            result["label_first_token_prediction"] = (
                label_first_token_prediction
            )
            result["label_token_ids"] = {
                label: list(ids) for label, ids in label_ids.items()
            }
            result["label_continuation_prefix"] = label_continuation_prefix
        if (
            label_continuation_scores is not None
            and label_token_logprobs is not None
            and label_scoring_times is not None
        ):
            if len(label_scoring_times) != 1:
                raise RuntimeError("continuation scoring did not record one duration")
            result["label_candidate_mean_logprobs"] = (
                label_continuation_scores
            )
            result["label_candidate_token_logprobs"] = label_token_logprobs
            result["label_scoring_time_ms"] = label_scoring_times[0]
        if selection_jaccard is not None and selection_exact_fraction is not None:
            result.update(
                {
                    "selection_reference_mean_jaccard": selection_jaccard,
                    "selection_reference_exact_layer_fraction": selection_exact_fraction,
                }
            )
        return result

    warmup_rows = rows
    if warmup_samples_per_task is not None:
        warmup_rows = []
        for task in tasks:
            task_rows = [row for row in rows if row["task"] == task]
            warmup_rows.extend(task_rows[:warmup_samples_per_task])
    try:
        for _ in range(warmup_passes):
            for row in warmup_rows:
                execute(row)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for row in rows:
            scored.append(execute(row))
    finally:
        store.close()

    by_task: dict[str, dict[str, float | int]] = {}
    for task in tasks:
        task_rows = [row for row in scored if row["task"] == task]
        ttfts = [float(row["ttft_ms"]) for row in task_rows]
        latencies = [float(row["latency_ms"]) for row in task_rows]
        first_token_ready_times = [
            float(row["first_token_ready_ms"]) for row in task_rows
        ]
        response_ready_times = [
            float(row["response_ready_ms"]) for row in task_rows
        ]
        evaluation_ready_times = [
            float(row["evaluation_ready_ms"]) for row in task_rows
        ]
        row_count = max(1, len(task_rows))

        def mean_metric(name: str) -> float:
            return sum(float(row[name]) for row in task_rows) / row_count

        task_summary: dict[str, float | int] = {
            "samples": len(task_rows),
            "accuracy": sum(row["correct"] for row in task_rows) / row_count,
            "mean_ttft_ms": sum(ttfts) / row_count,
            "p95_ttft_ms": percentile95(ttfts),
            "mean_logits_ready_ms": sum(ttfts) / row_count,
            "p95_logits_ready_ms": percentile95(ttfts),
            "mean_latency_ms": sum(latencies) / row_count,
            "p95_latency_ms": percentile95(latencies),
            "mean_first_token_ready_ms": (
                sum(first_token_ready_times) / row_count
            ),
            "p95_first_token_ready_ms": percentile95(
                first_token_ready_times
            ),
            "mean_response_ready_ms": sum(response_ready_times) / row_count,
            "p95_response_ready_ms": percentile95(response_ready_times),
            "mean_evaluation_ready_ms": (
                sum(evaluation_ready_times) / row_count
            ),
            "p95_evaluation_ready_ms": percentile95(
                evaluation_ready_times
            ),
            "mean_selected_kv_bytes": mean_metric("selected_kv_bytes"),
            "mean_minimum_transfer_kv_bytes": mean_metric("minimum_transfer_kv_bytes"),
            "mean_logical_attention_keep_ratio": mean_metric("logical_attention_keep_ratio"),            "mean_physical_prefetch_kv_bytes": mean_metric("physical_prefetch_kv_bytes"),
            "mean_read_amplification": mean_metric("read_amplification"),
            "mean_effective_keep_ratio": mean_metric("effective_mean_keep_ratio"),
            "mean_prefetch_gpu_source_fraction": mean_metric(
                "prefetch_gpu_source_fraction"
            ),
            "mean_prefetch_cpu_source_fraction": mean_metric(
                "prefetch_cpu_source_fraction"
            ),
            "mean_prefetch_disk_source_fraction": mean_metric(
                "prefetch_disk_source_fraction"
            ),
            "mean_ssd_prefetch_kv_bytes": mean_metric("critical_ssd_read_bytes"),
            "mean_total_ssd_read_bytes": mean_metric("total_ssd_read_bytes"),
            "mean_selector_key_bytes": mean_metric("selector_key_bytes"),
            "mean_selector_dequantized_key_bytes": mean_metric("selector_dequantized_key_bytes"),
            "mean_selector_index_enabled": mean_metric("selector_index_enabled"),
            "mean_selector_index_bits": mean_metric("selector_index_bits"),
            "mean_selector_index_group_size": mean_metric("selector_index_group_size"),
            "mean_selector_index_preloaded": mean_metric(
                "selector_index_preloaded"
            ),
            "mean_selector_index_preloaded_bytes": mean_metric(
                "selector_index_preloaded_bytes"
            ),
            "mean_selector_gpu_source_bytes": mean_metric("selector_gpu_source_bytes"),
            "mean_selector_cpu_source_bytes": mean_metric("selector_cpu_source_bytes"),
            "mean_selector_disk_source_bytes": mean_metric("selector_disk_source_bytes"),
            "mean_selector_load_ms": mean_metric("selector_load_ms"),
            "mean_selector_compute_ms": mean_metric("selector_compute_ms"),
            "mean_selector_wait_ms": mean_metric("selector_wait_ms"),
            "mean_selector_calls": mean_metric("selector_calls"),
            "mean_selector_fallbacks": mean_metric("selector_fallbacks"),
            "mean_selector_jaccard": mean_metric("selector_mean_jaccard"),
            "mean_promixed_decisions": mean_metric("promixed_decisions"),
            "mean_promixed_gqa_agreement": mean_metric("promixed_mean_gqa_agreement"),
            "mean_promixed_boundary_margin": mean_metric("promixed_mean_boundary_margin"),
            "mean_promixed_uncertainty": mean_metric("promixed_mean_uncertainty"),
            "mean_promixed_period": mean_metric("promixed_mean_period"),
            "mean_promixed_p1_decisions": mean_metric("promixed_p1_decisions"),
            "mean_promixed_p2_decisions": mean_metric("promixed_p2_decisions"),
            "mean_promixed_p4_decisions": mean_metric("promixed_p4_decisions"),
            "mean_promixed_p8_decisions": mean_metric("promixed_p8_decisions"),
            "mean_speculative_hit_tokens": mean_metric("inter_period_hit_tokens"),
            "mean_speculative_missing_tokens": mean_metric("inter_period_missing_tokens"),
            "mean_speculative_unused_tokens": mean_metric("inter_period_unused_tokens"),
            "mean_impress_prefetch_budget_seconds": mean_metric(
                "impress_mean_prefetch_budget_seconds"
            ),
            "mean_impress_next_prefetch_jobs": mean_metric(
                "impress_next_prefetch_jobs"
            ),
            "mean_impress_period_prefetch_jobs": mean_metric(
                "impress_period_prefetch_jobs"
            ),
            "mean_impress_period_prefetch_tokens": mean_metric(
                "impress_period_prefetch_tokens"
            ),
            "mean_prefetch_wait_ms": mean_metric("prefetch_wait_ms"),
            "mean_prefetch_elapsed_ms": mean_metric("prefetch_elapsed_ms"),
            "mean_prefetch_scheduler_current_submitted": mean_metric(
                "prefetch_scheduler_current_submitted"
            ),
            "mean_prefetch_scheduler_next_submitted": mean_metric(
                "prefetch_scheduler_next_submitted"
            ),
            "mean_prefetch_scheduler_period_submitted": mean_metric(
                "prefetch_scheduler_period_submitted"
            ),
            "mean_prefetch_scheduler_current_queue_wait_ms": mean_metric(
                "prefetch_scheduler_current_queue_wait_ms"
            ),
            "mean_prefetch_scheduler_next_queue_wait_ms": mean_metric(
                "prefetch_scheduler_next_queue_wait_ms"
            ),
            "mean_prefetch_scheduler_period_queue_wait_ms": mean_metric(
                "prefetch_scheduler_period_queue_wait_ms"
            ),
            "mean_prefetch_scheduler_total_execution_ms": mean_metric(
                "prefetch_scheduler_total_execution_ms"
            ),
            "mean_impress_deferred_compute_samples": mean_metric(
                "impress_deferred_compute_samples"
            ),
            "mean_impress_deferred_compute_pending": mean_metric(
                "impress_deferred_compute_pending"
            ),
            "mean_impress_period_prediction_jaccard": mean_metric(
                "impress_period_prediction_mean_jaccard"
            ),
            "mean_impress_period_prediction_precision": mean_metric(
                "impress_period_prediction_precision"
            ),
            "mean_impress_period_prediction_recall": mean_metric(
                "impress_period_prediction_recall"
            ),
            "mean_impress_period_prefetch_hit_tokens": mean_metric(
                "impress_period_prefetch_hit_tokens"
            ),
            "mean_impress_period_prefetch_missing_tokens": mean_metric(
                "impress_period_prefetch_missing_tokens"
            ),
            "mean_impress_period_prefetch_unused_tokens": mean_metric(
                "impress_period_prefetch_unused_tokens"
            ),
            "mean_impress_value_ordered_prefetch_jobs": mean_metric(
                "impress_value_ordered_prefetch_jobs"
            ),
            "mean_impress_value_prefetch_budget_scale": mean_metric(
                "impress_value_prefetch_budget_scale"
            ),
            "mean_cache_update_ms": mean_metric("cache_update_ms"),
            "mean_cache_update_deferred_ms": mean_metric("cache_update_deferred_ms"),
            "mean_cache_update_in_ttft": mean_metric("cache_update_in_ttft"),
            "mean_cache_update_deferred": mean_metric("cache_update_deferred"),
        }
        reference_rows = [
            row for row in task_rows if "selection_reference_mean_jaccard" in row
        ]
        if reference_rows:
            task_summary["mean_selection_reference_jaccard"] = sum(
                float(row["selection_reference_mean_jaccard"]) for row in reference_rows
            ) / len(reference_rows)
            task_summary["mean_selection_reference_exact_layer_fraction"] = sum(
                float(row["selection_reference_exact_layer_fraction"])
                for row in reference_rows
            ) / len(reference_rows)
        if accuracy_scoring == "label_continuation_loglikelihood":
            task_summary["mean_label_scoring_time_ms"] = mean_metric(
                "label_scoring_time_ms"
            )
        by_task[task] = task_summary
    ttfts = [float(row["ttft_ms"]) for row in scored]
    latencies = [float(row["latency_ms"]) for row in scored]
    first_token_ready_times = [
        float(row["first_token_ready_ms"]) for row in scored
    ]
    response_ready_times = [
        float(row["response_ready_ms"]) for row in scored
    ]
    evaluation_ready_times = [
        float(row["evaluation_ready_ms"]) for row in scored
    ]
    summary: dict[str, Any] = {
        "model_path": model_path,
        "plan": str(plan_path),
        "tasks": by_task,
        "overall": {
            "samples": len(scored),
            "accuracy": sum(row["correct"] for row in scored) / max(1, len(scored)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
            "mean_logits_ready_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_logits_ready_ms": percentile95(ttfts),
            "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
            "p95_latency_ms": percentile95(latencies),
            "mean_first_token_ready_ms": (
                sum(first_token_ready_times)
                / max(1, len(first_token_ready_times))
            ),
            "p95_first_token_ready_ms": percentile95(
                first_token_ready_times
            ),
            "mean_response_ready_ms": (
                sum(response_ready_times) / max(1, len(response_ready_times))
            ),
            "p95_response_ready_ms": percentile95(response_ready_times),
            "mean_evaluation_ready_ms": (
                sum(evaluation_ready_times)
                / max(1, len(evaluation_ready_times))
            ),
            "p95_evaluation_ready_ms": percentile95(evaluation_ready_times),
            "mean_effective_keep_ratio": sum(
                float(row["effective_mean_keep_ratio"]) for row in scored
            )
            / max(1, len(scored)),
        },
        "runtime": {
            "backend": "FlexGen Pcache KV_Division plus Qwen layerwise sparse attention",
            "method": method,
            "plan_method": plan_method,
            "as_baseline_mode": as_baseline_mode,
            "runtime_variant": runtime_variant(
                method=method,
                online_selection=online_selection,
                impress_async_prefetch=impress_async_prefetch,
                impress_reorder_enabled=flexgen_config.impress_reorder_path is not None,
                impress_selection_block_size=impress_selection_block_size,
                layer_budget_profile_enabled=budget_profile is not None,
                exact_layer_block_budget=exact_layer_block_budget,
                impress_period_prefetch_size=impress_period_prefetch_size,
                impress_period_prefetch_budget_scale=(
                    impress_period_prefetch_budget_scale
                ),
                impress_priority_prefetch=impress_priority_prefetch,
                impress_deferred_compute_timing=(
                    impress_deferred_compute_timing
                ),
                impress_rolling_period_prefetch=(
                    impress_rolling_period_prefetch
                ),
                impress_value_ordered_prefetch=(
                    impress_value_ordered_prefetch
                ),
                impress_value_prefetch_budget_scale=value_budget_scale,
                impress_selection_period_size=impress_selection_period_size,
                impress_known_period_prefetch=impress_known_period_prefetch,
                promixed_gqa_selection=promixed_gqa_selection,
                defer_cache_score_updates=defer_cache_score_updates,
                selector_index_bits=4 if flexgen_config.selector_index_dir is not None else 0,
            ),
            "model_compute_dtype": dtype,
            "accuracy_scoring": accuracy_scoring,
            "accuracy_scoring_protocol": (
                "mean token log-probability over each complete teacher-forced "
                "label continuation after the measured TTFT boundary"
                if accuracy_scoring == "label_continuation_loglikelihood"
                else None
            ),
            "generation_max_tokens": max_tokens,
            "pcache_storage_dtype": "float16",
            "keep_ratio": (1.0 if method == "as_lru" else plan_keep_ratio),
            "configured_plan_keep_ratio": plan_keep_ratio,
            "budget_semantics": (
                "full-kv-budget-independent"
                if method == "as_lru"
                else (
                    "h2o-logical-attention-retention-with-full-key-selector-transfer"
                    if method == "as_h2o_lru"
                    else "matched-sparse-kv-retention"
                )
            ),
            "actual_key_keep_ratio": (
                1.0 if method == "as_lru" else plan_keep_ratio
            ),
            "actual_value_keep_ratio": (
                1.0 if method == "as_lru" else plan_keep_ratio
            ),
            "actual_total_logical_kv_ratio": (
                1.0 if method == "as_lru" else plan_keep_ratio
            ),
            "logical_attention_keep_ratio": (
                1.0 if method == "as_lru" else plan_keep_ratio
            ),
            "minimum_transfer_ratio": (
                (1.0 + plan_keep_ratio) / 2.0
                if method == "as_h2o_lru"
                else None
            ),
            "selector_full_key_load_ratio": (
                1.0 if method == "as_h2o_lru" else None
            ),            "layer_budget_profile": (
                str(budget_profile.source_path) if budget_profile is not None else None
            ),
            "layer_budget_profile_sha256": (
                budget_profile.source_sha256 if budget_profile is not None else None
            ),
            "layer_budget_target_mean_ratio": (
                budget_profile.target_mean_ratio
                if budget_profile is not None
                else None
            ),
            "exact_layer_block_budget": exact_layer_block_budget,
            "layer_keep_ratios": (
                list(budget_profile.layer_ratios)
                if budget_profile is not None
                else None
            ),
            "layer_budget_calibration": (
                profile_to_json_dict(budget_profile)["calibration"]
                if budget_profile is not None
                else None
            ),
            "chunk_size": flexgen_config.chunk_size,
            "period_size": period_size,
            "subperiod_size": subperiod_size,
            "gpu_cache_mb": flexgen_config.gpu_cache_mb,
            "cpu_cache_mb": flexgen_config.cpu_cache_mb,
            "cache_type": flexgen_config.cache_type,
            "prefetch_time_budget": flexgen_config.prefetch_time_budget,
            "reused_existing_kv_chunks": flexgen_config.reuse_existing,
            "resumed_partial_kv_chunks": flexgen_config.resume_existing,
            "cache_score_updates": store.cache_score_updates,
            "cache_update_in_ttft": bool(
                online_selection
                and flexgen_config.cache_type == "CKLFU"
                and not defer_cache_score_updates
            ),
            "defer_cache_score_updates": defer_cache_score_updates,
            "cache_score_policy": (
                "LRU-recency"
                if method in {"as_lru", "as_h2o_lru"}
                else (
                    "cumulative-attention-times-frequency"
                    if method == "contigkv"
                    else "chunk-accesses-and-cumulative-important-tokens"
                )
            ),
            "online_selection": online_selection,
            "registered_store_tasks": list(registered_store_tasks),
            "selector_index_preloaded_tasks": list(
                store.selector_index_preloaded_tasks
            ),
            "selector_index_preload_ms": store.selector_index_preload_ms,
            "selector_index_preloaded_bytes": (
                store.selector_index_preloaded_bytes
            ),
            "nominal_cpu_cache_plus_selector_bytes": (
                int(flexgen_config.cpu_cache_mb * 1024 * 1024)
                + store.selector_index_preloaded_bytes
            ),
            "probe_query_heads": list(probe_query_heads),
            "selector_kv_head_ids": list(flexgen_config.selector_kv_head_ids),
            "selector_index_dir": (
                str(flexgen_config.selector_index_dir)
                if flexgen_config.selector_index_dir is not None
                else None
            ),
            "selector_compute_timing": (
                "CUDA-event score path plus host decision path"
            ),
            "selector_index_bits": 4 if flexgen_config.selector_index_dir is not None else None,
            "selector_index_group_size": (
                json.loads((flexgen_config.selector_index_dir / "manifest.json").read_text(encoding="utf-8")).get("group_size")
                if flexgen_config.selector_index_dir is not None else None
            ),
            "selector_index_manifest_sha256": (
                hashlib.sha256((flexgen_config.selector_index_dir / "manifest.json").read_bytes()).hexdigest()
                if flexgen_config.selector_index_dir is not None else None
            ),
            "similarity_alpha": similarity_alpha,
            "impress_selection_block_size": impress_selection_block_size,
            "impress_selection_period_size": impress_selection_period_size,
            "promixed_gqa_selection": promixed_gqa_selection,
            "selector_score_reduction": (
                "gqa-group-sum-per-kv-head-topk"
                if method == "as_h2o_lru"
                else (
                    "none-full-kv"
                    if method == "as_lru"
                    else (
                        "gpu-contiguous-block-sum"
                        if method == "impress" and impress_selection_block_size > 1
                        else "token-scores"
                    )
                )
            ),
            "attentionstore_semantics": (
                "full K and full V, synchronous layer loads, LRU residency"
                if method == "as_lru"
                else (
                    "full K selector load; per-GQA-KV-head H2O compact K/V attention; "
                    "selected V load only; LRU residency"
                    if method == "as_h2o_lru"
                    else None
                )
            ),
            "promixed_policy": promixed_policy,
            "impress_known_period_prefetch": impress_known_period_prefetch,
            "impress_period_prefetch_size": impress_period_prefetch_size,
            "impress_period_prefetch_budget_scale": (
                impress_period_prefetch_budget_scale
            ),
            "impress_priority_prefetch": impress_priority_prefetch,
            "impress_deferred_compute_timing": (
                impress_deferred_compute_timing
            ),
            "impress_rolling_period_prefetch": (
                impress_rolling_period_prefetch
            ),
            "impress_value_ordered_prefetch": (
                impress_value_ordered_prefetch
            ),
            "impress_value_prefetch_budget_scale": value_budget_scale,
            "impress_reorder_enabled": flexgen_config.impress_reorder_path is not None,
            "impress_reorder_manifest": (
                str(flexgen_config.impress_reorder_path)
                if flexgen_config.impress_reorder_path is not None
                else None
            ),
            "impress_reorder_sha256": (
                reorder_manifest_sha256(flexgen_config.impress_reorder_path)
                if flexgen_config.impress_reorder_path is not None
                else None
            ),
            "impress_async_inter_layer_prefetch": (
                online_selection and method == "impress" and impress_async_prefetch
            ),
            "impress_dynamic_prefetch_budget": (
                online_selection and method == "impress" and impress_async_prefetch
            ),
            "impress_predictive_period_prefetch": (
                online_selection
                and method == "impress"
                and impress_async_prefetch
                and impress_period_prefetch_size > 1
            ),
            "impress_predictive_lookahead_layers": (
                2
                if (
                    online_selection
                    and method == "impress"
                    and impress_async_prefetch
                    and impress_period_prefetch_size > 1
                )
                else None
            ),
            "online_scheduler": (
                "attentionstore-sync-layerwise-v1"
                if method in {"as_lru", "as_h2o_lru"}
                else (
                    "contig-full-period-v5+hyperinfer-async-v7+prism-ab-v1"
                    if online_selection
                    else "offline-plan-v1"
                )
            ),
            "online_plan_source": (
                "shape-only IMPRESS plan; AS baseline selection is request-time"
                if method in {"as_lru", "as_h2o_lru"}
                else (
                    "request-time selector; plan file supplies method/config only"
                    if online_selection
                    else "per-request offline plan"
                )
            ),
            "selection_reference_requests": sum(
                "selection_reference_mean_jaccard" in row for row in scored
            ),
            "warmup_passes": warmup_passes,
            "warmup_samples_per_task": warmup_samples_per_task,
            "warmup_requests": warmup_passes * len(warmup_rows),
            "response_ready_metric_valid_for_first_token": max_tokens == 1,
            "response_ready_excludes_accuracy_scoring": True,
            "evaluation_ready_includes_accuracy_scoring": True,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else None
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available() else None
            ),
        },
        "measurement": (
            "ttft_ms/logits_ready_ms ends when first-token logits are ready; "
            "first_token_ready_ms additionally includes cache-score maintenance "
            "and first-token ID selection; latency_ms ends when all requested "
            "token IDs are ready; response_ready_ms additionally includes token "
            "decoding and excludes benchmark-only accuracy scoring; "
            "evaluation_ready_ms/accuracy_scores_ready_ms additionally includes "
            "complete-label scoring. latency_ms and response_ready_ms describe "
            "one token only when max_tokens=1."
        ),
    }
    if accuracy_scoring == "label_continuation_loglikelihood":
        summary["overall"]["mean_label_scoring_time_ms"] = sum(
            float(row["label_scoring_time_ms"]) for row in scored
        ) / max(1, len(scored))
    (output / "scored_records.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in scored), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ContiguousKV or IMPRESS plans on a shared FlexGen Pcache Qwen runtime."
    )
    parser.add_argument("--model-path", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flexgen-root", required=True)
    parser.add_argument("--flexgen-kv-dir", required=True)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument(
        "--store-tasks",
        help=(
            "Ordered prefixes to register in Pcache before executing --tasks; "
            "this preserves persisted numeric prefix IDs in a single-task run."
        ),
    )
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--accuracy-scoring",
        choices=(
            "generation",
            "label_first_token_logit",
            "label_continuation_loglikelihood",
        ),
        default="generation",
        help=(
            "Score classification by generated text, constrained first-token "
            "logits, or complete teacher-forced label continuations."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--gpu-cache-mb", type=float, default=0.0)
    parser.add_argument("--cpu-cache-mb", type=float, default=0.0)
    parser.add_argument("--cache-type", choices=("LRU", "LFU", "CKLFU"), default="LRU")
    parser.add_argument("--prefetch-time-budget", type=float, default=10_000.0)
    parser.add_argument(
        "--online-selection",
        action="store_true",
        help=(
            "Compute ContiguousKV, IMPRESS, or AS+H2O selections during "
            "measured Re-Prefill."
        ),
    )
    parser.add_argument(
        "--as-baseline-mode",
        choices=("none", "as_lru", "as_h2o_lru"),
        default="none",
        help=(
            "Run the clean-room AttentionStore full-KV or AttentionStore+H2O "
            "full-K/selected-V LRU baseline using a chunk-64 shape plan."
        ),
    )
    parser.add_argument("--probe-query-heads", default="0,1,2")
    parser.add_argument("--selector-kv-head-ids", default="0,1,2")
    parser.add_argument("--selector-index-dir")
    parser.add_argument("--similarity-alpha", type=float, default=0.6)
    parser.add_argument(
        "--impress-selection-block-size",
        type=int,
        default=1,
        help=(
            "Vote over aligned importance blocks instead of individual tokens. "
            "Values above one require matching Pcache chunks and no reorder."
        ),
    )
    parser.add_argument(
        "--impress-selection-period-size",
        type=int,
        default=1,
        help=(
            "Reuse one online importance ranking across this many adjacent "
            "layers while retaining each layer's own sensitivity budget."
        ),
    )
    parser.add_argument(
        "--promixed-gqa-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one representative query head per physical GQA group and "
            "adaptively reuse selections over P1/P2/P4/P8."
        ),
    )
    parser.add_argument("--promixed-coverage-fraction", type=float, default=0.5)
    parser.add_argument("--promixed-margin-reference", type=float, default=0.05)
    parser.add_argument("--promixed-agreement-weight", type=float, default=0.75)
    parser.add_argument("--promixed-sensitivity-weight", type=float, default=0.1)
    parser.add_argument("--promixed-p1-threshold", type=float, default=0.90)
    parser.add_argument("--promixed-p2-threshold", type=float, default=0.82)
    parser.add_argument("--promixed-p4-threshold", type=float, default=0.68)
    parser.add_argument(
        "--promixed-adaptive-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use minimum one-block-per-group coverage when uncertainty selects "
            "the longest reuse horizon."
        ),
    )
    parser.add_argument(
        "--promixed-utility-max-weight",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--promixed-utility-mean-weight",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--promixed-utility-vote-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--defer-cache-score-updates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply CKLFU residency-score bookkeeping immediately after the "
            "measured TTFT boundary."
        ),
    )
    parser.add_argument(
        "--impress-known-period-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Queue every already-selected layer in a reused importance Period "
            "instead of predicting those selections one layer at a time."
        ),
    )
    parser.add_argument(
        "--impress-reorder-manifest",
        help="Importance-ranked physical token mapping used by the paper IMPRESS baseline.",
    )
    parser.add_argument(
        "--impress-async-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable HyperInfer's asynchronous next-layer extension. The paper "
            "IMPRESS baseline leaves this disabled."
        ),
    )
    parser.add_argument(
        "--layer-budget-profile",
        help=(
            "Offline per-layer FP16 retention profile. The profile mean must "
            "match the plan keep ratio."
        ),
    )
    parser.add_argument(
        "--exact-layer-block-budget",
        action="store_true",
        help=(
            "Allocate the nominal KV budget once across all model layers instead "
            "of independently rounding each layer upward."
        ),
    )
    parser.add_argument(
        "--impress-period-prefetch-size",
        type=int,
        default=1,
        help=(
            "Predict the Period leader's selected chunks for later layers as "
            "prefetch hints only; each layer still re-runs online selection."
        ),
    )
    parser.add_argument(
        "--impress-period-prefetch-budget-scale",
        type=float,
        default=1.0,
        help=(
            "Fraction of HyperInfer's dynamic time budget available to "
            "lower-priority Period predictions."
        ),
    )
    parser.add_argument(
        "--impress-priority-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prioritize current-layer misses over next-layer and predictive "
            "Period prefetch work."
        ),
    )
    parser.add_argument(
        "--impress-deferred-compute-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Harvest CUDA layer timings after natural synchronization points "
            "instead of synchronizing after every layer."
        ),
    )
    parser.add_argument(
        "--impress-rolling-period-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Predict each two-layer Period lookahead from the current layer "
            "instead of reusing the Period leader's selection."
        ),
    )
    parser.add_argument(
        "--impress-value-ordered-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve IMPRESS vote/attention order when a prefetch time budget "
            "can read only a subset of selected chunks."
        ),
    )
    parser.add_argument(
        "--impress-value-prefetch-budget-scale",
        type=float,
        default=1.0,
        help=(
            "Scale next-layer and Period prefetch budgets after value ordering; "
            "requires --impress-value-ordered-prefetch."
        ),
    )
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument("--reuse-flexgen-kv", action="store_true")
    persistence.add_argument(
        "--resume-flexgen-kv",
        action="store_true",
        help="Validate existing chunks and write only missing Pcache files.",
    )
    parser.add_argument("--period-size", type=int, default=8)
    parser.add_argument("--subperiod-size", type=int, default=4)
    parser.add_argument(
        "--expected-keep-ratio",
        type=float,
        help="Fail if the plan metadata uses a different fractional KV budget.",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=0,
        help="Run the selected requests this many times before recording metrics.",
    )
    parser.add_argument(
        "--warmup-samples-per-task",
        type=int,
        help="Cap each warm-up pass to the first N rows of every task.",
    )
    args = parser.parse_args()

    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    store_tasks = (
        [
            task.strip().lower()
            for task in args.store_tasks.split(",")
            if task.strip()
        ]
        if args.store_tasks
        else None
    )
    probe_query_heads = parse_head_ids(args.probe_query_heads)
    selector_kv_head_ids = parse_head_ids(args.selector_kv_head_ids)
    if args.online_selection:
        metadata = _load_plan_metadata(args.plan)
    else:
        _, metadata = _load_layer_plan(
            args.plan,
            str(_load_bundle_records(args.bundle_dir, tasks, 1)[0]["uid"]),
        )
    chunk_size = int(metadata.get("chunk_size", 0))
    if chunk_size <= 0:
        raise ValueError("plan metadata must declare a positive chunk_size")
    config = FlexGenPcacheConfig(
        flexgen_root=Path(args.flexgen_root),
        kv_dir=Path(args.flexgen_kv_dir),
        chunk_size=chunk_size,
        gpu_cache_mb=args.gpu_cache_mb,
        cpu_cache_mb=args.cpu_cache_mb,
        cache_type=args.cache_type,
        prefetch_time_budget=args.prefetch_time_budget,
        selector_kv_head_ids=selector_kv_head_ids,
        selector_index_dir=(
            Path(args.selector_index_dir)
            if args.selector_index_dir
            else None
        ),
        impress_reorder_path=(
            Path(args.impress_reorder_manifest)
            if args.impress_reorder_manifest
            else None
        ),
        reuse_existing=args.reuse_flexgen_kv,
        resume_existing=args.resume_flexgen_kv,
    )
    summary = run_flexgen_reprefill(
        model_path=args.model_path,
        bundle_dir=args.bundle_dir,
        store_root=args.store_root,
        plan_path=args.plan,
        tasks=tasks,
        store_tasks=store_tasks,
        samples_per_task=args.samples_per_task,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        accuracy_scoring=args.accuracy_scoring,
        device=args.device,
        dtype=args.dtype,
        flexgen_config=config,
        period_size=args.period_size,
        subperiod_size=args.subperiod_size,
        expected_keep_ratio=args.expected_keep_ratio,
        warmup_passes=args.warmup_passes,
        warmup_samples_per_task=args.warmup_samples_per_task,
        online_selection=args.online_selection,
        probe_query_heads=probe_query_heads,
        similarity_alpha=args.similarity_alpha,
        impress_async_prefetch=args.impress_async_prefetch,
        impress_selection_block_size=args.impress_selection_block_size,
        layer_budget_profile=args.layer_budget_profile,
        exact_layer_block_budget=args.exact_layer_block_budget,
        impress_period_prefetch_size=args.impress_period_prefetch_size,
        impress_period_prefetch_budget_scale=(
            args.impress_period_prefetch_budget_scale
        ),
        impress_priority_prefetch=args.impress_priority_prefetch,
        impress_deferred_compute_timing=args.impress_deferred_compute_timing,
        impress_rolling_period_prefetch=args.impress_rolling_period_prefetch,
        impress_value_ordered_prefetch=args.impress_value_ordered_prefetch,
        impress_value_prefetch_budget_scale=(
            args.impress_value_prefetch_budget_scale
        ),
        impress_selection_period_size=args.impress_selection_period_size,
        impress_known_period_prefetch=args.impress_known_period_prefetch,
        promixed_gqa_selection=args.promixed_gqa_selection,
        promixed_coverage_fraction=args.promixed_coverage_fraction,
        promixed_margin_reference=args.promixed_margin_reference,
        promixed_agreement_weight=args.promixed_agreement_weight,
        promixed_sensitivity_weight=args.promixed_sensitivity_weight,
        promixed_p1_threshold=args.promixed_p1_threshold,
        promixed_p2_threshold=args.promixed_p2_threshold,
        promixed_p4_threshold=args.promixed_p4_threshold,
        promixed_adaptive_coverage=args.promixed_adaptive_coverage,
        promixed_utility_max_weight=(
            args.promixed_utility_max_weight
        ),
        promixed_utility_mean_weight=(
            args.promixed_utility_mean_weight
        ),
        promixed_utility_vote_weight=(
            args.promixed_utility_vote_weight
        ),
        defer_cache_score_updates=args.defer_cache_score_updates,
        as_baseline_mode=args.as_baseline_mode,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
