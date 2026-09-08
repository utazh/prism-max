"""Opt-in packed-INT4 selector integration for the existing PRISM runner.

The original contiguous_fuxian package is never edited. install() patches only
the current Python process so the standard experiment CLI can be used unchanged
through python -m prism_gao.run_fused_selector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from .int4_selector import dequantize_int4_keys, int4_qk_logits
from .precision_integration import install_precision_hooks
from .period_control import wrap_selector
from .selector_resident_cache import (
    PackedSelectorLayer as ResidentPackedLayer,
    SelectorResidentCache,
    TensorWorkspaceCache,
)


@dataclass(frozen=True)
class PackedSelectorKeys:
    """Compressed selector payload transferred to the scoring stream."""

    codes: Any
    scales: Any
    head_dim: int
    group_size: int

    def numel(self) -> int:
        return int(self.codes.shape[0]) * int(self.codes.shape[1]) * self.head_dim

    def element_size(self) -> int:
        # Preserve the original runtime's dequantized-byte audit semantics.
        return 2


_INSTALLED = False
_ORIGINAL_SCORE = None
_ORIGINAL_LOAD_LAYER = None
_ORIGINAL_PRELOAD_TASK = None
_ORIGINAL_LOAD_SELECTOR_SYNC = None
_SELECTOR_WORKSPACE = TensorWorkspaceCache()


def _resident_enabled() -> bool:
    raw = os.environ.get("PRISM_GAO_SELECTOR_RESIDENT", "false").strip().lower()
    if raw not in {"true", "false"}:
        raise ValueError("PRISM_GAO_SELECTOR_RESIDENT must be true or false")
    return raw == "true"


def _preload_task_with_resident(index: Any, task: str) -> int:
    """Extend the existing pinned-host preload with an opt-in GPU copy."""
    if _ORIGINAL_PRELOAD_TASK is None:
        raise RuntimeError("selector preload hook was not installed")
    loaded = int(_ORIGINAL_PRELOAD_TASK(index, task))
    if not _resident_enabled():
        return loaded
    caches = getattr(index, "_prism_gao_resident_caches", None)
    if caches is None:
        caches = {}
        index._prism_gao_resident_caches = caches
    if task in caches:
        return loaded
    entry = index.tasks[task]

    def load_cpu(layer: int) -> ResidentPackedLayer:
        codes, scales, _ = index._cpu_cache[(task, int(layer))]
        return ResidentPackedLayer(
            codes=codes,
            scales=scales,
            meta={"head_dim": int(entry.head_dim), "group_size": int(entry.group_size)},
        )

    budget_mb = int(os.environ.get("PRISM_GAO_SELECTOR_GPU_CACHE_MB", "1024"))
    if budget_mb < 0:
        raise ValueError("PRISM_GAO_SELECTOR_GPU_CACHE_MB must be non-negative")
    torch = __import__("torch")
    cache = SelectorResidentCache(
        load_cpu,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_gpu_bytes=budget_mb * 1024 * 1024,
        pin_fallback=True,
    )
    cache.preload(range(int(entry.layers)))
    caches[task] = cache
    return loaded


def _load_layer_packed(
    index: Any, task: str, layer: int, *, device: Any
) -> tuple[PackedSelectorKeys, int, str]:
    """Transfer codes/scales without materializing an FP16 key tensor."""

    import numpy as np
    import torch

    entry = index.tasks[task]
    resident = getattr(index, "_prism_gao_resident_caches", {}).get(task)
    if resident is not None:
        layer_payload = resident.get(int(layer))
        return (
            PackedSelectorKeys(
                codes=layer_payload.codes,
                scales=layer_payload.scales,
                head_dim=int(entry.head_dim),
                group_size=int(entry.group_size),
            ),
            int(layer_payload.nbytes),
            "gpu" if resident.mode == "gpu" else "cpu",
        )
    cached = index._cpu_cache.get((task, layer))
    if cached is None:
        codes_path, scales_path = index.layer_paths(task, layer)
        codes = np.fromfile(codes_path, dtype=np.uint8).reshape(
            entry.prefix_tokens, entry.kv_heads, entry.head_dim // 2
        )
        scales = np.fromfile(scales_path, dtype=np.float16).reshape(
            entry.prefix_tokens,
            entry.kv_heads,
            entry.head_dim // entry.group_size,
        )
        codes_tensor = torch.from_numpy(codes)
        scales_tensor = torch.from_numpy(scales)
        compressed_bytes = codes_path.stat().st_size + scales_path.stat().st_size
        source = "disk"
    else:
        codes_tensor, scales_tensor, compressed_bytes = cached
        source = "cpu"

    target = torch.device(device)
    if target.type == "cuda":
        if not codes_tensor.is_pinned():
            codes_tensor = codes_tensor.pin_memory()
            scales_tensor = scales_tensor.pin_memory()
        codes_tensor = codes_tensor.to(target, non_blocking=True)
        scales_tensor = scales_tensor.to(target, non_blocking=True)
    else:
        codes_tensor = codes_tensor.to(target)
        scales_tensor = scales_tensor.to(target)

    return (
        PackedSelectorKeys(
            codes=codes_tensor.contiguous(),
            scales=scales_tensor.contiguous(),
            head_dim=int(entry.head_dim),
            group_size=int(entry.group_size),
        ),
        int(compressed_bytes),
        source,
    )


def _selector_slots(
    *,
    query_heads: Sequence[int] | None,
    selector_kv_head_ids: Sequence[int] | None,
    attention: Any,
    code_heads: int,
) -> tuple[list[int], list[int]]:
    """Return query heads and matching packed physical-KV-head slots."""

    from contiguous_fuxian.flexgen_qwen_reprefill import selector_key_slots

    num_query_heads = int(attention.config.num_attention_heads)
    num_kv_heads = int(attention.config.num_key_value_heads)
    if query_heads is None:
        normalized_query_heads = list(range(num_query_heads))
    else:
        normalized_query_heads = [int(head) for head in query_heads]

    if selector_kv_head_ids is None:
        if code_heads == num_kv_heads:
            loaded_ids = list(range(num_kv_heads))
        elif code_heads == len(normalized_query_heads):
            return normalized_query_heads, list(range(code_heads))
        else:
            raise ValueError("packed selector heads cannot be matched to query heads")
    else:
        loaded_ids = [int(head) for head in selector_kv_head_ids]
        if len(loaded_ids) != code_heads:
            raise ValueError(
                "packed selector payload does not match physical head IDs"
            )

    slots = selector_key_slots(
        normalized_query_heads,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        selector_kv_head_ids=loaded_ids,
    )
    return normalized_query_heads, list(slots)


def fused_qwen_online_prefix_head_scores(
    *,
    decoder_layer: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    selector_keys: Any,
    score_block_size: int = 1,
    query_heads: Sequence[int] | None,
    selector_kv_head_ids: Sequence[int] | None = None,
) -> Any:
    """Drop-in scorer that dispatches packed selector payloads to Triton."""

    if not isinstance(selector_keys, PackedSelectorKeys):
        if _ORIGINAL_SCORE is None:
            raise RuntimeError("fused selector integration was not installed")
        return _ORIGINAL_SCORE(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys,
            score_block_size=score_block_size,
            query_heads=query_heads,
            selector_kv_head_ids=selector_kv_head_ids,
        )
    if score_block_size <= 0:
        raise ValueError("online selector score block size must be positive")

    import torch
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    attention = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query_states = attention.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    suffix_keys = attention.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, suffix_keys = apply_rotary_pos_emb(
        query_states, suffix_keys, cos, sin
    )
    suffix_keys = repeat_kv(suffix_keys, attention.num_key_value_groups)

    normalized_query_heads, slots = _selector_slots(
        query_heads=query_heads,
        selector_kv_head_ids=selector_kv_head_ids,
        attention=attention,
        code_heads=int(selector_keys.codes.shape[1]),
    )
    device = query_states.device
    head_index = torch.tensor(
        normalized_query_heads, dtype=torch.long, device=device
    )
    slot_index = torch.tensor(slots, dtype=torch.long, device=device)
    if bool(torch.any(head_index < 0)) or bool(
        torch.any(head_index >= query_states.shape[1])
    ):
        raise ValueError("probe query head is outside the Qwen attention head range")

    query_states = query_states.index_select(1, head_index)
    suffix_keys = suffix_keys.index_select(1, head_index)
    if slots == list(range(int(selector_keys.codes.shape[1]))):
        codes = selector_keys.codes
        scales = selector_keys.scales
    else:
        codes = selector_keys.codes.index_select(1, slot_index).contiguous()
        scales = selector_keys.scales.index_select(1, slot_index).contiguous()
    if int(query_states.shape[-1]) != selector_keys.head_dim:
        raise ValueError("packed selector head dimension does not match Qwen")

    mode = os.environ.get("PRISM_GAO_SELECTOR_MODE", "exact").strip().lower()
    if mode == "direct":
        prefix_logits = int4_qk_logits(
            query_states[0].contiguous(),
            codes,
            scales,
            group_size=selector_keys.group_size,
            scaling=float(attention.scaling),
        ).unsqueeze(0)
    elif mode == "exact":
        canonical_shape = (
            int(codes.shape[0]),
            int(codes.shape[1]),
            int(selector_keys.head_dim),
        )
        fp16_workspace = _SELECTOR_WORKSPACE.get(
            canonical_shape, dtype=torch.float16, device=device
        )
        prefix_keys_fp16 = dequantize_int4_keys(
            codes,
            scales,
            group_size=selector_keys.group_size,
            out=fp16_workspace,
        )
        if query_states.dtype == torch.float16:
            prefix_keys = prefix_keys_fp16
        else:
            prefix_keys = _SELECTOR_WORKSPACE.get(
                canonical_shape, dtype=query_states.dtype, device=device
            )
            prefix_keys.copy_(prefix_keys_fp16)
        prefix_keys = prefix_keys.permute(1, 0, 2).unsqueeze(0).contiguous()
        prefix_logits = torch.matmul(
            query_states, prefix_keys.transpose(2, 3)
        ) * float(attention.scaling)
    else:
        raise ValueError("PRISM_GAO_SELECTOR_MODE must be exact or direct")
    suffix_logits = torch.matmul(
        query_states, suffix_keys.transpose(2, 3)
    ) * float(attention.scaling)
    logits = torch.cat((prefix_logits, suffix_logits), dim=-1)

    prefix_tokens = int(codes.shape[0])
    query_tokens = int(query_states.shape[2])
    suffix_future = torch.triu(
        torch.ones(
            (query_tokens, query_tokens), dtype=torch.bool, device=logits.device
        ),
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
        raise RuntimeError("fused Qwen selector produced non-finite prefix attention")
    return scores.detach().cpu()


def install() -> None:
    """Install process-local hooks; safe to call more than once."""

    global _INSTALLED, _ORIGINAL_LOAD_LAYER, _ORIGINAL_PRELOAD_TASK, _ORIGINAL_LOAD_SELECTOR_SYNC, _ORIGINAL_SCORE
    if _INSTALLED:
        return

    from contiguous_fuxian import flexgen_qwen_reprefill
    from contiguous_fuxian.flexgen_pcache import FlexGenLayerLoader
    from contiguous_fuxian.quantized_key_index import QuantizedKeyIndex

    _ORIGINAL_LOAD_LAYER = QuantizedKeyIndex.load_layer
    _ORIGINAL_PRELOAD_TASK = QuantizedKeyIndex.preload_task
    _ORIGINAL_LOAD_SELECTOR_SYNC = FlexGenLayerLoader._load_selector_keys_sync
    _ORIGINAL_SCORE = flexgen_qwen_reprefill.qwen_online_prefix_head_scores
    flexgen_qwen_reprefill.select_promixed_gqa_blocks = wrap_selector(
        flexgen_qwen_reprefill.select_promixed_gqa_blocks
    )

    def patched_load_layer(
        index: Any, task: str, layer: int, *, device: Any
    ) -> tuple[Any, int, str]:
        return _load_layer_packed(index, task, layer, device=device)

    def patched_preload_task(index: Any, task: str) -> int:
        return _preload_task_with_resident(index, task)

    def patched_load_selector_sync(loader: Any, layer: int) -> Any:
        index = getattr(loader, "_selector_index", None)
        task = getattr(loader, "_selector_index_task", None)
        resident = (
            getattr(index, "_prism_gao_resident_caches", {}).get(task)
            if index is not None and task is not None
            else None
        )
        if not _resident_enabled() or resident is None or resident.mode != "gpu":
            return _ORIGINAL_LOAD_SELECTOR_SYNC(loader, layer)
        import time as _time
        import torch

        started = _time.perf_counter()
        target = torch.device("cuda", torch.cuda.current_device())
        keys, key_bytes, source = index.load_layer(task, layer, device=target)
        loader._selector_load_ms += (_time.perf_counter() - started) * 1000
        loader._selector_key_bytes += int(key_bytes)
        loader._selector_dequantized_key_bytes += (
            int(keys.numel()) * int(keys.element_size())
        )
        loader._selector_source_bytes[source] += int(key_bytes)
        loader._selector_calls += 1
        return keys

    QuantizedKeyIndex.preload_task = patched_preload_task
    QuantizedKeyIndex.load_layer = patched_load_layer
    FlexGenLayerLoader._load_selector_keys_sync = patched_load_selector_sync
    flexgen_qwen_reprefill.qwen_online_prefix_head_scores = (
        fused_qwen_online_prefix_head_scores
    )
    install_precision_hooks(FlexGenLayerLoader)
    _INSTALLED = True
