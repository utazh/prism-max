"""Process-local Stage C integration for cost-aware 16/8/drop KV reads."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .mixed_precision_reader import MixedPrecisionPayloadReader
from .precision_run_coalescer import build_coalesced_16_8_drop_plan


_INSTALLED = False
_READERS: dict[Path, MixedPrecisionPayloadReader] = {}


def precision_mode() -> str:
    mode = os.environ.get("PRISM_GAO_PRECISION_MODE", "").strip().lower()
    if mode in {"", "off", "none"}:
        return ""
    if mode not in {"fp16", "naive", "coalesced"}:
        raise ValueError(
            "PRISM_GAO_PRECISION_MODE must be fp16, naive, or coalesced"
        )
    return mode


def _reader() -> MixedPrecisionPayloadReader:
    raw = os.environ.get(
        "PRISM_GAO_PAYLOAD_ROOT",
        "/home/panzihang/contiguous_fuxian_ssd/"
        "prism_ultra_payload_g32_v1",
    )
    root = Path(raw).resolve()
    reader = _READERS.get(root)
    if reader is None:
        reader = MixedPrecisionPayloadReader(root)
        _READERS[root] = reader
    return reader


def _unique_blocks(tokens: Sequence[int], block_size: int) -> tuple[int, ...]:
    seen: set[int] = set()
    result = []
    for token in tokens:
        block = int(token) // block_size
        if block not in seen:
            seen.add(block)
            result.append(block)
    return tuple(result)


def install_precision_hooks(loader_class: Any) -> None:
    """Patch one runtime class; original source files remain untouched."""

    global _INSTALLED
    if _INSTALLED:
        return
    original_configure: Callable[..., Any] = loader_class.configure_impress_layer
    original_resolve: Callable[..., Any] = loader_class._resolve_impress
    original_metrics: Callable[..., Any] = loader_class.metrics

    def configure(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_configure(self, *args, **kwargs)
        mode = precision_mode()
        if not mode:
            return result
        layer = int(kwargs["layer"])
        selected_tokens = kwargs["selected_tokens"]
        priority_tokens = kwargs.get("prefetch_priority_tokens")
        if priority_tokens is None:
            priority_tokens = selected_tokens
        block_size = int(self._config.chunk_size)
        selected_blocks = _unique_blocks(selected_tokens, block_size)
        priority_blocks = _unique_blocks(priority_tokens, block_size)
        if set(priority_blocks) != set(selected_blocks):
            raise RuntimeError(
                "precision priority blocks are not a permutation of selection"
            )
        total_blocks = math.ceil(self._info.prefix_tokens / block_size)
        fraction = float(os.environ.get("PRISM_GAO_FP16_FRACTION", "0.25"))
        if mode == "fp16":
            fraction = 1.0
            threshold = 1
        elif mode == "naive":
            threshold = 1
        else:
            threshold = int(
                os.environ.get("PRISM_GAO_MIN_INT8_RUN_BLOCKS", "4")
            )
        plan = build_coalesced_16_8_drop_plan(
            total_blocks=total_blocks,
            selected_blocks=selected_blocks,
            priority_blocks=priority_blocks,
            fp16_fraction=fraction,
            min_int8_run_blocks=threshold,
        )
        plans = getattr(self, "_prism_gao_precision_plans", None)
        if plans is None:
            plans = {}
            self._prism_gao_precision_plans = plans
        plans[layer] = plan
        self._prism_gao_precision_mode = mode
        return result

    def resolve(self: Any, layer: int) -> tuple[Any, Any]:
        mode = precision_mode()
        if not mode:
            return original_resolve(self, layer)
        if layer in self._loaded:
            return self._loaded[layer]
        plans = getattr(self, "_prism_gao_precision_plans", {})
        plan = plans.get(layer)
        if plan is None:
            raise RuntimeError(f"layer {layer} has no mixed precision plan")
        task = self._selector_index_task
        if not task:
            raise RuntimeError("mixed precision reader cannot identify task")
        started = time.perf_counter()
        key, value, stats = _reader().read_kv(
            task=task,
            layer=int(layer),
            tiers=plan.tiers,
            device="cuda",
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        selected = self._selected_by_layer[layer]
        if int(key.shape[0]) != len(selected) or int(value.shape[0]) != len(
            selected
        ):
            raise RuntimeError(
                f"mixed payload returned {key.shape[0]}/{value.shape[0]} "
                f"tokens for {len(selected)} selected positions"
            )
        self._prefetch_wait_ms += elapsed_ms
        self._prefetch_elapsed_ms += elapsed_ms
        self._physical_tokens += len(selected)
        self._physical_chunks.update(
            (layer, block) for block in plan.selected_blocks
        )
        totals = getattr(self, "_prism_gao_precision_totals", None)
        if totals is None:
            totals = {
                "read_bytes": 0,
                "pread_calls": 0,
                "read_ms": 0.0,
                "materialize_ms": 0.0,
                "layers": 0,
            }
            self._prism_gao_precision_totals = totals
        for field in ("read_bytes", "pread_calls", "layers"):
            totals[field] += int(stats[field] if field in stats else 1)
        totals["read_ms"] += float(stats["read_ms"])
        totals["materialize_ms"] += float(stats["materialize_ms"])
        self._update_loaded_layer_score(layer)
        self._loaded[layer] = (key, value)
        return key, value

    def metrics(self: Any) -> dict[str, Any]:
        result = original_metrics(self)
        mode = precision_mode()
        if not mode:
            return result
        plans = getattr(self, "_prism_gao_precision_plans", {})
        totals = getattr(
            self,
            "_prism_gao_precision_totals",
            {
                "read_bytes": 0,
                "pread_calls": 0,
                "read_ms": 0.0,
                "materialize_ms": 0.0,
                "layers": 0,
            },
        )
        selected_tokens = sum(
            len(tokens) for tokens in self._selected_by_layer
        )
        full_fp16_bytes = (
            selected_tokens
            * int(self._info.kv_heads)
            * int(self._info.head_dim)
            * 2
            * 2
        )
        result.update(
            {
                "prism_gao_precision_mode": mode,
                "prism_gao_fp16_fraction": float(
                    os.environ.get("PRISM_GAO_FP16_FRACTION", "0.25")
                ),
                "prism_gao_min_int8_run_blocks": (
                    1
                    if mode != "coalesced"
                    else int(
                        os.environ.get(
                            "PRISM_GAO_MIN_INT8_RUN_BLOCKS", "4"
                        )
                    )
                ),
                "prism_gao_fp16_blocks": sum(
                    len(plan.fp16_blocks) for plan in plans.values()
                ),
                "prism_gao_int8_blocks": sum(
                    len(plan.int8_blocks) for plan in plans.values()
                ),
                "prism_gao_promoted_int8_blocks": sum(
                    len(plan.promoted_int8_blocks) for plan in plans.values()
                ),
                "prism_gao_int8_runs_before": sum(
                    plan.int8_runs_before for plan in plans.values()
                ),
                "prism_gao_int8_runs_after": sum(
                    plan.int8_runs_after for plan in plans.values()
                ),
                "prism_gao_payload_layers": int(totals["layers"]),
                "prism_gao_payload_read_bytes": int(totals["read_bytes"]),
                "prism_gao_payload_pread_calls": int(totals["pread_calls"]),
                "prism_gao_payload_read_ms": float(totals["read_ms"]),
                "prism_gao_materialize_ms": float(totals["materialize_ms"]),
                "prism_gao_selected_fp16_reference_bytes": full_fp16_bytes,
                "prism_gao_payload_byte_ratio": int(totals["read_bytes"])
                / max(1, full_fp16_bytes),
            }
        )
        return result

    loader_class.configure_impress_layer = configure
    loader_class._resolve_impress = resolve
    loader_class.metrics = metrics
    _INSTALLED = True
