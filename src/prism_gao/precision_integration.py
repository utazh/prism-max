"""Process-local Stage C integration with one-thread host I/O prefetch.

This deliberately does not reuse the original Pcache speculative path.  It
keeps the original runner in no-prefetch mode and overlaps only the mixed
payload's CPU ``pread``/packing work.  That is the smallest safe change because
FP16/INT8 blocks live in a separate physical payload format.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .async_mixed_pipeline import (
    AsyncMixedPipeline,
    MixedCostCalibration,
    MixedPrecisionGate,
)
from .mixed_precision_reader import MixedPrecisionPayloadReader
from .precision_run_coalescer import (
    FP16,
    INT8,
    build_coalesced_16_8_drop_plan,
    encode_runs,
)


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


def host_async_enabled() -> bool:
    raw = os.environ.get("PRISM_GAO_HOST_ASYNC_PREFETCH", "false").strip().lower()
    if raw not in {"true", "false"}:
        raise ValueError("PRISM_GAO_HOST_ASYNC_PREFETCH must be true or false")
    return raw == "true"


def _reader() -> MixedPrecisionPayloadReader:
    raw = os.environ.get(
        "PRISM_GAO_PAYLOAD_ROOT",
        "/home/panzihang/contiguous_fuxian_ssd/prism_ultra_payload_g32_v1",
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


def _totals(self: Any) -> dict[str, Any]:
    totals = getattr(self, "_prism_gao_precision_totals", None)
    if totals is None:
        totals = {
            "read_bytes": 0,
            "pread_calls": 0,
            "read_ms": 0.0,
            "materialize_ms": 0.0,
            "layers": 0,
            "host_wait_ms": 0.0,
            "host_elapsed_ms": 0.0,
            "gate_candidate_layers": 0,
            "gate_active_layers": 0,
            "gate_fallback_layers": 0,
            "gate_predicted_gain_ms": 0.0,
            "materialize_events": [],
        }
        self._prism_gao_precision_totals = totals
    return totals


def _pipeline(self: Any) -> AsyncMixedPipeline:
    pipeline = getattr(self, "_prism_gao_mixed_pipeline", None)
    if pipeline is not None:
        return pipeline
    task = self._selector_index_task
    if not task:
        raise RuntimeError("mixed precision pipeline cannot identify task")
    reader = _reader()

    def read_host(layer: int, plan: Any):
        return reader.read_kv_host(task=task, layer=layer, tiers=plan.tiers)

    pipeline = AsyncMixedPipeline(
        read_host=read_host,
        materialize_gpu=reader.materialize_kv_gpu,
        device="cuda",
        max_workers=1,
    )
    self._prism_gao_mixed_pipeline = pipeline
    return pipeline


def _cost_gate() -> MixedPrecisionGate:
    return MixedPrecisionGate(
        MixedCostCalibration(
            io_gib_per_s=float(os.environ.get("PRISM_GAO_MIXED_IO_GIB_S", "3.0")),
            dequant_giga_elements_per_s=float(
                os.environ.get("PRISM_GAO_MIXED_DEQUANT_GELEM_S", "100.0")
            ),
            launch_overhead_us=float(os.environ.get("PRISM_GAO_MIXED_LAUNCH_US", "8.0")),
            exposed_io_fraction=float(
                os.environ.get("PRISM_GAO_MIXED_EXPOSED_IO_FRACTION", "0.25")
            ),
            minimum_gain_ms=float(os.environ.get("PRISM_GAO_MIXED_MINIMUM_GAIN_MS", "0.05")),
        )
    )


def _plan_cost_inputs(plan: Any, geometry: Any) -> dict[str, int]:
    fp16_bytes = 0
    mixed_bytes = 0
    int8_elements = 0
    for block, tier in enumerate(plan.tiers):
        if tier not in {FP16, INT8}:
            continue
        start = block * int(geometry.block_size)
        tokens = min(int(geometry.block_size), int(geometry.prefix_tokens) - start)
        elements = tokens * int(geometry.kv_heads) * int(geometry.head_dim) * 2
        fp16_bytes += elements * 2
        if tier == FP16:
            mixed_bytes += elements * 2
        else:
            mixed_bytes += elements + elements * 2 // int(geometry.group_size)
            int8_elements += elements
    return {
        "fp16_bytes": fp16_bytes,
        "mixed_bytes": mixed_bytes,
        "int8_elements": int8_elements,
        "int8_runs": sum(run.tier == INT8 for run in encode_runs(plan.tiers)),
    }


def _promote_all_int8(plan: Any) -> Any:
    tiers = tuple(FP16 if tier == INT8 else tier for tier in plan.tiers)
    promoted = tuple(sorted(set(plan.promoted_int8_blocks) | set(plan.int8_blocks)))
    return replace(
        plan, tiers=tiers, promoted_int8_blocks=promoted, int8_runs_after=0
    )


def install_precision_hooks(loader_class: Any) -> None:
    """Patch one runtime class; original source files remain untouched."""

    global _INSTALLED
    if _INSTALLED:
        return
    original_configure: Callable[..., Any] = loader_class.configure_impress_layer
    original_resolve: Callable[..., Any] = loader_class._resolve_impress
    original_metrics: Callable[..., Any] = loader_class.metrics
    original_close: Callable[..., Any] = loader_class.close
    original_record_compute: Callable[..., Any] = loader_class.record_impress_layer_compute

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
                os.environ.get("PRISM_GAO_MIN_INT8_RUN_BLOCKS", "2")
            )
        plan = build_coalesced_16_8_drop_plan(
            total_blocks=total_blocks,
            selected_blocks=selected_blocks,
            priority_blocks=priority_blocks,
            fp16_fraction=fraction,
            min_int8_run_blocks=threshold,
        )
        totals = _totals(self)
        if mode == "coalesced" and plan.int8_blocks:
            geometry = _reader().tasks[self._selector_index_task]
            cost_inputs = _plan_cost_inputs(plan, geometry)
            gate = _cost_gate()
            predicted_gain = gate.predicted_gain_ms(**cost_inputs)
            totals["gate_candidate_layers"] += 1
            totals["gate_predicted_gain_ms"] += float(predicted_gain)
            if gate.keep_mixed(**cost_inputs):
                totals["gate_active_layers"] += 1
            else:
                totals["gate_fallback_layers"] += 1
                plan = _promote_all_int8(plan)

        plans = getattr(self, "_prism_gao_precision_plans", None)
        if plans is None:
            plans = {}
            self._prism_gao_precision_plans = plans
        plans[layer] = plan
        self._prism_gao_precision_mode = mode

        if host_async_enabled():
            pipeline = _pipeline(self)
            pipeline.submit(layer, layer=layer, plan=plan)
            pipeline.poll()
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

        wait_started = time.perf_counter()
        if host_async_enabled():
            key, value, stats = _pipeline(self).resolve(layer)
        else:
            host = _reader().read_kv_host(
                task=task, layer=int(layer), tiers=plan.tiers
            )
            key, value, stats = _reader().materialize_kv_host(host, device="cuda")
        host_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        host_elapsed_ms = host_wait_ms
        blocked_ms = host_wait_ms
        _totals(self)["materialize_events"].append(
            (stats["materialize_started"], stats["materialize_finished"])
        )

        selected = self._selected_by_layer[layer]
        if int(key.shape[0]) != len(selected) or int(value.shape[0]) != len(selected):
            raise RuntimeError(
                f"mixed payload returned {key.shape[0]}/{value.shape[0]} "
                f"tokens for {len(selected)} selected positions"
            )
        self._prefetch_wait_ms += blocked_ms
        self._prefetch_elapsed_ms += host_elapsed_ms
        self._physical_tokens += len(selected)
        self._physical_chunks.update(
            (layer, block) for block in plan.selected_blocks
        )
        totals = _totals(self)
        totals["read_bytes"] += int(stats["read_bytes"])
        totals["pread_calls"] += int(stats["pread_calls"])
        totals["read_ms"] += float(stats["read_ms"])
        totals["materialize_ms"] += float(stats["materialize_ms"])
        totals["layers"] += 1
        totals["host_wait_ms"] += host_wait_ms
        totals["host_elapsed_ms"] += host_elapsed_ms
        self._update_loaded_layer_score(layer)
        self._loaded[layer] = (key, value)
        return key, value

    def metrics(self: Any) -> dict[str, Any]:
        result = original_metrics(self)
        index = getattr(self, "_selector_index", None)
        resident = None
        if index is not None:
            resident = getattr(index, "_prism_gao_resident_caches", {}).get(
                self._selector_index_task
            )
        result.update({
            "prism_gao_selector_resident_enabled": int(resident is not None),
            "prism_gao_selector_resident_mode": (resident.mode if resident is not None else "off"),
            "prism_gao_selector_resident_bytes": (int(resident.total_bytes) if resident is not None else 0),
        })
        mode = precision_mode()
        if not mode:
            return result
        plans = getattr(self, "_prism_gao_precision_plans", {})
        totals = _totals(self)
        ready_events = 0
        materialize_ms = 0.0
        for started, finished in totals["materialize_events"]:
            if finished.query():
                ready_events += 1
                materialize_ms += float(started.elapsed_time(finished))
        totals["materialize_ms"] = materialize_ms
        selected_tokens = sum(len(tokens) for tokens in self._selected_by_layer)
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
                "prism_gao_host_async_prefetch": int(host_async_enabled()),
                "prism_gao_fp16_fraction": float(
                    os.environ.get("PRISM_GAO_FP16_FRACTION", "0.25")
                ),
                "prism_gao_min_int8_run_blocks": (
                    1
                    if mode != "coalesced"
                    else int(
                        os.environ.get("PRISM_GAO_MIN_INT8_RUN_BLOCKS", "2")
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
                "prism_gao_host_prefetch_wait_ms": float(totals["host_wait_ms"]),
                "prism_gao_host_prefetch_elapsed_ms": float(
                    totals["host_elapsed_ms"]
                ),
                "prism_gao_selected_fp16_reference_bytes": full_fp16_bytes,
                "prism_gao_payload_byte_ratio": int(totals["read_bytes"])
                / max(1, full_fp16_bytes),
                "prism_gao_materialize_event_ready": ready_events,
                "prism_gao_materialize_event_total": len(totals["materialize_events"]),
                "prism_gao_gate_candidate_layers": int(totals["gate_candidate_layers"]),
                "prism_gao_gate_active_layers": int(totals["gate_active_layers"]),
                "prism_gao_gate_fallback_layers": int(totals["gate_fallback_layers"]),
                "prism_gao_gate_activation_rate": int(totals["gate_active_layers"])
                / max(1, int(totals["gate_candidate_layers"])),
                "prism_gao_gate_predicted_gain_ms": float(totals["gate_predicted_gain_ms"]),
            }
        )
        return result

    def record_compute(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_record_compute(self, *args, **kwargs)
        pipeline = getattr(self, "_prism_gao_mixed_pipeline", None)
        if pipeline is not None:
            pipeline.poll()
        return result

    def close(self: Any) -> None:
        pipeline = getattr(self, "_prism_gao_mixed_pipeline", None)
        if pipeline is not None:
            pipeline.close()
        return original_close(self)

    loader_class.configure_impress_layer = configure
    loader_class._resolve_impress = resolve
    loader_class.metrics = metrics
    loader_class.record_impress_layer_compute = record_compute
    loader_class.close = close
    _INSTALLED = True
