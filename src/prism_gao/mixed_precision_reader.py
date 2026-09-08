"""Read mixed FP16/INT8 KV payloads with optional host-side prefetch.

The file read/NumPy packing phase is separated from GPU materialization so a
single worker thread can overlap SSD/page-cache reads for future layers with the
current layer's GPU computation.  Attention still receives the same contiguous
FP16 K/V tensors as before.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .async_mixed_pipeline import pread_into_pinned
from .mixed_kv_materialize import mixed_materialize
from .precision_run_coalescer import DROP, FP16, INT8, encode_runs


@dataclass(frozen=True)
class TaskGeometry:
    prefix_tokens: int
    layers: int
    kv_heads: int
    head_dim: int
    group_size: int
    block_size: int


@dataclass(frozen=True)
class HostTensorPayload:
    fp16: torch.Tensor
    int8: torch.Tensor
    scales: torch.Tensor
    read_bytes: int
    pread_calls: int
    read_ms: float


@dataclass(frozen=True)
class HostKVPayload:
    task: str
    layer: int
    geometry: TaskGeometry
    selected_blocks: tuple[int, ...]
    tier_codes: tuple[int, ...]
    source_slots: tuple[int, ...]
    key: HostTensorPayload
    value: HostTensorPayload


class MixedPrecisionPayloadReader:
    """Direct reader for ``prism-ultra-payload-v1``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "prism-ultra-payload-v1":
            raise ValueError(f"unsupported payload format in {manifest_path}")
        self.tasks = {
            str(task): TaskGeometry(
                prefix_tokens=int(row["prefix_tokens"]),
                layers=int(row["layers"]),
                kv_heads=int(row["kv_heads"]),
                head_dim=int(row["head_dim"]),
                group_size=int(row["group_size"]),
                block_size=int(row["block_size"]),
            )
            for task, row in payload["tasks"].items()
        }
        self._fds: dict[Path, int] = {}

    def _fd(self, path: Path) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def _pread(self, path: Path, size: int, offset: int) -> bytes:
        data = os.pread(self._fd(path), int(size), int(offset))
        if len(data) != size:
            raise RuntimeError(
                f"short read from {path}: {len(data)} bytes, expected {size}"
            )
        return data

    def _paths(self, task: str, layer: int, tensor: str) -> dict[str, Path]:
        stem = self.root / task / f"layer_{layer:02d}_{tensor}"
        return {
            FP16: Path(f"{stem}.f16"),
            "int8_codes": Path(f"{stem}.int8.codes.i8"),
            "int8_scales": Path(f"{stem}.int8.scales.f16"),
        }

    def _read_tensor(
        self,
        *,
        task: str,
        layer: int,
        tensor: str,
        tiers: Sequence[str],
    ) -> HostTensorPayload:
        """Read physical precision runs into compact pinned tensors."""
        geometry = self.tasks[task]
        paths = self._paths(task, layer, tensor)
        h, d, g, b = geometry.kv_heads, geometry.head_dim, geometry.group_size, geometry.block_size
        runs = encode_runs(tiers)
        fp16_blocks = sum(run.length for run in runs if run.tier == FP16)
        int8_blocks = sum(run.length for run in runs if run.tier == INT8)
        block_elements = b * h * d
        scale_elements = block_elements // g
        fp16 = torch.zeros((fp16_blocks, block_elements), dtype=torch.float16, pin_memory=True)
        int8 = torch.zeros((int8_blocks, block_elements), dtype=torch.int8, pin_memory=True)
        scales = torch.zeros((int8_blocks, scale_elements), dtype=torch.float16, pin_memory=True)
        fp16_cursor = 0
        int8_cursor = 0
        read_bytes = 0
        pread_calls = 0
        started = time.perf_counter()

        for run in runs:
            if run.tier == DROP:
                continue
            start_token = run.start * b
            padded_tokens = run.length * b
            tokens = min(padded_tokens, max(0, geometry.prefix_tokens - start_token))
            if tokens <= 0:
                raise RuntimeError("precision plan selected a block past the prefix")
            if run.tier == FP16:
                token_bytes = h * d * 2
                destination = fp16[fp16_cursor : fp16_cursor + run.length].view(
                    padded_tokens, h, d
                )
                raw_view = destination[:tokens].reshape(-1).view(torch.uint8)
                pread_into_pinned(
                    self._fd(paths[FP16]),
                    nbytes=tokens * token_bytes,
                    offset=start_token * token_bytes,
                    out=raw_view,
                )
                fp16_cursor += run.length
                read_bytes += tokens * token_bytes
                pread_calls += 1
            elif run.tier == INT8:
                code_token_bytes = h * d
                scale_token_bytes = h * (d // g) * 2
                code_destination = int8[int8_cursor : int8_cursor + run.length].view(
                    padded_tokens, h, d
                )
                scale_destination = scales[int8_cursor : int8_cursor + run.length].view(
                    padded_tokens, h, d // g
                )
                pread_into_pinned(
                    self._fd(paths["int8_codes"]),
                    nbytes=tokens * code_token_bytes,
                    offset=start_token * code_token_bytes,
                    out=code_destination[:tokens].reshape(-1).view(torch.uint8),
                )
                pread_into_pinned(
                    self._fd(paths["int8_scales"]),
                    nbytes=tokens * scale_token_bytes,
                    offset=start_token * scale_token_bytes,
                    out=scale_destination[:tokens].reshape(-1).view(torch.uint8),
                )
                int8_cursor += run.length
                read_bytes += tokens * (code_token_bytes + scale_token_bytes)
                pread_calls += 2
            else:
                raise ValueError("unsupported precision tier " + repr(run.tier))

        return HostTensorPayload(
            fp16=fp16,
            int8=int8,
            scales=scales,
            read_bytes=int(read_bytes),
            pread_calls=int(pread_calls),
            read_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _slot_metadata(
        tiers: Sequence[str],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        selected_blocks: list[int] = []
        tier_codes: list[int] = []
        source_slots: list[int] = []
        cursors = {FP16: 0, INT8: 0}
        for block, tier in enumerate(tiers):
            if tier == DROP:
                continue
            selected_blocks.append(block)
            tier_codes.append(0 if tier == FP16 else 1)
            source_slots.append(cursors[tier])
            cursors[tier] += 1
        return tuple(selected_blocks), tuple(tier_codes), tuple(source_slots)

    def _validate_plan(
        self, *, task: str, layer: int, tiers: Sequence[str]
    ) -> TaskGeometry:
        if task not in self.tasks:
            raise KeyError(f"payload does not contain task {task!r}")
        geometry = self.tasks[task]
        if not 0 <= int(layer) < geometry.layers:
            raise ValueError(f"layer {layer} is outside payload geometry")
        total_blocks = math.ceil(geometry.prefix_tokens / geometry.block_size)
        if len(tiers) != total_blocks:
            raise ValueError(
                f"precision plan has {len(tiers)} blocks, expected {total_blocks}"
            )
        return geometry

    def read_kv_host(
        self,
        *,
        task: str,
        layer: int,
        tiers: Sequence[str],
    ) -> HostKVPayload:
        """Read and pack K/V on CPU only; safe to run in a worker thread."""

        geometry = self._validate_plan(task=task, layer=layer, tiers=tiers)
        selected_blocks, tier_codes, source_slots = self._slot_metadata(tiers)
        if not selected_blocks:
            raise ValueError("precision plan selected no blocks")
        key = self._read_tensor(task=task, layer=layer, tensor="key", tiers=tiers)
        value = self._read_tensor(task=task, layer=layer, tensor="value", tiers=tiers)
        return HostKVPayload(
            task=str(task),
            layer=int(layer),
            geometry=geometry,
            selected_blocks=selected_blocks,
            tier_codes=tier_codes,
            source_slots=source_slots,
            key=key,
            value=value,
        )

    @staticmethod
    def _materialize_tensor_no_sync(
        *,
        host: HostTensorPayload,
        tier: torch.Tensor,
        source: torch.Tensor,
        geometry: TaskGeometry,
        device: torch.device,
    ) -> torch.Tensor:
        fp16 = host.fp16.to(device=device, non_blocking=True)
        int8 = host.int8.to(device=device, non_blocking=True)
        scales = host.scales.to(device=device, non_blocking=True)
        return mixed_materialize(
            fp16_blocks=fp16,
            int8_blocks=int8,
            int8_scales=scales,
            tier_by_selected_slot=tier,
            source_slot_by_selected_slot=source,
            group_size=geometry.group_size,
            output_dtype=torch.float16,
            validate_metadata=False,
        )

    @staticmethod
    def _trim_blocks(
        blocks: torch.Tensor,
        *,
        selected_blocks: Sequence[int],
        geometry: TaskGeometry,
    ) -> torch.Tensor:
        """Flatten blocks and handle only the possibly partial final block."""

        shaped = blocks.reshape(
            len(selected_blocks),
            geometry.block_size,
            geometry.kv_heads,
            geometry.head_dim,
        )
        final_valid = geometry.prefix_tokens % geometry.block_size
        final_block = math.ceil(geometry.prefix_tokens / geometry.block_size) - 1
        if final_valid and selected_blocks[-1] == final_block:
            if len(selected_blocks) == 1:
                return shaped[0, :final_valid].contiguous()
            return torch.cat(
                (
                    shaped[:-1].reshape(
                        -1, geometry.kv_heads, geometry.head_dim
                    ),
                    shaped[-1, :final_valid],
                ),
                dim=0,
            ).contiguous()
        return shaped.reshape(-1, geometry.kv_heads, geometry.head_dim).contiguous()

    def materialize_kv_gpu(
        self,
        payload: HostKVPayload,
        plan: Any,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Enqueue pinned H2D plus fused materialization; never synchronize."""
        target = torch.device("cuda", torch.cuda.current_device())
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record(stream)
        tier = torch.tensor(payload.tier_codes, dtype=torch.int32, device=target)
        source = torch.tensor(payload.source_slots, dtype=torch.int32, device=target)
        key_blocks = self._materialize_tensor_no_sync(
            host=payload.key, tier=tier, source=source, geometry=payload.geometry, device=target
        )
        value_blocks = self._materialize_tensor_no_sync(
            host=payload.value, tier=tier, source=source, geometry=payload.geometry, device=target
        )
        key = self._trim_blocks(
            key_blocks, selected_blocks=payload.selected_blocks, geometry=payload.geometry
        )
        value = self._trim_blocks(
            value_blocks, selected_blocks=payload.selected_blocks, geometry=payload.geometry
        )
        finished.record(stream)
        return key, value, {
            "read_bytes": payload.key.read_bytes + payload.value.read_bytes,
            "pread_calls": payload.key.pread_calls + payload.value.pread_calls,
            "read_ms": payload.key.read_ms + payload.value.read_ms,
            "materialize_ms": 0.0,
            "materialize_started": started,
            "materialize_finished": finished,
            "selected_blocks": len(payload.selected_blocks),
            "selected_tokens": int(key.shape[0]),
        }

    def materialize_kv_host(
        self, payload: HostKVPayload, *, device: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Synchronous compatibility API used only by isolated microbenchmarks."""
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("mixed precision materialization requires CUDA")
        stream = torch.cuda.current_stream(target)
        key, value, stats = self.materialize_kv_gpu(payload, None, stream)
        stats["materialize_finished"].synchronize()
        stats["materialize_ms"] = stats["materialize_started"].elapsed_time(
            stats["materialize_finished"]
        )
        return key, value, stats

    def read_kv(
        self,
        *,
        task: str,
        layer: int,
        tiers: Sequence[str],
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        """Backward-compatible synchronous convenience path."""

        payload = self.read_kv_host(task=task, layer=layer, tiers=tiers)
        return self.materialize_kv_host(payload, device=device)
