"""Read mixed FP16/INT8 KV payloads and materialize selected blocks on GPU."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

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


class MixedPrecisionPayloadReader:
    """Minimal direct reader for prism-ultra-payload-v1."""

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
        geometry = self.tasks[task]
        paths = self._paths(task, layer, tensor)
        h = geometry.kv_heads
        d = geometry.head_dim
        g = geometry.group_size
        b = geometry.block_size
        fp16_parts: list[np.ndarray] = []
        int8_parts: list[np.ndarray] = []
        scale_parts: list[np.ndarray] = []
        read_bytes = 0
        pread_calls = 0
        started = time.perf_counter()

        for run in encode_runs(tiers):
            if run.tier == DROP:
                continue
            start_token = run.start * b
            padded_tokens = run.length * b
            tokens = min(
                padded_tokens,
                max(0, geometry.prefix_tokens - start_token),
            )
            if tokens <= 0:
                raise RuntimeError("precision plan selected a block past the prefix")
            if run.tier == FP16:
                token_bytes = h * d * 2
                raw = self._pread(
                    paths[FP16],
                    tokens * token_bytes,
                    start_token * token_bytes,
                )
                values = np.frombuffer(raw, dtype=np.float16).reshape(tokens, h, d)
                padded = np.zeros((padded_tokens, h, d), dtype=np.float16)
                padded[:tokens] = values
                fp16_parts.append(padded.reshape(run.length, -1))
                read_bytes += len(raw)
                pread_calls += 1
            elif run.tier == INT8:
                code_token_bytes = h * d
                scale_token_bytes = h * (d // g) * 2
                raw_codes = self._pread(
                    paths["int8_codes"],
                    tokens * code_token_bytes,
                    start_token * code_token_bytes,
                )
                raw_scales = self._pread(
                    paths["int8_scales"],
                    tokens * scale_token_bytes,
                    start_token * scale_token_bytes,
                )
                codes = np.frombuffer(raw_codes, dtype=np.int8).reshape(tokens, h, d)
                scales = np.frombuffer(raw_scales, dtype=np.float16).reshape(
                    tokens, h, d // g
                )
                padded_codes = np.zeros(
                    (padded_tokens, h, d), dtype=np.int8
                )
                padded_scales = np.zeros(
                    (padded_tokens, h, d // g), dtype=np.float16
                )
                padded_codes[:tokens] = codes
                padded_scales[:tokens] = scales
                int8_parts.append(padded_codes.reshape(run.length, -1))
                scale_parts.append(padded_scales.reshape(run.length, -1))
                read_bytes += len(raw_codes) + len(raw_scales)
                pread_calls += 2
            else:
                raise ValueError(f"unsupported precision tier {run.tier!r}")

        block_elements = b * h * d
        scale_elements = block_elements // g
        fp16 = (
            np.concatenate(fp16_parts, axis=0)
            if fp16_parts
            else np.empty((0, block_elements), dtype=np.float16)
        )
        int8 = (
            np.concatenate(int8_parts, axis=0)
            if int8_parts
            else np.empty((0, block_elements), dtype=np.int8)
        )
        scales = (
            np.concatenate(scale_parts, axis=0)
            if scale_parts
            else np.empty((0, scale_elements), dtype=np.float16)
        )
        return fp16, int8, scales, {
            "read_bytes": int(read_bytes),
            "pread_calls": int(pread_calls),
            "read_ms": (time.perf_counter() - started) * 1000.0,
        }

    @staticmethod
    def _slot_metadata(
        tiers: Sequence[str],
    ) -> tuple[list[int], list[int], list[int]]:
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
        return selected_blocks, tier_codes, source_slots

    def _materialize_tensor(
        self,
        *,
        host: tuple[np.ndarray, np.ndarray, np.ndarray],
        tier_codes: Sequence[int],
        source_slots: Sequence[int],
        geometry: TaskGeometry,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        if device.type != "cuda":
            raise ValueError("mixed precision materialization requires CUDA")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fp16 = torch.from_numpy(host[0]).to(device=device, non_blocking=False)
        int8 = torch.from_numpy(host[1]).to(device=device, non_blocking=False)
        scales = torch.from_numpy(host[2]).to(device=device, non_blocking=False)
        tier = torch.tensor(tier_codes, dtype=torch.int32, device=device)
        source = torch.tensor(source_slots, dtype=torch.int32, device=device)
        output = mixed_materialize(
            fp16_blocks=fp16,
            int8_blocks=int8,
            int8_scales=scales,
            tier_by_selected_slot=tier,
            source_slot_by_selected_slot=source,
            group_size=geometry.group_size,
            output_dtype=torch.float16,
        )
        torch.cuda.synchronize(device)
        return output, (time.perf_counter() - started) * 1000.0

    def read_kv(
        self,
        *,
        task: str,
        layer: int,
        tiers: Sequence[str],
        device: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
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

        selected_blocks, tier_codes, source_slots = self._slot_metadata(tiers)
        if not selected_blocks:
            raise ValueError("precision plan selected no blocks")
        key_host = self._read_tensor(
            task=task, layer=layer, tensor="key", tiers=tiers
        )
        value_host = self._read_tensor(
            task=task, layer=layer, tensor="value", tiers=tiers
        )
        target = torch.device(device)
        key_blocks, key_materialize_ms = self._materialize_tensor(
            host=key_host[:3],
            tier_codes=tier_codes,
            source_slots=source_slots,
            geometry=geometry,
            device=target,
        )
        value_blocks, value_materialize_ms = self._materialize_tensor(
            host=value_host[:3],
            tier_codes=tier_codes,
            source_slots=source_slots,
            geometry=geometry,
            device=target,
        )

        def trim(blocks: torch.Tensor) -> torch.Tensor:
            shaped = blocks.reshape(
                len(selected_blocks),
                geometry.block_size,
                geometry.kv_heads,
                geometry.head_dim,
            )
            pieces = []
            for slot, block in enumerate(selected_blocks):
                valid = min(
                    geometry.block_size,
                    geometry.prefix_tokens - block * geometry.block_size,
                )
                pieces.append(shaped[slot, :valid])
            return torch.cat(pieces, dim=0).contiguous()

        key = trim(key_blocks)
        value = trim(value_blocks)
        key_stats = key_host[3]
        value_stats = value_host[3]
        return key, value, {
            "read_bytes": int(key_stats["read_bytes"])
            + int(value_stats["read_bytes"]),
            "pread_calls": int(key_stats["pread_calls"])
            + int(value_stats["pread_calls"]),
            "read_ms": float(key_stats["read_ms"])
            + float(value_stats["read_ms"]),
            "materialize_ms": key_materialize_ms + value_materialize_ms,
            "selected_blocks": len(selected_blocks),
            "selected_tokens": int(key.shape[0]),
        }
