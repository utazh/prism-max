"""FlexGen Pcache bridge for physically sparse Qwen KV retrieval.

The paper builds both ContiguousKV and its baselines on a modified FlexGen
runtime.  This module keeps that storage and prefetch boundary separate from
the Qwen execution adapter: it stores full prefix KV tensors through the
existing ``Pcache`` implementation, asynchronously loads physical chunks, and
then gathers only the tokens selected by a method-specific plan.

No dropped token is represented by a zero vector.  The caller receives a
variable-length tensor containing only the selected token positions.
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .impress_reorder import load_task_reorder, reorder_manifest_sha256
from .quantized_key_index import QuantizedKeyIndex
from .sparse_qwen_reprefill import (
    PrefixStoreInfo,
    _read_layer_chunks,
    _tensor_path,
    read_store_info,
    selected_chunk_indices,
)


@dataclass(frozen=True)
class FlexGenPcacheConfig:
    """Parameters shared by both ContiguousKV and IMPRESS cache runs."""

    flexgen_root: Path
    kv_dir: Path
    chunk_size: int
    gpu_cache_mb: float = 0.0
    cpu_cache_mb: float = 0.0
    cache_type: str = "LRU"
    prefetch_time_budget: float = 10_000.0
    selector_kv_head_ids: tuple[int, ...] = (0, 1, 2)
    selector_index_dir: Path | None = None
    impress_reorder_path: Path | None = None
    reuse_existing: bool = False
    resume_existing: bool = False

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.gpu_cache_mb < 0 or self.cpu_cache_mb < 0:
            raise ValueError("cache capacities must be non-negative")
        if self.prefetch_time_budget <= 0:
            raise ValueError("prefetch_time_budget must be positive")
        if self.cache_type not in {"LRU", "LFU", "CKLFU"}:
            raise ValueError("cache_type must be one of LRU, LFU, or CKLFU")
        if not self.selector_kv_head_ids or any(head < 0 for head in self.selector_kv_head_ids):
            raise ValueError("selector_kv_head_ids must contain non-negative head indices")
        if self.selector_index_dir is not None and not (
            self.selector_index_dir / "manifest.json"
        ).is_file():
            raise FileNotFoundError(
                f"selector index manifest was not found: {self.selector_index_dir}"
            )
        if self.selector_index_dir is not None and self.impress_reorder_path is not None:
            raise ValueError("a selector index cannot be combined with physical token reorder")
        if self.impress_reorder_path is not None and not self.impress_reorder_path.is_file():
            raise FileNotFoundError(
                f"IMPRESS reorder manifest was not found: {self.impress_reorder_path}"
            )
        if self.reuse_existing and self.resume_existing:
            raise ValueError("reuse_existing and resume_existing are mutually exclusive")
        if not (self.flexgen_root / "my_pcache_fast.py").is_file():
            raise FileNotFoundError(
                "HyperInfer fast Pcache source was not found at "
                f"{self.flexgen_root / 'my_pcache_fast.py'}"
            )
        if self.kv_dir.exists() and not self.kv_dir.is_dir():
            raise ValueError(f"{self.kv_dir} must be a directory")
        has_files = self.kv_dir.exists() and any(self.kv_dir.iterdir())
        if (self.reuse_existing or self.resume_existing) and not has_files:
            raise ValueError(f"{self.kv_dir} has no existing HyperInfer KV chunks to reuse")
        if not (self.reuse_existing or self.resume_existing) and has_files:
            raise ValueError(
                f"{self.kv_dir} must be new or empty because Pcache clears its KV directory on startup"
            )


def retained_token_ids(
    tiers: Sequence[str],
    *,
    chunk_size: int,
    prefix_tokens: int,
) -> list[int]:
    """Expand retained physical chunk IDs to sorted, valid token positions."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")

    tokens: list[int] = []
    for chunk_index in selected_chunk_indices(tiers):
        start = chunk_index * chunk_size
        end = min(prefix_tokens, start + chunk_size)
        if start < end:
            tokens.extend(range(start, end))
    return tokens


def physical_token_ids_for_positions(
    positions: Sequence[int],
    *,
    chunk_size: int,
    prefix_tokens: int,
) -> list[int]:
    """Expand arbitrary useful positions to the complete physical chunks read."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    chunks = sorted(
        {
            int(position) // chunk_size
            for position in positions
            if 0 <= int(position) < prefix_tokens
        }
    )
    physical = []
    for chunk in chunks:
        start = chunk * chunk_size
        physical.extend(range(start, min(prefix_tokens, start + chunk_size)))
    return physical


def expected_chunk_count(*, prefix_tokens: int, chunk_size: int) -> int:
    """Return the number of physical chunks needed to cover a prefix."""

    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return math.ceil(prefix_tokens / chunk_size)


def selected_tokens_for_plan(
    layer_plan: Sequence[Sequence[str]],
    *,
    chunk_size: int,
    prefix_tokens: int,
    layer_token_selections: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """Resolve useful attention tokens separately from physical chunk plans."""

    if layer_token_selections is None:
        return [
            retained_token_ids(
                tiers,
                chunk_size=chunk_size,
                prefix_tokens=prefix_tokens,
            )
            for tiers in layer_plan
        ]
    if len(layer_token_selections) != len(layer_plan):
        raise ValueError("layer token selections must match the number of plan layers")

    normalized: list[list[int]] = []
    for layer, (tiers, token_ids) in enumerate(zip(layer_plan, layer_token_selections)):
        selected = sorted(set(int(token_id) for token_id in token_ids))
        if not selected:
            raise ValueError(f"layer {layer} has no useful attention tokens")
        for token_id in selected:
            if not 0 <= token_id < prefix_tokens:
                raise ValueError(f"layer {layer} token {token_id} is outside the prefix")
            if str(tiers[token_id // chunk_size]).lower() == "drop":
                raise ValueError(
                    f"layer {layer} token {token_id} belongs to a dropped physical chunk"
                )
        normalized.append(selected)
    return normalized


def gather_prefetched_tokens(
    physical_key: Any,
    physical_value: Any,
    physical_token_ids: Any,
    selected_token_ids: Sequence[int],
) -> tuple[Any, Any]:
    """Gather selected token positions from complete physical chunks.

    ``Pcache.prefetch_async`` returns every token in each touched chunk.  This
    helper maps a method's sparse token plan back into that physical response.
    It intentionally validates the mapping so an incomplete prefetch cannot be
    silently treated as a sparse cache hit.
    """

    import torch

    if physical_key.shape[0] != physical_value.shape[0]:
        raise ValueError("physical key/value token dimensions differ")
    if int(physical_token_ids.numel()) != int(physical_key.shape[0]):
        raise ValueError("physical token IDs do not describe the returned KV tensors")

    selected = torch.as_tensor(selected_token_ids, dtype=torch.long, device=physical_token_ids.device)
    if selected.numel() == 0:
        return physical_key[:0], physical_value[:0]
    lookup = torch.searchsorted(physical_token_ids, selected)
    if bool(torch.any(lookup >= physical_token_ids.numel())):
        raise RuntimeError("FlexGen prefetch omitted selected token positions")
    if not bool(torch.equal(physical_token_ids.index_select(0, lookup), selected)):
        raise RuntimeError("FlexGen prefetch token positions are not sorted or complete")
    key_lookup = lookup.to(physical_key.device)
    value_lookup = lookup.to(physical_value.device)
    return physical_key.index_select(0, key_lookup), physical_value.index_select(0, value_lookup)


def prefetch_source_tensor_tokens(
    pcache: Any,
    *,
    prefix_id: int,
    layer: int,
    token_ids: Sequence[int],
    chunk_size: int,
    prefix_tokens: int,
) -> dict[str, int]:
    """Count key/value tensor tokens by the read-only prefetch source tier.

    HyperInfer's fast prefetch copies into temporary buffers without changing
    ``device_map``, so the map remains the authoritative source after a future
    completes and before this loader applies its cache-score update.
    """

    counts = {"gpu": 0, "cpu": 0, "disk": 0}
    cache_layer = pcache.cache[prefix_id].layers[layer]
    chunk_count = int(cache_layer.chunk_num)
    for chunk in sorted(set(int(token_id) // chunk_size for token_id in token_ids)):
        if not 0 <= chunk < chunk_count:
            raise ValueError(f"physical chunk {chunk} is outside layer {layer}")
        chunk_start = chunk * chunk_size
        tokens = min(chunk_size, max(0, prefix_tokens - chunk_start))
        for map_index in (chunk, chunk + chunk_count):
            device = str(cache_layer.device_map[map_index])
            if device.startswith("cuda"):
                counts["gpu"] += tokens
            elif device == "cpu":
                counts["cpu"] += tokens
            elif device == "disk":
                counts["disk"] += tokens
            else:
                raise RuntimeError(f"unsupported HyperInfer cache device {device!r}")
    return counts


def _load_pcache_module(flexgen_root: Path):
    """Load HyperInfer's official fast Pcache implementation without copying it."""

    root = flexgen_root.resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("my_pcache_fast")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"my_pcache_fast resolved to {module_path}, outside requested FlexGen root {root}"
        ) from exc
    return module


class _ExistingMemmapSink:
    """Validate an existing memmap target while discarding duplicate writes."""

    def __init__(self, path: str | Path, *, dtype: Any, shape: Sequence[int], offset: int = 0) -> None:
        import numpy as np

        self.path = Path(path)
        expected_bytes = int(offset) + math.prod(int(size) for size in shape) * np.dtype(dtype).itemsize
        if not self.path.is_file():
            raise FileNotFoundError(f"missing persisted HyperInfer chunk {self.path}")
        actual_bytes = self.path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"persisted HyperInfer chunk size mismatch for {self.path}: "
                f"{actual_bytes} != {expected_bytes}"
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        # The existing file has already been byte-size validated. Pcache is
        # reconstructing its in-memory object graph, not updating the payload.
        return None

    def flush(self) -> None:
        return None


@contextmanager
def _reuse_existing_memmaps(pcache_module: Any, *, allow_missing: bool = False):
    """Validate existing chunks and optionally fill missing targets for resume."""

    original_memmap = pcache_module.np.memmap

    def existing_memmap(
        filename,
        dtype="uint8",
        mode="r+",
        offset=0,
        shape=None,
        order="C",
    ):
        if mode == "w+":
            if shape is None:
                raise ValueError("persisted HyperInfer memmaps require an explicit shape")
            if allow_missing and not Path(filename).is_file():
                return original_memmap(
                    filename,
                    dtype=dtype,
                    mode=mode,
                    offset=offset,
                    shape=shape,
                    order=order,
                )
            return _ExistingMemmapSink(filename, dtype=dtype, shape=shape, offset=offset)
        return original_memmap(
            filename,
            dtype=dtype,
            mode=mode,
            offset=offset,
            shape=shape,
            order=order,
        )

    pcache_module.np.memmap = existing_memmap
    try:
        yield
    finally:
        pcache_module.np.memmap = original_memmap


def _load_task_tensor(
    *,
    store_root: str | Path,
    task: str,
    info: PrefixStoreInfo,
    chunk_size: int,
    physical_to_logical: Sequence[Sequence[int]] | None = None,
):
    """Load one prepared prefix into Pcache's [layer, token, head, dim] layout."""

    import torch

    chunks = list(range(expected_chunk_count(prefix_tokens=info.prefix_tokens, chunk_size=chunk_size)))
    keys = []
    values = []
    for layer in range(info.layers):
        key = _read_layer_chunks(
            _tensor_path(store_root, task, layer, "key"),
            info,
            chunk_size,
            chunks,
        )
        value = _read_layer_chunks(
            _tensor_path(store_root, task, layer, "value"),
            info,
            chunk_size,
            chunks,
        )
        key = key.squeeze(0).permute(1, 0, 2).contiguous()
        value = value.squeeze(0).permute(1, 0, 2).contiguous()
        if physical_to_logical is not None:
            order = torch.tensor(physical_to_logical[layer], dtype=torch.long)
            key = key.index_select(0, order)
            value = value.index_select(0, order)
        # Pcache writes NumPy-backed chunks, so retain FP16 storage even if the
        # model's compute dtype is BF16. The Qwen adapter casts on use.
        keys.append(key.to(torch.float16))
        values.append(value.to(torch.float16))
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


class FlexGenPcacheStore:
    """Own a fresh Pcache instance and its task-to-prefix mapping."""

    def __init__(self, config: FlexGenPcacheConfig) -> None:
        config.validate()
        config.kv_dir.parent.mkdir(parents=True, exist_ok=True)
        pcache_module = _load_pcache_module(config.flexgen_root)
        pcache_type = pcache_module.Pcache
        self.config = config
        self._selector_index = (
            QuantizedKeyIndex(config.selector_index_dir)
            if config.selector_index_dir is not None
            else None
        )
        if self._selector_index is not None and (
            self._selector_index.selector_kv_head_ids != config.selector_kv_head_ids
        ):
            raise ValueError("selector index head IDs do not match the Pcache configuration")
        self._pcache_module = pcache_module
        original_clear_folder = pcache_module.clear_folder
        if config.reuse_existing or config.resume_existing:
            pcache_module.clear_folder = lambda _path: None
        try:
            self._pcache = pcache_type(
                cpu_size=config.cpu_cache_mb,
                gpu_size=config.gpu_cache_mb,
                kv_dir=str(config.kv_dir),
                cache_type=config.cache_type,
                disk_type="KV_Division",
                cpu_gather=True,
                head_ids=list(config.selector_kv_head_ids),
                chunk_size=config.chunk_size,
            )
        finally:
            pcache_module.clear_folder = original_clear_folder
        self._task_prefix_ids: dict[str, int] = {}
        self._task_infos: dict[str, PrefixStoreInfo] = {}
        self._physical_to_logical: dict[str, list[list[int]]] = {}
        self._logical_to_physical: dict[str, list[list[int]]] = {}
        self._attention_state: dict[tuple[int, int, int], tuple[float, int]] = {}
        self._impress_score_state: dict[tuple[int, int, int], tuple[int, int]] = {}
        self.cache_score_updates = 0
        self.selector_index_preload_ms = 0.0
        self.selector_index_preloaded_tasks: list[str] = []

    def _preload_selector_task(self, task: str) -> None:
        if (
            self._selector_index is None
            or task in self.selector_index_preloaded_tasks
        ):
            return
        preload_started = time.perf_counter()
        self._selector_index.preload_task(task)
        self.selector_index_preload_ms += (
            time.perf_counter() - preload_started
        ) * 1000
        self.selector_index_preloaded_tasks.append(task)

    def add_task(
        self,
        *,
        store_root: str | Path,
        task: str,
        preload_selector_index: bool = True,
    ) -> PrefixStoreInfo:
        """Register a task while optionally keeping its selector index cold."""

        if task in self._task_prefix_ids:
            if preload_selector_index:
                self._preload_selector_task(task)
            return self._task_infos[task]
        info = read_store_info(store_root, task)
        if self._selector_index is not None:
            self._selector_index.validate_task(
                task,
                prefix_tokens=info.prefix_tokens,
                layers=info.layers,
                kv_heads=info.kv_heads,
                head_dim=info.head_dim,
            )
            if preload_selector_index:
                self._preload_selector_task(task)
        physical_to_logical = logical_to_physical = None
        if self.config.impress_reorder_path is not None:
            physical_to_logical, logical_to_physical = load_task_reorder(
                self.config.impress_reorder_path,
                task=task,
                info=info,
            )
        key, value = _load_task_tensor(
            store_root=store_root,
            task=task,
            info=info,
            chunk_size=self.config.chunk_size,
            physical_to_logical=physical_to_logical,
        )
        prefix_id = len(self._task_prefix_ids)
        if self.config.reuse_existing or self.config.resume_existing:
            with _reuse_existing_memmaps(
                self._pcache_module,
                allow_missing=self.config.resume_existing,
            ):
                self._pcache.insert(prefix_id=prefix_id, key=key, value=value)
        else:
            self._pcache.insert(prefix_id=prefix_id, key=key, value=value)
        self._task_prefix_ids[task] = prefix_id
        self._task_infos[task] = info
        if physical_to_logical is not None and logical_to_physical is not None:
            self._physical_to_logical[task] = physical_to_logical
            self._logical_to_physical[task] = logical_to_physical
        del key, value
        return info

    @property
    def selector_index_preloaded_bytes(self) -> int:
        """Return compressed selector bytes resident for active tasks."""

        if self._selector_index is None:
            return 0
        return int(self._selector_index.preloaded_compressed_bytes)

    def new_layer_loader(
        self,
        *,
        task: str,
        layer_plan: Sequence[Sequence[str]],
        method: str,
        layer_token_selections: Sequence[Sequence[int]] | None = None,
        online_selection: bool = False,
        keep_ratio: float | None = None,
        layer_keep_ratios: Sequence[float] | None = None,
        layer_keep_blocks: Sequence[int] | None = None,
        probe_query_heads: Sequence[int] = (0, 1, 2),
        similarity_alpha: float = 0.6,
        impress_async_prefetch: bool = False,
        impress_period_prefetch_size: int = 1,
        impress_period_prefetch_budget_scale: float = 1.0,
        impress_priority_prefetch: bool = False,
        impress_deferred_compute_timing: bool = False,
        impress_rolling_period_prefetch: bool = False,
        impress_value_ordered_prefetch: bool = False,
        impress_value_prefetch_budget_scale: float = 1.0,
        promixed_policy: Mapping[str, float | bool] | None = None,
        defer_cache_score_updates: bool = False,
    ) -> "FlexGenLayerLoader":
        if task not in self._task_prefix_ids:
            raise KeyError(f"task {task!r} was not inserted into this Pcache")

        cache_score_updater: Callable[
            [int, Sequence[int], Sequence[float] | None], int
        ] | None = None
        if online_selection and self.config.cache_type == "CKLFU":
            if method == "contigkv":
                def update_contiguous_score(
                    layer: int,
                    selected: Sequence[int],
                    scores: Sequence[float] | None,
                ) -> int:
                    return self.update_attention_layer_scores(
                        task=task,
                        layer=layer,
                        selected_chunks=selected,
                        chunk_scores=scores,
                    )

                cache_score_updater = update_contiguous_score
            elif method == "impress":
                def update_impress_score(
                    layer: int,
                    selected: Sequence[int],
                    _scores: Sequence[float] | None,
                ) -> int:
                    return self.update_impress_layer_scores(
                        task=task,
                        layer=layer,
                        selected_tokens=selected,
                    )

                cache_score_updater = update_impress_score
        return FlexGenLayerLoader(
            pcache=self._pcache,
            prefix_id=self._task_prefix_ids[task],
            info=self._task_infos[task],
            layer_plan=layer_plan,
            config=self.config,
            method=method,
            layer_token_selections=layer_token_selections,
            online_selection=online_selection,
            keep_ratio=keep_ratio,
            layer_keep_ratios=layer_keep_ratios,
            layer_keep_blocks=layer_keep_blocks,
            probe_query_heads=probe_query_heads,
            similarity_alpha=similarity_alpha,
            impress_async_prefetch=impress_async_prefetch,
            impress_period_prefetch_size=impress_period_prefetch_size,
            impress_period_prefetch_budget_scale=impress_period_prefetch_budget_scale,
            impress_priority_prefetch=impress_priority_prefetch,
            impress_deferred_compute_timing=impress_deferred_compute_timing,
            impress_rolling_period_prefetch=impress_rolling_period_prefetch,
            impress_value_ordered_prefetch=impress_value_ordered_prefetch,
            impress_value_prefetch_budget_scale=(
                impress_value_prefetch_budget_scale
            ),
            promixed_policy=promixed_policy,
            defer_cache_score_updates=defer_cache_score_updates,
            selector_index=self._selector_index,
            selector_index_task=task if self._selector_index is not None else None,
            physical_to_logical=self._physical_to_logical.get(task),
            logical_to_physical=self._logical_to_physical.get(task),
            cache_score_updater=cache_score_updater,
        )

    def close(self) -> None:
        executor = getattr(self._pcache, "_prefetch_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)

    def update_attention_scores(
        self,
        *,
        task: str,
        layer_plan: Sequence[Sequence[str]],
        layer_chunk_scores: Sequence[Sequence[float]],
    ) -> int:
        """Apply the paper's cumulative-attention times frequency cache score."""

        if self.config.cache_type != "CKLFU":
            return 0
        if task not in self._task_prefix_ids:
            raise KeyError(f"task {task!r} was not inserted into this Pcache")
        if len(layer_plan) != len(layer_chunk_scores):
            raise ValueError("attention score layers must match the physical layer plan")

        updates = 0
        for layer, (tiers, scores) in enumerate(zip(layer_plan, layer_chunk_scores)):
            if len(tiers) != len(scores):
                raise ValueError(f"layer {layer} attention scores do not match its chunk plan")
            updates += self.update_attention_layer_scores(
                task=task,
                layer=layer,
                selected_chunks=selected_chunk_indices(tiers),
                chunk_scores=scores,
            )
        return updates

    def update_attention_layer_scores(
        self,
        *,
        task: str,
        layer: int,
        selected_chunks: Sequence[int],
        chunk_scores: Sequence[float] | None,
    ) -> int:
        """Update one loaded layer using the paper's scalar ``I_j * F_j`` score."""

        if self.config.cache_type != "CKLFU":
            return 0
        if task not in self._task_prefix_ids:
            raise KeyError(f"task {task!r} was not inserted into this Pcache")
        if chunk_scores is None:
            raise ValueError("ContiguousKV cache updates require chunk attention scores")

        prefix_id = self._task_prefix_ids[task]
        layers = self._pcache.cache[prefix_id].layers
        if not 0 <= layer < len(layers):
            raise ValueError(f"layer {layer} is outside the cached prefix")
        cache_layer = layers[layer]
        if len(chunk_scores) != len(cache_layer.key_tokens):
            raise ValueError(f"layer {layer} attention scores do not match its chunk count")

        chunks = sorted(set(int(chunk) for chunk in selected_chunks))
        if not chunks or any(not 0 <= chunk < len(chunk_scores) for chunk in chunks):
            raise ValueError(f"layer {layer} contains invalid selected chunks")
        updates = 0
        for chunk in chunks:
            score = float(chunk_scores[chunk])
            if not math.isfinite(score) or score < 0:
                raise ValueError(
                    f"layer {layer} chunk {chunk} has invalid attention score {score}"
                )
            state_key = (prefix_id, layer, chunk)
            cumulative, frequency = self._attention_state.get(state_key, (0.0, 0))
            cumulative += score
            frequency += 1
            self._attention_state[state_key] = (cumulative, frequency)
            cache_score = cumulative * frequency
            score_pair = [cache_score, cache_score]
            self._pcache.control.update_score(
                item=cache_layer.key_tokens[chunk],
                new_score=list(score_pair),
            )
            self._pcache.control.update_score(
                item=cache_layer.value_tokens[chunk],
                new_score=list(score_pair),
            )
            updates += 1
        self.cache_score_updates += updates
        return updates

    def update_impress_scores(
        self,
        *,
        task: str,
        layer_token_selections: Sequence[Sequence[int]],
    ) -> int:
        """Apply IMPRESS CKLFU's chunk-access and important-token counters."""

        if self.config.cache_type != "CKLFU":
            return 0
        if task not in self._task_prefix_ids or task not in self._task_infos:
            raise KeyError(f"task {task!r} was not inserted into this Pcache")
        info = self._task_infos[task]
        if len(layer_token_selections) != info.layers:
            raise ValueError("IMPRESS selections must cover every model layer")

        updates = 0
        for layer, token_ids in enumerate(layer_token_selections):
            updates += self.update_impress_layer_scores(
                task=task,
                layer=layer,
                selected_tokens=token_ids,
            )
        return updates

    def update_impress_layer_scores(
        self,
        *,
        task: str,
        layer: int,
        selected_tokens: Sequence[int],
    ) -> int:
        """Update one loaded IMPRESS layer with CKLFU's two counters."""

        if self.config.cache_type != "CKLFU":
            return 0
        if task not in self._task_prefix_ids or task not in self._task_infos:
            raise KeyError(f"task {task!r} was not inserted into this Pcache")
        info = self._task_infos[task]
        if not 0 <= layer < info.layers:
            raise ValueError(f"layer {layer} is outside the cached prefix")

        chunk_count = expected_chunk_count(
            prefix_tokens=info.prefix_tokens,
            chunk_size=self.config.chunk_size,
        )
        task_reorder = getattr(self, "_logical_to_physical", {}).get(task)
        counts: dict[int, int] = {}
        for token_id in sorted(set(int(token) for token in selected_tokens)):
            if not 0 <= token_id < info.prefix_tokens:
                raise ValueError(f"layer {layer} token {token_id} is outside the prefix")
            storage_token = (
                task_reorder[layer][token_id]
                if task_reorder is not None
                else token_id
            )
            chunk = storage_token // self.config.chunk_size
            counts[chunk] = counts.get(chunk, 0) + 1
        if not counts:
            raise ValueError(f"layer {layer} has no selected IMPRESS tokens")

        prefix_id = self._task_prefix_ids[task]
        cache_layer = self._pcache.cache[prefix_id].layers[layer]
        updates = 0
        for chunk, important_tokens in counts.items():
            if not 0 <= chunk < chunk_count:
                raise ValueError(f"layer {layer} chunk {chunk} is outside the prefix")
            state_key = (prefix_id, layer, chunk)
            accesses, cumulative_important = self._impress_score_state.get(
                state_key, (0, 0)
            )
            accesses += 1
            cumulative_important += important_tokens
            self._impress_score_state[state_key] = (accesses, cumulative_important)
            score_pair = [accesses, cumulative_important]
            self._pcache.control.update_score(
                item=cache_layer.key_tokens[chunk],
                new_score=list(score_pair),
            )
            self._pcache.control.update_score(
                item=cache_layer.value_tokens[chunk],
                new_score=list(score_pair),
            )
            updates += 1
        self.cache_score_updates += updates
        return updates


class FlexGenLayerLoader:
    """Schedule and consume Pcache loads one decoder layer at a time."""

    def __init__(
        self,
        *,
        pcache: Any,
        prefix_id: int,
        info: PrefixStoreInfo,
        layer_plan: Sequence[Sequence[str]],
        config: FlexGenPcacheConfig,
        method: str,
        layer_token_selections: Sequence[Sequence[int]] | None = None,
        online_selection: bool = False,
        keep_ratio: float | None = None,
        layer_keep_ratios: Sequence[float] | None = None,
        layer_keep_blocks: Sequence[int] | None = None,
        probe_query_heads: Sequence[int] = (0, 1, 2),
        similarity_alpha: float = 0.6,
        impress_async_prefetch: bool = False,
        impress_period_prefetch_size: int = 1,
        impress_period_prefetch_budget_scale: float = 1.0,
        impress_priority_prefetch: bool = False,
        impress_deferred_compute_timing: bool = False,
        impress_rolling_period_prefetch: bool = False,
        impress_value_ordered_prefetch: bool = False,
        impress_value_prefetch_budget_scale: float = 1.0,
        promixed_policy: Mapping[str, float | bool] | None = None,
        defer_cache_score_updates: bool = False,
        selector_index: QuantizedKeyIndex | None = None,
        selector_index_task: str | None = None,
        physical_to_logical: Sequence[Sequence[int]] | None = None,
        logical_to_physical: Sequence[Sequence[int]] | None = None,
        cache_score_updater: Callable[
            [int, Sequence[int], Sequence[float] | None], int
        ]
        | None = None,
    ) -> None:
        if len(layer_plan) != info.layers:
            raise ValueError(f"plan layers {len(layer_plan)} != prefix layers {info.layers}")
        expected_chunks = expected_chunk_count(
            prefix_tokens=info.prefix_tokens,
            chunk_size=config.chunk_size,
        )
        for layer, tiers in enumerate(layer_plan):
            if len(tiers) != expected_chunks:
                raise ValueError(
                    f"layer {layer} has {len(tiers)} chunks; expected {expected_chunks} "
                    f"for chunk size {config.chunk_size}"
                )

        self._pcache = pcache
        self._prefix_id = prefix_id
        self._info = info
        self._config = config
        if method not in {"contigkv", "impress"}:
            raise ValueError(f"unsupported sparse cache method {method!r}")
        if online_selection and (keep_ratio is None or not 0 < float(keep_ratio) <= 1):
            raise ValueError("online selection requires keep_ratio in (0, 1]")
        if layer_keep_ratios is not None:
            normalized_layer_ratios = tuple(float(ratio) for ratio in layer_keep_ratios)
            if not online_selection or method != "impress":
                raise ValueError(
                    "layer keep ratios are only valid for online IMPRESS/HyperInfer"
                )
            if len(normalized_layer_ratios) != info.layers:
                raise ValueError(
                    f"layer keep ratios cover {len(normalized_layer_ratios)} layers; "
                    f"expected {info.layers}"
                )
            if any(
                not math.isfinite(ratio) or not 0 < ratio <= 1
                for ratio in normalized_layer_ratios
            ):
                raise ValueError("layer keep ratios must be finite and in (0, 1]")
        else:
            normalized_layer_ratios = None
        if layer_keep_blocks is not None:
            normalized_layer_blocks = tuple(int(count) for count in layer_keep_blocks)
            if (
                not online_selection
                or method != "impress"
                or normalized_layer_ratios is None
            ):
                raise ValueError(
                    "exact layer block counts require online IMPRESS and layer ratios"
                )
            if len(normalized_layer_blocks) != info.layers:
                raise ValueError(
                    f"layer block counts cover {len(normalized_layer_blocks)} layers; "
                    f"expected {info.layers}"
                )
            if any(
                count < 1 or count > expected_chunks
                for count in normalized_layer_blocks
            ):
                raise ValueError(
                    f"layer block counts must be in [1, {expected_chunks}]"
                )
        else:
            normalized_layer_blocks = None
        if not probe_query_heads or any(int(head) < 0 for head in probe_query_heads):
            raise ValueError("probe_query_heads must contain non-negative head indices")
        if similarity_alpha <= 0:
            raise ValueError("similarity_alpha must be positive")
        if impress_async_prefetch and (method != "impress" or not online_selection):
            raise ValueError(
                "IMPRESS asynchronous prefetch is only valid for online IMPRESS"
            )
        if impress_period_prefetch_size <= 0:
            raise ValueError("IMPRESS predictive period size must be positive")
        if impress_period_prefetch_size > 1 and (
            method != "impress" or not online_selection or not impress_async_prefetch
        ):
            raise ValueError(
                "IMPRESS predictive periods require asynchronous online IMPRESS"
            )
        if impress_priority_prefetch and (
            method != "impress" or not online_selection or not impress_async_prefetch
        ):
            raise ValueError(
                "IMPRESS priority prefetch requires asynchronous online IMPRESS"
            )
        if impress_priority_prefetch and not callable(
            getattr(pcache, "prefetch_scheduler_metrics", None)
        ):
            raise ValueError(
                "IMPRESS priority prefetch requires the priority-enabled Pcache"
            )
        if impress_deferred_compute_timing and (
            method != "impress" or not online_selection or not impress_async_prefetch
        ):
            raise ValueError(
                "IMPRESS deferred compute timing requires asynchronous online IMPRESS"
            )
        if impress_rolling_period_prefetch and (
            method != "impress"
            or not online_selection
            or not impress_async_prefetch
            or impress_period_prefetch_size <= 1
        ):
            raise ValueError(
                "IMPRESS rolling Period prefetch requires predictive asynchronous "
                "online IMPRESS"
            )
        if impress_value_ordered_prefetch and (
            method != "impress" or not online_selection or not impress_async_prefetch
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
        period_budget_scale = float(impress_period_prefetch_budget_scale)
        if (
            not math.isfinite(period_budget_scale)
            or not 0 < period_budget_scale <= 1
        ):
            raise ValueError(
                "IMPRESS predictive-period budget scale must be finite and in (0, 1]"
            )
        if (physical_to_logical is None) != (logical_to_physical is None):
            raise ValueError("both IMPRESS reorder directions must be provided together")
        if physical_to_logical is not None:
            if method != "impress":
                raise ValueError("physical token reordering is only valid for IMPRESS")
            if len(physical_to_logical) != info.layers or len(logical_to_physical) != info.layers:
                raise ValueError("IMPRESS reorder mappings must cover every model layer")
            for layer, (forward, inverse) in enumerate(
                zip(physical_to_logical, logical_to_physical)
            ):
                if len(forward) != info.prefix_tokens or len(inverse) != info.prefix_tokens:
                    raise ValueError(f"IMPRESS reorder layer {layer} has the wrong token count")
                if any(
                    int(inverse[int(logical)]) != physical
                    for physical, logical in enumerate(forward)
                ):
                    raise ValueError(f"IMPRESS reorder layer {layer} directions are not inverse")

        if promixed_policy is not None:
            if method != "impress" or not online_selection:
                raise ValueError(
                    "ProMixed policies require online IMPRESS/HyperInfer selection"
                )
            normalized_promixed_policy = {
                str(key): (
                    bool(value)
                    if str(key) == "adaptive_coverage"
                    else float(value)
                )
                for key, value in promixed_policy.items()
            }
        else:
            normalized_promixed_policy = None
        if selector_index is not None:
            if (
                not online_selection
                or method not in {"contigkv", "impress"}
                or (method == "impress" and normalized_promixed_policy is None)
            ):
                raise ValueError(
                    "a selector index requires online ContiguousKV or ProMixed"
                )
            if selector_index_task is None:
                raise ValueError("a selector index requires a task name")
            if selector_index.selector_kv_head_ids != config.selector_kv_head_ids:
                raise ValueError("selector index head IDs do not match the loader")
            if physical_to_logical is not None:
                raise ValueError("a selector index cannot be physically reordered")
        if defer_cache_score_updates and cache_score_updater is None:
            raise ValueError(
                "deferred cache-score updates require an active cache-score policy"
            )

        self.method = method
        self.online_selection = online_selection
        self.keep_ratio = float(keep_ratio) if keep_ratio is not None else None
        self._layer_keep_ratios = normalized_layer_ratios
        self._layer_keep_blocks = normalized_layer_blocks
        self._exact_block_budget_target = (
            sum(normalized_layer_blocks)
            if normalized_layer_blocks is not None
            else None
        )
        self._exact_block_budget_consumed = 0
        self._exact_block_budget_last_layer = -1
        self.probe_query_heads = tuple(int(head) for head in probe_query_heads)
        self.similarity_alpha = float(similarity_alpha)
        self.impress_async_prefetch = bool(impress_async_prefetch)
        self.impress_period_prefetch_size = int(impress_period_prefetch_size)
        self.impress_period_prefetch_budget_scale = period_budget_scale
        self.impress_priority_prefetch = bool(impress_priority_prefetch)
        self.impress_deferred_compute_timing = bool(
            impress_deferred_compute_timing
        )
        self.impress_rolling_period_prefetch = bool(
            impress_rolling_period_prefetch
        )
        self.impress_value_ordered_prefetch = bool(
            impress_value_ordered_prefetch
        )
        self.impress_value_prefetch_budget_scale = value_budget_scale
        self.promixed_policy = normalized_promixed_policy
        self.defer_cache_score_updates = bool(defer_cache_score_updates)
        self._selector_index = selector_index
        self._selector_index_task = selector_index_task
        self._selector_index_group_size = (
            selector_index.tasks[selector_index_task].group_size
            if selector_index is not None and selector_index_task is not None
            else 0
        )
        self._cache_score_updater = cache_score_updater
        self._cache_score_updated_layers: set[int] = set()
        self._deferred_cache_score_layers: list[int] = []
        self._deferred_cache_score_layer_set: set[int] = set()
        self._cache_score_updates = 0
        self._cache_update_ms = 0.0
        self._cache_update_deferred_ms = 0.0
        self._physical_to_logical = (
            [[int(token) for token in row] for row in physical_to_logical]
            if physical_to_logical is not None
            else None
        )
        self._logical_to_physical = (
            [[int(token) for token in row] for row in logical_to_physical]
            if logical_to_physical is not None
            else None
        )
        self._selected_tier_by_layer = []
        for layer, tiers in enumerate(layer_plan):
            active = [str(tier) for tier in tiers if str(tier).lower() != "drop"]
            if not active:
                raise ValueError(f"layer {layer} has no retained tier label")
            self._selected_tier_by_layer.append(active[0])
        if online_selection:
            self._selected_by_layer = [[] for _ in layer_plan]
        else:
            self._selected_by_layer = selected_tokens_for_plan(
                layer_plan,
                chunk_size=config.chunk_size,
                prefix_tokens=info.prefix_tokens,
                layer_token_selections=layer_token_selections,
            )
            if any(not tokens for tokens in self._selected_by_layer):
                raise ValueError("every sparse attention layer must retain at least one physical chunk")
        self._prefetch_priority_by_layer = [
            list(tokens) for tokens in self._selected_by_layer
        ]
        self._online_layer_plan: list[list[str] | None] = [None for _ in layer_plan]
        self._online_chunk_scores: list[list[float] | None] = [None for _ in layer_plan]
        self._pending: dict[int, tuple[Any, float]] = {}
        self._speculative_pending: dict[int, tuple[Any, float]] = {}
        self._resolved_speculative: dict[int, tuple[Any, Any, Any]] = {}
        self._missing_pending: dict[int, tuple[Any, float]] = {}
        self._speculative_positions: dict[int, list[int]] = {}
        self._speculative_kind: dict[int, str] = {}
        self._speculative_requested_positions: dict[int, list[int]] = {}
        self._speculative_source_layer: dict[int, int] = {}
        self._speculative_prediction_recorded: set[int] = set()
        self._loaded: dict[int, tuple[Any, Any]] = {}
        self._prefetch_wait_ms = 0.0
        self._prefetch_elapsed_ms = 0.0
        self._physical_tokens = 0
        self._physical_chunks: set[tuple[int, int]] = set()
        self._source_tensor_tokens = {"gpu": 0, "cpu": 0, "disk": 0}
        self._inter_period_hit_tokens = 0
        self._inter_period_missing_tokens = 0
        self._inter_period_unused_tokens = 0
        self._speculative_physical_stats = {
            kind: {"hit": 0, "missing": 0, "unused": 0}
            for kind in ("next", "period")
        }
        self._speculative_prediction_stats = {
            kind: {
                "layers": 0,
                "requested": 0,
                "actual": 0,
                "overlap": 0,
                "jaccards": [],
                "layer_distances": [],
            }
            for kind in ("next", "period")
        }
        self._selector_key_bytes = 0
        self._selector_dequantized_key_bytes = 0
        self._selector_source_bytes = {"gpu": 0, "cpu": 0, "disk": 0}
        self._selector_load_ms = 0.0
        self._selector_compute_ms = 0.0
        self._selector_calls = 0
        self._selector_fallbacks = 0
        self._selector_similarities: list[float] = []
        self._selector_wait_ms = 0.0
        self._promixed_agreements: list[float] = []
        self._promixed_boundary_margins: list[float] = []
        self._promixed_uncertainties: list[float] = []
        self._promixed_periods: list[int] = []
        self._selector_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="contiguous-fuxian-selector",
        )
        self._selector_pending: dict[int, Future[Any]] = {}
        self._selector_stream: Any | None = None
        self._last_selector_compute_ms = 0.0
        self._impress_prefetch_time_budget = 0.0
        self._impress_prefetch_budgets: list[float] = []
        self._impress_next_prefetch_jobs = 0
        self._impress_period_prefetch_jobs = 0
        self._impress_period_prefetch_tokens = 0
        self._impress_value_ordered_prefetch_jobs = 0
        self._impress_compute_event_pairs: list[tuple[Any, Any]] = []
        self._impress_deferred_compute_samples = 0
        self._impress_deferred_compute_pending_max = 0
        scheduler_metrics = getattr(pcache, "prefetch_scheduler_metrics", None)
        self._prefetch_scheduler_start = (
            scheduler_metrics() if callable(scheduler_metrics) else {}
        )

    @property
    def layers(self) -> int:
        return self._info.layers

    @property
    def prefix_tokens(self) -> int:
        return self._info.prefix_tokens

    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size

    @property
    def selector_kv_head_ids(self) -> tuple[int, ...]:
        return self._config.selector_kv_head_ids

    @property
    def impress_reordered(self) -> bool:
        return self._logical_to_physical is not None

    def keep_ratio_for_layer(self, layer: int) -> float:
        """Return the calibrated online retention ratio for one model layer."""

        if not 0 <= layer < self.layers:
            raise ValueError(f"layer {layer} is outside the cached prefix")
        if self._layer_keep_ratios is None:
            if self.keep_ratio is None:
                raise RuntimeError("online keep ratio is not configured")
            return self.keep_ratio
        return self._layer_keep_ratios[layer]

    def keep_blocks_for_layer(self, layer: int) -> int | None:
        """Return this layer's share of the remaining exact global block budget."""

        if not 0 <= layer < self.layers:
            raise ValueError(f"layer {layer} is outside the cached prefix")
        if self._layer_keep_blocks is None:
            return None
        if layer != self._exact_block_budget_last_layer + 1:
            raise RuntimeError("exact block budget layers must be selected sequentially")
        target = int(self._exact_block_budget_target)
        remaining_budget = target - self._exact_block_budget_consumed
        remaining_layers = self.layers - layer
        blocks_per_layer = expected_chunk_count(
            prefix_tokens=self._info.prefix_tokens,
            chunk_size=self._config.chunk_size,
        )
        if not remaining_layers <= remaining_budget <= remaining_layers * blocks_per_layer:
            raise RuntimeError("remaining exact block budget is infeasible")
        planned_remaining = sum(self._layer_keep_blocks[layer:])
        scaled = (
            self._layer_keep_blocks[layer]
            * remaining_budget
            / planned_remaining
        )
        count = math.floor(scaled + 0.5)
        minimum = max(
            1,
            remaining_budget - (remaining_layers - 1) * blocks_per_layer,
        )
        maximum = min(
            blocks_per_layer,
            remaining_budget - (remaining_layers - 1),
        )
        return min(maximum, max(minimum, count))

    def max_blocks_for_layer(self, layer: int) -> int | None:
        """Return the largest selection that leaves one block for every later layer."""

        if self._layer_keep_blocks is None:
            return None
        if layer != self._exact_block_budget_last_layer + 1:
            raise RuntimeError("exact block budget layers must be selected sequentially")
        remaining_budget = (
            int(self._exact_block_budget_target)
            - self._exact_block_budget_consumed
        )
        return min(
            expected_chunk_count(
                prefix_tokens=self._info.prefix_tokens,
                chunk_size=self._config.chunk_size,
            ),
            remaining_budget - (self.layers - layer - 1),
        )

    def _storage_positions(self, layer: int, logical_positions: Sequence[int]) -> list[int]:
        """Map logical attention positions to physical IMPRESS storage positions."""

        positions = [int(position) for position in logical_positions]
        if self._logical_to_physical is None:
            return positions
        mapping = self._logical_to_physical[layer]
        return [mapping[position] for position in positions]

    def _logical_segment(self, layer: int, key: Any, value: Any, physical_ids: Any):
        """Restore a physical Pcache response to sorted logical token order."""

        if self._physical_to_logical is None:
            import torch

            order = torch.argsort(physical_ids)
            return (
                key.index_select(0, order.to(key.device)),
                value.index_select(0, order.to(value.device)),
                physical_ids.index_select(0, order),
            )
        import torch

        mapping = torch.tensor(
            self._physical_to_logical[layer],
            dtype=torch.long,
            device=physical_ids.device,
        )
        logical_ids = mapping.index_select(0, physical_ids.to(mapping.device))
        order = torch.argsort(logical_ids)
        logical_ids = logical_ids.index_select(0, order)
        return (
            key.index_select(0, order.to(key.device)),
            value.index_select(0, order.to(value.device)),
            logical_ids,
        )

    def configure_contiguous_period(
        self,
        *,
        period_start: int,
        selected_chunks: Sequence[int],
        chunk_scores: Sequence[float],
        period_size: int,
    ) -> None:
        if not self.online_selection or self.method != "contigkv":
            raise RuntimeError("dynamic ContiguousKV selection is not enabled")
        expected_chunks = expected_chunk_count(
            prefix_tokens=self._info.prefix_tokens,
            chunk_size=self._config.chunk_size,
        )
        if len(chunk_scores) != expected_chunks:
            raise ValueError("online chunk scores do not cover the prefix")
        normalized_scores = [float(score) for score in chunk_scores]
        if any(not math.isfinite(score) or score < 0 for score in normalized_scores):
            raise ValueError("online ContiguousKV chunk scores must be finite and non-negative")
        selected = sorted(set(int(chunk) for chunk in selected_chunks))
        if not selected or any(not 0 <= chunk < expected_chunks for chunk in selected):
            raise ValueError("online ContiguousKV selected invalid chunks")
        selected_set = set(selected)
        for layer in range(period_start, min(self.layers, period_start + period_size)):
            tier = self._selected_tier_by_layer[layer]
            self._online_layer_plan[layer] = [
                tier if chunk in selected_set else "drop" for chunk in range(expected_chunks)
            ]
            self._online_chunk_scores[layer] = list(normalized_scores)
            self._selected_by_layer[layer] = retained_token_ids(
                self._online_layer_plan[layer],
                chunk_size=self._config.chunk_size,
                prefix_tokens=self._info.prefix_tokens,
            )

    def configure_impress_layer(
        self,
        *,
        layer: int,
        selected_tokens: Sequence[int],
        prefetch_priority_tokens: Sequence[int] | None = None,
    ) -> None:
        if not self.online_selection or self.method != "impress":
            raise RuntimeError("dynamic IMPRESS selection is not enabled")
        if not 0 <= layer < self.layers:
            raise ValueError(f"online IMPRESS layer {layer} is outside the model")
        selected = sorted(set(int(token) for token in selected_tokens))
        if not selected or any(not 0 <= token < self._info.prefix_tokens for token in selected):
            raise ValueError("online IMPRESS selected invalid tokens")
        expected_chunks = expected_chunk_count(
            prefix_tokens=self._info.prefix_tokens,
            chunk_size=self._config.chunk_size,
        )
        storage_positions = self._storage_positions(layer, selected)
        touched = {token // self._config.chunk_size for token in storage_positions}
        if self._layer_keep_blocks is not None:
            if layer != self._exact_block_budget_last_layer + 1:
                raise RuntimeError("exact block budget layers must be configured sequentially")
            self._exact_block_budget_consumed += len(touched)
            self._exact_block_budget_last_layer = layer
            if self._exact_block_budget_consumed > int(self._exact_block_budget_target):
                raise RuntimeError("online selection exceeded the exact block budget")
        tier = self._selected_tier_by_layer[layer]
        self._online_layer_plan[layer] = [
            tier if chunk in touched else "drop" for chunk in range(expected_chunks)
        ]
        self._selected_by_layer[layer] = selected
        if prefetch_priority_tokens is None:
            priority = list(selected)
        else:
            priority = [int(token) for token in prefetch_priority_tokens]
            if len(priority) != len(selected) or set(priority) != set(selected):
                raise ValueError(
                    "IMPRESS prefetch priority must be a permutation of selected tokens"
                )
        self._prefetch_priority_by_layer[layer] = priority
        self._record_speculative_prediction_if_ready(layer)

    def _record_speculative_prediction_if_ready(self, layer: int) -> None:
        """Record prediction quality regardless of selection/request ordering."""

        if not hasattr(self, "_speculative_prediction_stats"):
            return
        recorded = getattr(self, "_speculative_prediction_recorded", None)
        if recorded is None:
            recorded = set()
            self._speculative_prediction_recorded = recorded
        if layer in recorded:
            return
        kind = getattr(self, "_speculative_kind", {}).get(layer)
        requested = getattr(self, "_speculative_requested_positions", {}).get(
            layer
        )
        actual = self._selected_by_layer[layer]
        if kind not in {"next", "period"} or requested is None or not actual:
            return
        predicted_set = set(requested)
        actual_set = set(actual)
        overlap = len(predicted_set & actual_set)
        union = len(predicted_set | actual_set)
        stats = self._speculative_prediction_stats[kind]
        stats["layers"] += 1
        stats["requested"] += len(predicted_set)
        stats["actual"] += len(actual_set)
        stats["overlap"] += overlap
        stats["jaccards"].append(overlap / max(1, union))
        source_layer = getattr(self, "_speculative_source_layer", {}).get(layer)
        if source_layer is not None:
            stats["layer_distances"].append(layer - source_layer)
        recorded.add(layer)
        self._speculative_requested_positions.pop(layer, None)
        self._speculative_source_layer.pop(layer, None)

    def _track_speculative_request(
        self,
        *,
        target: int,
        positions: Sequence[int],
        kind: str,
        source_layer: int,
    ) -> None:
        if not hasattr(self, "_speculative_kind"):
            self._speculative_kind = {}
        if not hasattr(self, "_speculative_requested_positions"):
            self._speculative_requested_positions = {}
        if not hasattr(self, "_speculative_source_layer"):
            self._speculative_source_layer = {}
        self._speculative_positions[target] = list(positions)
        self._speculative_kind[target] = kind
        self._speculative_requested_positions[target] = list(positions)
        self._speculative_source_layer[target] = int(source_layer)
        self._record_speculative_prediction_if_ready(target)

    def _load_selector_keys_sync(self, layer: int) -> Any:
        if not self.online_selection:
            raise RuntimeError("online selector key loading is not enabled")
        if not 0 <= layer < self.layers:
            raise ValueError(f"selector layer {layer} is outside the model")
        import torch

        started = time.perf_counter()
        if self._selector_index is not None:
            if self._selector_index_task is None:
                raise RuntimeError("selector index task was not configured")
            if torch.cuda.is_available():
                if self._selector_stream is None:
                    self._selector_stream = torch.cuda.Stream()
                target = torch.device("cuda", torch.cuda.current_device())
                with torch.cuda.stream(self._selector_stream):
                    keys, key_bytes, source = self._selector_index.load_layer(
                        self._selector_index_task,
                        layer,
                        device=target,
                    )
                    completed = torch.cuda.Event()
                    completed.record(self._selector_stream)
                completed.synchronize()
            else:
                keys, key_bytes, source = self._selector_index.load_layer(
                    self._selector_index_task,
                    layer,
                    device=torch.device("cpu"),
                )
        else:
            cache_layer = self._pcache.cache[self._prefix_id].layers[layer]
            source_device = str(cache_layer.head_token.device)
            if source_device.startswith("cuda"):
                source = "gpu"
            elif source_device in {"cpu", "disk"}:
                source = source_device
            else:
                raise RuntimeError(
                    f"unsupported selector key source {source_device!r}"
                )
            if torch.cuda.is_available():
                if self._selector_stream is None:
                    self._selector_stream = torch.cuda.Stream()
                with torch.cuda.stream(self._selector_stream):
                    keys = self._pcache.get_head(
                        prefix_id=self._prefix_id, layer=layer
                    )
                    if self._logical_to_physical is not None:
                        order = torch.tensor(
                            self._logical_to_physical[layer],
                            dtype=torch.long,
                            device=keys.device,
                        )
                        keys = keys.index_select(0, order)
                    completed = torch.cuda.Event()
                    completed.record(self._selector_stream)
                completed.synchronize()
            else:
                keys = self._pcache.get_head(
                    prefix_id=self._prefix_id, layer=layer
                )
                if self._logical_to_physical is not None:
                    order = torch.tensor(
                        self._logical_to_physical[layer],
                        dtype=torch.long,
                    )
                    keys = keys.index_select(0, order)
            key_bytes = int(keys.numel()) * int(keys.element_size())
        self._selector_load_ms += (time.perf_counter() - started) * 1000
        self._selector_key_bytes += key_bytes
        self._selector_dequantized_key_bytes += (
            int(keys.numel()) * int(keys.element_size())
        )
        self._selector_source_bytes[source] += key_bytes
        self._selector_calls += 1
        return keys

    def prefetch_selector_keys(self, layer: int) -> None:
        """Start loading a future layer's probing keys on a dedicated worker."""

        if not self.online_selection or not 0 <= layer < self.layers:
            return
        if layer in self._selector_pending:
            return
        self._selector_pending[layer] = self._selector_executor.submit(
            self._load_selector_keys_sync, layer
        )

    def load_selector_keys(self, layer: int) -> Any:
        """Resolve a prefetched selector key tensor or load it synchronously."""

        started = time.perf_counter()
        future = self._selector_pending.pop(layer, None)
        if future is None:
            keys = self._load_selector_keys_sync(layer)
        else:
            keys = future.result()
        self._selector_wait_ms += (time.perf_counter() - started) * 1000
        return keys

    def record_selector_compute(
        self,
        elapsed_ms: float,
        *,
        similarity: float | None = None,
        fallback: bool = False,
    ) -> None:
        elapsed = float(elapsed_ms)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("selector compute time must be finite and non-negative")
        self._selector_compute_ms += elapsed
        self._last_selector_compute_ms = elapsed
        if similarity is not None:
            normalized_similarity = float(similarity)
            if not math.isfinite(normalized_similarity) or not 0 <= normalized_similarity <= 1:
                raise ValueError("selector similarity must be finite and in [0, 1]")
            self._selector_similarities.append(normalized_similarity)
        if fallback:
            self._selector_fallbacks += 1

    def record_promixed_decision(
        self,
        *,
        agreement: float,
        boundary_margin: float,
        uncertainty: float,
        period: int,
    ) -> None:
        values = {
            "agreement": float(agreement),
            "boundary_margin": float(boundary_margin),
            "uncertainty": float(uncertainty),
        }
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in values.values()
        ):
            raise ValueError("ProMixed decision metrics must be finite and in [0, 1]")
        normalized_period = int(period)
        if normalized_period not in {1, 2, 4, 8}:
            raise ValueError("ProMixed decision period must be 1, 2, 4, or 8")
        self._promixed_agreements.append(values["agreement"])
        self._promixed_boundary_margins.append(values["boundary_margin"])
        self._promixed_uncertainties.append(values["uncertainty"])
        self._promixed_periods.append(normalized_period)

    def online_cache_plan_and_scores(self) -> tuple[list[list[str]], list[list[float]]]:
        if self.method != "contigkv" or not self.online_selection:
            raise RuntimeError("online ContiguousKV cache scores are unavailable")
        if any(plan is None for plan in self._online_layer_plan):
            raise RuntimeError("not every layer received an online ContiguousKV plan")
        if any(scores is None for scores in self._online_chunk_scores):
            raise RuntimeError("not every layer received online ContiguousKV scores")
        return (
            [list(plan) for plan in self._online_layer_plan if plan is not None],
            [list(scores) for scores in self._online_chunk_scores if scores is not None],
        )

    def selected_tokens_by_layer(self) -> list[list[int]]:
        """Return a defensive copy of the useful sparse-attention positions."""

        if any(not selected for selected in self._selected_by_layer):
            raise RuntimeError("not every layer has a configured sparse selection")
        return [list(selected) for selected in self._selected_by_layer]

    def schedule(self, layer: int) -> None:
        """Issue one asynchronous physical-chunk prefetch, if not already issued."""

        if not 0 <= layer < self.layers:
            return
        if (
            layer in self._pending
            or layer in self._speculative_pending
            or layer in self._resolved_speculative
            or layer in self._missing_pending
            or layer in self._loaded
        ):
            return
        if not self._selected_by_layer[layer]:
            raise RuntimeError(f"layer {layer} has no configured selected tokens")
        self._submit_positions(
            layer,
            self._selected_by_layer[layer],
            self._pending,
            prefetch_kind="current",
        )

    def _submit_positions(
        self,
        layer: int,
        positions: Sequence[int],
        pending: dict[int, tuple[Any, float]],
        *,
        time_budget: float | None = None,
        prefetch_kind: str = "default",
    ) -> None:
        if not positions:
            return
        import torch

        storage_positions = self._storage_positions(layer, positions)
        position_tensor = torch.tensor(storage_positions, dtype=torch.long)
        issued_at = time.perf_counter()
        priority_args: dict[str, Any] = {"prefetch_kind": prefetch_kind}
        if self.impress_priority_prefetch:
            priorities = {
                "current": 0,
                "next": 1,
                "period": 2,
            }
            if prefetch_kind not in priorities:
                raise ValueError(
                    f"unsupported priority prefetch kind {prefetch_kind!r}"
                )
            priority_args["priority"] = priorities[prefetch_kind]
        handle = self._pcache.prefetch_async(
            prefix_id=self._prefix_id,
            pos_id=position_tensor,
            layer=layer,
            k_v_num_tokens=self._info.prefix_tokens,
            time_budget=(
                self._config.prefetch_time_budget
                if time_budget is None
                else float(time_budget)
            ),
            **priority_args,
        )
        pending[layer] = (handle, issued_at)

    def schedule_range(self, start: int, end: int) -> None:
        for layer in range(max(0, start), min(self.layers, end)):
            self.schedule(layer)

    def schedule_inter_period(
        self,
        *,
        previous_period_start: int,
        target_period_start: int,
        period_size: int,
    ) -> None:
        """Speculatively prefetch the previous Period's chunks for the next Period."""

        if self.method != "contigkv" or not 0 <= previous_period_start < self.layers:
            return
        previous_positions = list(self._selected_by_layer[previous_period_start])
        for layer in range(
            max(0, target_period_start),
            min(self.layers, target_period_start + period_size),
        ):
            if layer in self._loaded or layer in self._speculative_pending:
                continue
            self._track_speculative_request(
                target=layer,
                positions=previous_positions,
                kind="period",
                source_layer=previous_period_start,
            )
            self._submit_positions(layer, previous_positions, self._speculative_pending)

    def schedule_inter_period_missing(self, period_start: int, period_size: int) -> None:
        """Issue only the current Period chunks absent from speculative prefetches."""

        for layer in range(
            max(0, period_start),
            min(self.layers, period_start + period_size),
        ):
            speculative = set(self._speculative_positions.get(layer, ()))
            if not speculative:
                continue
            missing = sorted(set(self._selected_by_layer[layer]) - speculative)
            self._inter_period_missing_tokens += len(missing)
            self._speculative_physical_stats["period"]["missing"] += len(missing)
            self._submit_positions(
                layer,
                missing,
                self._missing_pending,
                prefetch_kind="current",
            )

    def prime_period(self, period_start: int, subperiod_size: int, period_size: int) -> None:
        """Load the first subperiod before its Period begins computation."""

        self.schedule_range(period_start, period_start + period_size)
        for layer in range(max(0, period_start), min(self.layers, period_start + subperiod_size)):
            self.resolve(layer)

    def schedule_impress_missing(self, layer: int) -> None:
        """Load current IMPRESS tokens not covered by prior-layer speculation."""

        if (
            not self.online_selection
            or self.method != "impress"
            or not self.impress_async_prefetch
        ):
            return
        if layer not in self._resolved_speculative:
            return
        speculative = self._speculative_positions.get(layer, [])
        speculative_storage = self._storage_positions(layer, speculative)
        covered_chunks = {
            token // self._config.chunk_size
            for token in physical_token_ids_for_positions(
                speculative_storage,
                chunk_size=self._config.chunk_size,
                prefix_tokens=self._info.prefix_tokens,
            )
        }
        missing = [
            token
            for token in self._selected_by_layer[layer]
            if self._storage_positions(layer, [token])[0] // self._config.chunk_size
            not in covered_chunks
        ]
        self._inter_period_missing_tokens += len(missing)
        kind = self._speculative_kind.get(layer, "next")
        if kind in self._speculative_physical_stats:
            self._speculative_physical_stats[kind]["missing"] += len(missing)
        self._submit_positions(
            layer,
            missing,
            self._missing_pending,
            prefetch_kind="current",
        )

    def resolve_impress_speculation(self, layer: int) -> None:
        """Resolve prior-layer IMPRESS prefetch before selecting current tokens."""

        if (
            not self.online_selection
            or self.method != "impress"
            or not self.impress_async_prefetch
        ):
            return
        entry = self._speculative_pending.pop(layer, None)
        if entry is None:
            return
        segment = self._resolve_entry(layer, entry)
        self._resolved_speculative[layer] = segment
        self._speculative_positions[layer] = segment[2].tolist()

    def schedule_impress_next(self, layer: int) -> None:
        """Speculatively load this layer's IMPRESS tokens for the next layer."""

        if (
            not self.online_selection
            or self.method != "impress"
            or not self.impress_async_prefetch
        ):
            return
        target = layer + 1
        if target >= self.layers or self._impress_target_is_scheduled(target):
            return
        value_ordered = getattr(self, "impress_value_ordered_prefetch", False)
        selected = list(
            self._prefetch_priority_by_layer[layer]
            if value_ordered
            else self._selected_by_layer[layer]
        )
        if value_ordered and selected != sorted(selected):
            self._impress_value_ordered_prefetch_jobs += 1
        self._track_speculative_request(
            target=target,
            positions=selected,
            kind="next",
            source_layer=layer,
        )
        budget = self._impress_prefetch_time_budget
        if value_ordered:
            budget *= self.impress_value_prefetch_budget_scale
        self._impress_prefetch_budgets.append(budget)
        self._submit_positions(
            target,
            selected,
            self._speculative_pending,
            time_budget=budget,
            prefetch_kind="next",
        )
        self._impress_next_prefetch_jobs += 1

    def _impress_target_is_scheduled(self, layer: int) -> bool:
        return (
            layer in self._pending
            or layer in self._speculative_pending
            or layer in self._resolved_speculative
            or layer in self._missing_pending
            or layer in self._loaded
        )

    def schedule_impress_period(self, layer: int, period_size: int) -> None:
        """Queue one bounded-lookahead prediction from the current Period leader."""

        if (
            not self.online_selection
            or self.method != "impress"
            or not self.impress_async_prefetch
            or period_size <= 1
        ):
            return
        if not 0 <= layer < self.layers:
            return
        leader = layer - (layer % period_size)
        target = layer + 2
        period_end = min(self.layers, leader + period_size)
        if target >= period_end or self._impress_target_is_scheduled(target):
            return
        source_layer = (
            layer
            if getattr(self, "impress_rolling_period_prefetch", False)
            else leader
        )
        value_ordered = getattr(self, "impress_value_ordered_prefetch", False)
        selected = list(
            self._prefetch_priority_by_layer[source_layer]
            if value_ordered
            else self._selected_by_layer[source_layer]
        )
        if value_ordered and selected != sorted(selected):
            self._impress_value_ordered_prefetch_jobs += 1
        if not selected:
            raise RuntimeError(
                f"period prediction source layer {source_layer} has no selected tokens"
            )
        self._track_speculative_request(
            target=target,
            positions=selected,
            kind="period",
            source_layer=source_layer,
        )
        budget = (
            self._impress_prefetch_time_budget
            * self.impress_period_prefetch_budget_scale
        )
        if value_ordered:
            budget *= self.impress_value_prefetch_budget_scale
        self._impress_prefetch_budgets.append(budget)
        self._submit_positions(
            target,
            selected,
            self._speculative_pending,
            time_budget=budget,
            prefetch_kind="period",
        )
        self._impress_period_prefetch_jobs += 1
        self._impress_period_prefetch_tokens += len(selected)

    def record_impress_layer_compute(self, elapsed_ms: float) -> None:
        """Set HyperInfer's next-layer budget from selector plus compute time."""

        if (
            not self.online_selection
            or self.method != "impress"
            or not self.impress_async_prefetch
        ):
            return
        elapsed = float(elapsed_ms)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("IMPRESS layer compute time must be finite and non-negative")
        self._impress_prefetch_time_budget = (
            elapsed + self._last_selector_compute_ms
        ) / 1000.0

    def defer_impress_layer_compute(self, started: Any, finished: Any) -> None:
        """Queue CUDA event timing without forcing a per-layer host barrier."""

        if not self.impress_deferred_compute_timing:
            raise RuntimeError("deferred IMPRESS compute timing is not enabled")
        self._impress_compute_event_pairs.append((started, finished))
        self._impress_deferred_compute_pending_max = max(
            self._impress_deferred_compute_pending_max,
            len(self._impress_compute_event_pairs),
        )

    def resolve_deferred_impress_compute(self, *, wait: bool = False) -> int:
        """Harvest completed CUDA timings and update the next prefetch budget."""

        resolved = 0
        while self._impress_compute_event_pairs:
            started, finished = self._impress_compute_event_pairs[0]
            if wait:
                finished.synchronize()
            elif not finished.query():
                break
            self._impress_compute_event_pairs.pop(0)
            self.record_impress_layer_compute(started.elapsed_time(finished))
            self._impress_deferred_compute_samples += 1
            resolved += 1
        return resolved

    def _update_loaded_layer_score(
        self, layer: int, *, force: bool = False
    ) -> None:
        """Update residency now or queue it beyond the TTFT boundary."""

        updater = getattr(self, "_cache_score_updater", None)
        updated_layers = getattr(self, "_cache_score_updated_layers", set())
        if updater is None or layer in updated_layers:
            return
        if getattr(self, "defer_cache_score_updates", False) and not force:
            if layer not in self._deferred_cache_score_layer_set:
                self._deferred_cache_score_layers.append(layer)
                self._deferred_cache_score_layer_set.add(layer)
            return
        if self.method == "contigkv":
            scores = self._online_chunk_scores[layer]
            if scores is None:
                raise RuntimeError(f"layer {layer} has no online ContiguousKV scores")
            selected = sorted(
                {token // self._config.chunk_size for token in self._selected_by_layer[layer]}
            )
        else:
            scores = None
            selected = self._selected_by_layer[layer]

        started = time.perf_counter()
        updates = updater(layer, selected, scores)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._cache_score_updated_layers.add(layer)
        self._cache_score_updates += int(updates)
        self._cache_update_ms += elapsed_ms

    def flush_deferred_cache_score_updates(self) -> int:
        """Apply metadata-only CKLFU updates after first-token production."""

        if not self.defer_cache_score_updates:
            return 0
        layers = list(self._deferred_cache_score_layers)
        self._deferred_cache_score_layers.clear()
        self._deferred_cache_score_layer_set.clear()
        started = time.perf_counter()
        for layer in layers:
            self._update_loaded_layer_score(layer, force=True)
        self._cache_update_deferred_ms += (
            time.perf_counter() - started
        ) * 1000
        return len(layers)

    def resolve(self, layer: int) -> tuple[Any, Any]:
        """Wait for a layer's physical prefetch and gather the sparse plan."""

        if layer in self._loaded:
            return self._loaded[layer]
        if self.method == "impress" and (
            not self.online_selection or not self.impress_async_prefetch
        ):
            return self._resolve_impress(layer)
        segments = []
        if layer in self._resolved_speculative or layer in self._speculative_pending:
            if layer in self._resolved_speculative:
                speculative = self._resolved_speculative.pop(layer)
            else:
                speculative = self._resolve_entry(layer, self._speculative_pending.pop(layer))
            segments.append(speculative)
            physical_ids = set(speculative[2].tolist())
            useful_ids = set(self._selected_by_layer[layer])
            hit_tokens = len(physical_ids & useful_ids)
            unused_tokens = len(physical_ids - useful_ids)
            self._inter_period_hit_tokens += hit_tokens
            self._inter_period_unused_tokens += unused_tokens
            kind = self._speculative_kind.pop(layer, "next")
            if kind in self._speculative_physical_stats:
                self._speculative_physical_stats[kind]["hit"] += hit_tokens
                self._speculative_physical_stats[kind]["unused"] += unused_tokens
            if layer in self._missing_pending:
                segments.append(self._resolve_entry(layer, self._missing_pending.pop(layer)))
        else:
            self.schedule(layer)
            segments.append(self._resolve_entry(layer, self._pending.pop(layer)))

        if len(segments) == 1:
            physical_key, physical_value, physical_ids = segments[0]
        else:
            import torch

            physical_ids = torch.cat([segment[2] for segment in segments], dim=0)
            order = torch.argsort(physical_ids)
            if int(torch.unique(physical_ids).numel()) != int(physical_ids.numel()):
                raise RuntimeError(f"layer {layer} received duplicate speculative and missing tokens")
            physical_ids = physical_ids.index_select(0, order)
            physical_key = torch.cat([segment[0] for segment in segments], dim=0).index_select(
                0, order.to(segments[0][0].device)
            )
            physical_value = torch.cat([segment[1] for segment in segments], dim=0).index_select(
                0, order.to(segments[0][1].device)
            )

        key, value = gather_prefetched_tokens(
            physical_key,
            physical_value,
            physical_ids,
            self._selected_by_layer[layer],
        )
        self._update_loaded_layer_score(layer)
        self._loaded[layer] = (key, value)
        return key, value

    def _resolve_impress(self, layer: int) -> tuple[Any, Any]:
        """Use HyperInfer's synchronous get path for the no-prefetch baseline."""

        import torch

        selected = self._selected_by_layer[layer]
        storage_selected = self._storage_positions(layer, selected)
        physical_ids = physical_token_ids_for_positions(
            storage_selected,
            chunk_size=self._config.chunk_size,
            prefix_tokens=self._info.prefix_tokens,
        )
        self._record_physical_access(layer, physical_ids)
        positions = torch.tensor(storage_selected, dtype=torch.long)
        started = time.perf_counter()
        key, value = self._pcache.get(
            prefix_id=self._prefix_id,
            pos_id=positions,
            layer=layer,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._prefetch_wait_ms += elapsed_ms
        self._prefetch_elapsed_ms += elapsed_ms
        if int(key.shape[0]) != len(selected) or int(value.shape[0]) != len(selected):
            raise RuntimeError(
                f"HyperInfer synchronous get returned {key.shape[0]}/{value.shape[0]} "
                f"tokens for {len(selected)} selected positions"
            )
        self._update_loaded_layer_score(layer)
        self._loaded[layer] = (key, value)
        return key, value

    def _resolve_entry(self, layer: int, entry: tuple[Any, float]) -> tuple[Any, Any, Any]:
        handle, issued_at = entry
        wait_start = time.perf_counter()
        physical_key, physical_value, physical_ids = handle.result()
        finished_at = time.perf_counter()
        self._prefetch_wait_ms += (finished_at - wait_start) * 1000
        self._prefetch_elapsed_ms += (finished_at - issued_at) * 1000
        physical_ids = physical_ids.detach().cpu()
        physical_ids_cpu = physical_ids.tolist()
        self._record_physical_access(layer, physical_ids_cpu)
        return self._logical_segment(
            layer, physical_key, physical_value, physical_ids
        )

    def _record_physical_access(self, layer: int, physical_ids: Sequence[int]) -> None:
        source_counts = prefetch_source_tensor_tokens(
            self._pcache,
            prefix_id=self._prefix_id,
            layer=layer,
            token_ids=physical_ids,
            chunk_size=self._config.chunk_size,
            prefix_tokens=self._info.prefix_tokens,
        )
        for source, count in source_counts.items():
            self._source_tensor_tokens[source] += count
        self._physical_tokens += len(physical_ids)
        self._physical_chunks.update(
            (layer, token_id // self._config.chunk_size) for token_id in physical_ids
        )

    def _prefetch_scheduler_metrics_delta(self) -> dict[str, float | int]:
        """Return per-request executor metrics without prior-request carryover."""

        metric_names = (
            "submitted",
            "started",
            "completed",
            "failed",
            "cancelled",
            "queue_wait_ms",
            "execution_ms",
        )
        labels = ("current", "next", "period", "default")
        snapshot = getattr(self._pcache, "prefetch_scheduler_metrics", None)
        current = snapshot() if callable(snapshot) else {}
        initial = getattr(self, "_prefetch_scheduler_start", {})
        result: dict[str, float | int] = {}
        for metric_name in metric_names:
            total = 0.0
            for label in labels:
                end_value = float(current.get(label, {}).get(metric_name, 0.0))
                start_value = float(initial.get(label, {}).get(metric_name, 0.0))
                delta = max(0.0, end_value - start_value)
                key = f"prefetch_scheduler_{label}_{metric_name}"
                result[key] = (
                    int(round(delta))
                    if metric_name
                    in {"submitted", "started", "completed", "failed", "cancelled"}
                    else delta
                )
                total += delta
            total_key = f"prefetch_scheduler_total_{metric_name}"
            result[total_key] = (
                int(round(total))
                if metric_name
                in {"submitted", "started", "completed", "failed", "cancelled"}
                else total
            )
        return result

    def metrics(self) -> dict[str, Any]:
        if (
            self._layer_keep_blocks is not None
            and self._exact_block_budget_consumed != self._exact_block_budget_target
        ):
            raise RuntimeError(
                "online selection did not consume the exact global block budget"
            )
        selected_tokens = sum(len(tokens) for tokens in self._selected_by_layer)
        bytes_per_tensor_token = self._info.bytes_per_token_per_tensor
        bytes_per_token = bytes_per_tensor_token * 2
        source_total = sum(self._source_tensor_tokens.values())
        selector_source_total = sum(self._selector_source_bytes.values())
        critical_ssd_bytes = self._source_tensor_tokens["disk"] * bytes_per_tensor_token
        prediction_metrics: dict[str, float | int] = {}
        for kind in ("next", "period"):
            prediction = self._speculative_prediction_stats[kind]
            physical = self._speculative_physical_stats[kind]
            jaccards = prediction["jaccards"]
            layer_distances = prediction["layer_distances"]
            prefix = f"impress_{kind}_prediction"
            prediction_metrics.update(
                {
                    f"{prefix}_layers": prediction["layers"],
                    f"{prefix}_requested_tokens": prediction["requested"],
                    f"{prefix}_actual_tokens": prediction["actual"],
                    f"{prefix}_overlap_tokens": prediction["overlap"],
                    f"{prefix}_precision": prediction["overlap"]
                    / max(1, prediction["requested"]),
                    f"{prefix}_recall": prediction["overlap"]
                    / max(1, prediction["actual"]),
                    f"{prefix}_mean_jaccard": sum(jaccards)
                    / max(1, len(jaccards)),
                    f"{prefix}_mean_layer_distance": sum(layer_distances)
                    / max(1, len(layer_distances)),
                    f"impress_{kind}_prefetch_hit_tokens": physical["hit"],
                    f"impress_{kind}_prefetch_missing_tokens": physical["missing"],
                    f"impress_{kind}_prefetch_unused_tokens": physical["unused"],
                }
            )
        return {
            "cache_backend": "flexgen_pcache",
            "disk_type": "KV_Division",
            "cache_chunk_size": self._config.chunk_size,
            "selected_tokens": selected_tokens,
            "selected_tokens_by_layer": [
                len(tokens) for tokens in self._selected_by_layer
            ],
            "layer_keep_ratios": [
                self.keep_ratio_for_layer(layer) for layer in range(self.layers)
            ]
            if self.online_selection
            else [],
            "exact_layer_keep_blocks": (
                list(self._layer_keep_blocks)
                if self._layer_keep_blocks is not None
                else []
            ),
            "exact_block_budget_target": self._exact_block_budget_target,
            "exact_block_budget_consumed": (
                self._exact_block_budget_consumed
                if self._layer_keep_blocks is not None
                else None
            ),
            "effective_mean_keep_ratio": selected_tokens
            / max(1, self.layers * self._info.prefix_tokens),
            "selected_kv_bytes": selected_tokens * bytes_per_token,
            "physical_prefetch_tokens": self._physical_tokens,
            "physical_prefetch_chunks": len(self._physical_chunks),
            "physical_prefetch_kv_bytes": self._physical_tokens * bytes_per_token,
            "read_amplification": self._physical_tokens / max(1, selected_tokens),
            "prefetch_gpu_source_tensor_tokens": self._source_tensor_tokens["gpu"],
            "prefetch_cpu_source_tensor_tokens": self._source_tensor_tokens["cpu"],
            "prefetch_disk_source_tensor_tokens": self._source_tensor_tokens["disk"],
            "prefetch_gpu_source_fraction": self._source_tensor_tokens["gpu"] / max(1, source_total),
            "prefetch_cpu_source_fraction": self._source_tensor_tokens["cpu"] / max(1, source_total),
            "prefetch_disk_source_fraction": self._source_tensor_tokens["disk"] / max(1, source_total),
            "critical_ssd_read_bytes": critical_ssd_bytes,
            "selector_key_bytes": self._selector_key_bytes,
            "selector_dequantized_key_bytes": self._selector_dequantized_key_bytes,
            "selector_index_enabled": int(self._selector_index is not None),
            "selector_index_bits": self._selector_index.bits if self._selector_index is not None else 0,
            "selector_index_group_size": self._selector_index_group_size,
            "selector_index_preloaded": int(
                self._selector_index is not None
                and self._selector_index_task is not None
                and self._selector_index.task_is_preloaded(
                    self._selector_index_task
                )
            ),
            "selector_index_preloaded_bytes": (
                self._selector_index.preloaded_compressed_bytes
                if self._selector_index is not None
                else 0
            ),
            "selector_gpu_source_bytes": self._selector_source_bytes["gpu"],
            "selector_cpu_source_bytes": self._selector_source_bytes["cpu"],
            "selector_disk_source_bytes": self._selector_source_bytes["disk"],
            "selector_gpu_source_fraction": self._selector_source_bytes["gpu"]
            / max(1, selector_source_total),
            "selector_cpu_source_fraction": self._selector_source_bytes["cpu"]
            / max(1, selector_source_total),
            "selector_disk_source_fraction": self._selector_source_bytes["disk"]
            / max(1, selector_source_total),
            "selector_load_ms": self._selector_load_ms,
            "selector_compute_ms": self._selector_compute_ms,
            "selector_wait_ms": self._selector_wait_ms,
            "selector_calls": self._selector_calls,
            "selector_fallbacks": self._selector_fallbacks,
            "selector_mean_jaccard": sum(self._selector_similarities)
            / max(1, len(self._selector_similarities)),
            "promixed_decisions": len(self._promixed_periods),
            "promixed_mean_gqa_agreement": sum(self._promixed_agreements)
            / max(1, len(self._promixed_agreements)),
            "promixed_mean_boundary_margin": sum(
                self._promixed_boundary_margins
            )
            / max(1, len(self._promixed_boundary_margins)),
            "promixed_mean_uncertainty": sum(self._promixed_uncertainties)
            / max(1, len(self._promixed_uncertainties)),
            "promixed_mean_period": sum(self._promixed_periods)
            / max(1, len(self._promixed_periods)),
            "promixed_p1_decisions": self._promixed_periods.count(1),
            "promixed_p2_decisions": self._promixed_periods.count(2),
            "promixed_p4_decisions": self._promixed_periods.count(4),
            "promixed_p8_decisions": self._promixed_periods.count(8),
            "impress_mean_prefetch_budget_seconds": sum(self._impress_prefetch_budgets)
            / max(1, len(self._impress_prefetch_budgets)),
            "impress_period_prefetch_size": self.impress_period_prefetch_size,
            "impress_period_prefetch_budget_scale": (
                self.impress_period_prefetch_budget_scale
            ),
            "impress_next_prefetch_jobs": self._impress_next_prefetch_jobs,
            "impress_period_prefetch_jobs": self._impress_period_prefetch_jobs,
            "impress_period_prefetch_tokens": self._impress_period_prefetch_tokens,
            "impress_deferred_compute_timing": int(
                self.impress_deferred_compute_timing
            ),
            "impress_deferred_compute_samples": (
                self._impress_deferred_compute_samples
            ),
            "impress_deferred_compute_pending": len(
                self._impress_compute_event_pairs
            ),
            "impress_deferred_compute_pending_max": (
                self._impress_deferred_compute_pending_max
            ),
            "impress_rolling_period_prefetch": int(
                self.impress_rolling_period_prefetch
            ),
            "impress_value_ordered_prefetch": int(
                self.impress_value_ordered_prefetch
            ),
            "impress_value_prefetch_budget_scale": (
                self.impress_value_prefetch_budget_scale
            ),
            "impress_value_prefetch_budget_scaled": int(
                self.impress_value_prefetch_budget_scale != 1.0
            ),
            "impress_value_ordered_prefetch_jobs": (
                self._impress_value_ordered_prefetch_jobs
            ),
            "total_ssd_read_bytes": critical_ssd_bytes + self._selector_source_bytes["disk"],
            "inter_period_hit_tokens": self._inter_period_hit_tokens,
            "inter_period_missing_tokens": self._inter_period_missing_tokens,
            "inter_period_unused_tokens": self._inter_period_unused_tokens,
            "prefetch_wait_ms": self._prefetch_wait_ms,
            "prefetch_elapsed_ms": self._prefetch_elapsed_ms,
            "cache_score_updates": self._cache_score_updates,
            "cache_update_ms": self._cache_update_ms,
            "cache_update_deferred_ms": self._cache_update_deferred_ms,
            "cache_update_in_ttft": int(
                self._cache_score_updater is not None
                and not self.defer_cache_score_updates
            ),
            "cache_update_deferred": int(self.defer_cache_score_updates),
            **prediction_metrics,
            **self._prefetch_scheduler_metrics_delta(),
        }

    def close(self) -> None:
        """Join per-request selector work and release its worker thread."""

        if self._deferred_cache_score_layers:
            self.flush_deferred_cache_score_updates()
        if self.impress_deferred_compute_timing:
            self.resolve_deferred_impress_compute(wait=True)
        self._selector_executor.shutdown(wait=True)
