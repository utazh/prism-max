
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Any

import torch


@dataclass(frozen=True)
class PackedSelectorLayer:
    """Packed INT4 selector payload for one transformer layer."""

    codes: torch.Tensor
    scales: torch.Tensor
    meta: Mapping[str, Any]

    @property
    def nbytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


def _pin_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


class SelectorResidentCache:
    """
    Cache packed INT4 selector tensors once per persistent prefix.

    The fast path stores all packed codes/scales on the GPU.  If the byte
    budget is insufficient, the fallback stores them in pinned CPU memory so
    H2D copies can be non-blocking.

    `load_cpu(layer_id)` must return either:
      * PackedSelectorLayer
      * (codes, scales)
      * (codes, scales, meta)
    """

    def __init__(
        self,
        load_cpu: Callable[[int], Any],
        *,
        device: torch.device | str = "cuda",
        max_gpu_bytes: Optional[int] = None,
        pin_fallback: bool = True,
    ) -> None:
        self._load_cpu = load_cpu
        self.device = torch.device(device)
        self.max_gpu_bytes = max_gpu_bytes
        self.pin_fallback = bool(pin_fallback)

        self._gpu: Dict[int, PackedSelectorLayer] = {}
        self._cpu: Dict[int, PackedSelectorLayer] = {}
        self.mode: str = "empty"
        self.total_bytes: int = 0

    @staticmethod
    def _normalise(value: Any) -> PackedSelectorLayer:
        if isinstance(value, PackedSelectorLayer):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            codes, scales = value
            return PackedSelectorLayer(codes=codes, scales=scales, meta={})
        if isinstance(value, tuple) and len(value) == 3:
            codes, scales, meta = value
            return PackedSelectorLayer(codes=codes, scales=scales, meta=dict(meta))
        raise TypeError(
            "load_cpu(layer_id) must return PackedSelectorLayer, "
            "(codes, scales), or (codes, scales, meta)"
        )

    def preload(self, layer_ids: Iterable[int]) -> str:
        """Load every requested layer once. Returns ``gpu`` or ``pinned``."""
        staged: Dict[int, PackedSelectorLayer] = {}
        total = 0
        for layer_id in layer_ids:
            layer = self._normalise(self._load_cpu(int(layer_id)))
            layer = PackedSelectorLayer(
                codes=layer.codes.detach().contiguous().cpu(),
                scales=layer.scales.detach().contiguous().cpu(),
                meta=layer.meta,
            )
            staged[int(layer_id)] = layer
            total += layer.nbytes

        self.total_bytes = total
        if self.max_gpu_bytes is None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                # Conservative default: at most 10% of currently free memory.
                budget = int(free_bytes * 0.10)
            else:
                budget = 0
        else:
            budget = int(self.max_gpu_bytes)

        if (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and total <= budget
        ):
            self._gpu = {
                layer_id: PackedSelectorLayer(
                    codes=layer.codes.to(self.device, non_blocking=False),
                    scales=layer.scales.to(self.device, non_blocking=False),
                    meta=layer.meta,
                )
                for layer_id, layer in staged.items()
            }
            self._cpu.clear()
            self.mode = "gpu"
            return self.mode

        if not self.pin_fallback:
            self._cpu = staged
            self._gpu.clear()
            self.mode = "cpu"
            return self.mode

        self._cpu = {
            layer_id: PackedSelectorLayer(
                codes=_pin_if_needed(layer.codes),
                scales=_pin_if_needed(layer.scales),
                meta=layer.meta,
            )
            for layer_id, layer in staged.items()
        }
        self._gpu.clear()
        self.mode = "pinned"
        return self.mode

    def get(
        self,
        layer_id: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> PackedSelectorLayer:
        """
        Return packed tensors on the target device.

        GPU-resident mode performs no allocation and no transfer.  Pinned mode
        performs non-blocking H2D copies on `stream` when supplied.
        """
        layer_id = int(layer_id)
        if layer_id in self._gpu:
            return self._gpu[layer_id]
        if layer_id not in self._cpu:
            raise KeyError(f"selector layer {layer_id} was not preloaded")

        layer = self._cpu[layer_id]
        if self.device.type != "cuda":
            return layer

        ctx = (
            torch.cuda.stream(stream)
            if stream is not None
            else torch.cuda.device(self.device)
        )
        with ctx:
            return PackedSelectorLayer(
                codes=layer.codes.to(self.device, non_blocking=True),
                scales=layer.scales.to(self.device, non_blocking=True),
                meta=layer.meta,
            )

    def clear(self) -> None:
        self._gpu.clear()
        self._cpu.clear()
        self.mode = "empty"
        self.total_bytes = 0


class TensorWorkspaceCache:
    """Small reusable tensor workspace keyed by shape/dtype/device."""

    def __init__(self) -> None:
        self._items: MutableMapping[Tuple[Tuple[int, ...], torch.dtype, str], torch.Tensor] = {}

    def get(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        shape_tuple = tuple(int(x) for x in shape)
        device_obj = torch.device(device)
        key = (shape_tuple, dtype, str(device_obj))
        out = self._items.get(key)
        if out is None:
            out = torch.empty(shape_tuple, dtype=dtype, device=device_obj)
            self._items[key] = out
        return out

    def clear(self) -> None:
        self._items.clear()
