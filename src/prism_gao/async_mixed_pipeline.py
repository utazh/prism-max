
from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Hashable, Optional, Tuple, TypeVar

import torch


PlanT = TypeVar("PlanT")
PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class MixedCostCalibration:
    """
    Small measured cost model.

    `exposed_io_fraction` is the fraction of the FP16 host-read time that was
    visible on the critical path in the asynchronous FP16 baseline.
    """

    io_gib_per_s: float
    dequant_giga_elements_per_s: float
    launch_overhead_us: float = 8.0
    exposed_io_fraction: float = 1.0
    minimum_gain_ms: float = 0.05


class MixedPrecisionGate:
    """Disable INT8 for a layer when the measured cost model predicts no gain."""

    def __init__(self, calibration: MixedCostCalibration) -> None:
        self.c = calibration

    def predicted_gain_ms(
        self,
        *,
        fp16_bytes: int,
        mixed_bytes: int,
        int8_elements: int,
        int8_runs: int,
    ) -> float:
        if fp16_bytes < mixed_bytes:
            return float("-inf")
        io_bytes_per_s = self.c.io_gib_per_s * (1024.0 ** 3)
        saved_io_ms = (
            (fp16_bytes - mixed_bytes) / io_bytes_per_s * 1000.0
            * max(0.0, min(1.0, self.c.exposed_io_fraction))
        )
        dequant_ms = (
            int8_elements
            / (self.c.dequant_giga_elements_per_s * 1e9)
            * 1000.0
        )
        launch_ms = max(0, int8_runs) * self.c.launch_overhead_us / 1000.0
        return saved_io_ms - dequant_ms - launch_ms

    def keep_mixed(self, **kwargs: int) -> bool:
        return self.predicted_gain_ms(**kwargs) >= self.c.minimum_gain_ms


def pread_into_pinned(
    fd: int,
    *,
    nbytes: int,
    offset: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Read one contiguous file range directly into pinned CPU storage.

    This avoids the `bytes -> NumPy -> pinned tensor` copy and makes the later
    H2D transfer genuinely eligible for `non_blocking=True`.
    """
    if nbytes < 0 or offset < 0:
        raise ValueError("nbytes and offset must be non-negative")
    if out is None:
        out = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
    elif (
        out.dtype != torch.uint8
        or not out.is_pinned()
        or not out.is_contiguous()
        or int(out.numel()) != int(nbytes)
    ):
        raise ValueError("out must be contiguous pinned uint8 with exactly nbytes elements")
    if nbytes == 0:
        return out

    view = memoryview(out.numpy()).cast("B")
    completed = 0
    while completed < nbytes:
        count = os.preadv(fd, [view[completed:]], offset + completed)
        if count == 0:
            raise EOFError(
                f"short pread: requested={nbytes}, completed={completed}, offset={offset}"
            )
        completed += count
    return out


@dataclass
class _PendingHost(Generic[PlanT, PayloadT]):
    plan: PlanT
    future: Future[Tuple[PayloadT, float]]
    submitted_at: float


@dataclass
class _ReadyGpu(Generic[ResultT]):
    result: ResultT
    ready_event: torch.cuda.Event
    host_read_ms: float
    submitted_at: float


class AsyncMixedPipeline(Generic[PlanT, PayloadT, ResultT]):
    """
    Minimal two-stage pipeline:

      host worker: SSD/pread and CPU packing
      CUDA stream: non-blocking H2D and mixed dequant/materialisation

    CUDA work is enqueued from the model thread through `poll()`/`resolve()`.
    No CUDA operation is issued by the Python worker thread.

    Contracts:
      read_host(layer, plan) -> payload
      materialize_gpu(payload, plan, stream) -> result

    `materialize_gpu` must enqueue all copies/kernels on `stream`, use pinned
    host inputs for non-blocking H2D, and must not call synchronize().
    """

    def __init__(
        self,
        *,
        read_host: Callable[[int, PlanT], PayloadT],
        materialize_gpu: Callable[
            [PayloadT, PlanT, torch.cuda.Stream], ResultT
        ],
        device: torch.device | str = "cuda",
        max_workers: int = 1,
        stream_priority: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("AsyncMixedPipeline requires CUDA")
        self.device = torch.device(device)
        self.read_host = read_host
        self.materialize_gpu = materialize_gpu
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mixed-kv-host",
        )
        self.prefetch_stream = torch.cuda.Stream(
            device=self.device,
            priority=stream_priority,
        )
        self._pending: Dict[Hashable, _PendingHost[PlanT, PayloadT]] = {}
        self._ready: Dict[Hashable, _ReadyGpu[ResultT]] = {}

    def submit(self, key: Hashable, *, layer: int, plan: PlanT) -> None:
        if key in self._pending or key in self._ready:
            return

        def job() -> Tuple[PayloadT, float]:
            started = time.perf_counter()
            payload = self.read_host(int(layer), plan)
            return payload, (time.perf_counter() - started) * 1000.0

        self._pending[key] = _PendingHost(
            plan=plan,
            future=self.executor.submit(job),
            submitted_at=time.perf_counter(),
        )

    def _stage_one(
        self,
        key: Hashable,
        pending: _PendingHost[PlanT, PayloadT],
        *,
        block: bool,
    ) -> bool:
        if not block and not pending.future.done():
            return False
        payload, host_read_ms = pending.future.result()
        with torch.cuda.device(self.device), torch.cuda.stream(self.prefetch_stream):
            result = self.materialize_gpu(
                payload,
                pending.plan,
                self.prefetch_stream,
            )
            event = torch.cuda.Event(enable_timing=False, blocking=False)
            event.record(self.prefetch_stream)
        self._ready[key] = _ReadyGpu(
            result=result,
            ready_event=event,
            host_read_ms=host_read_ms,
            submitted_at=pending.submitted_at,
        )
        del self._pending[key]
        return True

    def poll(self) -> int:
        """Enqueue every completed host payload on the CUDA prefetch stream."""
        staged = 0
        for key, pending in list(self._pending.items()):
            staged += int(self._stage_one(key, pending, block=False))
        return staged

    def resolve(self, key: Hashable) -> ResultT:
        """
        Return the prepared K/V object.

        If preparation is still in flight, the current CUDA stream waits on a
        CUDA event.  There is deliberately no event.synchronize() and no
        torch.cuda.synchronize().
        """
        self.poll()
        if key not in self._ready:
            pending = self._pending.get(key)
            if pending is None:
                raise KeyError(f"no mixed-KV request for key={key!r}")
            self._stage_one(key, pending, block=True)

        ready = self._ready.pop(key)
        current = torch.cuda.current_stream(self.device)
        current.wait_event(ready.ready_event)
        return ready.result

    def cancel(self, key: Hashable) -> None:
        pending = self._pending.pop(key, None)
        if pending is not None:
            pending.future.cancel()
        self._ready.pop(key, None)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._ready.clear()

    def __enter__(self) -> "AsyncMixedPipeline[PlanT, PayloadT, ResultT]":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
