"""A dependency-free, priority-aware single-worker executor."""

from __future__ import annotations

import heapq
import math
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Dict, Tuple


_METRIC_FIELDS = (
    "submitted",
    "started",
    "completed",
    "failed",
    "cancelled",
    "queue_wait_ms",
    "execution_ms",
)


def _empty_metrics() -> Dict[str, float]:
    return {field_name: 0 for field_name in _METRIC_FIELDS}


@dataclass(order=True)
class _WorkItem:
    priority: Real
    sequence: int
    submitted_at: float = field(compare=False)
    label: str = field(compare=False)
    future: Future = field(compare=False)
    fn: Callable[..., Any] = field(compare=False)
    args: Tuple[Any, ...] = field(compare=False)
    kwargs: Dict[str, Any] = field(compare=False)


class PriorityExecutor:
    """Execute submitted callables serially in priority order.

    Smaller numeric priority values run first. Submission order breaks ties.
    A running callable is never preempted; priority only reorders queued work.

    ``metrics_snapshot`` returns cumulative milliseconds. Queue wait is recorded
    only for tasks that start, while execution time includes successful and
    failed tasks.
    """

    def __init__(self, thread_name_prefix: str = "priority-executor") -> None:
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix:
            raise ValueError("thread_name_prefix must be a non-empty string")

        self._condition = threading.Condition()
        self._metrics_lock = threading.Lock()
        self._queue = []
        self._metrics: Dict[str, Dict[str, float]] = {}
        self._next_sequence = 0
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._run,
            name=f"{thread_name_prefix}-0",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        priority: Real = 0,
        label: str = "default",
        **kwargs: Any,
    ) -> Future:
        """Schedule ``fn(*args, **kwargs)`` and return its ``Future``."""

        if not callable(fn):
            raise TypeError("fn must be callable")
        if (
            not isinstance(priority, Real)
            or isinstance(priority, bool)
            or not math.isfinite(float(priority))
        ):
            raise ValueError("priority must be a finite real number")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")

        future = Future()
        submitted_at = time.perf_counter()

        with self._condition:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            sequence = self._next_sequence
            self._next_sequence += 1
            item = _WorkItem(
                priority=priority,
                sequence=sequence,
                submitted_at=submitted_at,
                label=label,
                future=future,
                fn=fn,
                args=args,
                kwargs=kwargs,
            )
            future.add_done_callback(
                lambda completed_future, task=item: self._record_cancellation(
                    task, completed_future
                )
            )
            with self._metrics_lock:
                metrics = self._metrics.setdefault(label, _empty_metrics())
                metrics["submitted"] += 1
            heapq.heappush(self._queue, item)
            self._condition.notify()

        return future

    def shutdown(self, wait: bool = True) -> None:
        """Reject new work and let already submitted work finish."""

        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

        if wait:
            if threading.current_thread() is self._worker:
                raise RuntimeError("worker thread cannot wait for its own shutdown")
            self._worker.join()

    def metrics_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a detached, thread-safe copy of cumulative metrics by label."""

        with self._metrics_lock:
            return {
                label: dict(metrics)
                for label, metrics in sorted(self._metrics.items())
            }

    def __enter__(self) -> "PriorityExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown(wait=True)

    def _record_cancellation(self, item: _WorkItem, future: Future) -> None:
        if not future.cancelled():
            return
        with self._metrics_lock:
            self._metrics[item.label]["cancelled"] += 1

    def _record_started(self, item: _WorkItem, started_at: float) -> None:
        with self._metrics_lock:
            metrics = self._metrics[item.label]
            metrics["started"] += 1
            metrics["queue_wait_ms"] += (
                started_at - item.submitted_at
            ) * 1000.0

    def _record_finished(
        self, item: _WorkItem, *, failed: bool, elapsed_ms: float
    ) -> None:
        with self._metrics_lock:
            metrics = self._metrics[item.label]
            metrics["failed" if failed else "completed"] += 1
            metrics["execution_ms"] += elapsed_ms

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._shutdown:
                    self._condition.wait()
                if not self._queue:
                    return
                item = heapq.heappop(self._queue)

            if not item.future.set_running_or_notify_cancel():
                continue

            started_at = time.perf_counter()
            self._record_started(item, started_at)
            try:
                result = item.fn(*item.args, **item.kwargs)
            except BaseException as error:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                self._record_finished(item, failed=True, elapsed_ms=elapsed_ms)
                item.future.set_exception(error)
            else:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                self._record_finished(item, failed=False, elapsed_ms=elapsed_ms)
                item.future.set_result(result)

