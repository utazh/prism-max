"""Small Re-Prefill latency and read-amplification simulator.

The simulator is not a replacement for a full inference engine. It gives a
reproducible way to test the paper's systems claims without depending on a
specific vLLM/FlexGen fork: coarse physical chunks amplify reads, while
ContiguousChunk plus Period reuse lets later-layer I/O overlap computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

from .core import inter_period_prefetch_delta


@dataclass(frozen=True)
class ReadAmplification:
    useful_tokens: int
    physical_tokens_read: int
    amplification: float


@dataclass(frozen=True)
class LayerRequest:
    layer: int
    chunks: set[int]


@dataclass(frozen=True)
class KVShape:
    num_layers: int
    prefix_tokens: int
    contiguous_chunk_size: int = 16

    @property
    def num_contiguous_chunks(self) -> int:
        return ceil(self.prefix_tokens / self.contiguous_chunk_size)


@dataclass(frozen=True)
class PrefetchSimulation:
    total_ms: float
    unoptimized_ms: float
    compute_ms: float
    blocking_io_ms: float
    overlapped_io_ms: float
    periods: int


def estimate_read_amplification(
    selected_token_indices: Iterable[int],
    physical_chunk_size: int,
) -> ReadAmplification:
    """Compare useful selected tokens with full physical chunks read."""

    if physical_chunk_size <= 0:
        raise ValueError("physical_chunk_size must be positive")
    selected = set(selected_token_indices)
    useful = len(selected)
    if useful == 0:
        return ReadAmplification(0, 0, 1.0)
    physical_chunks = {idx // physical_chunk_size for idx in selected}
    physical = len(physical_chunks) * physical_chunk_size
    return ReadAmplification(useful, physical, physical / useful)


def simulate_contiguous_prefetch(
    requests: Sequence[LayerRequest],
    period_size: int,
    subperiod_size: int,
    chunk_load_ms: float,
    compute_ms: float,
) -> PrefetchSimulation:
    """Estimate Re-Prefill time with intra/inter Period prefetching.

    Model:
    - Serial baseline loads each layer's selected chunks, then computes.
    - First `subperiod_size` layers of a Period pay blocking I/O.
    - Later layers in the Period have their I/O overlapped with earlier compute.
    - At Period boundaries, chunks shared with the previous Period are assumed
      speculatively prefetched; only missing chunks block.
    """

    if period_size <= 0:
        raise ValueError("period_size must be positive")
    if subperiod_size <= 0:
        raise ValueError("subperiod_size must be positive")
    if chunk_load_ms < 0 or compute_ms < 0:
        raise ValueError("latencies must be non-negative")

    unoptimized_io = sum(len(req.chunks) * chunk_load_ms for req in requests)
    compute_total = len(requests) * compute_ms
    blocking_io = 0.0
    overlapped_io = 0.0
    periods = 0
    previous_period_chunks: set[int] = set()

    for start in range(0, len(requests), period_size):
        period = requests[start:start + period_size]
        if not period:
            continue
        periods += 1
        current_chunks = set(period[0].chunks)
        if start == 0:
            boundary_missing = current_chunks
        else:
            _, boundary_missing = inter_period_prefetch_delta(
                previous_period_chunks,
                current_chunks,
            )
        previous_period_chunks = set(current_chunks)

        for offset, req in enumerate(period):
            io_ms = len(req.chunks) * chunk_load_ms
            if offset == 0:
                blocking_io += len(boundary_missing) * chunk_load_ms
                overlapped_io += max(0.0, io_ms - len(boundary_missing) * chunk_load_ms)
            elif offset < subperiod_size:
                blocking_io += io_ms
            else:
                overlapped_io += io_ms

    total = compute_total + blocking_io
    return PrefetchSimulation(
        total_ms=total,
        unoptimized_ms=compute_total + unoptimized_io,
        compute_ms=compute_total,
        blocking_io_ms=blocking_io,
        overlapped_io_ms=overlapped_io,
        periods=periods,
    )


def _request_token_indices(
    request: LayerRequest,
    chunk_size: int,
    prefix_tokens: int,
) -> set[int]:
    out: set[int] = set()
    for chunk in request.chunks:
        start = chunk * chunk_size
        end = min(start + chunk_size, prefix_tokens)
        out.update(range(start, end))
    return out


def _sum_layer_read_amplification(
    requests: Sequence[LayerRequest],
    logical_chunk_size: int,
    physical_chunk_size: int,
    prefix_tokens: int,
) -> ReadAmplification:
    useful = 0
    physical = 0
    for request in requests:
        selected_tokens = _request_token_indices(
            request,
            logical_chunk_size,
            prefix_tokens,
        )
        ra = estimate_read_amplification(selected_tokens, physical_chunk_size)
        useful += ra.useful_tokens
        physical += ra.physical_tokens_read
    amplification = 1.0 if useful == 0 else physical / useful
    return ReadAmplification(useful, physical, amplification)


def compare_impress_contiguous(
    selected_requests: Sequence[LayerRequest],
    shape: KVShape,
    impress_chunk_size: int,
    budget_ratio: float,
    chunk_load_ms: float,
    compute_ms: float,
    period_size: int = 8,
    subperiod_size: int = 4,
) -> dict[str, dict[str, float]]:
    """Compare coarse IMPRESS-style reads with ContiguousKV-style reads."""

    del budget_ratio
    impress_ra = _sum_layer_read_amplification(
        selected_requests,
        logical_chunk_size=shape.contiguous_chunk_size,
        physical_chunk_size=impress_chunk_size,
        prefix_tokens=shape.prefix_tokens,
    )
    contig_ra = _sum_layer_read_amplification(
        selected_requests,
        logical_chunk_size=shape.contiguous_chunk_size,
        physical_chunk_size=shape.contiguous_chunk_size,
        prefix_tokens=shape.prefix_tokens,
    )
    sim = simulate_contiguous_prefetch(
        selected_requests,
        period_size=max(1, min(period_size, shape.num_layers)),
        subperiod_size=max(1, min(subperiod_size, shape.num_layers)),
        chunk_load_ms=chunk_load_ms,
        compute_ms=compute_ms,
    )
    return {
        "impress": {
            "physical_tokens_read": float(impress_ra.physical_tokens_read),
            "useful_tokens": float(impress_ra.useful_tokens),
            "read_amplification": impress_ra.amplification,
            "estimated_ms": len(selected_requests) * compute_ms
            + (impress_ra.physical_tokens_read / shape.contiguous_chunk_size) * chunk_load_ms,
        },
        "contiguous": {
            "physical_tokens_read": float(contig_ra.physical_tokens_read),
            "useful_tokens": float(contig_ra.useful_tokens),
            "read_amplification": contig_ra.amplification,
            "estimated_ms": sim.total_ms,
            "unoptimized_ms": sim.unoptimized_ms,
            "overlapped_io_ms": sim.overlapped_io_ms,
        },
    }
