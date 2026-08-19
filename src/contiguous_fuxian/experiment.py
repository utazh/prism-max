"""Reproduction experiments for ContiguousKV.

This file provides a deterministic synthetic workload that mirrors the paper's
reported structure: Qwen-like layer counts, 16-token ContiguousChunks, Period
reuse, and a coarse 64-token IMPRESS-style physical block baseline.
"""

from __future__ import annotations

import json
import random
from math import ceil
from pathlib import Path
from typing import Any, Sequence

from .core import AttentionGuidedCache, jaccard_similarity, select_period_chunks
from .simulator import KVShape, LayerRequest, compare_impress_contiguous


def make_synthetic_layer_scores(
    num_layers: int,
    num_chunks: int,
    hot_chunks: int,
    period_size: int,
    seed: int,
) -> list[list[float]]:
    """Create deterministic chunk scores with adjacent-period similarity."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if hot_chunks <= 0:
        raise ValueError("hot_chunks must be positive")
    if period_size <= 0:
        raise ValueError("period_size must be positive")

    rng = random.Random(seed)
    previous_hot: set[int] = set()
    layers: list[list[float]] = []
    for layer in range(num_layers):
        if layer % period_size == 0:
            reuse_count = min(len(previous_hot), max(0, hot_chunks // 2))
            reused = set(rng.sample(sorted(previous_hot), reuse_count)) if reuse_count else set()
            candidates = [idx for idx in range(num_chunks) if idx not in reused]
            fresh = set(rng.sample(candidates, max(0, hot_chunks - len(reused))))
            previous_hot = reused | fresh
        scores = [rng.random() * 0.05 for _ in range(num_chunks)]
        for chunk in previous_hot:
            scores[chunk] += 1.0 + rng.random() * 0.2
        layers.append([round(score, 6) for score in scores])
    return layers


def _period_heads(selected: Sequence[set[int]], period_size: int) -> list[set[int]]:
    return [set(selected[idx]) for idx in range(0, len(selected), period_size)]


def _mean_adjacent_jaccard(period_sets: Sequence[set[int]]) -> float:
    if len(period_sets) < 2:
        return 1.0
    values = [
        jaccard_similarity(period_sets[idx - 1], period_sets[idx])
        for idx in range(1, len(period_sets))
    ]
    return sum(values) / len(values)


def run_synthetic_reproduction(
    prefix_tokens: int = 6000,
    num_layers: int = 28,
    contiguous_chunk_size: int = 16,
    impress_chunk_size: int = 64,
    keep_ratio: float = 0.05,
    period_size: int = 8,
    subperiod_size: int = 4,
    seed: int = 42,
    chunk_load_ms: float = 0.08,
    compute_ms: float = 1.0,
) -> dict[str, Any]:
    """Run a deterministic systems-level reproduction of ContiguousKV."""

    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    num_chunks = ceil(prefix_tokens / contiguous_chunk_size)
    keep_chunks = max(1, ceil(num_chunks * keep_ratio))
    layer_scores = make_synthetic_layer_scores(
        num_layers=num_layers,
        num_chunks=num_chunks,
        hot_chunks=keep_chunks,
        period_size=period_size,
        seed=seed,
    )
    selected = select_period_chunks(
        layer_scores,
        keep_chunks=keep_chunks,
        period_size=period_size,
    )
    requests = [
        LayerRequest(layer=layer_idx, chunks=set(chunks))
        for layer_idx, chunks in enumerate(selected)
    ]
    comparison = compare_impress_contiguous(
        requests,
        shape=KVShape(
            num_layers=num_layers,
            prefix_tokens=prefix_tokens,
            contiguous_chunk_size=contiguous_chunk_size,
        ),
        impress_chunk_size=impress_chunk_size,
        budget_ratio=keep_ratio,
        chunk_load_ms=chunk_load_ms,
        compute_ms=compute_ms,
        period_size=period_size,
        subperiod_size=subperiod_size,
    )
    speedup = comparison["impress"]["estimated_ms"] / max(
        1e-9,
        comparison["contiguous"]["estimated_ms"],
    )
    period_sets = _period_heads(selected, period_size)
    cache_capacity = max(1, keep_chunks * max(1, subperiod_size))
    cache = AttentionGuidedCache(capacity=cache_capacity)
    total_evictions = 0
    for layer_idx, chunks in enumerate(selected):
        for chunk in chunks:
            attention_score = layer_scores[layer_idx][chunk]
            total_evictions += len(cache.touch((layer_idx, chunk), attention_score))
    return {
        "config": {
            "prefix_tokens": prefix_tokens,
            "num_layers": num_layers,
            "contiguous_chunk_size": contiguous_chunk_size,
            "impress_chunk_size": impress_chunk_size,
            "keep_ratio": keep_ratio,
            "keep_chunks": keep_chunks,
            "period_size": period_size,
            "subperiod_size": subperiod_size,
            "seed": seed,
            "chunk_load_ms": chunk_load_ms,
            "compute_ms": compute_ms,
        },
        "impress": comparison["impress"],
        "contiguous": comparison["contiguous"],
        "metrics": {
            "speedup_vs_impress": speedup,
            "mean_adjacent_period_jaccard": _mean_adjacent_jaccard(period_sets),
            "periods": len(period_sets),
        },
        "attention_cache": {
            "policy": "S=I*F",
            "capacity": cache_capacity,
            "resident_count": len(cache.resident_chunks()),
            "evictions": total_evictions,
            "top_resident_scores": [
                {
                    "chunk": str(chunk),
                    "score": round(cache.score(chunk), 6),
                }
                for chunk in sorted(
                    cache.resident_chunks(),
                    key=lambda key: (-cache.score(key), str(key)),
                )[:5]
            ],
        },
    }


def write_json_report(payload: dict[str, Any], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out
