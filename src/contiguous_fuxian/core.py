"""Core ContiguousKV algorithms from the paper.

This module is intentionally framework-free. It implements the data-management
parts that can be tested without a GPU: ContiguousChunk scoring, Period reuse,
inter-period prefetch deltas, and attention-guided cache scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence


ChunkId = Hashable


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def chunk_bounds(length: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return half-open token ranges for a prefix split into ContiguousChunks."""

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    _require_positive("chunk_size", chunk_size)
    return [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]


def contiguous_chunk_scores(
    token_scores: Sequence[float],
    chunk_size: int,
    length: int | None = None,
) -> list[float]:
    """Sum token-level importance scores inside each ContiguousChunk.

    The paper defines a chunk score as the accumulated attention importance of
    tokens in that chunk. `length` allows callers to score only a prefix of a
    longer score vector.
    """

    _require_positive("chunk_size", chunk_size)
    usable = len(token_scores) if length is None else length
    if usable < 0:
        raise ValueError(f"length must be non-negative, got {usable}")
    usable = min(usable, len(token_scores))
    return [
        float(sum(token_scores[start:end]))
        for start, end in chunk_bounds(usable, chunk_size)
    ]


def select_top_chunks(
    scores: Sequence[float],
    keep_chunks: int,
    recent_keep_chunks: int = 0,
) -> set[int]:
    """Select top-scoring chunks, optionally forcing a recent suffix to stay."""

    if keep_chunks < 0:
        raise ValueError(f"keep_chunks must be non-negative, got {keep_chunks}")
    if recent_keep_chunks < 0:
        raise ValueError(
            f"recent_keep_chunks must be non-negative, got {recent_keep_chunks}"
        )
    if not scores or keep_chunks == 0:
        return set()

    total = len(scores)
    forced = set(range(max(0, total - recent_keep_chunks), total))
    remaining_budget = max(0, keep_chunks - len(forced))
    candidates = [
        idx for idx in range(total)
        if idx not in forced
    ]
    candidates.sort(key=lambda idx: (-scores[idx], idx))
    return forced | set(candidates[:remaining_budget])


def select_period_chunks(
    layer_chunk_scores: Sequence[Sequence[float]],
    keep_chunks: int,
    period_size: int,
    recent_keep_chunks: int = 0,
) -> list[set[int]]:
    """Reuse first-layer selected chunk indices for every layer in a Period."""

    _require_positive("period_size", period_size)
    out: list[set[int]] = []
    for start in range(0, len(layer_chunk_scores), period_size):
        period_scores = layer_chunk_scores[start]
        selected = select_top_chunks(period_scores, keep_chunks, recent_keep_chunks)
        for _ in layer_chunk_scores[start:start + period_size]:
            out.append(set(selected))
    return out


def inter_period_prefetch_delta(
    previous_period_chunks: Iterable[int],
    current_period_chunks: Iterable[int],
) -> tuple[set[int], set[int]]:
    """Return prefetched hits and missing chunks for inter-Period prefetching."""

    previous = set(previous_period_chunks)
    current = set(current_period_chunks)
    return previous & current, current - previous


def build_layer_chunk_plan(
    num_chunks: int,
    layer_chunks: Sequence[Iterable[int]],
    keep_label: str = "keep",
    drop_label: str = "drop",
) -> list[list[str]]:
    """Build a layer x chunk keep/drop plan from selected chunk indices."""

    if num_chunks < 0:
        raise ValueError(f"num_chunks must be non-negative, got {num_chunks}")
    plan: list[list[str]] = []
    for chunks in layer_chunks:
        selected = set(chunks)
        plan.append([
            keep_label if chunk_idx in selected else drop_label
            for chunk_idx in range(num_chunks)
        ])
    return plan


def jaccard_similarity(left: Iterable[int], right: Iterable[int]) -> float:
    """Compute Jaccard similarity, treating two empty sets as identical."""

    a = set(left)
    b = set(right)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass
class _CacheEntry:
    cumulative_attention: float = 0.0
    frequency: int = 0

    @property
    def score(self) -> float:
        return self.cumulative_attention * self.frequency


@dataclass
class AttentionGuidedCache:
    """Min-score cache using the paper's S_j = I_j * F_j policy."""

    capacity: int
    _entries: dict[ChunkId, _CacheEntry] = field(default_factory=dict)
    _resident: set[ChunkId] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {self.capacity}")

    def touch(self, chunk_id: ChunkId, attention_score: float) -> list[ChunkId]:
        """Record an access and evict low-scored resident chunks if needed."""

        entry = self._entries.setdefault(chunk_id, _CacheEntry())
        entry.cumulative_attention += float(attention_score)
        entry.frequency += 1
        if self.capacity == 0:
            return [chunk_id]

        self._resident.add(chunk_id)
        evicted: list[ChunkId] = []
        while len(self._resident) > self.capacity:
            victim = min(
                self._resident,
                key=lambda key: (self.score(key), str(key)),
            )
            self._resident.remove(victim)
            evicted.append(victim)
        return evicted

    def score(self, chunk_id: ChunkId) -> float:
        entry = self._entries.get(chunk_id)
        return 0.0 if entry is None else entry.score

    def resident_chunks(self) -> list[ChunkId]:
        return sorted(self._resident, key=str)
