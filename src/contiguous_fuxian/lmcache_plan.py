"""Convert ContiguousKV attention plans into LMCache runtime plan JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


_SELECTED_SOURCE_TIERS = {"keep", "full", "fp16", "bf16"}
_UNSELECTED_SOURCE_TIERS = {"drop", "dropped", "skip"}
_RUNTIME_TIERS = {"int8", "int4", "drop"}


def policy_request_id(uid: str) -> str:
    """Return the request key consumed by the LMCache precision hook."""

    if not uid:
        raise ValueError("uid must not be empty")
    return f"cmpl-{uid}-score"


def _validate_grid(grid: Any, record_index: int) -> list[list[str]]:
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"record {record_index} has no layer_plan")
    if not all(isinstance(row, list) and row for row in grid):
        raise ValueError(f"record {record_index} has an empty layer_plan row")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError(f"record {record_index} layer_plan rows must have equal lengths")
    return grid


def _convert_tier(raw_tier: str, selected_tier: str, unselected_tier: str) -> str:
    normalized = str(raw_tier).strip().lower()
    if normalized in _RUNTIME_TIERS - {"drop"}:
        return normalized
    if normalized in _SELECTED_SOURCE_TIERS:
        return selected_tier
    if normalized in _UNSELECTED_SOURCE_TIERS:
        return unselected_tier
    raise ValueError(f"unsupported source plan tier: {raw_tier!r}")


def convert_probe_to_lmcache_plan(
    probe_payload: dict[str, Any],
    *,
    uid_prefix: str = "contiguous",
    selected_tier: str = "int8",
    unselected_tier: str = "drop",
) -> dict[str, Any]:
    """Convert qwen_attention_probe JSON into an LMCache layer-plan payload.

    A selected ContiguousChunk is stored at ``selected_tier`` while an
    unselected chunk is represented by ``unselected_tier``. The latter can be
    ``drop`` for the closest current runtime approximation to ContiguousKV, or
    ``int4`` when evaluating a no-drop precision-control variant.
    """

    if not uid_prefix:
        raise ValueError("uid_prefix must not be empty")
    if selected_tier not in _RUNTIME_TIERS - {"drop"}:
        raise ValueError("selected_tier must be int8 or int4")
    if unselected_tier not in _RUNTIME_TIERS:
        raise ValueError("unselected_tier must be int8, int4, or drop")

    records = probe_payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("probe payload must contain at least one record")

    request_prefixes: dict[str, list[str]] = {}
    layer_request_prefixes: dict[str, list[list[str]]] = {}
    records_meta: list[dict[str, Any]] = []
    all_tiers: Counter[str] = Counter()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        grid = _validate_grid(record.get("layer_plan"), index)
        uid = f"{uid_prefix}-{record.get('prompt_index', index)}"
        request_id = policy_request_id(uid)
        converted = [
            [_convert_tier(tier, selected_tier, unselected_tier) for tier in row]
            for row in grid
        ]
        active_chunk_tiers = [
            "base" if any(row[chunk] != "drop" for row in converted) else "drop"
            for chunk in range(len(converted[0]))
        ]
        request_prefixes[request_id] = active_chunk_tiers
        layer_request_prefixes[request_id] = converted
        all_tiers.update(tier for row in converted for tier in row)
        records_meta.append(
            {
                "uid": uid,
                "request_id": request_id,
                "prompt_index": record.get("prompt_index", index),
                "prefix_tokens": record.get("prefix_tokens"),
                "num_layers": len(converted),
                "num_chunks": len(converted[0]),
                "tiers": dict(Counter(tier for row in converted for tier in row)),
            }
        )

    return {
        "default": ["base"],
        "request_prefixes": request_prefixes,
        "layer_default": [],
        "layer_request_prefixes": layer_request_prefixes,
        "metadata": {
            "method": "contigkv",
            "source": "contiguous_fuxian.qwen_attention_probe",
            "model_path": probe_payload.get("model_path"),
            "contiguous_chunk_size": probe_payload.get("contiguous_chunk_size"),
            "keep_ratio": probe_payload.get("keep_ratio"),
            "period_size": probe_payload.get("period_size"),
            "subperiod_size": probe_payload.get("subperiod_size"),
            "selected_tier": selected_tier,
            "unselected_tier": unselected_tier,
            "runtime_tier_counts": dict(all_tiers),
            "records": records_meta,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a ContiguousKV attention probe into LMCache plan JSON."
    )
    parser.add_argument("--probe", required=True, help="qwen_attention_probe JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--uid-prefix", default="contiguous")
    parser.add_argument("--selected-tier", choices=("int8", "int4"), default="int8")
    parser.add_argument("--unselected-tier", choices=("drop", "int4", "int8"), default="drop")
    args = parser.parse_args()

    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    payload = convert_probe_to_lmcache_plan(
        probe,
        uid_prefix=args.uid_prefix,
        selected_tier=args.selected_tier,
        unselected_tier=args.unselected_tier,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
