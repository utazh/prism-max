"""Measure the smallest contiguous INT8 run that beats FP16 end to end."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from .mixed_precision_reader import MixedPrecisionPayloadReader
from .precision_run_coalescer import (
    DROP,
    FP16,
    INT8,
    choose_min_profitable_int8_run,
)


def _tiers(total: int, start: int, length: int, tier: str) -> tuple[str, ...]:
    values = [DROP] * total
    values[start : start + length] = [tier] * length
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--task", default="trec")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reader = MixedPrecisionPayloadReader(args.payload_root)
    geometry = reader.tasks[args.task]
    total = (
        geometry.prefix_tokens + geometry.block_size - 1
    ) // geometry.block_size
    lengths = (1, 2, 4, 8, 16)
    start = 64
    if start + max(lengths) > total:
        start = 0

    # Compile both kernel paths without using the measured layer files.
    for tier in (FP16, INT8):
        reader.read_kv(
            task=args.task,
            layer=geometry.layers - 1,
            tiers=_tiers(total, start, 1, tier),
            device=args.device,
        )

    rows = []
    for length_index, length in enumerate(lengths):
        samples = {FP16: [], INT8: []}
        for tier in (FP16, INT8):
            for repeat in range(args.repeats):
                layer = (length_index * args.repeats + repeat) % (
                    geometry.layers - 1
                )
                torch.cuda.synchronize()
                _, _, stats = reader.read_kv(
                    task=args.task,
                    layer=layer,
                    tiers=_tiers(total, start, length, tier),
                    device=args.device,
                )
                samples[tier].append(
                    float(stats["read_ms"]) + float(stats["materialize_ms"])
                )
        rows.append(
            (
                length,
                statistics.median(samples[FP16]),
                statistics.median(samples[INT8]),
            )
        )

    threshold = choose_min_profitable_int8_run(rows, margin_ratio=0.05)
    payload = {
        "task": args.task,
        "repeats": args.repeats,
        "safety_margin": 0.05,
        "measurements": [
            {"blocks": n, "fp16_ms": f, "int8_ms": q}
            for n, f, q in rows
        ],
        "min_profitable_int8_run_blocks": threshold,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
