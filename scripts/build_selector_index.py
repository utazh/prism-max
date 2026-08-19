#!/usr/bin/env python3
"""Build the ProMixed symmetric-INT4 all-GQA selector-key index."""

from __future__ import annotations

import argparse
import json

from contiguous_fuxian.quantized_key_index import build_quantized_key_index


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return items


def _head_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("head IDs must be integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantize raw Pcache selector heads into a ProMixed K4 index."
    )
    parser.add_argument("--source-pcache-dir", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", type=_csv, default=_csv("sst2,subj,trec,rte"))
    parser.add_argument(
        "--selector-kv-head-ids", type=_head_ids, default=_head_ids("0,1,2,3")
    )
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()
    manifest = build_quantized_key_index(
        source_pcache_dir=args.source_pcache_dir,
        store_root=args.store_root,
        tasks=args.tasks,
        output_dir=args.output_dir,
        selector_kv_head_ids=args.selector_kv_head_ids,
        group_size=args.group_size,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
