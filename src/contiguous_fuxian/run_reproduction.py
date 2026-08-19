"""Command-line entry point for ContiguousKV reproduction runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiment import run_synthetic_reproduction, write_json_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent ContiguousKV reproduction experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser(
        "synthetic",
        help="Run deterministic read-amplification and prefetch simulation.",
    )
    synthetic.add_argument("--prefix-tokens", type=int, default=6000)
    synthetic.add_argument("--num-layers", type=int, default=28)
    synthetic.add_argument("--contiguous-chunk-size", type=int, default=16)
    synthetic.add_argument("--impress-chunk-size", type=int, default=64)
    synthetic.add_argument("--keep-ratio", type=float, default=0.05)
    synthetic.add_argument("--period-size", type=int, default=8)
    synthetic.add_argument("--subperiod-size", type=int, default=4)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--chunk-load-ms", type=float, default=0.08)
    synthetic.add_argument("--compute-ms", type=float, default=1.0)
    synthetic.add_argument(
        "--output",
        default="src/contiguous_fuxian/results/synthetic_report.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "synthetic":
        payload = run_synthetic_reproduction(
            prefix_tokens=args.prefix_tokens,
            num_layers=args.num_layers,
            contiguous_chunk_size=args.contiguous_chunk_size,
            impress_chunk_size=args.impress_chunk_size,
            keep_ratio=args.keep_ratio,
            period_size=args.period_size,
            subperiod_size=args.subperiod_size,
            seed=args.seed,
            chunk_load_ms=args.chunk_load_ms,
            compute_ms=args.compute_ms,
        )
        out = write_json_report(payload, Path(args.output))
        print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
        print(f"wrote {out}")
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
