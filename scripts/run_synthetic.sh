#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/../.."
PYTHONPATH=src python -m contiguous_fuxian.run_reproduction synthetic \
  --output src/contiguous_fuxian/results/synthetic_report.json

