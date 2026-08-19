#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/contiguous_fuxian}"
REPEAT_IDS="${REPEAT_IDS:?set REPEAT_IDS to a space-separated list of grid IDs}"

for grid_id in $REPEAT_IDS; do
  GRID_ID="$grid_id" \
  GRID_DIR="$ROOT/results/flexgen_qwen_grid/$grid_id" \
    bash "$ROOT/scripts/run_hyperinfer_qwen_grid.sh"
done
