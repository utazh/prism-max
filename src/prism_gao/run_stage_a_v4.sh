#!/usr/bin/env bash
set -euo pipefail

# Exact selector only needs confirmation now; do not use direct for formal runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/src/prism_gao/results/stage_a_confirm_$STAMP}"
GPU="${GPU:-2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-64}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-16}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-16}"

run_one() {
  local task="$1" budget="$2" variant="$3" order="$4"
  local backend resident
  case "$variant" in
    fp16) backend=fp16; resident=false ;;
    exact_host) backend=k4; resident=false ;;
    exact_resident) backend=k4; resident=true ;;
    *) echo "unknown variant: $variant" >&2; return 2 ;;
  esac
  GPU="$GPU" REQUIRE_IDLE_RESERVE=false TASK="$task" BUDGET_TAG="$budget" \
  METHOD=promixed SELECTOR_BACKEND="$backend" SAMPLES_PER_TASK="$SAMPLES_PER_TASK" \
  SAMPLE_OFFSET="$SAMPLE_OFFSET" WARMUP_PASSES=1 \
  WARMUP_SAMPLES_PER_TASK="$WARMUP_SAMPLES_PER_TASK" RUN_ROOT="$RUN_ROOT/$order" \
  RUN_NAME="k${budget}_${variant}_${order}" PRISM_GAO_SELECTOR_MODE=exact \
  PRISM_GAO_SELECTOR_RESIDENT="$resident" PRISM_GAO_PERIOD_POLICY=original \
  PRISM_GAO_FIXED_PERIOD=0 PRISM_GAO_PRECISION_MODE=off \
  bash "$ROOT/src/prism_gao/run_cell.sh"
}

cells=("trec 010" "subj 025" "sst2 010" "rte 025")
for cell in "${cells[@]}"; do
  read -r task budget <<<"$cell"
  run_one "$task" "$budget" fp16 forward
  run_one "$task" "$budget" exact_host forward
  run_one "$task" "$budget" exact_resident forward
  run_one "$task" "$budget" exact_resident reverse
  run_one "$task" "$budget" exact_host reverse
  run_one "$task" "$budget" fp16 reverse
done

echo "Stage A confirmation complete: $RUN_ROOT"
