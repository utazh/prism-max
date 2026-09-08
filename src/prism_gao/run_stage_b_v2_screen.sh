#!/usr/bin/env bash
set -euo pipefail

# Place this file in src/prism_gao and run from the repository root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/src/prism_gao/results/stage_b_v2_$STAMP}"
GPU="${GPU:-2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-64}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-16}"
EXTENDED="${EXTENDED:-false}"

ratio_for_budget() {
  case "$1" in
    005) echo 0.05 ;;
    010) echo 0.10 ;;
    025) echo 0.25 ;;
    050) echo 0.50 ;;
    *) return 2 ;;
  esac
}

run_one() {
  local task="$1" budget="$2" policy="$3" order="$4"
  local ratio
  ratio="$(ratio_for_budget "$budget")"
  GPU="$GPU" REQUIRE_IDLE_RESERVE=false \
  TASK="$task" BUDGET_TAG="$budget" METHOD=promixed SELECTOR_BACKEND=k4 \
  SAMPLES_PER_TASK="$SAMPLES_PER_TASK" WARMUP_PASSES=1 \
  WARMUP_SAMPLES_PER_TASK="$WARMUP_SAMPLES_PER_TASK" \
  RUN_ROOT="$RUN_ROOT/$order" \
  RUN_NAME="k${budget}_promixed_exact_${policy}_${order}" \
  PRISM_GAO_SELECTOR_MODE=exact \
  PRISM_GAO_FIXED_PERIOD=0 \
  PRISM_GAO_PERIOD_POLICY="$policy" \
  PRISM_GAO_GLOBAL_KEEP_RATIO="$ratio" \
  PRISM_GAO_LOW_BUDGET_P4_THRESHOLD="${PRISM_GAO_LOW_BUDGET_P4_THRESHOLD:-0.76}" \
  PRISM_GAO_HIGH_BUDGET_MIN="${PRISM_GAO_HIGH_BUDGET_MIN:-0.40}" \
  bash "$ROOT/src/prism_gao/run_cell.sh"
}

cells=("trec 010" "subj 025" "trec 050")
if [[ "$EXTENDED" == "true" ]]; then
  # Cover all four datasets at the two budgets changed by budget_v2.
  cells+=(
    "sst2 010" "subj 010" "rte 010"
    "sst2 050" "subj 050" "rte 050"
  )
fi
for cell in "${cells[@]}"; do
  read -r task budget <<<"$cell"
  run_one "$task" "$budget" original forward
  run_one "$task" "$budget" budget_v2 forward
  run_one "$task" "$budget" budget_v2 reverse
  run_one "$task" "$budget" original reverse
done

echo "Stage B V2 screen complete: $RUN_ROOT"
