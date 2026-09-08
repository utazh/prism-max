#!/usr/bin/env bash
set -euo pipefail

# Place this file in src/prism_gao and run from the repository root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/src/prism_gao/results/stage_b_v4_eval_$STAMP}"
GPU="${GPU:-2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-64}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-16}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-16}"
PROFILE="${PROFILE:?set PROFILE to reuse_profile.json}"

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
  local task="$1" budget="$2" variant="$3"
  local ratio period policy
  ratio="$(ratio_for_budget "$budget")"
  case "$variant" in
    p1) period=1; policy=original ;;
    p2) period=2; policy=original ;;
    p4) period=4; policy=original ;;
    p8) period=8; policy=original ;;
    old) period=0; policy=original ;;
    profiled) period=0; policy=profiled ;;
    *) echo "unknown variant: $variant" >&2; return 2 ;;
  esac
  GPU="$GPU" REQUIRE_IDLE_RESERVE=false TASK="$task" BUDGET_TAG="$budget" \
  METHOD=promixed SELECTOR_BACKEND=k4 SAMPLES_PER_TASK="$SAMPLES_PER_TASK" \
  SAMPLE_OFFSET="$SAMPLE_OFFSET" WARMUP_PASSES=1 \
  WARMUP_SAMPLES_PER_TASK="$WARMUP_SAMPLES_PER_TASK" RUN_ROOT="$RUN_ROOT" \
  RUN_NAME="k${budget}_${variant}" PRISM_GAO_SELECTOR_MODE=exact \
  PRISM_GAO_SELECTOR_RESIDENT=true PRISM_GAO_FIXED_PERIOD="$period" \
  PRISM_GAO_PERIOD_POLICY="$policy" PRISM_GAO_REUSE_PROFILE="$PROFILE" \
  PRISM_GAO_GLOBAL_KEEP_RATIO="$ratio" PRISM_GAO_PRECISION_MODE=off \
  bash "$ROOT/src/prism_gao/run_cell.sh"
}

tasks=(sst2 subj trec rte)
budgets=(010 050)
variants=(p1 p2 p4 p8 old profiled)
for task in "${tasks[@]}"; do
  for budget in "${budgets[@]}"; do
    for variant in "${variants[@]}"; do
      run_one "$task" "$budget" "$variant"
    done
  done
done

echo "Stage B V4 held-out complete: $RUN_ROOT"
