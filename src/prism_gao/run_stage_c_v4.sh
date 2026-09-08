#!/usr/bin/env bash
set -euo pipefail

# Compare the original asynchronous FP16 path with the new payload reader in
# both FP16 and coalesced 16/8 modes. Place in src/prism_gao.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/src/prism_gao/results/stage_c_v4_$STAMP}"
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

fp16_fraction_for_budget() {
  # Fewer retained blocks means every selected block is more critical.
  case "$1" in
    005|010) echo "${LOW_BUDGET_FP16_FRACTION:-0.50}" ;;
    025|050) echo "${HIGH_BUDGET_FP16_FRACTION:-0.25}" ;;
    *) return 2 ;;
  esac
}

common_env() {
  :
}

run_original() {
  local task="$1" budget="$2" order="$3"
  local ratio
  ratio="$(ratio_for_budget "$budget")"
  GPU="$GPU" REQUIRE_IDLE_RESERVE=false \
  TASK="$task" BUDGET_TAG="$budget" METHOD=promixed SELECTOR_BACKEND=k4 \
  SAMPLES_PER_TASK="$SAMPLES_PER_TASK" SAMPLE_OFFSET="$SAMPLE_OFFSET" WARMUP_PASSES=1 \
  WARMUP_SAMPLES_PER_TASK="$WARMUP_SAMPLES_PER_TASK" \
  RUN_ROOT="$RUN_ROOT/$order" \
  RUN_NAME="k${budget}_original_async_${order}" \
  PRISM_GAO_SELECTOR_MODE=exact \
  PRISM_GAO_SELECTOR_RESIDENT=true \
  PRISM_GAO_FIXED_PERIOD=0 \
  PRISM_GAO_PERIOD_POLICY=profiled \
  PRISM_GAO_GLOBAL_KEEP_RATIO="$ratio" \
  PRISM_GAO_REUSE_PROFILE="$PROFILE" \
  PRISM_GAO_PRECISION_MODE=off \
  PRISM_GAO_HOST_ASYNC_PREFETCH=false \
  bash "$ROOT/src/prism_gao/run_cell.sh"
}

run_payload() {
  local task="$1" budget="$2" mode="$3" order="$4"
  local ratio fraction
  ratio="$(ratio_for_budget "$budget")"
  fraction="$(fp16_fraction_for_budget "$budget")"
  GPU="$GPU" REQUIRE_IDLE_RESERVE=false \
  TASK="$task" BUDGET_TAG="$budget" METHOD=promixed SELECTOR_BACKEND=k4 \
  SAMPLES_PER_TASK="$SAMPLES_PER_TASK" SAMPLE_OFFSET="$SAMPLE_OFFSET" WARMUP_PASSES=1 \
  WARMUP_SAMPLES_PER_TASK="$WARMUP_SAMPLES_PER_TASK" \
  RUN_ROOT="$RUN_ROOT/$order" \
  RUN_NAME="k${budget}_${mode}_pipeline_${order}" \
  PRISM_GAO_SELECTOR_MODE=exact \
  PRISM_GAO_SELECTOR_RESIDENT=true \
  PRISM_GAO_FIXED_PERIOD=0 \
  PRISM_GAO_PERIOD_POLICY=profiled \
  PRISM_GAO_GLOBAL_KEEP_RATIO="$ratio" \
  PRISM_GAO_REUSE_PROFILE="$PROFILE" \
  PRISM_GAO_PRECISION_MODE="$mode" \
  PRISM_GAO_HOST_ASYNC_PREFETCH=true \
  PRISM_GAO_FP16_FRACTION="$fraction" \
  PRISM_GAO_MIN_INT8_RUN_BLOCKS="${PRISM_GAO_MIN_INT8_RUN_BLOCKS:-2}" \
  bash "$ROOT/src/prism_gao/run_stage_c.sh"
}

cells=("sst2 010" "sst2 050" "subj 010" "subj 050" "trec 010" "trec 050" "rte 010" "rte 050")
for cell in "${cells[@]}"; do
  read -r task budget <<<"$cell"
  run_original "$task" "$budget" forward
  run_payload "$task" "$budget" fp16 forward
  run_payload "$task" "$budget" naive forward
  run_payload "$task" "$budget" coalesced forward
  run_payload "$task" "$budget" coalesced reverse
  run_payload "$task" "$budget" naive reverse
  run_payload "$task" "$budget" fp16 reverse
  run_original "$task" "$budget" reverse
done

echo "Stage C V4 comparison complete: $RUN_ROOT"
