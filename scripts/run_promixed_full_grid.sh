#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
RESERVE_GPU="${RESERVE_GPU:-1}"
TASKS="${TASKS:-sst2 subj trec rte}"
BUDGETS="${BUDGETS:-005 010 025 050}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1000000}"
WARMUP_PASSES="${WARMUP_PASSES:-1}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-32}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/promixed_full_grid_20260819}"
WAIT_SECONDS="${WAIT_SECONDS:-20}"
STABLE_SECONDS="${STABLE_SECONDS:-10}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-86400}"
VARIANT="${VARIANT:-k4}"
DRIVER_LOG="$RUN_ROOT/grid_driver.log"

if [[ "$GPU" == "$RESERVE_GPU" ]]; then
  echo "GPU and RESERVE_GPU must differ" >&2
  exit 2
fi
if [[ "$VARIANT" != "k4" && "$VARIANT" != "fp16" ]]; then
  echo "VARIANT must be k4 or fp16" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
exec 9>"/tmp/prism_promixed_gpu${GPU}.lock"
flock -n 9 || {
  echo "Another ProMixed run owns GPU lock $GPU" >&2
  exit 3
}
exec 8>"/tmp/prism_promixed_gpu${RESERVE_GPU}.reserve.lock"
flock -n 8 || {
  echo "Another ProMixed run owns reserve lock $RESERVE_GPU" >&2
  exit 3
}

gpu_has_compute_process() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
    grep -Eq '[0-9]'
}

wait_for_gpu_pair() {
  local announced=0
  while true; do
    if ! gpu_has_compute_process "$GPU" &&
       ! gpu_has_compute_process "$RESERVE_GPU"; then
      sleep "$STABLE_SECONDS"
      if ! gpu_has_compute_process "$GPU" &&
         ! gpu_has_compute_process "$RESERVE_GPU"; then
        if (( announced )); then
          echo "[$(date -Is)] GPU $GPU and reserve GPU $RESERVE_GPU are stably idle" |
            tee -a "$DRIVER_LOG"
        fi
        return
      fi
    fi
    if (( ! announced )); then
      echo "[$(date -Is)] waiting for idle GPU $GPU plus idle reserve GPU $RESERVE_GPU" |
        tee -a "$DRIVER_LOG"
      announced=1
    fi
    sleep "$WAIT_SECONDS"
  done
}

archive_incomplete_output() {
  local output="$1"
  if [[ -d "$output" && ! -f "$output/summary.json" ]]; then
    local archived="${output}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "$output" "$archived"
    echo "[$(date -Is)] archived incomplete output as $archived" |
      tee -a "$DRIVER_LOG"
  fi
}

run_one() {
  local task="$1"
  local budget="$2"
  local output="$RUN_ROOT/$task/k${budget}_promixed_${VARIANT}"
  if [[ -f "$output/summary.json" ]]; then
    echo "[$(date -Is)] skipping completed $task k$budget $VARIANT" |
      tee -a "$DRIVER_LOG"
    return
  fi
  archive_incomplete_output "$output"

  while true; do
    wait_for_gpu_pair
    echo "[$(date -Is)] launching $task k$budget $VARIANT" |
      tee -a "$DRIVER_LOG"
    set +e
    GPU="$GPU" \
    RESERVE_GPU="$RESERVE_GPU" \
    TASK="$task" \
    BUDGET_TAG="$budget" \
    VARIANT="$VARIANT" \
    SAMPLES_PER_TASK="$SAMPLES_PER_TASK" \
    WARMUP_PASSES="$WARMUP_PASSES" \
    WARMUP_SAMPLES_PER_TASK="$WARMUP_SAMPLES_PER_TASK" \
    RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" \
    RUN_ROOT="$RUN_ROOT" \
      "$ROOT/scripts/run_promixed_screen.sh"
    local status=$?
    set -e
    if (( status == 0 )); then
      break
    fi
    if (( status == 3 )); then
      echo "[$(date -Is)] GPU availability changed before launch; retrying" |
        tee -a "$DRIVER_LOG"
      continue
    fi
    echo "[$(date -Is)] failed $task k$budget with status $status" |
      tee -a "$DRIVER_LOG"
    exit "$status"
  done
  sleep "$SETTLE_SECONDS"
}

echo "[$(date -Is)] ProMixed independent full grid starts; experiment GPU $GPU, reserve GPU $RESERVE_GPU" |
  tee -a "$DRIVER_LOG"
for task in $TASKS; do
  for budget in $BUDGETS; do
    run_one "$task" "$budget"
  done
done
echo "[$(date -Is)] ProMixed independent full grid completed" |
  tee -a "$DRIVER_LOG"
