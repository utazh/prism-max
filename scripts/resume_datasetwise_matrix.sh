#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/panzihang/src/prism_datasetwise_logit_20260804}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/k005_datasetwise_full_loglikelihood_20260804}"
EXPECTED_SUMMARIES="${EXPECTED_SUMMARIES:-48}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"

while true; do
  completed="$(find "$RUN_ROOT" -name summary.json | wc -l)"
  if [[ "$completed" -ge "$EXPECTED_SUMMARIES" ]]; then
    exit 0
  fi

  if ROOT="$ROOT" RUN_ROOT="$RUN_ROOT" \
    GPU="${GPU:-3}" \
    TASKS="${TASKS:-sst2 subj trec rte}" \
    BUDGETS="${BUDGETS:-010 025 050}" \
    FAMILIES="${FAMILIES:-contigkv ours impress}" \
    SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1000000}" \
    WARMUP_PASSES="${WARMUP_PASSES:-1}" \
    WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-32}" \
    OURS_SELECTION_PERIOD_SIZE="${OURS_SELECTION_PERIOD_SIZE:-8}" \
    OURS_KNOWN_PERIOD_PREFETCH="${OURS_KNOWN_PERIOD_PREFETCH:-true}" \
    ACCURACY_SCORING="${ACCURACY_SCORING:-label_continuation_loglikelihood}" \
    SETTLE_SECONDS="${SETTLE_SECONDS:-0}" \
    RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}" \
    bash "$ROOT/scripts/run_datasetwise_matrix.sh"; then
    exit 0
  fi

  sleep "$RETRY_SECONDS"
done
