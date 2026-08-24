#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-2}"
RESERVE_GPU="${RESERVE_GPU:-3}"
REQUIRE_IDLE_RESERVE="${REQUIRE_IDLE_RESERVE:-true}"
TASK="${TASK:-trec}"
BUDGET_TAG="${BUDGET_TAG:-010}"
VARIANT="${VARIANT:-k4}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-128}"
WARMUP_PASSES="${WARMUP_PASSES:-1}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-1}"
PROMIXED_COVERAGE_FRACTION="${PROMIXED_COVERAGE_FRACTION:-0.5}"
PROMIXED_MARGIN_REFERENCE="${PROMIXED_MARGIN_REFERENCE:-0.05}"
PROMIXED_AGREEMENT_WEIGHT="${PROMIXED_AGREEMENT_WEIGHT:-0.75}"
PROMIXED_SENSITIVITY_WEIGHT="${PROMIXED_SENSITIVITY_WEIGHT:-0.1}"
PROMIXED_P1_THRESHOLD="${PROMIXED_P1_THRESHOLD:-0.90}"
PROMIXED_P2_THRESHOLD="${PROMIXED_P2_THRESHOLD:-0.82}"
PROMIXED_P4_THRESHOLD="${PROMIXED_P4_THRESHOLD:-0.68}"
PROMIXED_ADAPTIVE_COVERAGE="${PROMIXED_ADAPTIVE_COVERAGE:-false}"
PROMIXED_UTILITY_MAX_WEIGHT="${PROMIXED_UTILITY_MAX_WEIGHT:-0.55}"
PROMIXED_UTILITY_MEAN_WEIGHT="${PROMIXED_UTILITY_MEAN_WEIGHT:-0.35}"
PROMIXED_UTILITY_VOTE_WEIGHT="${PROMIXED_UTILITY_VOTE_WEIGHT:-0.10}"
DEFER_CACHE_SCORE_UPDATES="${DEFER_CACHE_SCORE_UPDATES:-false}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/promixed_screen_20260819}"

MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
SELECTOR_INDEX="${SELECTOR_INDEX:-$ROOT/assets/selector_index_k4_g32}"

gpu_has_compute_process() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | grep -Eq '[0-9]'
}

guard_gpus() {
  if [[ "$REQUIRE_IDLE_RESERVE" != "true" &&
        "$REQUIRE_IDLE_RESERVE" != "false" ]]; then
    echo "REQUIRE_IDLE_RESERVE must be true or false" >&2
    exit 2
  fi
  if [[ "$REQUIRE_IDLE_RESERVE" == "true" &&
        "$GPU" == "$RESERVE_GPU" ]]; then
    echo "GPU and RESERVE_GPU must differ" >&2
    exit 2
  fi
  if gpu_has_compute_process "$GPU"; then
    echo "Refusing to start: experiment GPU $GPU is occupied" >&2
    exit 3
  fi
  if [[ "$REQUIRE_IDLE_RESERVE" == "true" ]]; then
    if gpu_has_compute_process "$RESERVE_GPU"; then
      echo "Refusing to start: reserve GPU $RESERVE_GPU is occupied" >&2
      exit 3
    fi
  fi
}

budget_ratio() {
  case "$1" in
    005) echo 0.05 ;;
    010) echo 0.10 ;;
    025) echo 0.25 ;;
    050) echo 0.50 ;;
    *) echo "Unsupported budget tag: $1" >&2; return 2 ;;
  esac
}

budget_profile() {
  case "$1" in
    005) echo "$ROOT/configs/layer_budget_k005_sensitivity.json" ;;
    010) echo "$ROOT/configs/layer_budget_k010_sensitivity.json" ;;
    025) echo "$ROOT/configs/layer_budget_k025_scaled.json" ;;
    050) echo "$ROOT/configs/layer_budget_k050_sensitivity.json" ;;
    *) echo "Unsupported budget tag: $1" >&2; return 2 ;;
  esac
}

case "$VARIANT" in
  fp16) index_args=() ;;
  k4) index_args=(--selector-index-dir "$SELECTOR_INDEX") ;;
  *) echo "VARIANT must be fp16 or k4" >&2; exit 2 ;;
esac
case "$PROMIXED_ADAPTIVE_COVERAGE" in
  true) adaptive_coverage_args=(--promixed-adaptive-coverage) ;;
  false) adaptive_coverage_args=(--no-promixed-adaptive-coverage) ;;
  *)
    echo "PROMIXED_ADAPTIVE_COVERAGE must be true or false" >&2
    exit 2
    ;;
esac

case "$DEFER_CACHE_SCORE_UPDATES" in
  true) cache_score_args=(--defer-cache-score-updates) ;;
  false) cache_score_args=(--no-defer-cache-score-updates) ;;
  *)
    echo "DEFER_CACHE_SCORE_UPDATES must be true or false" >&2
    exit 2
    ;;
esac


KEEP_RATIO="$(budget_ratio "$BUDGET_TAG")"
PROFILE="$(budget_profile "$BUDGET_TAG")"
PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_ours.json"
OUTPUT="$RUN_ROOT/$TASK/k${BUDGET_TAG}_promixed_${VARIANT}"
LOG="$RUN_ROOT/$TASK/k${BUDGET_TAG}_promixed_${VARIANT}.log"

[[ -f "$PLAN" && -f "$PROFILE" ]] || {
  echo "Missing plan or layer-budget profile" >&2
  exit 2
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Output already exists: $OUTPUT" >&2
  exit 4
}
if [[ "$VARIANT" == "k4" && ! -f "$SELECTOR_INDEX/manifest.json" ]]; then
  echo "K4 selector index is missing: $SELECTOR_INDEX" >&2
  exit 2
fi

guard_gpus
mkdir -p "$(dirname "$OUTPUT")"
if [[ "$REQUIRE_IDLE_RESERVE" == "true" ]]; then
  resource_note="GPU $RESERVE_GPU reserved"
else
  resource_note="no idle reserve enforced"
fi
echo "[$(date -Is)] ProMixed $TASK k$BUDGET_TAG $VARIANT on GPU $GPU; $resource_note; deferred=$DEFER_CACHE_SCORE_UPDATES adaptive_coverage=$PROMIXED_ADAPTIVE_COVERAGE" | tee "$LOG"
guard_gpus
CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$ROOT/src" \
timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" \
  -m contiguous_fuxian.flexgen_qwen_reprefill \
  --model-path "$MODEL_PATH" \
  --bundle-dir "$BUNDLE_DIR" \
  --store-root "$STORE_ROOT" \
  --plan "$PLAN" \
  --output-dir "$OUTPUT" \
  --flexgen-root "$FLEXGEN_ROOT" \
  --flexgen-kv-dir "$KV_DIR" \
  --tasks "$TASK" \
  --store-tasks sst2,subj,trec,rte \
  --samples-per-task "$SAMPLES_PER_TASK" \
  --max-tokens 1 \
  --accuracy-scoring label_continuation_loglikelihood \
  --device cuda \
  --dtype bfloat16 \
  --gpu-cache-mb 55 \
  --cpu-cache-mb 131 \
  --cache-type CKLFU \
  --prefetch-time-budget 10000 \
  --online-selection \
  --reuse-flexgen-kv \
  --period-size 8 \
  --subperiod-size 4 \
  --expected-keep-ratio "$KEEP_RATIO" \
  --warmup-passes "$WARMUP_PASSES" \
  --warmup-samples-per-task "$WARMUP_SAMPLES_PER_TASK" \
  --probe-query-heads 0,7,14,21 \
  --selector-kv-head-ids 0,1,2,3 \
  --similarity-alpha 0.6 \
  --impress-selection-block-size 16 \
  --impress-selection-period-size 8 \
  --promixed-gqa-selection \
  --promixed-coverage-fraction "$PROMIXED_COVERAGE_FRACTION" \
  --promixed-margin-reference "$PROMIXED_MARGIN_REFERENCE" \
  --promixed-agreement-weight "$PROMIXED_AGREEMENT_WEIGHT" \
  --promixed-sensitivity-weight "$PROMIXED_SENSITIVITY_WEIGHT" \
  --promixed-p1-threshold "$PROMIXED_P1_THRESHOLD" \
  --promixed-p2-threshold "$PROMIXED_P2_THRESHOLD" \
  --promixed-p4-threshold "$PROMIXED_P4_THRESHOLD" \
  --promixed-utility-max-weight "$PROMIXED_UTILITY_MAX_WEIGHT" \
  --promixed-utility-mean-weight "$PROMIXED_UTILITY_MEAN_WEIGHT" \
  --promixed-utility-vote-weight "$PROMIXED_UTILITY_VOTE_WEIGHT" \
  "${adaptive_coverage_args[@]}" \
  --impress-async-prefetch \
  --impress-period-prefetch-size 1 \
  --no-impress-priority-prefetch \
  --no-impress-deferred-compute-timing \
  --no-impress-rolling-period-prefetch \
  --impress-value-ordered-prefetch \
  --impress-value-prefetch-budget-scale 0.9 \
  --layer-budget-profile "$PROFILE" \
  --exact-layer-block-budget \
  --impress-known-period-prefetch \
  "${cache_score_args[@]}" \
  "${index_args[@]}" \
  >>"$LOG" 2>&1

echo "[$(date -Is)] completed $OUTPUT" | tee -a "$LOG"
