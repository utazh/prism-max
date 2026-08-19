#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/prism_phase_b_valuebudget_20260730}"
STABLE_ROOT="${STABLE_ROOT:-/home/panzihang/src/prism_phase_ab_20260730}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/valuebudget_formal_128_20260730}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$STABLE_ROOT/data/paper_task_bundles_32}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
PLAN="${PLAN:-$ROOT/configs/qwen25_k050_hyperinfer_block16.json}"
PROFILE="${PROFILE:-$STABLE_ROOT/results/prism_phase_ab_20260730/calibration/layer_budget_profile_delta0125.json}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1}"
BASELINE="${BASELINE:-$STABLE_ROOT/results/prism_phase_ab_20260730/formal_phase_ab_delta0125_p4_s025_128}"
CANDIDATE="$RUN_ROOT/formal_leader_value_budget_s090_128"

guard_gpu() {
  local active
  active="$(
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader |
      tr -d '[:space:]'
  )"
  [[ -z "$active" ]] || {
    echo "GPU $GPU has active compute process $active; refusing to overlap" >&2
    exit 3
  }
}

for required in \
  "$PYTHON" \
  "$BUNDLE_DIR/metadata.json" \
  "$PLAN" \
  "$PROFILE" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  "$BASELINE/summary.json" \
  "$BASELINE/scored_records.jsonl"; do
  [[ -f "$required" ]] || {
    echo "Required file is missing: $required" >&2
    exit 2
  }
done

mkdir -p "$RUN_ROOT"
exec 9>"/tmp/prism_phase_ab_gpu${GPU}.lock"
flock -n 9 || {
  echo "Another PRISM run owns GPU lock $GPU" >&2
  exit 3
}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src"

if [[ ! -f "$CANDIDATE/summary.json" ]]; then
  guard_gpu
  timeout 43200s "$PYTHON" -m contiguous_fuxian.flexgen_qwen_reprefill \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$PLAN" \
    --output-dir "$CANDIDATE" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$KV_DIR" \
    --tasks "sst2,subj,trec,rte" \
    --samples-per-task 32 \
    --max-tokens 4 \
    --device cuda \
    --dtype bfloat16 \
    --gpu-cache-mb 55 \
    --cpu-cache-mb 131 \
    --cache-type CKLFU \
    --prefetch-time-budget 10000 \
    --online-selection \
    --probe-query-heads 0,1,2 \
    --selector-kv-head-ids 0 \
    --similarity-alpha 1.0 \
    --impress-selection-block-size 16 \
    --impress-async-prefetch \
    --impress-period-prefetch-size 4 \
    --impress-period-prefetch-budget-scale 0.25 \
    --no-impress-priority-prefetch \
    --no-impress-deferred-compute-timing \
    --no-impress-rolling-period-prefetch \
    --impress-value-ordered-prefetch \
    --impress-value-prefetch-budget-scale 0.90 \
    --reuse-flexgen-kv \
    --period-size 8 \
    --subperiod-size 4 \
    --expected-keep-ratio 0.50 \
    --warmup-passes 1 \
    --layer-budget-profile "$PROFILE" \
    >"$RUN_ROOT/formal_leader_value_budget_s090_128.log" 2>&1
fi

set +e
"$PYTHON" "$ROOT/scripts/analyze_valuebudget_formal.py" \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --expected-scale 0.90 \
  --output "$RUN_ROOT/formal_comparison.json" \
  >"$RUN_ROOT/formal_comparison.log" 2>&1
analysis_status=$?
set -e
[[ "$analysis_status" == "0" ]] &&
  touch "$RUN_ROOT/GATE_PASS" ||
  touch "$RUN_ROOT/GATE_FAIL"
sha256sum \
  "$ROOT/src/contiguous_fuxian/paper_plan_generator.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
  "$ROOT/vendor/flexgen/my_pcache_fast.py" \
  "$ROOT/scripts/analyze_valuebudget_formal.py" \
  "$ROOT/scripts/run_valuebudget_formal_128.sh" \
  "$PLAN" \
  "$PROFILE" \
  "$BUNDLE_DIR/metadata.json" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  "$BASELINE/summary.json" \
  >"$RUN_ROOT/source_and_input_sha256.txt"
cat "$RUN_ROOT/formal_comparison.json"
exit "$analysis_status"
