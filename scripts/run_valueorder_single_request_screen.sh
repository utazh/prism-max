#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/prism_phase_b_valueorder_20260730}"
STABLE_ROOT="${STABLE_ROOT:-/home/panzihang/src/prism_phase_ab_20260730}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/valueorder_single_request_20260730}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$STABLE_ROOT/data/paper_task_bundles_32}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
PLAN="${PLAN:-$ROOT/configs/qwen25_k050_hyperinfer_block16.json}"
PROFILE="${PROFILE:-$STABLE_ROOT/results/prism_phase_ab_20260730/calibration/layer_budget_profile_delta0125.json}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1}"

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
  "$FLEXGEN_ROOT/my_pcache_fast.py"; do
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

write_hashes() {
  sha256sum \
    "$ROOT/src/contiguous_fuxian/paper_plan_generator.py" \
    "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
    "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
    "$ROOT/vendor/flexgen/my_pcache_fast.py" \
    "$ROOT/scripts/analyze_abba_screen.py" \
    "$ROOT/scripts/analyze_valueorder_pilot.py" \
    "$ROOT/scripts/run_valueorder_single_request_screen.sh" \
    "$PLAN" \
    "$PROFILE" \
    "$BUNDLE_DIR/metadata.json" \
    "$KV_DIR/.contiguous_fuxian_complete" \
    >"$RUN_ROOT/source_and_input_sha256.txt"
}

"$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v \
  >"$RUN_ROOT/unit_tests.log" 2>&1

run_variant() {
  local label="$1"
  local tasks="$2"
  local samples_per_task="$3"
  local rolling="$4"
  local value_ordered="$5"
  local output="$RUN_ROOT/$label"
  local rolling_args=(--no-impress-rolling-period-prefetch)
  local value_args=(--no-impress-value-ordered-prefetch)
  [[ "$rolling" == "1" ]] && rolling_args=(--impress-rolling-period-prefetch)
  [[ "$value_ordered" == "1" ]] && value_args=(--impress-value-ordered-prefetch)
  if [[ -f "$output/summary.json" ]]; then
    echo "Skipping completed $label"
    return
  fi
  guard_gpu
  timeout 21600s "$PYTHON" -m contiguous_fuxian.flexgen_qwen_reprefill \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$PLAN" \
    --output-dir "$output" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$KV_DIR" \
    --tasks "$tasks" \
    --samples-per-task "$samples_per_task" \
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
    --reuse-flexgen-kv \
    --period-size 8 \
    --subperiod-size 4 \
    --expected-keep-ratio 0.50 \
    --warmup-passes 0 \
    --layer-budget-profile "$PROFILE" \
    "${rolling_args[@]}" \
    "${value_args[@]}" \
    >"$RUN_ROOT/$label.log" 2>&1
}

run_variant "pilot_baseline_4" "sst2,subj,trec,rte" 1 0 0
run_variant "pilot_leader_value_4" "sst2,subj,trec,rte" 1 0 1
run_variant "pilot_rolling_value_4" "sst2,subj,trec,rte" 1 1 1

set +e
"$PYTHON" "$ROOT/scripts/analyze_valueorder_pilot.py" \
  --baseline "$RUN_ROOT/pilot_baseline_4" \
  --leader-value "$RUN_ROOT/pilot_leader_value_4" \
  --rolling-value "$RUN_ROOT/pilot_rolling_value_4" \
  --output "$RUN_ROOT/valueorder_pilot_report.json" \
  --selected-output "$RUN_ROOT/selected_candidate.txt" \
  >"$RUN_ROOT/valueorder_pilot_report.log" 2>&1
pilot_status=$?
set -e
write_hashes
if [[ "$pilot_status" != "0" ]]; then
  touch "$RUN_ROOT/PILOT_GATE_FAIL"
  cat "$RUN_ROOT/valueorder_pilot_report.json"
  exit "$pilot_status"
fi
touch "$RUN_ROOT/PILOT_GATE_PASS"
selected="$(tr -d '[:space:]' <"$RUN_ROOT/selected_candidate.txt")"
case "$selected" in
  leader_value)
    candidate_rolling=0
    feature_args=(--feature-flag-metric impress_value_ordered_prefetch)
    ;;
  rolling_value)
    candidate_rolling=1
    feature_args=(
      --feature-flag-metric impress_value_ordered_prefetch
      --feature-flag-metric impress_rolling_period_prefetch
    )
    ;;
  *)
    echo "Unsupported selected candidate: $selected" >&2
    exit 2
    ;;
esac

run_variant "screen_baseline_a_32" "sst2,subj,trec,rte" 8 0 0
run_variant "screen_candidate_a_32" "sst2,subj,trec,rte" 8 "$candidate_rolling" 1
run_variant "screen_candidate_b_32" "sst2,subj,trec,rte" 8 "$candidate_rolling" 1
run_variant "screen_baseline_b_32" "sst2,subj,trec,rte" 8 0 0

set +e
"$PYTHON" "$ROOT/scripts/analyze_abba_screen.py" \
  --reference-a "$RUN_ROOT/screen_baseline_a_32" \
  --candidate-a "$RUN_ROOT/screen_candidate_a_32" \
  --candidate-b "$RUN_ROOT/screen_candidate_b_32" \
  --reference-b "$RUN_ROOT/screen_baseline_b_32" \
  --candidate-name "$selected" \
  "${feature_args[@]}" \
  --output "$RUN_ROOT/valueorder_screen_report.json" \
  >"$RUN_ROOT/valueorder_screen_report.log" 2>&1
analysis_status=$?
set -e
[[ "$analysis_status" == "0" ]] && touch "$RUN_ROOT/GATE_PASS" || touch "$RUN_ROOT/GATE_FAIL"
write_hashes
cat "$RUN_ROOT/valueorder_screen_report.json"
exit "$analysis_status"
