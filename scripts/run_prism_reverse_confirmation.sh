#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/prism_phase_ab_20260730}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-1}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/prism_phase_ab_20260730}"
MODEL_PATH="/data1/llm/Qwen/Qwen2.5-7B-Instruct"
BUNDLE_DIR="$ROOT/data/paper_task_bundles_32"
STORE_ROOT="/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42"
PLAN="$ROOT/configs/qwen25_k050_hyperinfer_block16.json"
PROFILE="$RUN_ROOT/calibration/layer_budget_profile_delta0125.json"
FLEXGEN_ROOT="/home/panzihang/src/hyperinfer_fuxian_20260720/HyperInfer/h2o_flexgen/flexgen"
KV_DIR="/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1"

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

run_variant() {
  local label="$1"
  local period="$2"
  local budget_scale="$3"
  local output="$RUN_ROOT/$label"
  if [[ -f "$output/summary.json" ]]; then
    echo "Skipping completed $label"
    return
  fi
  guard_gpu
  timeout 7200s "$PYTHON" -m contiguous_fuxian.flexgen_qwen_reprefill \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$PLAN" \
    --output-dir "$output" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$KV_DIR" \
    --tasks sst2,subj,trec,rte \
    --samples-per-task 8 \
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
    --impress-period-prefetch-size "$period" \
    --impress-period-prefetch-budget-scale "$budget_scale" \
    --reuse-flexgen-kv \
    --period-size 8 \
    --subperiod-size 4 \
    --expected-keep-ratio 0.50 \
    --warmup-passes 1 \
    --layer-budget-profile "$PROFILE" \
    >"$RUN_ROOT/$label.log" 2>&1
}

# Reverse the formal ordering to expose order-dependent cache or system effects.
run_variant "confirmation_reverse_phase_ab_p4_s025_32" 4 0.25
run_variant "confirmation_reverse_phase_a_p1_32" 1 1.0

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
labels = {
    "phase_ab_first": "confirmation_reverse_phase_ab_p4_s025_32",
    "phase_a_second": "confirmation_reverse_phase_a_p1_32",
}

def load_records(label):
    return {
        row["uid"]: row
        for row in map(
            json.loads,
            (root / labels[label] / "scored_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }

ab = load_records("phase_ab_first")
a = load_records("phase_a_second")
if set(a) != set(ab):
    raise SystemExit("reverse confirmation uid sets differ")
deltas = [float(ab[uid]["ttft_ms"]) - float(a[uid]["ttft_ms"]) for uid in a]
payload = {
    "order": ["phase_ab_first", "phase_a_second"],
    "requests": len(a),
    "phase_ab_mean_ttft_ms": statistics.fmean(
        float(row["ttft_ms"]) for row in ab.values()
    ),
    "phase_a_mean_ttft_ms": statistics.fmean(
        float(row["ttft_ms"]) for row in a.values()
    ),
    "phase_ab_vs_phase_a_mean_ttft_delta_ms": statistics.fmean(deltas),
    "phase_ab_vs_phase_a_mean_ttft_change_percent": (
        statistics.fmean(float(row["ttft_ms"]) for row in ab.values())
        / statistics.fmean(float(row["ttft_ms"]) for row in a.values())
        - 1.0
    )
    * 100.0,
    "phase_ab_faster_requests": sum(delta < 0 for delta in deltas),
    "prediction_mismatches": sum(
        a[uid]["prediction"] != ab[uid]["prediction"] for uid in a
    ),
    "selection_mismatches": sum(
        a[uid]["layer_token_selection_sha256"]
        != ab[uid]["layer_token_selection_sha256"]
        for uid in a
    ),
    "mean_period_prefetch_jobs": statistics.fmean(
        float(row["impress_period_prefetch_jobs"]) for row in ab.values()
    ),
    "mean_period_prefetch_tokens": statistics.fmean(
        float(row["impress_period_prefetch_tokens"]) for row in ab.values()
    ),
}
(root / "confirmation_reverse_comparison.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY
