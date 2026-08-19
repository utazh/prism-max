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
OFFICIAL_REPO="/home/panzihang/src/hyperinfer_fuxian_20260720/HyperInfer"
FLEXGEN_ROOT="$OFFICIAL_REPO/h2o_flexgen/flexgen"
KV_DIR="/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1"
BASELINE_SUMMARY="/home/panzihang/src/hyperinfer_fuxian_20260720/results/qwen25_k050_ssd_hybrid_block16_alpha1_128_v1/k050_hyperinfer_block16/summary.json"

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
  "$FLEXGEN_ROOT/my_pcache_fast.py" \
  "$BASELINE_SUMMARY"; do
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

run_formal() {
  local label="$1"
  local period="$2"
  local budget_scale="$3"
  local output="$RUN_ROOT/$label"
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
    --tasks sst2,subj,trec,rte \
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

run_formal "formal_phase_a_delta0125_p1_128" 1 1.0
run_formal "formal_phase_ab_delta0125_p4_s025_128" 4 0.25

"$PYTHON" - "$RUN_ROOT" "$BASELINE_SUMMARY" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
paths = {
    "HyperInfer+block16": baseline_path,
    "Phase-A-layer-budget": root / "formal_phase_a_delta0125_p1_128" / "summary.json",
    "Phase-A+B-period-prefetch": (
        root / "formal_phase_ab_delta0125_p4_s025_128" / "summary.json"
    ),
}

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

baseline = read(baseline_path)
baseline_ttft = float(baseline["overall"]["mean_ttft_ms"])
baseline_accuracy = float(baseline["overall"]["accuracy"])
methods = []
for name, path in paths.items():
    summary = read(path)
    overall = summary["overall"]
    methods.append(
        {
            "method": name,
            "samples": int(overall["samples"]),
            "accuracy": float(overall["accuracy"]),
            "accuracy_change_vs_hyperinfer": (
                float(overall["accuracy"]) - baseline_accuracy
            ),
            "mean_ttft_ms": float(overall["mean_ttft_ms"]),
            "p95_ttft_ms": float(overall["p95_ttft_ms"]),
            "ttft_speedup_vs_hyperinfer_percent": (
                1.0 - float(overall["mean_ttft_ms"]) / baseline_ttft
            )
            * 100.0,
            "effective_mean_keep_ratio": overall.get("mean_effective_keep_ratio"),
            "runtime_variant": summary["runtime"]["runtime_variant"],
        }
    )

def records(directory):
    return {
        row["uid"]: row
        for row in (
            json.loads(line)
            for line in (directory / "scored_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }

a = records(root / "formal_phase_a_delta0125_p1_128")
ab = records(root / "formal_phase_ab_delta0125_p4_s025_128")
integrity = {
    "requests": len(a),
    "prediction_mismatches_phase_a_vs_ab": sum(
        a[uid]["prediction"] != ab[uid]["prediction"] for uid in a
    ),
    "selection_mismatches_phase_a_vs_ab": sum(
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
    "mean_phase_a_prefetch_wait_ms": statistics.fmean(
        float(row["prefetch_wait_ms"]) for row in a.values()
    ),
    "mean_phase_ab_prefetch_wait_ms": statistics.fmean(
        float(row["prefetch_wait_ms"]) for row in ab.values()
    ),
    "mean_phase_a_missing_tokens": statistics.fmean(
        float(row["inter_period_missing_tokens"]) for row in a.values()
    ),
    "mean_phase_ab_missing_tokens": statistics.fmean(
        float(row["inter_period_missing_tokens"]) for row in ab.values()
    ),
}
payload = {"methods": methods, "phase_b_integrity_and_prefetch": integrity}
(root / "formal_comparison_moderate.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY

sha256sum \
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
  "$ROOT/src/contiguous_fuxian/layer_budget.py" \
  "$ROOT/scripts/calibrate_layer_budget.py" \
  "$ROOT/scripts/run_prism_formal_moderate.sh" \
  "$PLAN" \
  "$PROFILE" \
  "$FLEXGEN_ROOT/my_pcache_fast.py" \
  "$BUNDLE_DIR/metadata.json" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  >"$RUN_ROOT/source_and_input_sha256_moderate.txt"
