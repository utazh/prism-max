#!/usr/bin/env bash
set -euo pipefail

# Physical sparse-Qwen runner: only ContiguousKV versus IMPRESS.
ROOT="/home/panzihang/src/contiguous_fuxian"
GPU="${GPU:-3}"
TASKS="${TASKS:-sst2,subj,trec,rte}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
RATIOS="${RATIOS:-0.05 0.10 0.25 0.50}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles}"
STORE_ROOT="${STORE_ROOT:-/data1/contiguous_fuxian_sparse_kv/qwen25_7b_paper_seed42}"
PLAN_DIR="${PLAN_DIR:-$ROOT/results/sparse_qwen_plans/qwen25_7b_seed42}"
GRID_ID="${GRID_ID:-$(date +%Y%m%d_%H%M%S)}"
GRID_DIR="$ROOT/results/sparse_qwen_grid/$GRID_ID"
mkdir -p "$GRID_DIR"

if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU $GPU has an active compute process; refusing to start" >&2
  exit 2
fi

for ratio in $RATIOS; do
  tag="k$(printf '%s' "$ratio" | tr -d '.')"
  for method in contig impress; do
    run_dir="$GRID_DIR/${tag}_${method}"
    plan="$PLAN_DIR/${tag}_${method}.json"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT/src" \
      timeout 1800s /home/panzihang/venvs/vllm-stable/bin/python3 \
      -m contiguous_fuxian.sparse_qwen_reprefill run \
      --model-path "$MODEL_PATH" \
      --bundle-dir "$BUNDLE_DIR" \
      --store-root "$STORE_ROOT" \
      --plan "$plan" \
      --output-dir "$run_dir" \
      --tasks "$TASKS" \
      --samples-per-task "$SAMPLES_PER_TASK" \
      --device cuda \
      --dtype bfloat16 \
      --max-tokens 4 \
      > "$GRID_DIR/${tag}_${method}.log" 2>&1
  done
  PYTHONPATH="$ROOT/src" /home/panzihang/venvs/vllm-stable/bin/python3 \
    -m contiguous_fuxian.sparse_qwen_compare \
    --contiguous-run "$GRID_DIR/${tag}_contig" \
    --impress-run "$GRID_DIR/${tag}_impress" \
    --output "$GRID_DIR/${tag}_comparison.json" \
    > "$GRID_DIR/${tag}_comparison.log" 2>&1
done

echo "$GRID_DIR"
