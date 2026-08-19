#!/usr/bin/env bash
set -euo pipefail

# Run only ContiguousKV versus IMPRESS, matching the paper's four KV budgets.
ROOT="/home/panzihang/src/contiguous_fuxian"
GPU="${GPU:-3}"
TASKS="${TASKS:-sst2,subj,trec,rte}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
RATIOS="${RATIOS:-0.10 0.25 0.50}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
L1_SIZE_GB="${L1_SIZE_GB:-1}"
POST_WARM_SLEEP_S="${POST_WARM_SLEEP_S:-3}"
GRID_ID="${GRID_ID:-$(date +%Y%m%d_%H%M%S)}"
GRID_DIR="$ROOT/results/paper_grid/$GRID_ID"
L2_ROOT="${L2_ROOT:-/data1/contiguous_fuxian_l2/paper_grid/$GRID_ID}"
mkdir -p "$GRID_DIR" "$L2_ROOT"

if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU $GPU has an active compute process; refusing to start" >&2
  exit 2
fi

for ratio in $RATIOS; do
  tag="k$(printf '%s' "$ratio" | tr -d '.')"
  contig_plan="$GRID_DIR/${tag}_contig.json"
  impress_plan="$GRID_DIR/${tag}_impress.json"
  plan_log="$GRID_DIR/${tag}_plan.log"

  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT/src" \
    timeout 1800s /home/panzihang/venvs/vllm-stable/bin/python3 \
    -m contiguous_fuxian.paper_plan_generator \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --tasks "$TASKS" \
    --samples-per-task "$SAMPLES_PER_TASK" \
    --keep-ratio "$ratio" \
    --contiguous-output "$contig_plan" \
    --impress-output "$impress_plan" \
    --subperiod-size 4 \
    --int8-layers "${LMCACHE_INT8_LAYERS:-0}" \
    --max-prompt-tokens "$MAX_MODEL_LEN" \
    --device cuda \
    --allow-gpu \
    > "$plan_log" 2>&1

  contig_log="$GRID_DIR/${tag}_contig_launcher.log"
  env GPU="$GPU" TASKS="$TASKS" SAMPLES_PER_TASK="$SAMPLES_PER_TASK" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" L1_SIZE_GB="$L1_SIZE_GB" L1_INIT_SIZE_GB=1 \
    POST_WARM_SLEEP_S="$POST_WARM_SLEEP_S" L2_BASE_PATH="$L2_ROOT/${tag}_contig" \
    BUNDLE_DIR="$BUNDLE_DIR" "$ROOT/scripts/run_paper_lmcache.sh" contigkv "$contig_plan" \
    > "$contig_log" 2>&1
  contig_run="$(tail -n 1 "$contig_log")"

  impress_log="$GRID_DIR/${tag}_impress_launcher.log"
  env GPU="$GPU" TASKS="$TASKS" SAMPLES_PER_TASK="$SAMPLES_PER_TASK" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" L1_SIZE_GB="$L1_SIZE_GB" L1_INIT_SIZE_GB=1 \
    POST_WARM_SLEEP_S="$POST_WARM_SLEEP_S" L2_BASE_PATH="$L2_ROOT/${tag}_impress" \
    BUNDLE_DIR="$BUNDLE_DIR" "$ROOT/scripts/run_paper_lmcache.sh" impress "$impress_plan" \
    > "$impress_log" 2>&1
  impress_run="$(tail -n 1 "$impress_log")"

  PYTHONPATH="$ROOT/src" /home/panzihang/venvs/vllm-stable/bin/python3 \
    -m contiguous_fuxian.paper_compare \
    --contiguous-run "$contig_run" \
    --impress-run "$impress_run" \
    --output "$GRID_DIR/${tag}_comparison.json" \
    > "$GRID_DIR/${tag}_comparison.log" 2>&1
done

echo "$GRID_DIR"
