#!/usr/bin/env bash
# New audited entrypoint. Historical run_prism_max_cell.sh remains unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TASK="${TASK:-trec}"
SELECTION="${SELECTION:-legacy}"
SELECTOR_BACKEND="${SELECTOR_BACKEND:-k4}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval_strict}"
SELECTOR_INDEX="${SELECTOR_INDEX:-$ROOT/assets/selector_index_k4_g32}"
RUN_DIR="${RUN_DIR:-$ROOT/results/review_fixed8/${TASK}_${SELECTION}_${SELECTOR_BACKEND}_$(date +%Y%m%dT%H%M%S)_$$}"
extra=()
case "$SELECTOR_BACKEND" in
  k4) extra+=(--selector-index-dir "$SELECTOR_INDEX") ;;
  fp16) ;;
  *) echo 'SELECTOR_BACKEND must be k4 or fp16' >&2; exit 2 ;;
esac
if [[ -n "${LAYER_BUDGET_PROFILE:-}" ]]; then
  extra+=(--layer-budget-profile "$LAYER_BUDGET_PROFILE")
fi
dry=false
for arg in "$@"; do [[ "$arg" != --dry-run ]] || dry=true; done
if [[ "$dry" == false ]]; then
  exec 7>/tmp/prism_max_storage.lock
  flock -n 7 || { echo 'Shared PRISM storage is in use' >&2; exit 3; }
  exec 9>"/tmp/prism_max_gpu${GPU}.lock"
  flock -n 9 || { echo "GPU lock $GPU is in use" >&2; exit 3; }
  processes="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)"
  if grep -Eq '[0-9]' <<<"$processes"; then
    echo "Refusing to run on occupied GPU $GPU" >&2; exit 3
  fi
fi
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m contiguous_fuxian.audited_reprefill \
  --model-path "$MODEL_PATH" --bundle-dir "$BUNDLE_DIR" --store-root "$STORE_ROOT" \
  --flexgen-root "$ROOT/vendor/flexgen" --flexgen-kv-dir "$KV_DIR" \
  --output-dir "$RUN_DIR" --tasks "$TASK" --store-tasks sst2,subj,trec,rte \
  --selection "$SELECTION" --keep-ratio "${KEEP_RATIO:-0.1}" \
  --samples-per-task "${SAMPLES_PER_TASK:-128}" \
  --gpu-cache-mb "${GPU_CACHE_MB:-55}" --cpu-cache-mb "${CPU_CACHE_MB:-131}" \
  "${extra[@]}" "$@"
