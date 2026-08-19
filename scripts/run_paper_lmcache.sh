#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?method required: contigkv or impress}"
PLAN="${2:?LMCache plan JSON required}"
BUNDLE_DIR="${BUNDLE_DIR:-/home/panzihang/src/contiguous_fuxian/data/paper_task_bundles}"
TASKS="${TASKS:-sst2,subj,trec,rte}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
GPU="${GPU:-3}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL="${SERVED_MODEL:-qwen2.5-7b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
NUM_BLOCKS="${NUM_BLOCKS:-768}"
L1_SIZE_GB="${L1_SIZE_GB:-24}"
L1_INIT_SIZE_GB="${L1_INIT_SIZE_GB:-1}"
L2_BASE_PATH="${L2_BASE_PATH:-}"
POST_WARM_SLEEP_S="${POST_WARM_SLEEP_S:-0}"
ROOT="/home/panzihang/src/contiguous_fuxian"
LMCACHE_REPO="/home/panzihang/src/experience/LMCache-precision-allfull"
STAMP="$(date +%Y%m%d_%H%M%S)"

case "$METHOD" in
  contigkv)
    POLICY="paper-contigkv"
    CHUNK_SIZE=16
    ;;
  impress)
    POLICY="paper-impress"
    CHUNK_SIZE=64
    ;;
  *)
    echo "unknown method: $METHOD" >&2
    exit 2
    ;;
esac

if ss -ltn | awk '{print $4}' | grep -Eq ':(8000|8080|5555)$'; then
  echo "ports 8000, 8080, or 5555 are occupied" >&2
  exit 2
fi

RUN_DIR="$ROOT/results/paper_runs/${STAMP}_${METHOD}_c${CHUNK_SIZE}"
mkdir -p "$RUN_DIR"

cleanup() {
  set +e
  [ -f "$RUN_DIR/vllm.pid" ] && kill "$(cat "$RUN_DIR/vllm.pid")" 2>/dev/null || true
  [ -f "$RUN_DIR/lmcache.pid" ] && kill "$(cat "$RUN_DIR/lmcache.pid")" 2>/dev/null || true
  sleep 2
  [ -f "$RUN_DIR/vllm.pid" ] && kill -9 "$(cat "$RUN_DIR/vllm.pid")" 2>/dev/null || true
  [ -f "$RUN_DIR/lmcache.pid" ] && kill -9 "$(cat "$RUN_DIR/lmcache.pid")" 2>/dev/null || true
}
trap cleanup EXIT

export LMCACHE_PRECISION_POLICY="$POLICY"
export LMCACHE_ENABLE_FIDELITY_CACHE=True
export LMCACHE_DEFAULT_FIDELITY=base
export LMCACHE_BASE_CODEC=int8
export LMCACHE_INT8_LAYERS="${LMCACHE_INT8_LAYERS:-0}"
export LMCACHE_CHUNK_SIZE="$CHUNK_SIZE"
export LMCACHE_PRECISION_PLAN_JSON="$PLAN"
export LMCACHE_LAYER_GROUP_PLAN=1
export LMCACHE_LAYER_OBJECT_PACK=1
export LMCACHE_LAYER_TIMING=1
export PYTHONPATH="$LMCACHE_REPO:$ROOT/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

L2_ARGS=()
if [ -n "$L2_BASE_PATH" ]; then
  mkdir -p "$L2_BASE_PATH"
  L2_ARGS+=(--l2-adapter "{\"type\":\"fs\",\"base_path\":\"$L2_BASE_PATH\",\"relative_tmp_dir\":\".tmp\"}" --l2-store-policy skip_l1)
fi

source /home/panzihang/venvs/vllm-stable/bin/activate
python -m contiguous_fuxian.lmcache_server \
  --l1-size-gb "$L1_SIZE_GB" \
  --l1-init-size-gb "$L1_INIT_SIZE_GB" \
  --eviction-policy LRU \
  --disable-prometheus \
  --host 127.0.0.1 \
  --port 5555 \
  --http-host 127.0.0.1 \
  --http-port 8080 \
  --chunk-size "$CHUNK_SIZE" \
  "${L2_ARGS[@]}" \
  > "$RUN_DIR/lmcache.log" 2>&1 &
echo $! > "$RUN_DIR/lmcache.pid"
sleep 5

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs 1 \
  --num-gpu-blocks-override "$NUM_BLOCKS" \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
  > "$RUN_DIR/vllm.log" 2>&1 &
echo $! > "$RUN_DIR/vllm.pid"

python -m contiguous_fuxian.paper_client \
  --bundle-dir "$BUNDLE_DIR" \
  --tasks "$TASKS" \
  --samples-per-task "$SAMPLES_PER_TASK" \
  --policy "$POLICY" \
  --model "$SERVED_MODEL" \
  --output-dir "$RUN_DIR/client" \
  --post-warm-sleep-s "$POST_WARM_SLEEP_S" \
  > "$RUN_DIR/client.log" 2>&1

echo "$RUN_DIR"
