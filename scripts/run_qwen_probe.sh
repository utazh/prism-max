#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/../.."

MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-}"
ALLOW_GPU_ARGS=()
if [[ "$DEVICE" == cuda* ]]; then
  if [[ "${ALLOW_GPU:-0}" != "1" ]]; then
    echo "Refusing to use CUDA without ALLOW_GPU=1. This avoids disrupting shared server experiments." >&2
    exit 2
  fi
  ALLOW_GPU_ARGS+=(--allow-gpu)
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

PYTHONPATH=src python -m contiguous_fuxian.qwen_attention_probe \
  --model-path "$MODEL_PATH" \
  --output src/contiguous_fuxian/results/qwen_attention_probe.json \
  --max-prompts "${MAX_PROMPTS:-1}" \
  --contiguous-chunk-size "${CONTIGUOUS_CHUNK_SIZE:-16}" \
  --keep-ratio "${KEEP_RATIO:-0.05}" \
  --period-size "${PERIOD_SIZE:-8}" \
  --subperiod-size "${SUBPERIOD_SIZE:-4}" \
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-2048}" \
  --device "$DEVICE" \
  "${ALLOW_GPU_ARGS[@]}"
