#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="./fewshots_datasets/output"
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*
WEIGHT_PATH="${WEIGHT_PATH:-${HOME}/opt_weights}"
HF_CACHE_DIR="${HYPERINFER_HF_CACHE:-${HF_HOME:-}}"
RUNTIME_ARGS=(--path "$WEIGHT_PATH")
if [[ -n "$HF_CACHE_DIR" ]]; then
    RUNTIME_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
fi

model="facebook/opt-6.7b"
sele_percents=25
# sim_threds=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0)
sim_threds=(0)

datasets=("./fewshots_datasets/input/openbookqa.jsonl" )

result_prefix="./fewshots_datasets"  # 这里需要替换成实际的结果文件路径前缀，例如 /home/user/results

for sim_thred in "${sim_threds[@]}"
do
    input_file="${datasets[0]}"
    task="openbookqa"
    result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

    time "$PYTHON_BIN" ./flex_opt.py "${RUNTIME_ARGS[@]}" \
            --gpu-batch-size 1 \
            --overlap false \
            --model $model \
            --prefix-aware-inf \
            --sele-load \
            --sele-load-by-percent \
            --sele-head 0 1 2 \
            --sele-percent "$sele_percents" \
            --sim-thred "$sim_thred" \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache \
            --fill-keys-zero
    
#     time python3 ./evaluate_task_result.py \
#             --result-file "$result_file" \
#             --task-name "$task" \
#             --model-type opt \
#             --exp "sele_load_by-percent_${sele_percents}_sim_thred${sim_thred}-prekv-ours"
done
# time python3 ./scripts/extract_load_ratio.py \
#             --figure11
# time python3 ./scripts/extract_acc.py\
#             --figure11
# 分析log以及精度
