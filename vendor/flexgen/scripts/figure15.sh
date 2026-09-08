#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="./fewshots_datasets/output"
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

# 引入包含 run_sele_load_by_percent 函数的脚本文件，假设其名为 original_script.sh，根据实际情况修改路径
source ./scripts/overall_acc.sh

# 定义多组参数值
sele_percent_value1=25
sim_thred_value1=0.3
sele_percent_value2=50
sim_thred_value2=0.5
sele_percent_value3=10
sim_thred_value3=0.17
sele_percent_value4=5
sim_thred_value4=0.1

# 将参数值组合成数组
sele_percent_values=($sele_percent_value1 $sele_percent_value2 $sele_percent_value3 $sele_percent_value4)
sim_thred_values=($sim_thred_value1 $sim_thred_value2 $sim_thred_value3 $sim_thred_value4)

# 通过 for 循环依次调用 run_sele_load_by_percent 函数并传入不同组的参数值
# for ((i = 0; i < ${#sele_percent_values[@]}; i++)); do
#     sele_percent="${sele_percent_values[i]}"
#     sim_thred="${sim_thred_values[i]}"
#     # run_sele_load_by_percent "$sele_percent" "$sim_thred"
#     run_full_load_sele_inf "$sele_percent"
# done
run_sele_load_by_percent 25 0.3
run_full_load_sele_inf 25
run_full_load_full_inf

time "$PYTHON_BIN" ./scripts/extract_acc.py --figure15
