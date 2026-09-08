# models=("opt-1.3b" "opt-6.7b" "opt-13b" "opt-30b")
models=("opt-6.7b" "opt-13b" "opt-30b")
sele_percents=(50 25 25 25)
sim_threds=(0.5 0.3 0.3 0.3)
gpu_size="10240"
cpu_size="32768"

datasets=("copa-expand" "rte-expand" "openbookqa-small" "piqa-small")
dataset_paths=(
  "./fewshots_datasets/input/copa-expand.jsonl"
  "./fewshots_datasets/input/rte-expand.jsonl"
  "./fewshots_datasets/input/openbookqa-small.jsonl"
  "./fewshots_datasets/input/piqa-small.jsonl"
)
declare -A PAD_MUL=(
  # copa-expand
  ["copa-expand|opt-6.7b"]=57
  ["copa-expand|opt-13b"]=54
  ["copa-expand|opt-30b"]=25   # 新测

  # rte-expand
  ["rte-expand|opt-6.7b"]=10
  ["rte-expand|opt-13b"]=10
  ["rte-expand|opt-30b"]=5     # 新测

  # openbookqa-small
  ["openbookqa-small|opt-6.7b"]=54
  ["openbookqa-small|opt-13b"]=54
  ["openbookqa-small|opt-30b"]=27  # 新测

  # piqa-small
  ["piqa-small|opt-6.7b"]=27
  ["piqa-small|opt-13b"]=27
  ["piqa-small|opt-30b"]=10   # 新测
)

declare -A SUF_MUL=(
  # copa-expand
  ["copa-expand|opt-6.7b"]=33
  ["copa-expand|opt-13b"]=33
  ["copa-expand|opt-30b"]=1    # 新测（你给的是 25-1）

  # rte-expand
  ["rte-expand|opt-6.7b"]=8
  ["rte-expand|opt-13b"]=8
  ["rte-expand|opt-30b"]=1     # 新测（5-1）

  # openbookqa-small
  ["openbookqa-small|opt-6.7b"]=32
  ["openbookqa-small|opt-13b"]=32
  ["openbookqa-small|opt-30b"]=15   # 新测（27-15）

  # piqa-small
  ["piqa-small|opt-6.7b"]=15
  ["piqa-small|opt-13b"]=15
  ["piqa-small|opt-30b"]=6   # 新测（14-10）
)
# 以下是在 gpu size = 2048, cpu size = 4096 测出的 padding mul 和 suffix mul
# 新测出的padding_mul=(57-33 10-8 54-32 27-15) 6.7b 
#                    (54-33 10-8 54-32 27-15) 13b
#                    (25-10 (这里需要修改，compute 时间太长了，73883 MiB 0.71:0.53) 5-2 (78757 MiB 0.56:0.50)  27-15 (77545 MiB 0.70:0.60) 14-10 (79970 MiB 0.78:0.67)) ) 30b

# gpu size = 10240, cpu size = 32768
# 30b copa(25-1), rte(5-1), obqa(27-15), piqa(14-10)

get_val() {
  local map_name=$1 key=$2
  local val
  # 间接展开关联数组
  eval 'val="${'"$map_name"'[$key]-__MISSING__}"'
  if [[ "$val" == "__MISSING__" ]]; then
    echo "未找到参数：$map_name[$key]" >&2
    exit 1
  fi
  echo "$val"
}

for di in "${!datasets[@]}"; do
    dataset="${datasets[$di]}"
    dataset_path="${dataset_paths[$di]}"
    sele_percent="${sele_percents[$di]}"
    sim_thred="${sim_threds[$di]}"

    for model in "${models[@]}"; do
        key="${dataset}|${model}"
        padding_mul="$(get_val PAD_MUL "$key")"
        suffix_mul="$(get_val SUF_MUL "$key")"
        model_path="facebook/${model}"
        
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-dump --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --suffix-mul "$suffix_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size"
        # python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-dump --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --suffix-mul "$suffix_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size"
        
        # echo python3 flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --full-load --sele-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-percent "$sele_percent" --no-cache --padding-mul "$padding_mul"  --no-prefetch --generate-mapping-list
        # python3 flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --full-load --sele-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-percent "$sele_percent" --no-cache --padding-mul "$padding_mul"  --no-prefetch --generate-mapping-list

        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-dump --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" --logits --recompute --sep-layer false --suffix-mul "$suffix_mul"
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-dump --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" --logits --recompute --sep-layer false --suffix-mul "$suffix_mul"
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-load --suffix-comp --cache-type LRU --suffix-mul "$suffix_mul" #as
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-load --suffix-comp --cache-type LRU --suffix-mul "$suffix_mul"
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp --cache-type LRU --disk-type KV_Division --sele-percent "$sele_percent" --suffix-mul "$suffix_mul" # as + h2o +lru
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp --cache-type LRU --disk-type KV_Division --sele-percent "$sele_percent" --suffix-mul "$suffix_mul"
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp --cache-type LFU --disk-type KV_Division --sele-percent "$sele_percent" --suffix-mul "$suffix_mul" # as + h2o + lfu
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp --cache-type LFU --disk-type KV_Division --sele-percent "$sele_percent" --suffix-mul "$suffix_mul" 
        echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type CKLFU --sele-percent "$sele_percent" --sim-thred "$sim_thred" --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --reorder --suffix-mul "$suffix_mul" # impress
        time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type CKLFU --sele-percent "$sele_percent" --sim-thred "$sim_thred" --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --reorder --suffix-mul "$suffix_mul"
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type LRU --sele-percent "$sele_percent" --sim-thred "$sim_thred" --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --suffix-mul "$suffix_mul"  #impress - reorder - cklfu
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type LRU --sele-percent "$sele_percent" --sim-thred "$sim_thred" --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --suffix-mul "$suffix_mul" 
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type LRU --sele-percent "$sele_percent" --sim-thred "$sim_thred" --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --reorder --suffix-mul "$suffix_mul" # impress - cklfu
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type LRU --sele-percent "$sele_percent" --sim-thred "$sim_thred" --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --reorder --suffix-mul "$suffix_mul"
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp --cache-type LRU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" # hyperinfer - reorder - cklfu
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp --cache-type LRU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul"
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp --cache-type LRU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" --reorder # hyperinfer - cklfu
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp --cache-type LRU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" --reorder
        # echo python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" --reorder # hyperinfer 
        # time python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf --input-path "$dataset_path" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" --reorder
    done

done
# mkdir logs/performance
# mv logs/*.log logs/performance
# cd scripts
# python performance_log_analys.py
# cd ..