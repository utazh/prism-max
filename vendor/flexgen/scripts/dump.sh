# models=("opt-1.3b" "opt-6.7b" "opt-13b" "opt-30b")
models=("opt-6.7b" "opt-13b" "opt-30b")
sele_percents=(50 25 25 25)
sim_threds=(0.5 0.3 0.3 0.3)
gpu_size="10240"
cpu_size="32768"

datasets=("./fewshots_datasets/input/input_expand/copa-expand.jsonl" "./fewshots_datasets/input/input_expand/rte-expand.jsonl" "./fewshots_datasets/input/openbookqa-small.jsonl" "./fewshots_datasets/input/piqa-small.jsonl")
padding_muls=(57 10 54 22)
for i in {0,1,2,3,}; do
    sele_percent=${sele_percents[$i]}
    sim_thred=${sim_threds[$i]}
    dataset=${datasets[$i]}
    padding_mul=${padding_muls[$i]}
    for model in "${models[@]}"; do
    
        if [ "$model" == "opt-30b" ]; then
            padding_mul=$(($padding_mul / 2))
        fi

        model_path="facebook/${model}"
        echo python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-dump --input-path "$dataset" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size"
        python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-dump --input-path "$dataset" --model-type opt --logits --gen-len 1 --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size"
        
        echo python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model "$model_path" \
            --prefix-aware-inf \
            --full-load \
            --sele-inf \
            --sele-percent "$sele_percent" \
            --input-path "$dataset" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache \
            --padding-mul "$padding_mul" \
            --generate-mapping-list 
        python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model "$model_path" \
            --prefix-aware-inf \
            --full-load \
            --sele-inf \
            --sele-percent "$sele_percent" \
            --input-path "$dataset" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache \
            --padding-mul "$padding_mul" \
            --generate-mapping-list 


    done
done