# 此README用来指导复现SPEC的实验

## 后续命令都在h2o_flexgen/flexgen下运行

```bash 
cd h2o_flexgen/flexgen
```

## SPEC是IMPRESS的扩刊工作，在impress的基础上增加了投机预取的技术，进一步隐藏关键路径上的I/O

### 我们通过一些离线的profile找到合适的padding mul和suffix_mul，即前缀和后缀的长度，让计算时间和关键路径上的load时间比例有利于预取。
#### datasets=("copa-expand.jsonl" "e-expand.jsonl" "openbookqa-small.jsonl" "piqa-small.jsonl")
#### opt-6.7b padding_mul-suffix_mul = (57-33 10-8 54-32 27-15) 
#### opt-13b  padding_mul-suffix_mul = (54-33 10-8 54-32 27-15)
#### opt-30b  padding_mul-suffix_mul = (25-10  5-2 27-15 14-10)

### dump prefix kv，示例：
```bash
time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --input-path "$input_path" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --prefix-dump \
            --no-cache \
            --padding-mul 57 \
            
# input_path和model根据需要填写或者更换
```
### 开启 reorder 需要执行 generate-mapping-list ：
```bash
time python3 flex_opt.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf --full-load --sele-inf --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --sele-percent 50 --no-cache --padding-mul 10  --no-prefetch --generate-mapping-list
```

### 开启自适应预取策略（默认），示例(rte 6.7b)：
```bash
time python3 flex_opt.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --sele-load --sele-percent 25 --sele-load-by-percent --sim-thred 0.3 --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul 10 --sep-layer false --chunk-size 64 --suffix-mul 8 --reorder
```

### 不开启预取，回退到 IMPRESS，示例：
```bash
time python3 flex_opt.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf  --input-path ./fewshots_datasets/input/copa-expand.jsonl --model-type opt --logits --gen-len 1 --sele-load --sele-percent 25 --sele-load-by-percent --sim-thred 0.3 --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul 57 --sep-layer false --reorder --chunk-size 64 --suffix-mul 55 --no-prefetch
```

### 30b + copa IMPRESS
```bash
time python3 flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-30b --prefix-aware-inf  --input-path ./fewshots_datasets/input/copa-expand.jsonl --model-type opt --logits --gen-len 1 --sele-load --sele-percent 50 --sele-load-by-percent --sim-thred 0.5 --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul 25 --sep-layer false --reorder --chunk-size 64 --suffix-mul 10 --no-prefetch
```


### 开启固定比例预取，示例（这个还没有调好，还有问题）：
```bash 
time python3 flex_opt.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf  --input-path ./fewshots_datasets/input/copa-expand.jsonl --model-type opt --logits --gen-len 1 --sele-load --sele-percent 25 --sele-load-by-percent --sim-thred 0.3 --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul 57 --sep-layer false --reorder --prefetch-ratio 10 --solid-ratio-prefetch --chunk-size 64 --suffix-mul 55
```

### 输出两个时间日志的差异和加速比（在 time_logs/下执行，更换模型时需要修改脚本内模型层数，如13b是40层），取前k个，示例：
```bash
python3 cal_parallel_benefits_sorted.py file1.txt file2.txt -k $样本数量
```

## prefix dump
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-dump --input-path fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10
```

## generate maiiping list
```bash
python3 flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf --full-load --sele-inf --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --sele-percent 50 --no-cache --padding-mul 10  --no-prefetch --generate-mapping-list
```

## 重计算 时间=dt_all
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-dump --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --recompute --sep-layer false --suffix-mul 8
```

## as-like 时间=dt_suffix+dt_load
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-load --suffix-comp --cache-type LRU --suffix-mul 8
```

## as_h2o+lru 时间=dt_key+dt_sele+dt_value+dt_suffix
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp --cache-type LRU --disk-type KV_Division --sele-percent 25 --suffix-mul 8
```

## as+h2o+lfu 时间=dt_key+dt_sele+dt_value+dt_suffix
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp --cache-type LFU --disk-type KV_Division --sele-percent 25 --suffix-mul 8
```

## impress-cklfu-reorder (ITF) 时间=dt_load_head+dt_sele+dt_load+dt_suffix
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type LRU --sele-percent 25 --sim-thred 0.3 --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --suffix-mul 8
```

## impress-cklfu 时间=dt_load_head+dt_sele+dt_load+dt_suffix
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type LRU --sele-percent 25 --sim-thred 0.3 --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --reorder --suffix-mul 8
```

## impress 时间=dt_load_head+dt_sele+dt_load+dt_suffix
```bash
python flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --padding-mul 10 --gpu-size 10240 --cpu-size 32768 --logits --sep-layer false --prefix-aware-inf --sele-load --suffix-comp --cache-type CKLFU --sele-percent 25 --sim-thred 0.3 --disk-type KV_Division --sele-load-by-percent --fill-keys-zero --no-prefetch --reorder --suffix-mul 8
```

## impress+prefetch 时间=dt_load_head+dt_sele+dt_suffix
```bash
python3 flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --sele-load --sele-percent 25 --sele-load-by-percent --sim-thred 0.3 --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul 10 --sep-layer false --suffix-mul 8
```

## impress+prefetch+reorder 时间=dt_load_head+dt_sele+dt_suffix
```bash
python3 flex_opt_bk.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b --prefix-aware-inf --input-path ./fewshots_datasets/input/rte-expand.jsonl --model-type opt --logits --gen-len 1 --sele-load --sele-percent 25 --sele-load-by-percent --sim-thred 0.3 --suffix-comp --cache-type CKLFU --fill-keys-zero --disk-type KV_Division --gpu-size 10240 --cpu-size 32768 --padding-mul 10 --sep-layer false --suffix-mul 8 --reorder
```

### 复现论文所有图例
```bash
./scripts/spec_performance.sh # 后面完善
```