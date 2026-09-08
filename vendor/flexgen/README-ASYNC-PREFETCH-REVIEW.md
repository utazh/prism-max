# Async prefetch review notes

This README records the current async prefetch prototype used for the review experiments.
The original files are kept unchanged. The prototype uses:

- `flex_opt_async_prefetch.py`
- `my_pcache_async.py`

Run all commands from this directory:

```bash
cd /home/zrd/IMPRESS-code/h2o_flexgen/flexgen
```

## Environment

Use the migrated conda environment:

```bash
source /home/zrd/miniconda3/etc/profile.d/conda.sh
conda activate impress
```

The HuggingFace cache is on disk:

```bash
export HF_HOME=/mnt/disk0/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/disk0/huggingface_cache/hub
export TRANSFORMERS_CACHE=/mnt/disk0/huggingface_cache/hub
```

The FlexGen numpy weights are separate from the HuggingFace cache. Current usable
converted OPT weights are under:

```bash
/mnt/disk0/data/lrc/opt_weights
```

Use them with:

```bash
--path /mnt/disk0/data/lrc/opt_weights
```

The prefix KV cache directory is a symlink:

```bash
cache -> /mnt/disk0/zrd/cache
```

The old local directory was backed up as:

```bash
/mnt/disk0/zrd/cache.local_backup_20260613_171912
```

## TTFT definition

For the async prefetch path, the logged `ttft_sum` follows this critical-path
definition:

```text
TTFT = sum over layers(
    exposed prefetch wait
  + probe-head key load
  + important-token selection
  + synchronous miss KV load
  + compute path
)
```

The script also prints:

- `async prefetch jobs`
- `wait_time`
- `prefetched_tokens`
- `hit_tokens`
- `miss_tokens`
- `p99`

`wait_time` is the part of async prefetch that was not hidden by computation.

## Prefix dump

Before running prefix-aware inference, dump prefix KV for the same dataset name,
model, and `padding-mul`.

Example for OPT-13B on RTE:

```bash
CUDA_VISIBLE_DEVICES=1 python flex_opt_async_prefetch.py \
  --path /mnt/disk0/data/lrc/opt_weights \
  --gpu-batch-size 1 \
  --overlap false \
  --model facebook/opt-13b \
  --prefix-dump \
  --input-path ./fewshots_datasets/input/rte-expand.jsonl \
  --model-type opt \
  --logits \
  --gen-len 1 \
  --padding-mul 10 \
  --gpu-size 10240 \
  --cpu-size 32768 \
  --sep-layer false \
  --suffix-mul 8
```

## Async prefetch on

Do not pass `--no-prefetch`.

```bash
CUDA_VISIBLE_DEVICES=1 python flex_opt_async_prefetch.py \
  --path /mnt/disk0/data/lrc/opt_weights \
  --gpu-batch-size 1 \
  --overlap false \
  --model facebook/opt-13b \
  --prefix-aware-inf \
  --input-path ./fewshots_datasets/input/rte-expand.jsonl \
  --model-type opt \
  --logits \
  --gen-len 1 \
  --sele-load \
  --sele-percent 25 \
  --sele-load-by-percent \
  --sim-thred 0.3 \
  --suffix-comp \
  --cache-type CKLFU \
  --fill-keys-zero \
  --disk-type KV_Division \
  --gpu-size 10240 \
  --cpu-size 32768 \
  --padding-mul 10 \
  --sep-layer false \
  --chunk-size 64 \
  --suffix-mul 8
```

## Async prefetch off

Add `--no-prefetch` to fall back to synchronous IMPRESS-style loading.

```bash
CUDA_VISIBLE_DEVICES=1 python flex_opt_async_prefetch.py \
  --path /mnt/disk0/data/lrc/opt_weights \
  --gpu-batch-size 1 \
  --overlap false \
  --model facebook/opt-13b \
  --prefix-aware-inf \
  --input-path ./fewshots_datasets/input/rte-expand.jsonl \
  --model-type opt \
  --logits \
  --gen-len 1 \
  --sele-load \
  --sele-percent 25 \
  --sele-load-by-percent \
  --sim-thred 0.3 \
  --suffix-comp \
  --cache-type CKLFU \
  --fill-keys-zero \
  --disk-type KV_Division \
  --gpu-size 10240 \
  --cpu-size 32768 \
  --padding-mul 10 \
  --sep-layer false \
  --chunk-size 64 \
  --suffix-mul 8 \
  --no-prefetch
```

## Three Review Baselines

The review prefix-sensitivity experiments compare three systems:

- `AS`: full prefix-KV loading and full prefix-KV inference.
- `IMPRESS`: selective KV loading, but all selected KVs are loaded synchronously
  on the critical path.
- `HyperInfer`: selective KV loading plus layer-aware asynchronous prefetch.

The following commands use the single-sample prefix-sweep setting with
CPU-resident KVs. Change `GPU_SIZE` and `CPU_SIZE` for SSD-backed runs.

```bash
cd /home/zrd/IMPRESS-code/h2o_flexgen/flexgen
source /home/zrd/miniconda3/etc/profile.d/conda.sh
conda activate impress

export HF_HOME=/mnt/disk0/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/mnt/disk0/huggingface_cache/hub
export TRANSFORMERS_CACHE=/mnt/disk0/huggingface_cache/hub
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="facebook/opt-13b"
WEIGHT_PATH="/mnt/disk0/data/lrc/opt_weights"
INPUT_PATH="fewshots_datasets/input/rte-single-prefix937-suffix50.jsonl"
OUT_DIR="logs_review"
PAD=16
SUFFIX_MUL=20
SELE_PERCENT=25
SIM_THRED=0.3
CHUNK_SIZE=64
GPU_SIZE=0
CPU_SIZE=24000

COMMON_ARGS="\
  --path ${WEIGHT_PATH} \
  --gpu-batch-size 1 \
  --overlap false \
  --model ${MODEL} \
  --prefix-aware-inf \
  --input-path ${INPUT_PATH} \
  --model-type opt \
  --logits \
  --gen-len 1 \
  --suffix-comp \
  --cache-type LRU \
  --disk-type KV_Division \
  --gpu-size ${GPU_SIZE} \
  --cpu-size ${CPU_SIZE} \
  --padding-mul ${PAD} \
  --sep-layer false \
  --chunk-size ${CHUNK_SIZE} \
  --suffix-mul ${SUFFIX_MUL} \
  --reorder"
```

### AS full KV

```bash
python flex_opt_async_prefetch.py ${COMMON_ARGS} \
  --full-load \
  --no-prefetch \
  --log-file "${OUT_DIR}/as_full_load_pad${PAD}.log"
```

### IMPRESS

```bash
python flex_opt_async_prefetch.py ${COMMON_ARGS} \
  --sele-load \
  --sele-percent "${SELE_PERCENT}" \
  --sele-load-by-percent \
  --sim-thred "${SIM_THRED}" \
  --fill-keys-zero \
  --no-prefetch \
  --log-file "${OUT_DIR}/impress_pad${PAD}.log"
```

### HyperInfer

```bash
python flex_opt_async_prefetch.py ${COMMON_ARGS} \
  --sele-load \
  --sele-percent "${SELE_PERCENT}" \
  --sele-load-by-percent \
  --sim-thred "${SIM_THRED}" \
  --fill-keys-zero \
  --log-file "${OUT_DIR}/hyperinfer_pad${PAD}.log"
```

## One-Layer KV I/O Profile

Set `PROFILE_KV_IO_LAYER` to print one-layer critical-path I/O records. The
layer id is zero-indexed. For example, this profiles layer 20:

```bash
export PROFILE_KV_IO_LAYER=20
mkdir -p logs_review/kv_io_profile_layer20
```

Then run the three commands above and use log files such as:

```bash
--log-file logs_review/kv_io_profile_layer20/as_full_load_pad16_layer20.log
--log-file logs_review/kv_io_profile_layer20/impress_pad16_layer20.log
--log-file logs_review/kv_io_profile_layer20/hyperinfer_pad16_layer20.log
```

The script prints lines like:

```text
[kv_io_profile] tag=as_full_kv_load,layer=20,bytes=307036160,gb=0.307036,time_s=0.57825288,bandwidth_gbps=0.530972
```

The reported bandwidth is end-to-end KV-load bandwidth for that code path. It
includes cache lookup, chunk gather, tensor indexing, and CPU-to-GPU movement;
it is not a pure PCIe copy microbenchmark.

## Fast pcache prototype

The newer prototype keeps the original async-prefetch files untouched and uses:

- `flex_opt_fast_pcache.py`
- `my_pcache_fast.py`

The fast pcache path targets CPU-resident `KV_Division` chunks with
`--gpu-size 0`. It avoids the old per-chunk cache promotion/eviction path and
gathers chunk data into a pinned CPU buffer before a larger H2D copy. The async
prefetch path also uses direct chunk-range gather when the prefetch expands
selected tokens to full chunks.

Use the same `COMMON_ARGS` block above, but replace the script name:

```bash
mkdir -p logs_review/fast_pcache_layer20
export PROFILE_KV_IO_LAYER=20
```

AS full KV:

```bash
python flex_opt_fast_pcache.py ${COMMON_ARGS} \
  --full-load \
  --no-prefetch \
  --log-file logs_review/fast_pcache_layer20/as_full_load_pad${PAD}_layer20.log
```

IMPRESS:

```bash
python flex_opt_fast_pcache.py ${COMMON_ARGS} \
  --sele-load \
  --sele-percent "${SELE_PERCENT}" \
  --sele-load-by-percent \
  --sim-thred "${SIM_THRED}" \
  --fill-keys-zero \
  --no-prefetch \
  --log-file logs_review/fast_pcache_layer20/impress_pad${PAD}_layer20.log
```

HyperInfer:

```bash
python flex_opt_fast_pcache.py ${COMMON_ARGS} \
  --sele-load \
  --sele-percent "${SELE_PERCENT}" \
  --sele-load-by-percent \
  --sim-thred "${SIM_THRED}" \
  --fill-keys-zero \
  --log-file logs_review/fast_pcache_layer20/hyperinfer_pad${PAD}_layer20.log
```

## Logs

For review experiments, put logs in:

```bash
logs_review/
```

Recommended naming:

```bash
logs_review/opt13b_rte_prefetch_on.log
logs_review/opt13b_rte_prefetch_off.log
```

Quick summary command:

```bash
grep -E 'ttft_sum|p99|async prefetch|suffix time|total time' logs_review/*.log
```
