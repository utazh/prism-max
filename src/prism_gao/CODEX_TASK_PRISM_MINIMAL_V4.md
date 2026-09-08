
# PRISM 最小改动 V4：让 A/B/C 各自有明确收益来源

适用代码：`prism_gaov2.zip`

本任务不重写主 attention，不改变 ProMixed 的 block 选择预算，也不做 IMPRESS 式物理重排。只做三个局部改动：

1. A：让 packed INT4 selector index 对 persistent prefix 常驻 GPU（不足时常驻 pinned CPU），继续使用 exact 数值路径；
2. B：把 fixed-period 校准结果写成 workload profile，当前 uncertainty 只作为少量安全降级；
3. C：把当前“host 异步”补成真正的 `SSD host read -> nonblocking H2D -> CUDA stream dequant/materialize -> CUDA event` 流水，并用成本 gate 禁止负收益层使用 INT8。

辅助代码：

- `selector_resident_cache.py`
- `profiled_reuse.py`
- `async_mixed_pipeline.py`
- `reuse_calibration.example.csv`
- `tests/test_minimal_v4.py`

---

## 0. 先确认当前代码的问题

在最新代码中用以下命令定位接入点：

```bash
rg -n "load_layer_packed|QuantizedKeyIndex|selector.int4|exact|dequant|tl\.dot" src
rg -n "fixed.period|promixed.*period|uncertainty|margin|agreement" src
rg -n "ThreadPoolExecutor|materialize|precision.run|synchronize|cuda\.Stream|cuda\.Event" src
```

需要保持的事实：

- Stage A 的 formal 路径继续走 exact：INT4 unpack/dequant 后使用原 BF16/FP16 matmul；不要把 `direct tl.dot` 放进正式结果。
- Stage B 不再继续手调一个全局 P1/P2/P4/P8 阈值表。
- Stage C 不能调用 `torch.cuda.synchronize()` 或 `event.synchronize()`；当前层只能 `current_stream.wait_event(ready_event)`。

---

# A. Packed INT4 selector 常驻

## A1. 安装 helper

```bash
cp selector_resident_cache.py src/prism_gao/
```

在模型/runner 初始化、persistent prefix 已确定后：

```python
from prism_gao.selector_resident_cache import (
    SelectorResidentCache,
    TensorWorkspaceCache,
)

self.selector_cache = SelectorResidentCache(
    self.quantized_key_index.load_layer_packed,
    device=self.device,
    max_gpu_bytes=args.selector_gpu_cache_mb * 1024 * 1024,
    pin_fallback=True,
)
cache_mode = self.selector_cache.preload(range(self.num_layers))
logger.info(
    "selector cache mode=%s bytes=%d",
    cache_mode,
    self.selector_cache.total_bytes,
)
self.selector_dequant_workspace = TensorWorkspaceCache()
```

增加参数：

```python
parser.add_argument(
    "--selector-gpu-cache-mb",
    type=int,
    default=1024,
    help="GPU byte budget for packed INT4 selector index; fallback is pinned CPU.",
)
```

每层 selector 原来是：

```python
packed = self.quantized_key_index.load_layer_packed(layer_idx)
```

改为：

```python
packed = self.selector_cache.get(
    layer_idx,
    stream=self.selector_stream,
)
```

## A2. 复用 exact dequant 输出

现有 exact Triton kernel 增加一个可选 `out` 参数，不要每层 `torch.empty`：

```python
shape = (prefix_tokens, num_kv_heads, head_dim)
dequant_out = self.selector_dequant_workspace.get(
    shape,
    dtype=torch.bfloat16,
    device=self.device,
)
dequant_int4_exact(
    packed.codes,
    packed.scales,
    out=dequant_out,
    # 原有 layout/group-size 参数保持不变
)
prefix_logits = torch.matmul(query, dequant_out.transpose(-1, -2))
```

要求：

- 结果 hash、selected block、prediction 必须与旧 exact 路径完全一致；
- prefix 改变时按 prefix hash 重建 cache；
- 单独报告一次性的 preload 时间，不把它重复计入每个请求；
- formal benchmark 使用 warm persistent-prefix；cold-start 另表报告。

## A3. 对比

每个 cell 按双顺序：

```text
FP16 -> K4-exact-disk -> K4-exact-resident
K4-exact-resident -> K4-exact-disk -> FP16
```

优先跑：

```text
SST2/k010
SUBJ/k025
TREC/k010
RTE/k025
```

通过条件：

```text
resident exact 与旧 exact 的 selection/prediction 完全一致
resident exact 相对 exact-disk 的 response-ready 宏平均下降 >= 3%
resident exact 相对 FP16 selector 在 warm persistent-prefix 下不更慢
```

若第三项仍不满足，不继续优化 unpack；进入 Quest 风格“两阶段索引”：

```text
block min/max upper bound -> 取 2K candidate blocks -> exact K4 只评分 candidates
```

这是 A 的下一步，不在本补丁内实现。

---

# B. Workload-profiled adaptive reuse

这不是在测试集上硬编码答案。必须使用独立 calibration IDs，例如每个 task/budget 16 条；正式评估用不重叠的 64 条。

## B1. 跑固定周期校准

先只覆盖最有区分度的两个预算：

```text
tasks: SST2, SUBJ, TREC, RTE
budgets: 0.10, 0.50
periods: 1, 2, 4, 8
calibration samples: 16
```

将每次结果汇总为 CSV：

```csv
task,budget,period,correct,total,mean_ttft_ms
trec,0.10,1,11,16,340.0
trec,0.10,2,11,16,310.0
trec,0.10,4,11,16,285.0
trec,0.10,8,10,16,240.0
```

生成 profile：

```bash
PYTHONPATH=src python -m prism_gao.profiled_reuse \
  --input-csv results/reuse_calibration.csv \
  --output-json results/reuse_profile.json \
  --quality-slack-examples 1 \
  --latency-tie-fraction 0.015 \
  --guard-threshold 0.95
```

规则非常简单：

1. 先找到 calibration accuracy 最好的 fixed period；
2. 允许最多少 1 条；
3. 在这些候选中选 TTFT 最低的；
4. 若 TTFT 相差不超过 1.5%，优先更高 accuracy，再优先更长 period。

## B2. Runtime 接入

```python
from prism_gao.profiled_reuse import ProfiledAdaptiveReuse

self.reuse_policy = ProfiledAdaptiveReuse.from_json(
    args.promixed_reuse_profile
)
```

原来：

```python
period = period_from_uncertainty(uncertainty, ...)
```

改为：

```python
period = self.reuse_policy.choose(
    task=args.task,
    budget=global_keep_ratio,
    uncertainty=uncertainty,
)
```

这个 policy 的行为是：

```text
正常请求：直接使用该 task/budget 在 calibration 上最好的 base period
极高 uncertainty（默认 >= 0.95）：只缩短一级，例如 P8 -> P4、P4 -> P2
```

不要再叠加旧的 budget-v2 阈值。只保留一个策略来源。

增加 CLI：

```python
parser.add_argument("--promixed-reuse-profile", type=str, default="")
```

没有 profile 时退回旧 adaptive，保证可复现。

## B3. 正式比较

```text
P1
P2
P4
P8
old adaptive
profiled adaptive
```

先跑 4 tasks × {k010,k050} × 64 held-out samples。通过后再补 k005/k025。

通过条件：

```text
profiled adaptive accuracy >= 各 cell 最佳 fixed accuracy - 1/64
profiled adaptive TTFT 明显低于 P1，并接近该 cell 最快的合格 fixed period
4-task 宏平均同时优于 old adaptive 的 accuracy 和 response-ready
```

论文名称建议使用：

```text
workload-profiled adaptive cross-layer reuse
```

不要写成完全 task-agnostic online adaptation。

---

# C. 真正异步的 16/8/drop

当前 host worker 只隐藏 `pread`。如果 H2D、INT8 dequant 和 materialize 在当前层调用并同步，它们仍在关键路径上，因此 mixed 可能输给原 FP16 async。

## C1. 安装 helper

```bash
cp async_mixed_pipeline.py src/prism_gao/
```

## C2. Host reader 直接读入 pinned memory

对于每个连续 FP16/INT8 run，替换：

```text
os.pread -> bytes -> numpy -> torch tensor
```

为：

```python
from prism_gao.async_mixed_pipeline import pread_into_pinned

raw = pread_into_pinned(
    fd,
    nbytes=run_nbytes,
    offset=run_file_offset,
)
```

`raw` 是 pinned CPU tensor，后续才能真正使用：

```python
raw_gpu = raw.to(device, non_blocking=True)
```

继续保留现有 run coalescing：

```text
短 INT8 run -> FP16
drop 不变
原始 block 顺序不变
```

## C3. 建立 pipeline

```python
from prism_gao.async_mixed_pipeline import AsyncMixedPipeline

self.mixed_pipeline = AsyncMixedPipeline(
    read_host=self.mixed_reader.read_kv_host,
    materialize_gpu=self.mixed_reader.materialize_kv_gpu,
    device=self.device,
    max_workers=1,
)
```

`materialize_kv_gpu(payload, plan, stream)` 必须：

```python
def materialize_kv_gpu(payload, plan, stream):
    # 当前已在 with torch.cuda.stream(stream) 中
    # 1. pinned CPU -> GPU，全部 non_blocking=True
    # 2. 调用现有 mixed materialize Triton kernel
    # 3. 返回统一 BF16/FP16 K/V
    # 4. 禁止任何 synchronize()
    return key_out, value_out
```

leader 得到 reuse window 后立即提交 future layers：

```python
for target_layer, plan in plans_for_window:
    self.mixed_pipeline.submit(
        (request_id, target_layer),
        layer=target_layer,
        plan=plan,
    )
```

每完成一个 transformer layer 后调用一次：

```python
self.mixed_pipeline.poll()
```

目标层真正使用 KV 时：

```python
key, value = self.mixed_pipeline.resolve(
    (request_id, layer_idx)
)
```

`resolve` 只执行：

```python
torch.cuda.current_stream().wait_event(ready_event)
```

删除 mixed path 内所有：

```python
torch.cuda.synchronize()
event.synchronize()
stream.synchronize()
```

## C4. 加一个 layer-level cost gate

先通过 microbenchmark 得到：

```text
SSD/host 实际 GiB/s
INT8 materialize/dequant giga-elements/s
FP16 async baseline 中 host-read 有多少比例实际暴露在关键路径
```

接入：

```python
from prism_gao.async_mixed_pipeline import (
    MixedCostCalibration,
    MixedPrecisionGate,
)

self.mixed_gate = MixedPrecisionGate(
    MixedCostCalibration(
        io_gib_per_s=args.mixed_io_gib_s,
        dequant_giga_elements_per_s=args.mixed_dequant_gelem_s,
        launch_overhead_us=args.mixed_launch_us,
        exposed_io_fraction=args.mixed_exposed_io_fraction,
        minimum_gain_ms=0.05,
    )
)
```

生成 coalesced plan 后：

```python
keep_mixed = self.mixed_gate.keep_mixed(
    fp16_bytes=plan.fp16_baseline_bytes,
    mixed_bytes=plan.actual_read_bytes,
    int8_elements=plan.int8_elements,
    int8_runs=plan.int8_run_count,
)

if not keep_mixed:
    plan = plan.promote_all_int8_to_fp16()
```

这是最重要的 non-regression 规则：

```text
异步 FP16 已经完全隐藏 I/O时，不使用 INT8；
只有预计节省的“暴露 I/O”大于反量化和 launch 成本时才启用 INT8。
```

第一版不要写 full mixed attention。

## C5. 公平比较

三组必须共用同一个新 pipeline：

```text
payload_fp16_async
naive_16_8_async
coalesced_gated_16_8_async
```

另列原系统：

```text
original_pcache_fp16_async
```

算法收益看：

```text
coalesced_gated_16_8_async vs payload_fp16_async
```

系统收益看：

```text
coalesced_gated_16_8_async vs original_pcache_fp16_async
```

运行双顺序，并保持同样 warmup：

```text
FP16 -> mixed
mixed -> FP16
```

先跑 4 tasks × {k010,k050} × 64 samples。

记录：

```text
host_read_ms
host_future_wait_ms
H2D bytes
int8_dequant/materialize_ms
CUDA event 是否在 resolve 前 ready
precision gate activation rate
INT8 runs before/after
response-ready
accuracy
```

通过条件：

```text
mixed gate 开启的 cell/layer：response-ready 相对 payload FP16 有正收益
所有 cell 宏平均不慢于 payload FP16
准确率每个 64-sample cell 最多少 1 条
```

如果 gate activation 接近 0，说明原 FP16 async 已经把 SSD I/O 几乎完全隐藏。在这种情况下，16/8/drop 无法仅靠减少读取字节获得 TTFT 收益；下一步只能使用直接消费 INT8 KV 的 fused attention kernel，而不是继续调整 run 阈值。

---

# 最小执行顺序

```text
1. A resident cache：4 个 cell，双顺序
2. B calibration：4 tasks × 2 budgets × 4 periods × 16 samples
3. B held-out：4 tasks × 2 budgets × 64 samples
4. C microbenchmark：得到 4 个 cost 参数
5. C async：4 tasks × 2 budgets × 64 samples，双顺序
6. 三阶段均通过后，再补 k005/k025 和正式 ABBA
```

---

# 单元测试

```bash
cp profiled_reuse.py async_mixed_pipeline.py selector_resident_cache.py \
  src/prism_gao/

PYTHONPATH=src pytest -q \
  tests/test_minimal_v4.py
```

本辅助包自身已通过：

```text
compileall: pass
pytest: 5 passed
```

CUDA 路径必须在目标 RTX 3090 上运行，不能用 CPU 测试替代。
