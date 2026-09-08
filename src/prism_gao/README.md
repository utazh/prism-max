# prism_gao

这是一个完全隔离的 PRISM 最小实验目录。原来的
`src/contiguous_fuxian`、`scripts` 和 `results` 没有被修改。

## 已实现

- `int4_selector.py`
  - `direct`：INT4 unpack/dequant 与 QK 直接融合。
  - `exact`（默认）：单 Triton kernel 完成 unpack/dequant，再沿用原
    PyTorch BF16 matmul；它与旧路径逐元素、逐选择完全一致。
- `fused_selector_integration.py`
  - 只在当前 Python 进程内 hook 原 runner。
  - packed codes/scales 直接传到 selector，不在 load 阶段生成完整 key。
- `period_control.py`
  - 通过环境变量切换 adaptive/P1/P4/P8，只覆盖复用周期。
- `run_cell.sh` / `run_stage_c.sh`
  - 隔离实验脚本；Stage C 使用同步、无预取的物理读取对照。
- `precision_run_coalescer.py`
  - 16/8/drop 标签规划与短 INT8 run 提升。
- `mixed_kv_materialize.py`
  - FP16 copy + INT8 dequant + original-order scatter 的 Triton 原型。
  - 已通过 mixed reader 接入服务器现有三精度物理条带。
- `mixed_precision_reader.py` / `precision_integration.py`
  - 分组物理读取、融合 materialize 和进程内 runner hook。
- `analyze_ab.py`
  - 对比 summary 和逐请求选择/预测是否完全相同。
- `generate_three_stage_report.py`
  - 生成 GPT_REVIEW_REPORT.md 与机器可读汇总。

## 为什么默认不用 direct

真实 TREC/k010 的 64 条 A/B 中，direct 虽然更快，但由于 Triton
`tl.dot` 与 PyTorch BF16 matmul 的舍入顺序不同：

- 64/64 请求的 `layer_token_selection_sha256` 不同；
- 13/64 最终分类不同；
- 8/64 correctness 不同。

因此 direct 只保留作研究原型，不作为当前可用优化。

默认 exact 路径的反量化结果与旧 PyTorch 路径 bitwise equal，并且两个
64 条实验的选择哈希、Period、预测和正确性全部相同。

## 64 条端到端结果

同一 RTX 3090（GPU 2），1 次 warmup、16 条 warmup samples。

| task/budget | 路径 | accuracy | mean TTFT | p95 TTFT | selector load | selector compute |
|---|---|---:|---:|---:|---:|---:|
| TREC/k010 | torch | 0.625 | 268.57 ms | 330.50 ms | 10.62 ms | 60.62 ms |
| TREC/k010 | exact | 0.625 | 257.31 ms | 330.59 ms | 3.58 ms | 59.03 ms |
| SUBJ/k025 | torch | 0.765625 | 472.53 ms | 544.05 ms | 10.35 ms | 80.04 ms |
| SUBJ/k025 | exact | 0.765625 | 438.77 ms | 518.74 ms | 3.53 ms | 74.48 ms |

mean TTFT 分别下降 4.19% 和 7.15%。这是 64 条筛选实验，不应替代完整
数据集的多次 ABBA 正式验证。

详细机器可读对比：

- `results/trec64/ab_exact.json`
- `results/subj64/ab_exact.json`

## 复现

从项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  /home/panzihang/venvs/vllm-stable/bin/python \
  -m pytest -q src/prism_gao/tests
```

运行 exact selector cell：

```bash
GPU=2 REQUIRE_IDLE_RESERVE=false \
PRISM_GAO_SELECTOR_MODE=exact \
TASK=trec BUDGET_TAG=010 METHOD=promixed SELECTOR_BACKEND=k4 \
SAMPLES_PER_TASK=64 WARMUP_PASSES=1 WARMUP_SAMPLES_PER_TASK=16 \
RUN_NAME=k010_promixed_k4_fused_exact_64 \
RUN_ROOT=/home/panzihang/src/prism_max/src/prism_gao/results/recheck \
bash src/prism_gao/run_cell.sh
```

`PRISM_GAO_SELECTOR_MODE=direct` 只用于数值敏感性实验。
