# prism-max

PRISM / ProMixed 稀疏 KV re-prefill 实验的代码和历史结果备份。此仓库来自服务器当前恢复后的 `prism_max` 目录，发布快照基于 Git 提交 `61318220e7df80488292b54b00a87134a031f169`（2026-09-08 归档）。原目录中未跟踪或被忽略的代码、数据、索引和结果也随本次快照纳入版本管理。

本次发布只整理阅读入口、清单和校验记录，没有修改算法或重新运行 GPU 性能实验。它保存当前能够读取的恢复目录；无法据此保证误删前所有未备份文件都已找回。

## 从哪里开始读

1. [ProMixed 实验报告](PROMIXED_EXPERIMENT_REPORT.md)：方法背景、GQA 代表头和原始实验记录。
2. [完整五方法格子报告](results/prism_max_five_method_strict_response_warm1_nodefer_r1_20260828_v2/prism_max_five_method_strict_response_warm1_nodefer_r1.md)：对应历史运行的延迟、准确率与预算对照。
3. [实验索引](docs/EXPERIMENT_INDEX.md)：按原始 summary 定位历史格子；[机器可读索引](docs/experiment_index.json) 保留 runtime、measurement 和逐请求文件位置。
4. [准确率协议审计](ACCURACY_PROTOCOL_K005_AUDIT_20260804_ZH.md)、[阶段 AB 结果](PRISM_PHASE_AB_RESULTS_20260730_ZH.md)、[Paper counts 结果](PAPER_COUNTS_410_RESULTS_ZH.md)。

## 代码地图

| 路径 | 内容 |
|---|---|
| [src/contiguous_fuxian/promixed.py](src/contiguous_fuxian/promixed.py) | GQA 感知选块和 ProMixed 原有自适应层复用决策 |
| [src/contiguous_fuxian/quantized_key_index.py](src/contiguous_fuxian/quantized_key_index.py) | 压缩 key selector 索引与 INT4 解量化 |
| [src/contiguous_fuxian/flexgen_pcache.py](src/contiguous_fuxian/flexgen_pcache.py) | KV 后端接入与缓存流程 |
| [src/contiguous_fuxian/flexgen_qwen_reprefill.py](src/contiguous_fuxian/flexgen_qwen_reprefill.py) | Qwen re-prefill、选择及计时 |
| [vendor/flexgen/my_pcache_fast.py](vendor/flexgen/my_pcache_fast.py) | 保存的真实 payload 后端 |
| [scripts](scripts) / [configs](configs) / [tests](tests) | 运行脚本、层预算与单元测试 |
| [src/prism_gao](src/prism_gao) / [research](research) | 原目录内保存的其他实验代码与研究材料 |
| [results](results) | 原始日志、逐请求结果、汇总与历史归档 |
| [data](data) / [assets](assets) | 目录内现存的任务数据和 selector 索引 |

## 读取结果时的口径

以每个格子的 `summary.json` 中 `runtime`、`measurement` 及相邻 `scored_records.jsonl` 为准。完整五方法格子使用 response-ready 作为主要延迟口径；logits-ready 是中间阶段，evaluation-ready 包含准确率评估。准确率使用完整标签 continuation 的 mean-log-likelihood 协议。其他历史试验可能使用不同口径。

例如 [TREC k010 ProMixed 原始 summary](results/prism_max_five_method_strict_response_warm1_nodefer_r1_20260828_v2/trec/k010_promixed_k4_nodefer_warm1_response_r1/summary.json) 与 [逐请求结果](results/prism_max_five_method_strict_response_warm1_nodefer_r1_20260828_v2/trec/k010_promixed_k4_nodefer_warm1_response_r1/scored_records.jsonl) 可配对读取。`runtime_variant` 中的 adaptive-p8 描述该历史运行的自适应周期设置，不能解释为每层固定 P8。性能结论应指向对应运行的实际字段，不能从本次归档推断新的加速收益。

## 运行与校验

代码保留原服务器路径约定。入口 [run_prism_max_cell.sh](scripts/run_prism_max_cell.sh) 通过环境变量接收 Python、GPU、任务 bundle、模型、现有 KV、selector 索引和新结果目录。模型权重及完整 KV 位于原目录之外，需要另行提供；本仓库的 `assets` 是 selector 索引。复现时应先核对脚本的依赖与现有 KV 条件，并选择新的 `RUN_ROOT` 以保留历史结果。

发布副本执行了以下 CPU 测试（没有启动 GPU 推理），结果为 **98 passed, 17 subtests passed**：

```bash
cd /home/panzihang/src/prism_max_github_publish_20260908/repo
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" \
  /home/panzihang/venvs/vllm-stable/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_flexgen_pcache.py tests/test_flexgen_qwen_reprefill.py \
  tests/test_promixed.py tests/test_quantized_key_index.py tests/test_prism_max_runner.py
```

[完整测试日志](docs/publication/05_cpu_unit_tests.log) 和 [运行脚本](docs/publication/05_cpu_unit_tests.sh) 已保存。此测试覆盖不等于 GPU 性能或全部历史实验的重新验证。

## 快照完整性

[发布说明](docs/PUBLICATION.md) 记录范围、清单和校验。原有 `.gitignore` 原样保留，本次快照显式纳入现存实验文件；后续新增结果如需备份，也需要显式添加。仓库保留原有三个分支及 24 个可达历史提交，当前完整快照位于 `prism-max` 分支。
