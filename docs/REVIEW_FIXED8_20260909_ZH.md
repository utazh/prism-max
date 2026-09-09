# PRISM 固定 P8 审查补丁与最小选块实验

基准提交：`5dd35b501a3a4025dda5c81b6aa348d19af4f7a5`。
本补丁没有更新历史实验表，没有运行真实大模型/GPU/SSD 基准。新增 CPU 回归测试 48 项通过；另有 1000 个有效输入的历史选块结果等价检查，以及 100 个 INT4 量化结果字节等价检查。它们不代表模型质量、端到端性能或所有原有测试已通过。

## 代码交付范围

* `promixed.py`：新增显式固定周期、真正关闭轮询覆盖的开关；历史默认行为不变。新的 `selection_experiment` 是运行范围内的 ContextVar，不替换全局函数。原解码器实际消费返回的 `decision.period`，因此新入口确实在 0/8/16/24 层重新选块，不是仅修改日志。
* `quantized_key_index.py`：固定周期实验可以只预加载锚点层索引；默认入口仍预加载全部层。校验整数码、偶数非空宽度、非法字节、非有限/负 scales 等。合法既有 INT4 量化公式和字节格式不变。
* `sparse_qwen_reprefill.py`：写入 `.bf16` 前显式转换为 BF16，修复 `prepare --dtype float16` 写出 FP16 位模式但读取为 BF16 的错误。默认 BF16 构建不受这项修复影响。旧 FP16 构建的错误缓存需要另建新目录重建，不能靠新 reader 修复。
* `group_coverage.py`：实验性的组覆盖均衡贪心选择，以及 raw mean / normalized mean / max 对照。它不是已证实有收益的算法。
* `audited_reprefill.py`、`run_prism_fixed8.sh`：完整新运行入口，仍调用原来的 `run_flexgen_reprefill`、decoder 和 Pcache；检查 prefix token hash、实际周期日志，避免覆盖原结果。FP16 选择器仍走原 Pcache 路径，不冒称已经实现“CPU 常驻 FP16 的公平对照”。

## 先运行

在仓库根目录，使用原服务器能够运行 PRISM 的 Python 环境：

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_review_fixed8.py

# 不启动 GPU、不写结果目录；但会读取本机模型配置、store metadata 和 bundle。
PYTHON=/home/panzihang/venvs/vllm-stable/bin/python \
  bash scripts/run_prism_fixed8.sh --dry-run

# 真正运行：固定 P8 + 原 GQA 选择规则 + 均匀层预算 + INT4 索引。
# 模型/KV/selector 路径沿用原服务器默认值，也可通过环境变量覆盖。
PYTHON=/home/panzihang/venvs/vllm-stable/bin/python \
  TASK=trec SELECTION=legacy SAMPLES_PER_TASK=128 \
  RUN_DIR="$PWD/results/review_fixed8/trec_legacy_k010_r1" \
  bash scripts/run_prism_fixed8.sh

# 单独替换选块规则，其他配置保持一致。
PYTHON=/home/panzihang/venvs/vllm-stable/bin/python \
  TASK=trec SELECTION=balanced SAMPLES_PER_TASK=128 \
  RUN_DIR="$PWD/results/review_fixed8/trec_balanced_k010_r1" \
  bash scripts/run_prism_fixed8.sh
```

`KEEP_RATIO` 可取 0.05/0.10/0.25/0.50；`SELECTOR_BACKEND=fp16` 关闭 INT4 sidecar。`SELECTION=mean` / `normalized_mean` / `max` 提供同代表头、同固定周期的对照。`--no-coverage` 才是真正关闭旧的按组轮询保留；历史 `coverage_fraction=0` 本来仍有最少组覆盖，本补丁保留旧默认语义，避免悄悄改变历史实验。

新入口默认**均匀层预算**。与历史分层预算比较时显式设置：

```bash
LAYER_BUDGET_PROFILE="$PWD/configs/layer_budget_k010_sensitivity.json" \
  RUN_DIR="$PWD/results/review_fixed8/trec_fixed8_profile_r1" \
  bash scripts/run_prism_fixed8.sh
```

先让 legacy 和 balanced 使用相同的均匀预算，再比较层预算分配；不要将两种变化同时记到新选择算法名下。精确预算以真实 block/token 计数为准，尾块可能不满 16 token。

`--no-anchor-preload` 用于隔离索引常驻空间优化。28 层固定 P8 时只需预加载 0/8/16/24 四层；在每层大小相同时，减少的是索引常驻空间的 6/7，**不是整个系统内存的 6/7**。保留磁盘中原有全部层索引，不删除任何缓存。

`--trace` 输出 `selector_invocation_trace.jsonl`，含分数矩阵、选中索引、每组保留注意力质量以及相对本组 top-k 的覆盖比。此日志包括 warmup 和不同层预算的选择函数调用，不带请求/层归因，不能把每条当作独立请求。开启会增加计时内开销，summary 明确标记，禁止作为正式延迟结果。

## 核实的关键问题

1. TREC 2026-08-20 的原始 FP16 selector 消融为 67.7419%，同阶段 K4 为 63.9113%，不是 K4 更准。参考 `results/promixed_ablation_20260820/trec/k010_promixed_fp16/summary.json` 和 `results/promixed_full_grid_20260819/trec/k010_promixed_k4/summary.json`。两者还同时改变驻留路径和实际自适应周期，不能把时延差纯归因于 INT4 数值计算。
2. 当前 `qwen_online_prefix_head_scores` 先将 selector K 转成 Q 的计算 dtype，softmax 用 float；K4 不是独享 FP32 打分。它按 query 维求和，没有发现“把 softmax 的 key 维加成常数”的错误。ContiguousKV 使用全部 Q heads，不是只使用 CLI 里的 0/1/2。
3. 跨组 Jaccard/边界差不是跨层稳定性测量。固定 P8 后可以删除在线周期决策这条研究支线。旧 decoder 的模 8 fallback selector 预取可能发生在自适应非 leader 层，值得查看未消费 selector 请求；不能把 selector_calls 与决策数不等直接判为唯一原因。
4. 2026-08-28 strict 五方法中的 TREC K10：payload GPU/CPU/disk 来源约为 93.59%/5.30%/1.11%。`total_ssd_read_bytes` 是按软件来源映射估算，不是 NVMe 设备计数；原协议设备是 `/dev/sda4`，也不能不核实硬件就写成 NVMe。
5. k010 层预算配置明确说明直接由 k050 比例缩放，未在 k010 重新标定。保留它作为对照，不建议继续把它当核心贡献。
6. 495 与 496 的 TREC 样本数不同来自 strict bundle 显式排除 `trec-0`；协议样本上限为 1000000。不能错误归因为 `-1` 切片。不同 bundle 不可混用为成对比较。

## 最小方法假设：共享物理块的组覆盖均衡

当一个物理块包含所有 KV heads 时，不能无限合并各组 top-k，否则超出物理预算；也不应让大多数头重复偏好的块占满预算。对每组计算已选块的分数和，除以该组单独保留 k 块时可取得的分数和，得到相对覆盖比 r_g(S)。实验目标为 `sum_g sqrt(r_g(S))`。每次加入使此目标增加最多的块，直到 k 块。

这替代固定 50% 轮询配额及 0.55/0.35/0.10 融合权重，而非再叠加模块。平方根使重复照顾已高覆盖组的收益递减，但不保证每组最低覆盖，更不保证答案质量。代码复杂度为 O(G*B*k)，是 CPU NumPy 参考实现；必须将选块开销计入 response-ready。它使用经典的凹覆盖目标，不应声称发明了新优化理论或首次考虑 GQA。

必须对比同 heads/同 P8/同物理预算的 mean、normalized_mean、max、legacy RR+fusion；还需与 Ada-KV、CompactAttention 在缓存粒度和执行路径上区分。先用真实 trace 检查少数组的覆盖确实不足、与错误相关，再跑质量实验；如果当前规则没有这种缺陷或新方法无改善，就不要强讲这个故事。

## 分级精度后续

本补丁没有实现图里的 FP16/INT8/INT4 payload 多档存储。后续先固定索引，测真实 KV8/4 的质量，再测物理搬运；不把 selector 量化混成 payload 量化。优先比较统一 8 位以及 K/V 非对称量化，再决定是否值得做按块混合。已就绪的高精度副本不应为了“服从位宽策略”而绕路到磁盘。高命中场景先测索引/CPU控制/复制开销，不能靠人为清空缓存制造 SSD 故事。

阅读近邻：ContiguousKV (arXiv:2601.13631)、SpeCache (ICML 2025)、Ada-KV (NeurIPS 2025)、CompactAttention (arXiv:2605.16839)、KVTuner (ICML 2025)、MiKV (arXiv:2402.18096)、CacheGen (SIGCOMM 2024)、Tutti (arXiv:2605.03375)、RateQuant (arXiv:2605.06675)。SemKV (arXiv:2608.28911) 是本次检索到的新预印本，只按摘要作为待核查线索，未复核全文实验。
