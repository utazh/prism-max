# ContiguousKV Figure 9-11 模仿实验报告

## 1. 结论摘要

本轮已经完成 Qwen2.5-7B、四个数据集、三个方法、四档 KV 预算的主矩阵：

- 数据集：SST-2、SUBJ、TREC、RTE。
- 方法：IMPRESS、ContiguousKV、Ours。
- KV 预算：5%、10%、25%、50%。
- 每个方法-预算组合测量 128 个顺序单请求，共 12 x 128 = 1536 条正式测量记录。
- Figure 9 对齐四个预算的 Accuracy；Figure 10 对齐 5%/25% 的 Mean TTFT；Figure 11 对齐 5% 下 SST-2/RTE 的 P95 TTFT。

结果不是“全面复现原文趋势”：

1. TTFT 与 P95 的主方向符合原文：ContiguousKV 显著快于 IMPRESS，且低预算优势更大。
2. Accuracy 不符合原文：论文中 ContiguousKV 的 Accuracy 通常优于 IMPRESS；本复现中 ContiguousKV 在四档预算的总体 Accuracy 都低于 IMPRESS，尤其 5% 的 SUBJ 明显失真。
3. Ours 在四档预算的 Accuracy 都不低于 ContiguousKV；在 25%/50% 时还同时降低 Mean TTFT，但在 5%/10% 时延迟高于 ContiguousKV。
4. Ours 相对 IMPRESS 的 Mean TTFT 在四档预算都显著更低；Accuracy 基本相当，但 10% 时低 2.34 个百分点。

因此，当前最可靠的论文叙事是：Ours 修复了低预算 ContiguousKV 的精度问题，并在中高预算形成比 ContiguousKV 更好的精度-延迟折中；不能声称 Ours 在所有预算全面支配 ContiguousKV。

## 2. 论文请求数的更正

ContiguousKV Table 1 中的 100/110/120/80 不是评测请求数，而是 SST-2/SUBJ/TREC/RTE 共享前缀中的 few-shot 示例数。论文没有披露 Figure 9-11 的评测请求总数。

本实验不能伪造“与原文请求数一致”，因此采用可审计的统一协议：每数据集固定 32 个评测请求，总计 128 个。128 表示 128 个依次执行的独立单请求，不是 128 并发。

共享前缀仍按论文 Table 1 构造：

| 数据集 | Few-shot 示例数 | 论文前缀长度 | 实际 Qwen tokenizer 长度 | 评测请求数 |
|---|---:|---:|---:|---:|
| SST-2 | 100 | 3.8k | 3811 | 32 |
| SUBJ | 110 | 4.4k | 4412 | 32 |
| TREC | 120 | 5.0k | 4940 | 32 |
| RTE | 80 | 6.0k | 6011 | 32 |

## 3. 实验设置

- 模型：`/data1/llm/Qwen/Qwen2.5-7B-Instruct`。
- 框架：FlexGen Pcache + Qwen layerwise sparse Re-Prefill 适配。
- 计算精度：BF16；KV 外存精度：FP16。
- 执行：batch size 1，顺序单请求；每个运行先完整 warmup 128 个请求，再测量同一组 128 个请求。
- 数据与随机性：固定 UID，任务 bundle 使用 seed 42。
- 缓存：GPU 55 MiB、CPU 131 MiB、CKLFU。
- 外存：服务器 `/dev/sda4`，设备型号 INTEL SSDSC2KB96，`ROTA=0`，是本机 SSD；不是论文的 Samsung 990 Pro。
- GPU：NVIDIA RTX 3090 24GB，仅使用 GPU 3；实验结束后 GPU 3 为 15 MiB、0% utilization。
- 论文环境差异：论文使用 A800 80GB、GPU/CPU 10GB/24GB 缓存以及 Samsung 990 Pro。因此绝对 TTFT 不能与原文柱高直接相等，只能比较方法间和预算间趋势。

## 4. 总体结果

| KV 预算 | 方法 | Accuracy | 实际预算 | Mean TTFT (ms) | P95 TTFT (ms) |
|---:|---|---:|---:|---:|---:|
| 5% | IMPRESS | 0.8203 | 5.64% | 2192.7 | 3181.6 |
| 5% | ContiguousKV | 0.6328 | 4.96% | 297.7 | 380.0 |
| 5% | Ours | **0.8359** | 4.91% | 527.9 | 659.6 |
| 10% | IMPRESS | **0.8125** | 13.28% | 3024.3 | 4229.0 |
| 10% | ContiguousKV | 0.7188 | 9.95% | **425.8** | **573.3** |
| 10% | Ours | 0.7891 | 9.90% | 728.1 | 905.8 |
| 25% | IMPRESS | 0.8203 | 29.42% | 4221.7 | 6124.2 |
| 25% | ContiguousKV | 0.7422 | 24.98% | 1504.3 | 2319.2 |
| 25% | Ours | **0.8359** | 24.90% | **1272.2** | **2176.2** |
| 50% | IMPRESS | 0.8281 | 54.89% | 4852.1 | 7433.6 |
| 50% | ContiguousKV | 0.8281 | 50.02% | 3661.7 | 5582.8 |
| 50% | Ours | **0.8359** | 49.94% | **3129.6** | **5273.2** |

注意：IMPRESS 使用 64-token 物理块，因此 10%/25%/50% 的实际保留率分别达到 13.28%/29.42%/54.89%。Ours 使用精确的全模型 16-token block 预算，四档预算均为 128/128 请求 `target == consumed`。

## 5. Figure 9：Accuracy 对比

论文报告 ContiguousKV 相对 IMPRESS 在 5%/10%/25%/50% 的平均 Accuracy 提升为 7.69%/4.81%/3.58%/1.63%（跨数据集与模型的论文汇总）。本复现中，ContiguousKV 在 0/4 个总体预算点超过 IMPRESS，因此 Accuracy 趋势不符合原文。

主要问题来自低预算数据集：

- 5% SUBJ：ContiguousKV 0.1875，IMPRESS 0.8125，Ours 0.8438。
- 10% SUBJ：ContiguousKV 0.5000，IMPRESS 0.7500，Ours 0.8125。
- 5% TREC：ContiguousKV 0.6562，IMPRESS/Ours 均为 0.7188。

Ours 相对 ContiguousKV 的总体 Accuracy 提升：

- 5%：+20.31 个百分点，McNemar `p=2.56e-6`。
- 10%：+7.03 个百分点，`p=0.0784`。
- 25%：+9.38 个百分点，`p=0.0118`。
- 50%：+0.78 个百分点，`p=1.0`。

Ours 相对 IMPRESS 的 Accuracy 差值为 +1.56/-2.34/+1.56/+0.78 个百分点，四个点的 McNemar 检验均不显著。因此可以说 Accuracy 接近 IMPRESS，不能声称稳定显著超过 IMPRESS。

### 10% 非单调现象

Ours 的 Accuracy 从 5% 的 0.8359 降到 10% 的 0.7891。配对检查表明这不是记录错位：128 个 UID 完全相同，且 128/128 个请求的选择集合都发生了变化。5% -> 10% 共出现 1 个错误变正确、7 个正确变错误，其中 TREC 为 1 个变正确、5 个变错误。

这说明当前在线选择在增加预算后不是严格单调包含旧集合，更多 KV 可能改变后续层的隐藏状态和选择结果。该点应作为方法限制披露；后续可测试“预算嵌套约束”或单独校准 10% 层预算，但不能删除这个结果。

## 6. Figure 10：Mean TTFT 对比

原文的主要趋势得到复现：ContiguousKV 在 5%/25% 的四数据集共 8 个单元中全部快于 IMPRESS，并且低预算加速更明显。

总体速度比：

| 预算 | ContiguousKV 相对 IMPRESS | Ours 相对 IMPRESS | Ours 相对 ContiguousKV |
|---:|---:|---:|---:|
| 5% | 7.37x | 4.15x | 慢 77.34% |
| 10%（扩展） | 7.10x | 4.15x | 慢 70.98% |
| 25% | 2.81x | 3.32x | **快 15.43%** |
| 50%（扩展） | 1.33x | 1.55x | **快 14.53%** |

Ours 与 ContiguousKV 的配对 TTFT 95% 区间：

- 5%：`[+211.2, +248.3] ms`，Ours 明确更慢。
- 10%：`[+282.3, +322.5] ms`，Ours 明确更慢。
- 25%：`[-273.4, -192.6] ms`，Ours 明确更快。
- 50%：`[-649.7, -414.8] ms`，Ours 明确更快。

这些区间都不跨 0，说明在这 128 个固定请求内，延迟方向稳定。不过每个方法-预算组合本轮只测量一轮；该配对 bootstrap 反映请求级变异，不替代跨轮系统重复实验。

## 7. Figure 11：5% P95 尾延迟

| 数据集 | IMPRESS (ms) | ContiguousKV (ms) | Ours (ms) |
|---|---:|---:|---:|
| SST-2 | 3093.6 | **261.3** | 399.1 |
| RTE | 3254.1 | **363.6** | 688.6 |

这两个数据集都复现了论文的方向：ContiguousKV 的 P95 显著低于 IMPRESS。Ours 也大幅低于 IMPRESS，但没有超过 ContiguousKV；因此当前 Figure 11 不能作为 Ours 优于 ContiguousKV 的证据。

## 8. 异步预取审计

10% Ours 每个请求平均：

- 提交 55 个异步调度任务并完成 55 个：28 current、13 next-layer、14 period；failed=0，cancelled=0。
- 14 个 period 预取任务，平均 period-prefetch 7385.7 tokens。
- 平均 period hit 1122.9 tokens、missing 5276.9 tokens、unused 354.4 tokens。

这证明预取路径实际执行，而不是仅设置了 CLI 开关。它也显示预取仍有优化空间：period 命中覆盖有限，大量目标仍在当前层关键路径中补拉取。

## 9. 可用于论文与不可直接声称的结论

可以支持：

1. Ours 在 25% 和 50% 同时提高 ContiguousKV Accuracy 并降低 Mean/P95 TTFT。
2. Ours 在所有预算都显著快于 IMPRESS，Accuracy 与 IMPRESS 大体相当。
3. 层敏感度 + 在线重要性 + 16-token block + 异步预取形成了有意义的中高预算折中。

当前不能支持：

1. “完整复现了 ContiguousKV Accuracy”——低预算 Accuracy 与原文方向明显冲突。
2. “Ours 在所有预算都优于 ContiguousKV”——5%/10% 的 Mean/P95 TTFT 更慢。
3. “10% Accuracy 随预算单调提高”——实际出现反向变化。
4. “已与原文绝对数值公平对齐”——硬件、缓存容量、评测请求 N 和模型规模范围不同。

## 10. 产物

- `report/figure9_accuracy_qwen25_7b.png`
- `report/figure10_mean_ttft_qwen25_7b.png`
- `report/figure11_p95_qwen25_7b.png`
- `report/extended_mean_ttft_all_budgets.png`
- `report/extended_p95_all_budgets.png`
- `report/fig9_11_report.md`
- `report/fig9_11_report.json`
- `report/fig9_11_overall.csv`
- `report/fig9_11_by_task.csv`
- `report/fig9_11_comparisons.csv`
- `source_and_input_sha256.txt`
- `unit_tests.log`
