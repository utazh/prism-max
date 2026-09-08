# 5% KV 预算：准确率协议与独立数据集审计

日期：2026-08-04

## 结论

1. 旧的“贪心生成最多 4 token + 字符串标签匹配”确实会误判部分格式回声，但不是 ContiguousKV 低准确率的唯一原因。
2. HyperInfer 官方 `--logits --gen-len 1` 路径实际对完整候选 continuation 的 token log-probability 求平均，再在候选之间取最大值；它不是仅比较标签首 token。
3. 同一 128 条 ContiguousKV 请求中：旧生成匹配、首 token 约束、完整候选平均 log-prob 的正确数分别是 81、89、81。完整协议相对旧协议有 9 条错误变正确，同时有 9 条正确变错误，净准确率不变。
4. 新的 P8 已知周期预取只改变调度，不改变选择。四个数据集的 128 条请求中，开关前后选择哈希、首 token logits、预测和正确性均为 0 个不一致。

## 评测协议

### 旧协议

- 最多贪心生成 4 token。
- 对生成文本做规范化后与标签做严格/前缀匹配。
- 旧记录：`server_results/prism_fig9_11_20260803/k005_contiguouskv/scored_records.jsonl`。

### 首 token 约束协议

- 只比较每个合法标签第一个 continuation token 的 raw logit。
- 单 token 标签下，它与该位置的 log-softmax 排序等价。
- 多 token 标签下，它不等价于 HyperInfer 官方候选打分。

### 完整候选协议

- 第一个标签 token 使用 query prefill 最后位置的 logits。
- 后续标签 token 在同一稀疏 KV 状态的独立 cache 分支上 teacher-force。
- 每个候选按 token log-probability 的平均值打分。
- 补评分发生在 TTFT 截止点之后，写入逐 token 分数与独立补评分耗时。
- 实现：`src/contiguous_fuxian/flexgen_qwen_reprefill.py` 和 `src/contiguous_fuxian/paper_client.py`。

Qwen tokenizer 下，SST-2 标签均为单 token；SubJ、TREC、RTE 均存在多 token 标签。因此首 token 结果不能作为后三个数据集的正式准确率。

## 128 条协议翻转

| 协议 | 正确数 | Accuracy |
|---|---:|---:|
| 旧生成字符串匹配 | 81/128 | 63.2813% |
| 合法标签首 token | 89/128 | 69.5313% |
| 完整候选平均 log-prob | 81/128 | 63.2813% |

| 转换 | 错 -> 对 | 对 -> 错 | 对 -> 对 | 错 -> 错 |
|---|---:|---:|---:|---:|
| 旧生成 -> 首 token | 12 | 4 | 77 | 35 |
| 首 token -> 完整候选 | 2 | 10 | 79 | 37 |
| 旧生成 -> 完整候选 | 9 | 9 | 72 | 38 |

旧协议的 47 条错误中，按 `Sentence/Subjectivity/Type/Sentiment` 宽松标记有 43 条格式回声。首 token 修正其中 12 条，完整候选修正其中 9 条；43 条并不都是无害截断，很多文本同时包含错误标签。

## 5% 预算独立筛选

每个数据集独立进程，N=32，warm-up=32，Qwen2.5-7B-Instruct，BF16 计算，Pcache FP16 存储，GPU/CPU cache 为 55/131 MiB。准确率使用完整候选平均 log-prob。Ours 使用 16-token block、逐层敏感度预算、P8 在线重要性复用、value-order 0.9 和已知周期预取。

| 数据集 | 方法 | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |
|---|---|---:|---:|---:|
| SST-2 | IMPRESS | 0.8438 | 807.81 | 976.19 |
| SST-2 | ContiguousKV | 0.8125 | 197.72 | 214.79 |
| SST-2 | Ours | 0.8125 | 169.17 | 204.91 |
| SubJ | IMPRESS | 0.6250 | 848.41 | 1153.06 |
| SubJ | ContiguousKV | 0.3750 | 242.90 | 274.65 |
| SubJ | Ours | 0.5000 | 188.12 | 213.82 |
| TREC | IMPRESS | 0.4375 | 1001.04 | 1283.56 |
| TREC | ContiguousKV | 0.4688 | 372.00 | 402.46 |
| TREC | Ours | 0.5312 | 197.34 | 228.42 |
| RTE | IMPRESS | 0.9375 | 1457.08 | 1774.23 |
| RTE | ContiguousKV | 0.8750 | 311.97 | 351.55 |
| RTE | Ours | 0.9062 | 235.30 | 272.64 |

| 数据集 | Ours accuracy - ContiguousKV | Mean TTFT 降低 | P95 降低 |
|---|---:|---:|---:|
| SST-2 | +0.00 pp | 14.44% | 4.60% |
| SubJ | +12.50 pp | 22.55% | 22.15% |
| TREC | +6.25 pp | 46.95% | 43.24% |
| RTE | +3.12 pp | 24.58% | 22.45% |

这些是筛选结果，不作为最终论文数字。正式结果必须使用每个数据集的完整评测 split，并分别报告，禁止汇总为一个 410 条总体指标。

## 5% 预算全量正式结果

每个任务独立运行；warm-up 固定为该任务前 32 条，随后测量完整 eval split。下表不计算跨数据集总体数。

| 数据集 | 方法 | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |
|---|---|---:|---:|---:|---:|
| SST-2 | IMPRESS | 868 | 0.9343 | 725.02 | 1009.56 |
| SST-2 | ContiguousKV | 868 | 0.8952 | 200.30 | 228.39 |
| SST-2 | Ours | 868 | 0.9182 | 153.71 | 184.09 |
| SubJ | IMPRESS | 999 | 0.6577 | 893.47 | 1212.02 |
| SubJ | ContiguousKV | 999 | 0.4905 | 227.94 | 262.53 |
| SubJ | Ours | 999 | 0.5255 | 187.58 | 221.79 |
| TREC | IMPRESS | 496 | 0.3589 | 977.91 | 1306.83 |
| TREC | ContiguousKV | 496 | 0.5081 | 341.04 | 389.70 |
| TREC | Ours | 496 | 0.5444 | 198.56 | 229.77 |
| RTE | IMPRESS | 273 | 0.8974 | 1328.65 | 1758.42 |
| RTE | ContiguousKV | 273 | 0.8645 | 295.05 | 337.89 |
| RTE | Ours | 273 | 0.8755 | 238.89 | 296.83 |

### Ours 与 ContiguousKV 的配对结果

| 数据集 | Accuracy delta | W->C/C->W | McNemar p | Mean TTFT 降低 | 配对均值差 95% CI (ms) | P95 降低 | SSD 降低 | Ours 更快请求 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | +2.30 pp | 27/7 | 0.000821 | 23.26% | [-48.61, -44.51] | 19.40% | 82.70% | 848/868 |
| SubJ | +3.50 pp | 122/87 | 0.01847 | 17.71% | [-42.63, -38.13] | 15.52% | 74.25% | 952/999 |
| TREC | +3.63 pp | 49/31 | 0.05666 | 41.78% | [-146.40, -138.38] | 41.04% | 77.20% | 495/496 |
| RTE | +1.10 pp | 10/7 | 0.6291 | 19.03% | [-60.66, -51.91] | 12.15% | 73.78% | 261/273 |

四个数据集的 TTFT 配对区间均完全小于 0，支持 Ours 的速度优势。准确率提升在 SST-2 和 SubJ 达到 0.05 显著性；TREC 接近但未达到，RTE 不显著。因此不能声称四个数据集准确率都显著更高。

Ours 也不是对 IMPRESS 的无条件精度支配：SST-2、SubJ、RTE 的准确率分别低 1.61、13.21、2.20 pp；TREC 高 18.55 pp。Ours 对 IMPRESS 的主要稳定优势是平均 TTFT 降低约 78.8%-82.0%，而不是所有任务精度最高。

## 已知周期预取验证

P8 在线重要性选择在周期首层一次得到当前周期各层的实际选择；新路径立即提交这些已知选择。相对原预测式 next/period 预取：

- `impress_next_prefetch_jobs`：13 -> 3。
- `impress_period_prefetch_jobs`：14 -> 0。
- read amplification：约 1.13-1.16 -> 1.06。
- 四个数据集均无 prefetch failed/cancelled。
- 128 条请求的选择哈希、首 token logits、预测和正确性完全一致。

周期边界层 8/16/24 仍会先接收上一层推测，再由新周期真实选择补取缺块；因此该机制应表述为“周期内部已知选择预取”，不能声称所有层均为 100% 精确预取。

## 数据集范围

- SST-2：GLUE 官方 train/dev，完整评测 868 条（排除 4 条校准 UID）。
- SubJ：Cornell Subjectivity 官方原始语料，确定性 90/10 split，完整评测 999 条（排除 4 条校准 UID）。
- TREC：官方 train_5500/TREC_10，完整评测 496 条（排除 4 条校准 UID）。
- RTE：GLUE 官方 train/dev，完整评测 273 条（排除 4 条校准 UID）。

ContiguousKV 论文没有公开 exact few-shot 样本、模板和 SubJ split；当前共享前缀采用确定性分层、长度匹配近似。这是与原文不能消除的协议差异，不能把绝对准确率差异全部归因于算法实现。
