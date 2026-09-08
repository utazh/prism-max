# PRISM 三阶段最小实验报告（供 GPT 审核）

## 实验边界

本轮只做筛选实验：每个 cell 64 条，1 次 warmup（16 条），单一运行顺序；没有跑完整数据集，也没有做多轮 ABBA。所有新增实现和结果均位于 /home/panzihang/src/prism_max/src/prism_gao，原 src/contiguous_fuxian 未修改。

## 实现审计

- A：packed INT4 codes/scales 直接进入 GPU；exact 只融合 unpack/dequant，仍沿用原 BF16 matmul。
- B：固定周期钩子只覆盖 decision.period，不改当次 selected_blocks 与 priority_blocks。
- C：重要性只分配精度，标签写回原物理 block 顺序；短 INT8 run 提升为 FP16，K/V 由融合 materialize 恢复原顺序。

## 阶段 A：INT4 selector

| task/budget | accuracy torch/exact | mean TTFT torch→exact | selector load torch→exact | exact 审计差异 |
|---|---:|---:|---:|---:|
| trec/k010 | 0.625000/0.625000 | 268.57→257.31 ms (-4.19%) | 10.62→3.58 ms | 0 |
| subj/k025 | 0.765625/0.765625 | 472.53→438.77 ms (-7.15%) | 10.35→3.53 ms | 0 |

结论：exact 路径可作为默认实现。它在两个 64 条实验中保持选择、预测和正确性完全一致，mean TTFT 分别下降 4.19% 和 7.15%。

direct 路径不应作为正式实现：TREC 64 条中选择哈希变化 64 条，分类变化 13 条，correctness 变化 8 条。

## 阶段 B：固定周期与自适应周期

| task/budget | mode | accuracy | mean TTFT | p95 | selector calls | mean period |
|---|---|---:|---:|---:|---:|---:|
| subj/k025 | adaptive | 0.765625 | 446.05 ms | 523.01 ms | 7.56 | 4.39 |
| subj/k025 | p1 | 0.828125 | 765.55 ms | 916.10 ms | 28.00 | 1.00 |
| subj/k025 | p4 | 0.781250 | 479.63 ms | 552.74 ms | 7.00 | 4.00 |
| subj/k025 | p8 | 0.578125 | 438.21 ms | 495.59 ms | 4.00 | 8.00 |
| trec/k010 | adaptive | 0.625000 | 308.14 ms | 361.51 ms | 7.88 | 5.10 |
| trec/k010 | p1 | 0.640625 | 604.22 ms | 688.69 ms | 28.00 | 1.00 |
| trec/k010 | p4 | 0.687500 | 329.31 ms | 375.50 ms | 7.00 | 4.00 |
| trec/k010 | p8 | 0.671875 | 246.77 ms | 291.78 ms | 4.00 | 8.00 |
| trec/k050 | adaptive | 0.593750 | 894.87 ms | 1264.86 ms | 4.38 | 7.00 |
| trec/k050 | p1 | 0.578125 | 1342.77 ms | 1686.16 ms | 28.00 | 1.00 |
| trec/k050 | p4 | 0.625000 | 859.59 ms | 1028.18 ms | 7.00 | 4.00 |
| trec/k050 | p8 | 0.609375 | 866.41 ms | 989.16 ms | 4.00 | 8.00 |

B 阶段说明固定周期是明确的速度/准确率控制变量。P1 的 selector 调用最多且明显最慢；P8 通常最快，但不保证准确率最高。因此当前不应仅凭 TTFT 把自适应策略替换成固定 P8。
具体地，TREC/k010 的 P8 同时比 adaptive 快 61.37 ms 且多对 3/64 条，TREC/k050 的 P4 也同时更快且多对 2/64 条。
但 SUBJ/k025 的 P8 虽快 7.84 ms，却少对 12/64 条；当前自适应阈值还不是稳定 Pareto 最优，需要按任务/不确定度重新校准。

## 阶段 C：16/8/drop 与物理 run 合并

break-even 微基准采用 5% 安全边际，最小可盈利 INT8 连续 run 为 2 个 block。

新 FP16 reader 对照 Stage B adaptive：两个任务的选择、预测和正确性差异均为 0/64。

| task/budget | mode | accuracy | mean TTFT | payload MiB | byte ratio | read ms | materialize ms | pread | FP16/INT8/promoted blocks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj/k025 | fp16 | 0.765625 | 360.93 ms | 60.16 | 1.000 | 68.63 | 40.91 | 1836.3 | 1932.0/0.0/0.0 |
| subj/k025 | naive | 0.781250 | 304.00 ms | 39.15 | 0.651 | 62.57 | 32.20 | 4257.6 | 497.0/1435.0/0.0 |
| subj/k025 | coalesced | 0.781250 | 291.21 ms | 48.09 | 0.799 | 53.86 | 33.49 | 2710.1 | 1107.7/824.3/610.7 |
| trec/k010 | fp16 | 0.625000 | 218.43 ms | 26.81 | 1.000 | 35.12 | 26.71 | 990.6 | 865.0/0.0/0.0 |
| trec/k010 | naive | 0.578125 | 208.44 ms | 17.45 | 0.651 | 35.28 | 23.83 | 2158.4 | 224.0/641.0/0.0 |
| trec/k010 | coalesced | 0.593750 | 205.82 ms | 23.17 | 0.864 | 29.48 | 25.26 | 1225.0 | 615.2/249.8/391.2 |

run 合并机制本身有效：相对 naive，coalesced 将 SUBJ/TREC 的 pread 分别减少 36.3%/43.2%，mean TTFT 再下降 4.2%/1.3%。
准确率方面，SUBJ 与 naive 相同且比 FP16 多对 1/64；TREC 比 naive 多对 1/64，但仍比 FP16 少对 2/64。
因此 C 证明了碎片治理有收益，但尚未证明混合精度在所有任务上无损。

### C 阶段逐请求审计

| task/budget | candidate vs FP16 | selection hash diff | prediction diff | correct diff |
|---|---|---:|---:|---:|
| subj/k025 | naive | 64 | 1 | 1 |
| subj/k025 | coalesced | 64 | 1 | 1 |
| trec/k010 | naive | 63 | 5 | 5 |
| trec/k010 | coalesced | 64 | 5 | 4 |

C 阶段是同步、无预取的隔离物理读取对照。低精度会改变后续 hidden state，进而反馈到在线 selector，所以报告同时列出选择哈希和预测差异，不能把字节下降直接等价为无损加速。

## 请 GPT 重点审核

1. 是否接受 Stage A exact 为当前默认 selector，并拒绝 direct 正式结果。
2. Stage B 结果应只用于调阈值，还是足以改动自适应复用规则。
3. Stage C 在当前结果下是否值得扩展到 16/8/4；若 16/8 没有稳定正收益，不建议立即扩大实现。
4. 审核通过后再跑完整数据集和多轮 ABBA；本报告不把筛选结果包装成正式论文结论。

机器可读汇总：/home/panzihang/src/prism_max/src/prism_gao/results/three_stage_summary.json
