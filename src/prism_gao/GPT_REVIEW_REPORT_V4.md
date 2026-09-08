# PRISM_GAO V4 最小实验审查报告

生成时间：2026-09-02T03:38:36.667269+08:00

## 结论摘要

V4 补丁已严格按 `CODEX_TASK_PRISM_MINIMAL_V4.md` 的最小执行顺序完成。代码只位于 `src/prism_gao`，`src/contiguous_fuxian` 的 Git 状态为 clean；GPU2 上的 22 项测试全部通过。

三个阶段没有同时达到 V4 的严格通过条件：A 对真正 FP16 selector 明显更快，但 GPU-resident 相对原 exact-host 的端到端增益只有 0.81%；B 的独立 calibration profile 更快但少对 2/512 条；C 的新异步 payload 流水明显更快，但 16/8 算法本身没有收益，成本门控正确地全部回退到 FP16。

| 阶段 | 主要结果 | 严格结论 |
|---|---|---|
| A | resident exact 相对 FP16 response-ready -20.24%，相对 exact-host -0.81% | 等价性通过；>=3% host 增益未通过 |
| B | profiled 413/512, 615.25 ms；old 415/512, 625.78 ms | 速度快 1.68%，但少对 2 条，未通过 |
| C | 新 gated 流水相对 original -30.26%；相对同流水 FP16 +3.97% | 系统流水通过；分级精度未通过 |

## 实验范围与完整性

- A：SST2/k010、SUBJ/k025、TREC/k010、RTE/k025，64 条 held-out，forward/reverse 双顺序。
- B：4 tasks × {k010,k050}；16 条 calibration 与 64 条 held-out 完全不重叠；P1/P2/P4/P8/old/profiled。
- C：4 tasks × {k010,k050} × 4 variants × forward/reverse，共 64 个完整运行。
- 没有跑完整数据集，也没有做多轮 ABBA；本轮只用于筛选方向。

## 阶段 A：selector resident cache

实现包括 packed INT4 resident cache、可复用 dequant 输出 workspace、identity head-index 快路径和 resident hit 的同步开销消除。原 exact 路径已把索引预加载到 pinned CPU，因此本机没有可重复测量的“每次从磁盘 exact-disk”基线，公平基线记为 exact-host。

| variant | 宏平均 response-ready | selector load | correct/256 |
|---|---:|---:|---:|
| FP16 | 540.47 ms | 152.850 ms | 199/256 |
| exact-host | 434.59 ms | 3.599 ms | 205/256 |
| exact-resident | 431.08 ms | 0.540 ms | 205/256 |

resident 相对 host 的 selector load 下降 85%，但 response-ready 只下降 0.81%。8 个顺序审计（512 order-samples）中 selection hash、prediction、correct 均为 0 mismatch。

判定：可保留 resident exact；若下一轮目标仍是降低 selector 总耗时，应进入 V4 文档建议的 Quest 风格两阶段 candidate pruning，而不是继续优化 unpack。

## 阶段 B：workload-profiled adaptive reuse

profile 使用独立 16 条 calibration 选择 task/budget 的 base period，uncertainty >= 0.95 时仅缩短一级。四个任务的 calibration/held-out UID 交集均为 0。

| policy | correct/512 | 宏平均 response-ready |
|---|---:|---:|
| p1 | 406/512 | 937.52 ms |
| p2 | 407/512 | 770.87 ms |
| p4 | 417/512 | 662.20 ms |
| p8 | 398/512 | 573.26 ms |
| old | 415/512 | 625.78 ms |
| profiled | 413/512 | 615.25 ms |

| task/budget | profiled P | profiled/best-fixed correct | old→profiled ms | quality gate |
|---|---:|---:|---:|---|
| rte/k010 | 8 | 53/57 | 391.08→306.07 | fail |
| rte/k050 | 8 | 57/58 | 1451.11→1409.25 | pass |
| sst2/k010 | 8 | 62/62 | 252.05→219.38 | pass |
| sst2/k050 | 8 | 64/64 | 707.68→631.67 | pass |
| subj/k010 | 2 | 44/44 | 272.38→451.82 | pass |
| subj/k050 | 4 | 58/58 | 785.62→824.99 | pass |
| trec/k010 | 8 | 40/42 | 270.37→245.54 | fail |
| trec/k050 | 8 | 35/37 | 875.92→833.28 | fail |

严格 profile 仅有 5/8 cells 满足 best fixed -1 条；宏平均比 old 快 1.68%，但少对 2 条。

另保留一个明确标注为 post-hoc 的候选：只把 RTE/k010 改成 P4 后，组合为 416/512、623.84 ms，相对 old 同时略好；它是在 held-out 结果后选择，不能作为正式证据，只能放到下一批新样本验证。

## 阶段 C：true async 16/8/drop

正式路径使用 pinned `preadv`、non-blocking H2D、独立 CUDA stream/event 和 `current_stream.wait_event()`；没有在当前层调用全局同步。成本微基准中 1/2/4/8/16-block INT8 均慢于 FP16，因此没有 profitable run length。

| task/budget | original | payload FP16 | naive 16/8 | gated | gated vs original | gated vs FP16 |
|---|---:|---:|---:|---:|---:|---:|
| rte/k010 | 316.92 | 294.10 | 351.91 | 309.11 | -2.47% | +5.10% |
| rte/k050 | 1384.23 | 627.76 | 802.42 | 655.04 | -52.68% | +4.35% |
| sst2/k010 | 207.12 | 216.16 | 252.73 | 220.14 | +6.28% | +1.84% |
| sst2/k050 | 722.23 | 439.36 | 565.89 | 469.94 | -34.93% | +6.96% |
| subj/k010 | 374.47 | 327.27 | 349.69 | 334.51 | -10.67% | +2.21% |
| subj/k050 | 784.21 | 519.20 | 692.82 | 537.27 | -31.49% | +3.48% |
| trec/k010 | 248.06 | 254.72 | 285.27 | 266.47 | +7.42% | +4.61% |
| trec/k050 | 814.53 | 575.66 | 733.67 | 591.08 | -27.43% | +2.68% |

| variant | correct/512 | 宏平均 response-ready |
|---|---:|---:|
| original_async | 413/512 | 606.47 ms |
| fp16_pipeline | 413/512 | 406.78 ms |
| naive_pipeline | 414/512 | 504.30 ms |
| coalesced_pipeline | 413/512 | 422.94 ms |

gated 的 CUDA materialize event 为 224/224 在 resolve 前 ready；所有候选层 gate activation 为 0，payload ratio 回到 1.0。相对 payload FP16，gated 多出的约 3.97% 是 plan/gate 计算开销。逐 UID 审计中 gated 与 FP16 的 selection/prediction/correct 全部 0 mismatch。naive 改变了 1014 个 order-sample 的选择哈希、2 个 prediction 和 2 个 correctness。

判定：新 payload async pipeline 值得保留，尤其 k050；当前 16/8/drop 不值得启用。要让 C 的“算法收益”成立，下一步必须让 attention 直接消费 INT8 KV（fused dequant-attention），否则单纯减少 SSD 字节抵不过 host 读取碎片和 materialize 成本。

## 建议给 GPT 的下一步决策

1. A 冻结 resident exact；只在愿意实现 candidate pruning 时继续 A。
2. B 不把 strict profile 宣称为 Pareto 改进；用下一批未见样本验证 post-hoc hybrid，失败则保留 old adaptive。
3. C 将 payload FP16 async 作为独立系统优化；暂停 16/8/drop，除非实现直接消费 INT8 的 fused attention kernel。
4. 暂不进入完整数据集和多轮 ABBA，因为 B、C 的算法判据尚未通过。

## 产物与复现

- 机器路径：`/home/panzihang/src/prism_max/src/prism_gao`
- 汇总 JSON：`src/prism_gao/results/v4_experiment_summary.json`
- A/B/C 原始根目录和 driver log 均保留在 `src/prism_gao/results/`。
- 测试：`PYTHONPATH=src /home/panzihang/venvs/vllm-stable/bin/python -m pytest -q src/prism_gao/tests` → `22 passed`。
