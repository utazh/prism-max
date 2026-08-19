# PRISM-KV 阶段 A/B 实验报告

日期：2026-07-30

## 1. 结论

1. 已在修改前完整备份 ContiguousKV、IMPRESS、HyperInfer 三套复现代码。原目录未修改，所有新代码和实验都位于独立目录 `/home/panzihang/src/prism_phase_ab_20260730`。
2. 阶段 A 已实现：保留 HyperInfer 的逐层在线重要性算法，用离线逐层敏感度生成每层 KV 保留预算。正式配置为 7 个高预算层 `0.625`、14 个中预算层 `0.5`、7 个低预算层 `0.375`，名义均值仍为 `0.5`。
3. 阶段 B 已实现：保留 16-token 连续块和每层真实 HyperInfer 选择，增加 P4 周期领导层的推测预取。预测只影响预取，不改变最终选择。
4. 阶段 A 的 128 请求结果由 `107/128` 提升到 `108/128`，但只有一个净正确样本，McNemar 双侧精确检验 `p=1.0`，目前只能称为“未损害准确率并出现一个正向样本”，不能宣称显著提升。
5. 阶段 B 在 A 后运行的 128 请求正式实验中达到 `3763.60 ms`，比 A 快 `5.38%`；但 B 先运行的 32 请求反序确认中反而慢 `2.68%`。共同 32 请求的两顺序探索性平衡估计为 B 快 `1.81%`，95% bootstrap CI 为 `[-141.92, -4.40] ms`。因此 `5.38%` 不能作为稳定 headline，当前较稳妥的判断是“可能有约 2% 的均值收益，但需要随机交错重复实验确认”。
6. 阶段 C 的高低精度与丢弃没有开始，KV 存储仍为 FP16，模型计算为 BF16。

## 2. 修改前备份

- 服务器压缩包：`/home/panzihang/src/reproduction_backups/reproduction_baselines_20260730_170807.tar.gz`
- 本地压缩包：`server_backups/reproduction_baselines_20260730_170807.tar.gz`
- SHA256：`a770a237441c911a7b00683ae828fb0444297eeaf4a56137e8cd80021871466c`
- 校验：服务器与本地哈希一致，`gzip -t` 通过；压缩包包含 9,533 个条目和 8,140 个源文件哈希。
- 冻结原目录：
  - `/home/panzihang/src/contiguous_fuxian`
  - `/home/panzihang/src/hyperinfer_fuxian_20260720`
  - `/home/panzihang/src/impress_fuxian`

## 3. 实验协议

| 项目 | 本次配置 |
|---|---|
| 模型 | Qwen2.5-7B-Instruct |
| GPU | NVIDIA RTX 3090 24 GB，固定 GPU1 |
| CPU | Intel Xeon Silver 4314，64 CPU |
| SSD | Intel SSDSC2KB960G，`/dev/sda4` |
| 请求方式 | 单请求、串行、batch=1 |
| 任务 | SST-2、SUBJ、TREC、RTE，各 32 请求 |
| 前缀 | 100/110/120/80 examples，实际 3811/4412/4940/6011 tokens |
| KV 预算 | 名义 50% |
| 存储/计算精度 | FP16 KV / BF16 model compute |
| 块大小 | 16 tokens |
| Cache | GPU 55 MiB、CPU 131 MiB、CKLFU |
| 计时 | 每个正式变体 1 次 warmup，随后 128 请求 |
| 准确率 | 最多生成 4 token 的现有 strict 标签匹配 |

校准使用 `sst2-0/subj-0/trec-0/rte-0` 四个请求，不使用标签，只比较单层降预算后的首 token logit JS 偏移。为避免校准请求混入评价造成误读，审计同时报告排除这四个请求后的 124 请求结果。

## 4. 阶段 A

高预算层：`1, 4, 11, 12, 14, 16, 20`

低预算层：`17, 19, 23, 24, 25, 26, 27`

其余层为 `0.5`。层 1 因四个校准请求均触发官方选择器 fallback，被保护为高预算，不能进入低预算组。

激进档 `0.75/0.5/0.25` 在 16 请求试点中从 `13/16` 降到 `12/16`，因此正式实验改用 `0.625/0.5/0.375`。正式阶段 A：

- Accuracy：`108/128 = 0.84375`；冻结 hybrid 基线为 `107/128 = 0.8359375`。
- 排除四个校准请求后：`105/124`；基线为 `104/124`。
- Mean TTFT：`3977.40 ms`；基线 `3949.45 ms`，慢 `0.71%`。
- 配对 TTFT bootstrap 95% CI：`[-27.00, 81.81] ms`，跨过 0。
- 实际平均保留率：基线推算 `0.51805`，阶段 A `0.51393`。
- 已选 KV token/bytes 比基线减少 `0.79%`，SSD 总读取减少 `0.93%`。

结论：阶段 A 在近似等预算下保留了准确率并多答对一个 SUBJ 样本，但准确率和 TTFT 都没有达到统计显著。

## 5. 阶段 B 与预取证据

阶段 A 每请求有 27 个全预算相邻层预取。阶段 B 将其改为：

- 13 个相邻层预取；
- 14 个 P4 周期领导层推测预取；
- 周期预取预算为当前动态预算的 `0.25`；
- 每层到达后仍重新运行真实 HyperInfer 选择，并补读 missing set。

128 请求审计结果：

- 128/128 请求发生周期预取；
- 共 1,792 个周期预取 job，4,762,068 个预测 token；
- 阶段 A/B 的 128 个 prediction mismatch 为 0；
- 阶段 A/B 的 128 个逐层选择 SHA256 mismatch 为 0；
- 无用预取 token：`2328.63 -> 1248.50/request`，降低 `46.38%`；
- 总 SSD 读取：`126.41 -> 124.34 MiB/request`，降低 `1.64%`；
- 暴露 prefetch wait：`2984.32 -> 2887.82 ms`，降低 `3.23%`；
- 命中覆盖：`9.78% -> 7.08%`，反而降低；
- missing token：增加 `2.99%`。

因此预取“确实执行”且不改变模型选择，但本次收益来源主要是降低过度预取和无用 I/O，不是实现论文所述的接近连续、高覆盖预取。

## 6. 50% SSD 正式结果

下表 Mean TTFT/Accuracy 来自各自 `summary.json`；P95 和 SSD bytes 从 128 条原始记录统一重算，以保持与旧复现表相同的线性 P95 口径。

| 方法 | Accuracy | Mean TTFT | P95 TTFT | SSD 读取/请求 |
|---|---:|---:|---:|---:|
| ContiguousKV | 0.828125 | 3906.59 ms | **6040.64 ms** | 128.97 MiB |
| IMPRESS | 0.828125 | 5674.26 ms | 9032.33 ms | 239.96 MiB |
| HyperInfer | 0.828125 | 4857.69 ms | 7278.41 ms | 240.00 MiB |
| Hybrid：HyperInfer + block16 async | 0.835938 | 3949.45 ms | 6553.18 ms | 127.60 MiB |
| 阶段 A：Hybrid + layer budget | **0.843750** | 3977.40 ms | 6309.62 ms | 126.41 MiB |
| 阶段 A+B：再加 P4 节流预取 | **0.843750** | **3763.60 ms** | 6332.11 ms | **124.34 MiB** |

仅按这一次 128 正式运行，阶段 A+B 相比：

- ContiguousKV：Mean TTFT 低 `3.66%`，Accuracy 高 `1.5625 pp`，但 P95 高 `4.83%`；
- IMPRESS：Mean TTFT 低 `33.67%`，即 `1.508x`；
- HyperInfer：Mean TTFT 低 `22.52%`，即 `1.291x`；
- Hybrid：Mean TTFT 低 `4.71%`，Accuracy 高 `0.78125 pp`。

这些百分比是该次正式运行的观测值。由于反序确认出现方向反转，不应把阶段 A+B 的 `3763.60 ms` 单独写成已稳定超过 ContiguousKV。

## 7. 顺序稳健性

| 顺序与样本 | A Mean TTFT | B Mean TTFT | B 相对 A |
|---|---:|---:|---:|
| 16 请求试点，A -> B | 4264.73 ms | 4350.75 ms | 慢 2.02% |
| 128 请求正式，A -> B | 3977.40 ms | 3763.60 ms | 快 5.38% |
| 32 请求确认，B -> A | 3839.04 ms | 3941.89 ms | 慢 2.68% |
| 共同 32 请求，两顺序平衡估计 | 4006.07 ms | 3933.52 ms | 快 1.81% |

反序确认中 B 的无用预取和 SSD bytes 仍更低，但长前缀任务的暴露等待波动抵消了 I/O 节省。下一轮若继续优化 B，应采用请求级随机交错 ABBA，并改造官方单 worker FIFO，使当前层 missing load 的优先级高于未来周期预测；否则无法把调度收益与运行顺序分开。

## 8. 与 ContiguousKV 原文对比

一致项：

- 同为 Qwen2.5-7B、SST-2/SUBJ/TREC/RTE；
- few-shot 数量和前缀长度与论文表 1 基本一致；
- 16-token 连续块；
- 单请求 Re-Prefill、SSD 外存、Accuracy/TTFT/P95 指标；
- 论文在 50% 预算下报告 ContiguousKV 对 IMPRESS 平均准确率提高约 `1.63%`；本次阶段 A+B 对已复现 IMPRESS 高 `1.5625 pp`，方向和量级接近。

不同项及不能直接对齐的原因：

- 原文使用 A800 80 GB、Samsung 990 Pro NVMe 7.45 GB/s、10 GB GPU cache 和 24 GB CPU cache；本次为 RTX 3090、Intel SATA SSD、55/131 MiB Pcache。
- 原文默认 Period=8、SubPeriod=4，并在 Period 内复用同一选择；本次为了保持 HyperInfer 准确率，每层重新选择，额外预测周期为 P4。
- 原文有专用预取 buffer 和完整 intra/inter-period pipeline；官方 HyperInfer Pcache 是单 worker FIFO + RLock，本次没有重写该底层。
- 原文 Figure 10 的主要 TTFT 对比为 5%/25% 预算；本次为 50%，绝对 TTFT 和原文 `3.85x` 加速不能直接比较。
- 原文报告 SSD token load 平均降低约 `16.33x`；本次 50% 下阶段 A+B 相比 IMPRESS 仅降低约 `1.93x`，趋势一致但幅度未复现。
- 本次只有 32 请求/任务并采用 strict 4-token 生成匹配；原文准确率协议和完整样本规模不同。

结论：已有复现和本次 A/B 保持了“连续块减少 I/O、重要性选择保持质量”的总趋势，但尚未完整复现 ContiguousKV 原文的高覆盖两级预取及其加速幅度。

## 9. 产物与验证

- 正式审计：`results/prism_phase_ab_20260730/phase_ab_audit.json`
- 源码/输入/结果哈希：`results/prism_phase_ab_20260730/final_source_and_results_sha256.txt`
- 层预算：`results/prism_phase_ab_20260730/calibration/layer_budget_profile_delta0125.json`
- 逐层敏感度：`results/prism_phase_ab_20260730/calibration/layer_sensitivity.json`
- 阶段 A 正式记录：`results/prism_phase_ab_20260730/formal_phase_a_delta0125_p1_128/`
- 阶段 A+B 正式记录：`results/prism_phase_ab_20260730/formal_phase_ab_delta0125_p4_s025_128/`
- 反序确认：`results/prism_phase_ab_20260730/confirmation_reverse_*`
- 测试：服务器 `64/64` 通过；核心 Python 编译通过；三个 shell 入口 `bash -n` 通过。
- 审计 SHA256：`70d2d235b2c89a64b19eb03116eae49d3bb5b8dbeb1a524681d5e09bd7f98bc8`
