# PRISM-Gao V3 实验审核报告

生成时间：2026-09-01T11:15:50.462934+00:00
范围：V3 representative 64-sample cells only; no full grid or full ABBA

## 总结论

- **Stage A：等价性通过，性能门槛失败。** exact 与 torch 的 selection hash、prediction、correctness 全部一致；但四 cell 宏 response-ready 变慢。
- **Stage B：三 cell 初筛通过，9-cell 扩展失败。** k010 更快但净少 3/256；k050 多对 2/256但更慢，统一预算规则不是 Pareto 改进。
- **Stage C：host-async FP16 成功，16/8 coalesced 失败。** FP16 payload 与 original_async 输出完全一致且宏延迟明显下降；混合精度因 INT8 碎片/pread 增长和质量回退未过门槛。
- 按停止条件未运行完整网格、五-cell Stage C、完整 ABBA 或 INT4 mixed attention。

## 静态与实现验证

- V3 patch SHA256：`a683767d861be149f32d5d36a21da4eebe096f4dbf2f91a05d839aebc6d6ff45`
- `compileall` 通过；三个 shell 脚本语法通过；GPU2 单测 **17 passed**。
- 修改仅位于 `src/prism_gao`；应用前备份：`src/prism_gao/v3_backup_20260901_165338`。

## Stage A：exact selector 双顺序

| cell | torch acc | exact acc | torch response ms | exact response ms | 变化 | 审计差异 |
|---|---:|---:|---:|---:|---:|---:|
| rte/k025 | 0.921875 | 0.921875 | 670.62 | 664.09 | -0.97% | 0 |
| sst2/k010 | 0.921875 | 0.921875 | 233.29 | 221.17 | -5.20% | 0 |
| subj/k025 | 0.765625 | 0.765625 | 425.10 | 451.03 | +6.10% | 0 |
| trec/k010 | 0.625000 | 0.625000 | 254.74 | 270.39 | +6.14% | 0 |

- 宏平均：395.94 → 401.67 ms（+1.45%）。
- 判定：等价性 `True`；性能门槛 `False`。

## Stage B：budget_v2

### 三 cell 初筛

- 宏 accuracy：0.661458 → 0.671875。
- 宏 response-ready：590.40 → 549.42 ms（-6.94%）。
- 初筛门槛：`True`。

### 9-cell 扩展

| cell | original acc | V2 acc | original ms | V2 ms | 时间变化 |
|---|---:|---:|---:|---:|---:|
| rte/k010 | 0.906250 | 0.906250 | 394.98 | 366.19 | -7.29% |
| rte/k050 | 0.921875 | 0.921875 | 1417.86 | 1491.79 | +5.21% |
| sst2/k010 | 0.921875 | 0.937500 | 217.23 | 205.87 | -5.23% |
| sst2/k050 | 0.968750 | 0.968750 | 682.82 | 733.21 | +7.38% |
| subj/k010 | 0.656250 | 0.593750 | 270.07 | 268.26 | -0.67% |
| subj/k025 | 0.765625 | 0.765625 | 490.07 | 486.18 | -0.79% |
| subj/k050 | 0.890625 | 0.890625 | 792.29 | 838.13 | +5.79% |
| trec/k010 | 0.625000 | 0.625000 | 296.48 | 260.59 | -12.11% |
| trec/k050 | 0.593750 | 0.625000 | 848.47 | 882.75 | +4.04% |

- 9-cell 宏：accuracy 0.805556 → 0.803819；response 601.14 → 614.77 ms（+2.27%）。
- k010×4：accuracy 0.777344 → 0.765625，response -6.61%。
- k050×4：accuracy 0.843750 → 0.851562，response +5.46%。
- 扩展门槛：`False`。

## Stage C：host async + 16/8/drop

| cell | original acc/ms | payload FP16 acc/ms | coalesced acc/ms | coalesced vs FP16 | byte ratio | FP16→coal preads |
|---|---:|---:|---:|---:|---:|---:|
| subj/k025 | 0.765625/463.60 | 0.765625/272.70 | 0.781250/274.24 | +0.57% | 0.799 | 1836→2710 |
| trec/k010 | 0.625000/250.73 | 0.625000/237.37 | 0.593750/213.53 | -10.04% | 0.938 | 1053→1165 |
| trec/k050 | 0.625000/873.10 | 0.625000/464.96 | 0.609375/549.68 | +18.22% | 0.770 | 3321→5627 |

- payload FP16 vs original_async 宏 response：529.15 → 325.01 ms（-38.58%）；三 cell accuracy 完全一致，双顺序 selection/prediction/correctness 审计均为 0。
- coalesced vs payload FP16 宏 response：325.01 → 345.82 ms（+6.40%）；宏 accuracy 0.671875 → 0.661458。
- 每个 cell 的 host wait 均小于 payload read，说明 host 预取真实隐藏了读取。
- 相对旧 naive，同计划的 coalesced pread：SUBJ/k025 下降 36.35%，TREC/k010 下降 46.01%；但相对 FP16 payload，INT8 的 codes+scales 与碎片 run 仍增加 pread。
- 62.5% FP16 回退：accuracy 0.625000 → 0.593750，response 254.77 → 247.11 ms（-3.01%），质量仍失败。
- host-prefetch 门槛：`True`；coalesced 主门槛：`False`。

## 建议给 GPT 的审核重点

1. Stage A exact 可以作为数学等价实现保留，但本轮不能宣称稳定加速。
2. Stage B 的最小预算 heuristic 只呈现任务相关 trade-off；当前应保留 original，不再把 0.76/高预算 P4 cap 设为默认。
3. Stage C 最值得保留的是 **payload FP16 host-async reader**：输出与 original_async 完全一致，且三 cell 宏 response-ready 下降 38.58%。
4. 16/8/drop 当前瓶颈不是 materialize kernel，而是低精度 run 导致的 pread 数量/读取时间和质量传播；不要进入 INT4 或完整 mixed attention。
5. 下一步若继续，先单独审计 payload FP16 host-async 的系统公平性与缓存状态，再设计减少 INT8 codes/scales 双 pread 的物理布局；不建议继续叠加 period heuristic。

## 结果位置

- `stage_a`：`/home/panzihang/src/prism_max/src/prism_gao/results/stage_a_confirm_20260901_165455`
- `stage_b_screen`：`/home/panzihang/src/prism_max/src/prism_gao/results/stage_b_v2_20260901_171810`
- `stage_b_extended`：`/home/panzihang/src/prism_max/src/prism_gao/results/stage_b_v2_extended_20260901_173809`
- `stage_c_screen`：`/home/panzihang/src/prism_max/src/prism_gao/results/stage_c_async_20260901_183636`
- `stage_c_fallback`：`/home/panzihang/src/prism_max/src/prism_gao/results/stage_c_trec_k010_fp16_0625_20260901_190306`

机器可读汇总：`src/prism_gao/results/v3_experiment_summary.json`。
