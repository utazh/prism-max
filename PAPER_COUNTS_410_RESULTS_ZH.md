# ContiguousKV 论文样本规模实验（Qwen2.5-7B）

## 实验协议

- 数据集与请求数：SST-2 100、SubJ 110、TREC 120、RTE 80，共410条。
- 执行方式：batch size 1的顺序单请求；每个配置先完整暖身410条，再测量410条。
- 模型：`/data1/llm/Qwen/Qwen2.5-7B-Instruct`，BF16计算，FP16 KV存储。
- 系统：RTX 3090 GPU 3，Intel SSDSC2KB96 SSD，55 MiB GPU cache，131 MiB CPU cache，CKLFU。
- 方法：复现IMPRESS、复现ContiguousKV、Ours（HyperInfer在线重要性 + 离线层敏感度 + 16-token连续块 + 精确预算 + P4/异步value-order预取）。
- IMPRESS重排使用的16条校准UID从三种方法的评测集中统一排除，校准/测试重叠为0。

## 总体结果

| KV预算 | 方法 | 正确数 | 准确率 | 实际预算 | 平均TTFT (ms) | P95 (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 338/410 | 82.44% | 5.55% | 2033.6 | 3255.4 |
| 5% | ContiguousKV | 264/410 | 64.39% | 4.97% | 295.3 | 409.3 |
| 5% | Ours | 342/410 | 83.41% | 4.91% | 587.4 | 800.5 |
| 10% | IMPRESS | 332/410 | 80.98% | 13.27% | 2831.6 | 4150.3 |
| 10% | ContiguousKV | 305/410 | 74.39% | 9.95% | 440.5 | 601.3 |
| 10% | Ours | 339/410 | 82.68% | 9.90% | 682.2 | 898.2 |
| 25% | IMPRESS | 343/410 | 83.66% | 29.44% | 9224.8 | 34240.5 |
| 25% | ContiguousKV | 317/410 | 77.32% | 24.99% | 1368.1 | 2352.8 |
| 25% | Ours | 344/410 | 83.90% | 24.91% | 1287.1 | 2471.1 |
| 50% | IMPRESS | 350/410 | 85.37% | 55.15% | 16198.3 | 43934.3 |
| 50% | ContiguousKV | 333/410 | 81.22% | 50.02% | 3611.0 | 5976.4 |
| 50% | Ours | 353/410 | 86.10% | 49.94% | 3158.0 | 5410.3 |

## 主要结论

1. Ours在4个预算下的准确率都高于ContiguousKV，分别高19.02、8.29、6.59和4.88个百分点。McNemar检验`p`值分别为`1.40e-15`、`1.74e-5`、`8.98e-4`和`1.19e-3`，均有统计意义。
2. Ours相对ContiguousKV在5%和10%下分别慢98.96%和54.88%；在25%和50%下平均TTFT分别快5.92%和12.55%，配对bootstrap 95%区间均不跨0。
3. 25%下Ours的P95比ContiguousKV慢5.03%，因此该点不是全面支配；50%下Ours的P95快9.47%。
4. Ours相对IMPRESS的准确率高0.98、1.71、0.24和0.73个百分点，但四个McNemar检验均未达统计显著；平均TTFT分别快71.11%、75.91%、86.05%和80.50%。
5. IMPRESS的10%、25%和50%实际预算分别膨胀到13.27%、29.44%和55.15%，而Ours基本精确命中目标预算。
6. Ours四个预算共提交90,198个异步预取任务，完成90,198个，失败0，取消0；所有1,640条Ours请求的精确块预算均完全满足。

## 与原文趋势对照

- 速度趋势复现：原文图10的5%/25%八个数据集-预算单元中，当前ContiguousKV在8/8都快于IMPRESS；图11的5% SST-2/RTE中，ContiguousKV的P95在2/2都快于IMPRESS。
- 准确率趋势未复现：原文报告ContiguousKV在5/10/25/50%下平均高于IMPRESS 7.69/4.81/3.58/1.63%；当前Qwen-7B复现中ContiguousKV在0/4预算高于IMPRESS。
- 主要失配来自未开源的ContiguousKV关键块选择语义，尤其SubJ在5%下仅22.73%。当前实现已复现16-token存储、8层Period、4层SubPeriod和异步路径，但不能声称准确率路径忠实复现。

## 限制

- 论文未公开完整trace、精确提示模板和ContiguousKV选择代码；本实验按表1的`100/110/120/80`解释为评测数量，但仍沿用每个数据集一个共享长前缀的现有复现协议。
- 论文使用A800 80GB、10/24GB GPU/CPU cache和Samsung 990 Pro NVMe；当前服务器是RTX 3090、55/131 MiB cache和Intel SATA SSD，绝对TTFT不能与论文柱状图直接比值。
- IMPRESS在25%和50%下出现大工作集尾延迟；设备读量与本进程读量一致，但正式写作前仍应做反序重复以排除顺序和持续I/O状态影响。

## 产物

- 服务器根目录：`/home/panzihang/src/prism_paper_counts_20260803`
- 正式结果：`results/qwen7_paper_counts_410_20260803`
- 主报告：`report/fig9_11_report.md`
- 审计：`report/paper_count_audit.md`
- 图：`report/figure9_accuracy_qwen25_7b.png`、`figure10_mean_ttft_qwen25_7b.png`、`figure11_p95_qwen25_7b.png`
