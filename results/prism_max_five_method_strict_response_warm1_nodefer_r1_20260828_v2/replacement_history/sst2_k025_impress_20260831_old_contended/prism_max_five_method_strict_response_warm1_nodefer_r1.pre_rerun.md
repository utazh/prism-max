# Five-method Re-prefill Grid

Primary latency is **response-ready**; logits-ready is retained as a phase metric. Accuracy uses complete-label continuation log-likelihood. Tasks are not sample-pooled.

AS+LRU uses full prefix K/V in every cell. The 5%, 10%, 25%, and 50% labels identify comparison blocks only; each cell is timed independently and none claims sparse AS+LRU retention.

For AS+H2O+LRU, `k_actual` is the compact K/V **logical-attention ratio** and selected-value transfer ratio. The selector still loads full K, so minimum transfer is `(1+k_actual)/2`. The LRU name follows the baseline title and figure legend in the ContiguousKV paper.

## Detailed results

| Dataset | Budget | Method | Samples | Accuracy | Response-ready mean / P95 (ms) | Logits-ready phase mean / P95 (ms) | Run source |
|---|---:|---|---:|---:|---:|---:|---|
| sst2 | 5% | IMPRESS | 867 | 93.54% | 926.039 / 1091.369 | 925.758 / 1091.066 | independent k005 timing |
| sst2 | 5% | ContiguousKV | 867 | 89.62% | 218.116 / 240.596 | 217.845 / 240.266 | independent k005 timing |
| sst2 | 5% | ProMixed | 867 | 92.39% | 152.238 / 187.829 | 151.993 / 187.575 | independent k005 timing |
| sst2 | 5% | AS+LRU | 867 | 94.81% | 3795.687 / 4348.082 | 3795.339 / 4347.848 | independent k005 timing; full K/V |
| sst2 | 5% | AS+H2O+LRU | 867 | 94.46% | 4865.977 / 5672.923 | 4865.654 / 5672.563 | independent k005 timing |
| sst2 | 10% | IMPRESS | 867 | 94.12% | 1001.314 / 1308.148 | 1001.049 / 1307.824 | independent k010 timing |
| sst2 | 10% | ContiguousKV | 867 | 92.50% | 341.279 / 425.053 | 340.986 / 424.746 | independent k010 timing |
| sst2 | 10% | ProMixed | 867 | 93.89% | 230.838 / 274.084 | 230.579 / 273.585 | independent k010 timing |
| sst2 | 10% | AS+LRU | 867 | 94.81% | 3132.821 / 3401.642 | 3132.454 / 3401.352 | independent k010 timing; full K/V |
| sst2 | 10% | AS+H2O+LRU | 867 | 95.16% | 3655.590 / 4362.784 | 3655.267 / 4362.487 | independent k010 timing |
| sst2 | 25% | IMPRESS | 867 | 94.12% | 4346.254 / 16727.386 | 4345.955 / 16726.864 | independent k025 timing |
| sst2 | 25% | ContiguousKV | 867 | 94.46% | 500.037 / 562.401 | 499.635 / 562.167 | independent k025 timing |
| sst2 | 25% | ProMixed | 867 | 94.00% | 349.417 / 413.239 | 349.164 / 412.876 | independent k025 timing |
| sst2 | 25% | AS+LRU | 867 | 94.81% | 3235.363 / 3436.805 | 3234.975 / 3436.453 | independent k025 timing; full K/V |
| sst2 | 25% | AS+H2O+LRU | 867 | 94.69% | 5221.750 / 6180.753 | 5221.434 / 6180.426 | independent k025 timing |
| sst2 | 50% | IMPRESS | 867 | 94.81% | 1643.896 / 1960.450 | 1643.647 / 1960.155 | independent k050 timing |
| sst2 | 50% | ContiguousKV | 867 | 94.35% | 683.120 / 796.642 | 682.474 / 796.372 | independent k050 timing |
| sst2 | 50% | ProMixed | 867 | 94.58% | 628.256 / 724.385 | 627.989 / 724.109 | independent k050 timing |
| sst2 | 50% | AS+LRU | 867 | 94.81% | 3189.759 / 3330.283 | 3189.451 / 3329.952 | independent k050 timing; full K/V |
| sst2 | 50% | AS+H2O+LRU | 867 | 94.81% | 3753.021 / 4127.092 | 3752.758 / 4126.809 | independent k050 timing |
| subj | 5% | IMPRESS | 998 | 65.73% | 995.367 / 1168.152 | 995.112 / 1167.873 | independent k005 timing |
| subj | 5% | ContiguousKV | 998 | 49.10% | 222.330 / 244.469 | 222.065 / 244.164 | independent k005 timing |
| subj | 5% | ProMixed | 998 | 60.42% | 168.454 / 208.049 | 168.220 / 207.748 | independent k005 timing |
| subj | 5% | AS+LRU | 998 | 85.37% | 3571.284 / 3820.237 | 3571.012 / 3819.969 | independent k005 timing; full K/V |
| subj | 5% | AS+H2O+LRU | 998 | 77.66% | 3390.273 / 3688.355 | 3390.008 / 3687.971 | independent k005 timing |
| subj | 10% | IMPRESS | 998 | 67.43% | 1184.040 / 1419.297 | 1183.773 / 1419.016 | independent k010 timing |
| subj | 10% | ContiguousKV | 998 | 44.29% | 274.622 / 320.663 | 274.374 / 320.444 | independent k010 timing |
| subj | 10% | ProMixed | 998 | 67.94% | 238.407 / 295.822 | 238.175 / 295.620 | independent k010 timing |
| subj | 10% | AS+LRU | 998 | 85.37% | 3366.349 / 3596.942 | 3366.071 / 3596.614 | independent k010 timing; full K/V |
| subj | 10% | AS+H2O+LRU | 998 | 87.27% | 3690.550 / 4240.848 | 3690.289 / 4240.581 | independent k010 timing |
| subj | 25% | IMPRESS | 998 | 78.86% | 1538.959 / 1993.112 | 1538.722 / 1992.758 | independent k025 timing |
| subj | 25% | ContiguousKV | 998 | 46.19% | 491.195 / 562.361 | 490.894 / 562.022 | independent k025 timing |
| subj | 25% | ProMixed | 998 | 82.46% | 407.982 / 486.474 | 407.742 / 486.209 | independent k025 timing |
| subj | 25% | AS+LRU | 998 | 85.37% | 3309.136 / 3704.364 | 3308.864 / 3704.076 | independent k025 timing; full K/V |
| subj | 25% | AS+H2O+LRU | 998 | 87.27% | 3892.294 / 4557.308 | 3892.034 / 4557.039 | independent k025 timing |
| subj | 50% | IMPRESS | 998 | 86.17% | 2248.704 / 2597.780 | 2248.431 / 2597.445 | independent k050 timing |
| subj | 50% | ContiguousKV | 998 | 62.32% | 886.571 / 1310.964 | 886.235 / 1310.743 | independent k050 timing |
| subj | 50% | ProMixed | 998 | 85.47% | 795.827 / 979.569 | 795.555 / 979.240 | independent k050 timing |
| subj | 50% | AS+LRU | 998 | 85.37% | 3297.463 / 3692.283 | 3297.195 / 3692.004 | independent k050 timing; full K/V |
| subj | 50% | AS+H2O+LRU | 998 | 86.37% | 4308.932 / 5627.789 | 4308.659 / 5627.584 | independent k050 timing |
| trec | 5% | IMPRESS | 495 | 35.76% | 1052.409 / 1267.407 | 1052.158 / 1267.126 | independent k005 timing |
| trec | 5% | ContiguousKV | 495 | 50.71% | 321.149 / 350.611 | 320.894 / 350.342 | independent k005 timing |
| trec | 5% | ProMixed | 495 | 58.18% | 195.617 / 232.310 | 195.365 / 232.031 | independent k005 timing |
| trec | 5% | AS+LRU | 495 | 56.77% | 5613.309 / 6064.893 | 5612.955 / 6064.463 | independent k005 timing; full K/V |
| trec | 5% | AS+H2O+LRU | 495 | 51.11% | 3848.671 / 4167.549 | 3848.390 / 4167.230 | independent k005 timing |
| trec | 10% | IMPRESS | 495 | 47.07% | 1433.622 / 2719.797 | 1433.368 / 2719.529 | independent k010 timing |
| trec | 10% | ContiguousKV | 495 | 54.95% | 468.755 / 522.982 | 468.488 / 522.682 | independent k010 timing |
| trec | 10% | ProMixed | 495 | 64.04% | 222.301 / 263.633 | 222.086 / 263.335 | independent k010 timing |
| trec | 10% | AS+LRU | 495 | 56.77% | 3832.430 / 4070.043 | 3832.151 / 4069.693 | independent k010 timing; full K/V |
| trec | 10% | AS+H2O+LRU | 495 | 60.81% | 4352.485 / 4668.040 | 4352.220 / 4667.782 | independent k010 timing |
| trec | 25% | IMPRESS | 495 | 62.22% | 2207.359 / 2595.168 | 2207.109 / 2594.869 | independent k025 timing |
| trec | 25% | ContiguousKV | 495 | 50.51% | 740.945 / 863.312 | 740.684 / 863.096 | independent k025 timing |
| trec | 25% | ProMixed | 495 | 61.41% | 484.862 / 568.004 | 484.589 / 567.735 | independent k025 timing |
| trec | 25% | AS+LRU | 495 | 56.77% | 3721.346 / 3936.081 | 3721.058 / 3935.817 | independent k025 timing; full K/V |
| trec | 25% | AS+H2O+LRU | 495 | 62.42% | 4643.339 / 5345.904 | 4643.066 / 5345.432 | independent k025 timing |
| trec | 50% | IMPRESS | 495 | 61.01% | 2589.693 / 3177.276 | 2589.435 / 3176.952 | independent k050 timing |
| trec | 50% | ContiguousKV | 495 | 48.69% | 1254.372 / 1767.986 | 1254.057 / 1767.751 | independent k050 timing |
| trec | 50% | ProMixed | 495 | 59.39% | 891.463 / 1026.430 | 891.195 / 1026.163 | independent k050 timing |
| trec | 50% | AS+LRU | 495 | 56.77% | 3678.152 / 4071.274 | 3677.878 / 4070.949 | independent k050 timing; full K/V |
| trec | 50% | AS+H2O+LRU | 495 | 58.38% | 4626.001 / 5445.385 | 4625.725 / 5445.095 | independent k050 timing |
| rte | 5% | IMPRESS | 272 | 89.71% | 1257.245 / 1679.225 | 1256.994 / 1678.960 | independent k005 timing |
| rte | 5% | ContiguousKV | 272 | 86.40% | 297.019 / 347.167 | 296.757 / 346.844 | independent k005 timing |
| rte | 5% | ProMixed | 272 | 89.71% | 228.601 / 301.427 | 228.371 / 301.141 | independent k005 timing |
| rte | 5% | AS+LRU | 272 | 88.97% | 6717.192 / 7134.968 | 6716.883 / 7134.699 | independent k005 timing; full K/V |
| rte | 5% | AS+H2O+LRU | 272 | 89.71% | 4770.792 / 5096.580 | 4770.495 / 5096.167 | independent k005 timing |
| rte | 10% | IMPRESS | 272 | 88.97% | 1876.906 / 2374.170 | 1876.646 / 2373.898 | independent k010 timing |
| rte | 10% | ContiguousKV | 272 | 85.29% | 407.025 / 484.676 | 406.740 / 484.350 | independent k010 timing |
| rte | 10% | ProMixed | 272 | 88.24% | 354.045 / 446.367 | 353.794 / 446.141 | independent k010 timing |
| rte | 10% | AS+LRU | 272 | 88.97% | 5433.914 / 5773.631 | 5433.605 / 5773.344 | independent k010 timing; full K/V |
| rte | 10% | AS+H2O+LRU | 272 | 89.71% | 4991.873 / 5716.491 | 4991.577 / 5716.206 | independent k010 timing |
| rte | 25% | IMPRESS | 272 | 88.24% | 2925.837 / 3542.959 | 2925.564 / 3542.714 | independent k025 timing |
| rte | 25% | ContiguousKV | 272 | 87.13% | 779.864 / 913.274 | 779.547 / 912.952 | independent k025 timing |
| rte | 25% | ProMixed | 272 | 88.97% | 636.635 / 762.585 | 636.382 / 762.354 | independent k025 timing |
| rte | 25% | AS+LRU | 272 | 88.97% | 4629.540 / 5073.692 | 4629.251 / 5073.316 | independent k025 timing; full K/V |
| rte | 25% | AS+H2O+LRU | 272 | 88.97% | 5177.315 / 6016.399 | 5177.045 / 6016.085 | independent k025 timing |
| rte | 50% | IMPRESS | 272 | 88.60% | 3772.563 / 4469.867 | 3772.262 / 4469.557 | independent k050 timing |
| rte | 50% | ContiguousKV | 272 | 88.97% | 1996.236 / 2684.347 | 1995.911 / 2683.901 | independent k050 timing |
| rte | 50% | ProMixed | 272 | 88.24% | 1473.071 / 2011.715 | 1472.731 / 2011.493 | independent k050 timing |
| rte | 50% | AS+LRU | 272 | 88.97% | 4604.864 / 4902.464 | 4604.569 / 4902.167 | independent k050 timing; full K/V |
| rte | 50% | AS+H2O+LRU | 272 | 88.60% | 5411.178 / 6341.239 | 5410.908 / 6340.964 | independent k050 timing |

## Macro average by budget

Each row is the unweighted macro average of four dataset cells.

| Budget | Method | Cells | Accuracy | Response-ready mean / cell-P95 (ms) | Logits-ready phase mean / cell-P95 (ms) |
|---:|---|---:|---:|---:|---:|
| 5% | IMPRESS | 4 | 71.18% | 1057.765 / 1301.538 | 1057.506 / 1301.256 |
| 5% | ContiguousKV | 4 | 68.96% | 264.653 / 295.711 | 264.390 / 295.404 |
| 5% | ProMixed | 4 | 75.17% | 186.227 / 232.404 | 185.987 / 232.124 |
| 5% | AS+LRU | 4 | 81.48% | 4924.368 / 5342.045 | 4924.047 / 5341.745 |
| 5% | AS+H2O+LRU | 4 | 78.23% | 4218.928 / 4656.352 | 4218.637 / 4655.983 |
| 10% | IMPRESS | 4 | 74.40% | 1373.971 / 1955.353 | 1373.709 / 1955.067 |
| 10% | ContiguousKV | 4 | 69.26% | 372.920 / 438.343 | 372.647 / 438.056 |
| 10% | ProMixed | 4 | 78.52% | 261.398 / 319.976 | 261.158 / 319.670 |
| 10% | AS+LRU | 4 | 81.48% | 3941.378 / 4210.564 | 3941.070 / 4210.251 |
| 10% | AS+H2O+LRU | 4 | 83.24% | 4172.624 / 4747.041 | 4172.338 / 4746.764 |
| 25% | IMPRESS | 4 | 80.86% | 2754.602 / 6214.656 | 2754.338 / 6214.301 |
| 25% | ContiguousKV | 4 | 69.57% | 628.010 / 725.337 | 627.690 / 725.059 |
| 25% | ProMixed | 4 | 81.71% | 469.724 / 557.576 | 469.469 / 557.293 |
| 25% | AS+LRU | 4 | 81.48% | 3723.846 / 4037.736 | 3723.537 / 4037.416 |
| 25% | AS+H2O+LRU | 4 | 83.34% | 4733.675 / 5525.091 | 4733.395 / 5524.745 |
| 50% | IMPRESS | 4 | 82.65% | 2563.714 / 3051.343 | 2563.444 / 3051.027 |
| 50% | ContiguousKV | 4 | 73.58% | 1205.075 / 1639.985 | 1204.669 / 1639.691 |
| 50% | ProMixed | 4 | 81.92% | 947.154 / 1185.525 | 946.868 / 1185.251 |
| 50% | AS+LRU | 4 | 81.48% | 3692.560 / 3999.076 | 3692.273 / 3998.768 |
| 50% | AS+H2O+LRU | 4 | 82.04% | 4524.783 / 5385.376 | 4524.513 / 5385.113 |

## Overall macro average

Each row is the unweighted macro average of 16 dataset-budget cells.

| Method | Cells | Accuracy | Response-ready mean / cell-P95 (ms) | Logits-ready phase mean / cell-P95 (ms) |
|---|---:|---:|---:|---:|
| IMPRESS | 16 | 77.27% | 1937.513 / 3130.723 | 1937.249 / 3130.413 |
| ContiguousKV | 16 | 70.34% | 617.665 / 774.844 | 617.349 / 774.553 |
| ProMixed | 16 | 79.33% | 466.126 / 573.870 | 465.871 / 573.585 |
| AS+LRU | 16 | 81.48% | 4070.538 / 4397.355 | 4070.232 / 4397.045 |
| AS+H2O+LRU | 16 | 81.71% | 4412.502 / 5078.465 | 4412.221 / 5078.151 |
