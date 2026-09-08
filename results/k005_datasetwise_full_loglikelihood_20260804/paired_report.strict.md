# Paired dataset-wise comparisons

Every row is paired by UID within one dataset and one KV budget. No cross-dataset aggregate is computed.

Strict holdout filtering excludes every UID used to calibrate the layer-budget profile.

## RTE

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 272 | +1.10 | 10/7 | 0.6291 | +18.99% | [-60.58, -51.75] | +12.15% | +73.78% | 260/272 |
| Ours vs IMPRESS | 272 | -2.21 | 8/14 | 0.2863 | +82.00% | [-1114.95, -1063.58] | +83.12% | +86.15% | 272/272 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 272 | +1.47 | 7/3 | 0.3438 | +21.16% | [-91.39, -75.77] | +19.65% | +74.57% | 265/272 |
| Ours vs IMPRESS | 272 | -2.21 | 9/15 | 0.3075 | +84.51% | [-1732.93, -1666.05] | +85.07% | +91.60% | 272/272 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 272 | -1.10 | 3/6 | 0.5078 | +32.47% | [-275.76, -241.01] | +32.71% | +77.47% | 268/272 |
| Ours vs IMPRESS | 272 | -2.21 | 9/15 | 0.3075 | +82.97% | [-2649.94, -2579.78] | +82.89% | +95.30% | 272/272 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 272 | -2.21 | 4/10 | 0.1796 | +31.67% | [-773.49, -689.74] | +34.36% | +73.34% | 266/272 |
| Ours vs IMPRESS | 272 | -1.84 | 6/11 | 0.3323 | +59.14% | [-2335.42, -2231.40] | +55.39% | +91.36% | 272/272 |

## SST2

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 867 | +2.31 | 27/7 | 0.0008214 | +23.25% | [-48.54, -44.49] | +19.36% | +82.70% | 847/867 |
| Ours vs IMPRESS | 867 | -1.61 | 11/25 | 0.02882 | +78.80% | [-578.42, -564.82] | +81.77% | +91.05% | 867/867 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 867 | +0.35 | 13/10 | 0.6776 | +20.60% | [-56.66, -49.89] | +18.58% | +82.05% | 843/867 |
| Ours vs IMPRESS | 867 | -1.27 | 13/24 | 0.09887 | +78.64% | [-769.93, -746.39] | +81.22% | +91.83% | 866/867 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 867 | -0.58 | 4/9 | 0.2668 | +25.60% | [-116.41, -106.69] | +26.04% | +86.36% | 859/867 |
| Ours vs IMPRESS | 867 | -0.23 | 11/13 | 0.8388 | +72.90% | [-882.53, -860.88] | +76.82% | +92.68% | 867/867 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 867 | -0.35 | 5/8 | 0.5811 | +10.98% | [-89.42, -64.44] | +12.96% | +84.54% | 665/867 |
| Ours vs IMPRESS | 867 | -0.81 | 6/13 | 0.1671 | +57.95% | [-871.28, -843.93] | +60.33% | +94.08% | 865/867 |

## SUBJ

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 998 | +3.41 | 121/87 | 0.0219 | +17.70% | [-42.62, -38.12] | +15.52% | +74.25% | 951/998 |
| Ours vs IMPRESS | 998 | -13.23 | 12/144 | 6.703e-30 | +79.01% | [-715.07, -696.92] | +81.70% | +84.97% | 998/998 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 998 | +11.22 | 371/259 | 9.279e-06 | +21.62% | [-68.05, -60.98] | +20.67% | +73.93% | 987/998 |
| Ours vs IMPRESS | 998 | -11.92 | 40/159 | 5.425e-18 | +78.43% | [-860.88, -837.70] | +81.39% | +84.88% | 998/998 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 998 | +32.97 | 377/48 | 1.962e-64 | +13.46% | [-69.81, -57.65] | +16.27% | +73.65% | 842/998 |
| Ours vs IMPRESS | 998 | +0.30 | 72/69 | 0.8663 | +73.27% | [-1137.14, -1111.17] | +76.53% | +91.38% | 998/998 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 998 | +23.65 | 253/17 | 4.089e-55 | +10.23% | [-104.84, -79.44] | +13.61% | +76.05% | 809/998 |
| Ours vs IMPRESS | 998 | -0.20 | 35/37 | 0.9063 | +62.56% | [-1370.28, -1334.08] | +65.31% | +93.33% | 998/998 |

## TREC

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 495 | +3.64 | 49/31 | 0.05666 | +41.76% | [-146.26, -138.32] | +41.04% | +77.20% | 494/495 |
| Ours vs IMPRESS | 495 | +18.59 | 105/13 | 4.764e-19 | +79.70% | [-795.16, -764.66] | +82.42% | +83.72% | 495/495 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 495 | -10.91 | 18/72 | 8.071e-09 | +41.39% | [-193.27, -182.96] | +43.39% | +78.87% | 494/495 |
| Ours vs IMPRESS | 495 | -3.03 | 33/48 | 0.1193 | +79.04% | [-1023.77, -985.22] | +81.88% | +83.50% | 495/495 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 495 | +4.44 | 67/45 | 0.04674 | +40.82% | [-312.16, -291.38] | +40.52% | +79.82% | 490/495 |
| Ours vs IMPRESS | 495 | -7.27 | 30/66 | 0.0003056 | +78.85% | [-1648.60, -1611.16] | +80.01% | +92.91% | 495/495 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 495 | +11.11 | 63/8 | 1.027e-11 | +24.53% | [-312.53, -272.42] | +32.17% | +80.94% | 474/495 |
| Ours vs IMPRESS | 495 | -1.21 | 15/21 | 0.405 | +66.89% | [-1847.75, -1783.39] | +65.03% | +94.49% | 495/495 |
