# Paired dataset-wise comparisons

Every row is paired by UID within one dataset and one KV budget. No cross-dataset aggregate is computed.

## RTE

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 273 | +1.10 | 10/7 | 0.6291 | +19.03% | [-60.66, -51.91] | +12.15% | +73.78% | 261/273 |
| Ours vs IMPRESS | 273 | -2.20 | 8/14 | 0.2863 | +82.02% | [-1116.77, -1064.39] | +83.12% | +86.15% | 273/273 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 273 | +1.47 | 7/3 | 0.3438 | +21.23% | [-91.77, -76.05] | +19.65% | +74.58% | 266/273 |
| Ours vs IMPRESS | 273 | -2.20 | 9/15 | 0.3075 | +84.51% | [-1732.28, -1665.46] | +85.07% | +91.60% | 273/273 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 273 | -1.10 | 3/6 | 0.5078 | +32.49% | [-276.04, -241.39] | +32.71% | +77.48% | 269/273 |
| Ours vs IMPRESS | 273 | -2.20 | 9/15 | 0.3075 | +82.97% | [-2649.12, -2579.34] | +82.89% | +95.30% | 273/273 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 273 | -2.20 | 4/10 | 0.1796 | +31.67% | [-773.17, -689.55] | +34.36% | +73.35% | 267/273 |
| Ours vs IMPRESS | 273 | -1.83 | 6/11 | 0.3323 | +59.19% | [-2339.61, -2234.45] | +55.74% | +91.36% | 273/273 |

## SST2

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 868 | +2.30 | 27/7 | 0.0008214 | +23.26% | [-48.61, -44.51] | +19.40% | +82.70% | 848/868 |
| Ours vs IMPRESS | 868 | -1.61 | 11/25 | 0.02882 | +78.80% | [-578.12, -564.68] | +81.77% | +91.05% | 868/868 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 868 | +0.35 | 13/10 | 0.6776 | +20.61% | [-56.69, -49.95] | +18.58% | +82.05% | 844/868 |
| Ours vs IMPRESS | 868 | -1.27 | 13/24 | 0.09887 | +78.65% | [-770.49, -746.88] | +81.25% | +91.83% | 867/868 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 868 | -0.58 | 4/9 | 0.2668 | +25.61% | [-116.49, -106.67] | +26.17% | +86.36% | 860/868 |
| Ours vs IMPRESS | 868 | -0.23 | 11/13 | 0.8388 | +72.90% | [-882.55, -860.85] | +76.82% | +92.68% | 868/868 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 868 | -0.35 | 5/8 | 0.5811 | +11.00% | [-89.40, -64.56] | +12.96% | +84.54% | 666/868 |
| Ours vs IMPRESS | 868 | -0.81 | 6/13 | 0.1671 | +57.97% | [-872.19, -844.11] | +60.41% | +94.08% | 866/868 |

## SUBJ

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 999 | +3.50 | 122/87 | 0.01847 | +17.71% | [-42.63, -38.13] | +15.52% | +74.25% | 952/999 |
| Ours vs IMPRESS | 999 | -13.21 | 12/144 | 6.703e-30 | +79.01% | [-715.09, -697.00] | +81.70% | +84.97% | 999/999 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 999 | +11.31 | 372/259 | 7.826e-06 | +21.64% | [-68.13, -61.03] | +20.69% | +73.94% | 988/999 |
| Ours vs IMPRESS | 999 | -11.91 | 40/159 | 5.425e-18 | +78.43% | [-860.86, -837.59] | +81.39% | +84.88% | 999/999 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 999 | +32.93 | 377/48 | 1.962e-64 | +13.47% | [-69.84, -57.85] | +16.27% | +73.65% | 843/999 |
| Ours vs IMPRESS | 999 | +0.30 | 72/69 | 0.8663 | +73.28% | [-1137.78, -1111.72] | +76.53% | +91.38% | 999/999 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 999 | +23.62 | 253/17 | 4.089e-55 | +10.24% | [-104.84, -79.45] | +13.61% | +76.05% | 810/999 |
| Ours vs IMPRESS | 999 | -0.20 | 35/37 | 0.9063 | +62.56% | [-1370.14, -1333.77] | +65.31% | +93.33% | 999/999 |

## TREC

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 496 | +3.63 | 49/31 | 0.05666 | +41.78% | [-146.40, -138.38] | +41.04% | +77.20% | 495/496 |
| Ours vs IMPRESS | 496 | +18.55 | 105/13 | 4.764e-19 | +79.70% | [-794.59, -764.45] | +82.42% | +83.73% | 496/496 |

### 10% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 496 | -11.09 | 18/73 | 5.009e-09 | +41.37% | [-193.16, -182.73] | +43.39% | +78.87% | 495/496 |
| Ours vs IMPRESS | 496 | -3.23 | 33/49 | 0.09703 | +79.04% | [-1024.09, -985.74] | +81.88% | +83.49% | 496/496 |

### 25% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 496 | +4.23 | 67/46 | 0.05943 | +40.83% | [-312.04, -291.36] | +40.52% | +79.83% | 491/496 |
| Ours vs IMPRESS | 496 | -7.46 | 30/67 | 0.0002189 | +78.85% | [-1648.56, -1611.12] | +80.01% | +92.91% | 496/496 |

### 50% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 496 | +11.09 | 63/8 | 1.027e-11 | +24.53% | [-312.57, -272.07] | +32.17% | +80.94% | 475/496 |
| Ours vs IMPRESS | 496 | -1.21 | 15/21 | 0.405 | +66.88% | [-1847.32, -1783.63] | +65.03% | +94.49% | 496/496 |
