# Paired dataset-wise comparisons

Every row is paired by UID within one dataset and one KV budget. No cross-dataset aggregate is computed.

## RTE

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 273 | +1.10 | 10/7 | 0.6291 | +19.03% | [-60.66, -51.91] | +12.15% | +73.78% | 261/273 |
| Ours vs IMPRESS | 273 | -2.20 | 8/14 | 0.2863 | +82.02% | [-1116.77, -1064.39] | +83.12% | +86.15% | 273/273 |

## SST2

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 868 | +2.30 | 27/7 | 0.0008214 | +23.26% | [-48.61, -44.51] | +19.40% | +82.70% | 848/868 |
| Ours vs IMPRESS | 868 | -1.61 | 11/25 | 0.02882 | +78.80% | [-578.12, -564.68] | +81.77% | +91.05% | 868/868 |

## SUBJ

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 999 | +3.50 | 122/87 | 0.01847 | +17.71% | [-42.63, -38.13] | +15.52% | +74.25% | 952/999 |
| Ours vs IMPRESS | 999 | -13.21 | 12/144 | 6.703e-30 | +79.01% | [-715.09, -697.00] | +81.70% | +84.97% | 999/999 |

## TREC

### 5% KV budget

| Comparison | N | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT reduction | Paired delta 95% CI (ms) | P95 reduction | SSD reduction | Faster requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours vs ContiguousKV | 496 | +3.63 | 49/31 | 0.05666 | +41.78% | [-146.40, -138.38] | +41.04% | +77.20% | 495/496 |
| Ours vs IMPRESS | 496 | +18.55 | 105/13 | 4.764e-19 | +79.70% | [-794.59, -764.45] | +82.42% | +83.73% | 496/496 |
