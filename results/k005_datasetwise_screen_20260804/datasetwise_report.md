# Dataset-wise ContiguousKV comparison

Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.

## SST2

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.8438 | 877.37 | 1201.73 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.8125 | 211.82 | 234.84 | 4.00 | 1 |
| 5% | Ours | 32 | 0.8125 | 208.42 | 241.14 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +0.00 | +1.60% | -2.68% |

## SUBJ

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.7188 | 873.54 | 1147.58 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.5000 | 246.10 | 272.96 | 4.00 | 1 |
| 5% | Ours | 32 | 0.5938 | 223.16 | 248.63 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +9.38 | +9.32% | +8.92% |

## TREC

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.7500 | 1106.51 | 1319.19 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.5938 | 350.72 | 379.38 | 4.00 | 1 |
| 5% | Ours | 32 | 0.6250 | 253.97 | 282.84 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +3.12 | +27.59% | +25.45% |
