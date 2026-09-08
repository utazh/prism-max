# Dataset-wise ContiguousKV comparison

Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.
Accuracy scoring: `label_continuation_loglikelihood`.

## RTE

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.9375 | 1457.08 | 1774.23 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.8750 | 311.97 | 351.55 | 4.00 | 1 |
| 5% | Ours | 32 | 0.9062 | 235.30 | 272.64 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +3.12 | +24.58% | +22.45% |

## SST2

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.8438 | 807.81 | 976.19 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.8125 | 197.72 | 214.79 | 4.00 | 1 |
| 5% | Ours | 32 | 0.8125 | 169.17 | 204.91 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +0.00 | +14.44% | +4.60% |

## SUBJ

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.6250 | 848.41 | 1153.06 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.3750 | 242.90 | 274.65 | 4.00 | 1 |
| 5% | Ours | 32 | 0.5000 | 188.12 | 213.82 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +12.50 | +22.55% | +22.15% |

## TREC

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.4375 | 1001.04 | 1283.56 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.4688 | 372.00 | 402.46 | 4.00 | 1 |
| 5% | Ours | 32 | 0.5312 | 197.34 | 228.42 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
| 5% | +6.25 | +46.95% | +43.24% |
