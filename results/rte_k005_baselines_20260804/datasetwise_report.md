# Dataset-wise ContiguousKV comparison

Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.

## RTE

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | IMPRESS | 32 | 0.9375 | 1423.66 | 1842.65 | 28.00 | 1 |
| 5% | ContiguousKV | 32 | 0.8750 | 332.63 | 363.25 | 4.00 | 1 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
