# Dataset-wise ContiguousKV comparison

Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.

## RTE

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | Ours | 32 | 0.8750 | 294.84 | 328.30 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
