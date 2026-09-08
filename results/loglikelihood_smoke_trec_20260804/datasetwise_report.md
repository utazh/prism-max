# Dataset-wise ContiguousKV comparison

Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.
Accuracy scoring: `label_continuation_loglikelihood`.

## TREC

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | Ours | 1 | 1.0000 | 1558.71 | 1558.71 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
