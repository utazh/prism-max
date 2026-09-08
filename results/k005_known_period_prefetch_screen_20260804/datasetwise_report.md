# Dataset-wise ContiguousKV comparison

Each dataset is an independent process and workload. No request-weighted or unweighted cross-dataset headline metric is reported.

## RTE

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | Ours | 32 | 0.8750 | 236.97 | 305.30 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|

## SST2

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | Ours | 32 | 0.8125 | 171.03 | 205.19 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|

## SUBJ

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | Ours | 32 | 0.5938 | 185.02 | 210.95 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|

## TREC

| KV budget | Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Selector calls | Selection period |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5% | Ours | 32 | 0.6250 | 208.78 | 244.25 | 4.00 | 8 |

| KV budget | Ours accuracy delta vs ContiguousKV (pp) | Ours mean TTFT reduction vs ContiguousKV | Ours P95 reduction vs ContiguousKV |
|---:|---:|---:|---:|
