# PRISM-Max matched ABBA comparison

Task `subj`, KV budget `050`, selector backend `k4`, score mode `nodefer`. Strict protocol validation passed for 998 paired UIDs.

Each ready time is averaged per UID over the two repeats before the paired bootstrap. Negative paired deltas favor ProMixed; positive reduction percentages favor ProMixed.
Response-ready is the primary latency; logits-ready is retained as a phase metric.

| Boundary | ContiguousKV mean/P95 (ms) | ProMixed mean/P95 (ms) | Mean/P95 reduction | Paired delta 95% CI (ms) | Faster UIDs |
|---|---:|---:|---:|---:|---:|
| Response ready (primary) | 914.87/1202.22 | 808.04/1040.64 | +11.68%/+13.44% | [-116.52, -97.36] | 840/998 |
| Logits ready (phase) | 914.48/1201.96 | 807.76/1040.30 | +11.67%/+13.45% | [-116.32, -97.03] | 840/998 |
| First token ready | 914.73/1202.10 | 807.90/1040.48 | +11.68%/+13.44% | [-116.43, -97.22] | 840/998 |
| All requested token IDs ready (latency) | 914.77/1202.13 | 807.94/1040.51 | +11.68%/+13.44% | [-116.37, -97.14] | 840/998 |
| Evaluation ready | 950.63/1239.37 | 851.11/1100.06 | +10.47%/+11.24% | [-109.52, -89.84] | 823/998 |

## Exclusion provenance

Input-bundle exclusions were pre-applied before all runs: `subj-0`. No analysis-time exclusion manifest was applied.

## Accuracy

ContiguousKV `0.5551`, ProMixed `0.8547`, delta `+29.96 pp`; W→C/C→W `316/17`, exact McNemar p `1.706e-72`.

## Run order

- `reference_r1`
- `candidate_r1`
- `candidate_r2`
- `reference_r2`
