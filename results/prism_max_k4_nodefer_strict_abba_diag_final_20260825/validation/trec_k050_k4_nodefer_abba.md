# PRISM-Max matched ABBA comparison

Task `trec`, KV budget `050`, selector backend `k4`, score mode `nodefer`. Strict protocol validation passed for 495 paired UIDs.

Each ready time is averaged per UID over the two repeats before the paired bootstrap. Negative paired deltas favor ProMixed; positive reduction percentages favor ProMixed.
Response-ready is the primary latency; logits-ready is retained as a phase metric.

| Boundary | ContiguousKV mean/P95 (ms) | ProMixed mean/P95 (ms) | Mean/P95 reduction | Paired delta 95% CI (ms) | Faster UIDs |
|---|---:|---:|---:|---:|---:|
| Response ready (primary) | 1236.50/1717.75 | 901.90/1157.46 | +27.06%/+32.62% | [-354.95, -315.01] | 473/495 |
| Logits ready (phase) | 1236.07/1717.47 | 901.61/1157.18 | +27.06%/+32.62% | [-354.59, -314.96] | 473/495 |
| First token ready | 1236.36/1717.62 | 901.77/1157.33 | +27.06%/+32.62% | [-354.77, -314.69] | 473/495 |
| All requested token IDs ready (latency) | 1236.40/1717.66 | 901.81/1157.36 | +27.06%/+32.62% | [-355.11, -314.86] | 473/495 |
| Evaluation ready | 1342.78/1832.68 | 1010.94/1269.84 | +24.71%/+30.71% | [-352.34, -311.84] | 472/495 |

## Exclusion provenance

Input-bundle exclusions were pre-applied before all runs: `trec-0`. No analysis-time exclusion manifest was applied.

## Accuracy

ContiguousKV `0.6141`, ProMixed `0.5939`, delta `-2.02 pp`; W→C/C→W `5/15`, exact McNemar p `0.04139`.

## Run order

- `reference_r1`
- `candidate_r1`
- `candidate_r2`
- `reference_r2`
