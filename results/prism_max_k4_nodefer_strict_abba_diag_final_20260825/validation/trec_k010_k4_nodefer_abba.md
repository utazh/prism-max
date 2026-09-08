# PRISM-Max matched ABBA comparison

Task `trec`, KV budget `010`, selector backend `k4`, score mode `nodefer`. Strict protocol validation passed for 495 paired UIDs.

Each ready time is averaged per UID over the two repeats before the paired bootstrap. Negative paired deltas favor ProMixed; positive reduction percentages favor ProMixed.
Response-ready is the primary latency; logits-ready is retained as a phase metric.

| Boundary | ContiguousKV mean/P95 (ms) | ProMixed mean/P95 (ms) | Mean/P95 reduction | Paired delta 95% CI (ms) | Faster UIDs |
|---|---:|---:|---:|---:|---:|
| Response ready (primary) | 413.16/451.01 | 249.70/288.62 | +39.56%/+36.00% | [-167.69, -159.31] | 491/495 |
| Logits ready (phase) | 412.90/450.78 | 249.45/288.37 | +39.59%/+36.03% | [-167.68, -159.14] | 491/495 |
| First token ready | 413.03/450.90 | 249.58/288.51 | +39.57%/+36.01% | [-167.65, -159.24] | 491/495 |
| All requested token IDs ready (latency) | 413.07/450.93 | 249.61/288.54 | +39.57%/+36.01% | [-167.70, -159.16] | 491/495 |
| Evaluation ready | 513.05/561.99 | 346.82/396.87 | +32.40%/+29.38% | [-171.09, -161.26] | 491/495 |

## Exclusion provenance

Input-bundle exclusions were pre-applied before all runs: `trec-0`. No analysis-time exclusion manifest was applied.

## Accuracy

ContiguousKV `0.5434`, ProMixed `0.6404`, delta `+9.70 pp`; W→C/C→W `97/49`, exact McNemar p `8.769e-05`.

## Run order

- `reference_r1`
- `candidate_r1`
- `candidate_r2`
- `reference_r2`
