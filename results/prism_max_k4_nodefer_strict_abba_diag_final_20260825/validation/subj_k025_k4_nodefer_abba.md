# PRISM-Max matched ABBA comparison

Task `subj`, KV budget `025`, selector backend `k4`, score mode `nodefer`. Strict protocol validation passed for 998 paired UIDs.

Each ready time is averaged per UID over the two repeats before the paired bootstrap. Negative paired deltas favor ProMixed; positive reduction percentages favor ProMixed.
Response-ready is the primary latency; logits-ready is retained as a phase metric.

| Boundary | ContiguousKV mean/P95 (ms) | ProMixed mean/P95 (ms) | Mean/P95 reduction | Paired delta 95% CI (ms) | Faster UIDs |
|---|---:|---:|---:|---:|---:|
| Response ready (primary) | 441.74/595.62 | 440.30/507.60 | +0.33%/+14.78% | [-6.79, +3.85] | 454/998 |
| Logits ready (phase) | 441.37/595.32 | 440.04/507.30 | +0.30%/+14.78% | [-6.66, +4.03] | 453/998 |
| First token ready | 441.62/595.46 | 440.17/507.45 | +0.33%/+14.78% | [-6.80, +3.84] | 454/998 |
| All requested token IDs ready (latency) | 441.65/595.50 | 440.20/507.50 | +0.33%/+14.78% | [-6.78, +3.85] | 454/998 |
| Evaluation ready | 475.96/633.39 | 476.41/549.50 | -0.09%/+13.24% | [-4.99, +5.75] | 442/998 |

## Exclusion provenance

Input-bundle exclusions were pre-applied before all runs: `subj-0`. No analysis-time exclusion manifest was applied.

## Accuracy

ContiguousKV `0.4860`, ProMixed `0.8246`, delta `+33.87 pp`; W→C/C→W `373/35`, exact McNemar p `1.699e-72`.

## Run order

- `reference_r1`
- `candidate_r1`
- `candidate_r2`
- `reference_r2`
