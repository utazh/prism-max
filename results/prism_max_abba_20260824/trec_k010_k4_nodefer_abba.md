# PRISM-Max matched ABBA comparison

Task `trec`, KV budget `010`, selector backend `k4`, score mode `nodefer`. Strict protocol validation passed for 128 paired UIDs.

Each ready time is averaged per UID over the two repeats before the paired bootstrap. Negative paired deltas favor ProMixed; positive reduction percentages favor ProMixed.

| Boundary | ContiguousKV mean/P95 (ms) | ProMixed mean/P95 (ms) | Mean/P95 reduction | Paired delta 95% CI (ms) | Faster UIDs |
|---|---:|---:|---:|---:|---:|
| Logits ready | 465.04/520.54 | 274.61/330.60 | +40.95%/+36.49% | [-197.25, -183.58] | 128/128 |
| First token ready | 465.18/520.70 | 274.76/330.73 | +40.94%/+36.48% | [-197.34, -183.44] | 128/128 |
| All requested token IDs ready (latency) | 465.22/520.75 | 274.79/330.76 | +40.93%/+36.49% | [-197.33, -183.41] | 128/128 |
| Response ready | 465.32/520.85 | 274.90/330.84 | +40.92%/+36.48% | [-197.46, -183.54] | 128/128 |
| Evaluation ready | 563.49/623.19 | 371.98/433.04 | +33.99%/+30.51% | [-198.90, -184.18] | 128/128 |

## Accuracy

ContiguousKV `0.5547`, ProMixed `0.5938`, delta `+3.91 pp`; W→C/C→W `22/17`, exact McNemar p `0.5224`.

## Run order

- `reference_r1`
- `candidate_r1`
- `candidate_r2`
- `reference_r2`
