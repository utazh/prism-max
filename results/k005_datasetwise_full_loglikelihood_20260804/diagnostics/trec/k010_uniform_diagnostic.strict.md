# TREC 10% Uniform vs Sensitivity diagnostic

Strict holdout excludes `trec-0`. All methods use complete label-continuation log-likelihood scoring.

| Method | N | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) | Observed KV | Selector calls |
|---|---:|---:|---:|---:|---:|---:|
| ContiguousKV | 495 | 0.5495 | 454.61 | 514.61 | 9.96% | 4.00 |
| Sensitivity | 495 | 0.4404 | 266.45 | 291.29 | 9.93% | 4.00 |
| Uniform | 495 | 0.4424 | 265.27 | 309.01 | 9.93% | 4.00 |
| IMPRESS | 495 | 0.4707 | 1270.93 | 1607.89 | 13.21% | 28.00 |

| Comparison | Accuracy delta (pp) | W->C / C->W | McNemar p | Mean TTFT delta (ms) | Paired CI95 (ms) | Faster requests |
|---|---:|---:|---:|---:|---:|---:|
| Uniform vs Sensitivity | +0.20 | 13/12 | 1 | -1.17 | [-6.11, 3.62] | 285/495 |
| Uniform vs ContiguousKV | -10.71 | 19/72 | 1.967e-08 | -189.34 | [-194.73, -183.97] | 494/495 |
| Uniform vs IMPRESS | -2.83 | 38/52 | 0.1702 | -1005.66 | [-1024.87, -986.68] | 495/495 |

## NUM class

| Method | Correct / N | Predictions |
|---|---:|---|
| ContiguousKV | 68/113 | `{"ENTY": 43, "LOC": 2, "NUM": 68}` |
| Sensitivity | 16/113 | `{"DESC": 19, "ENTY": 78, "NUM": 16}` |
| Uniform | 21/113 | `{"DESC": 16, "ENTY": 76, "NUM": 21}` |
| IMPRESS | 34/113 | `{"ABBR": 7, "DESC": 8, "ENTY": 12, "HUM": 52, "NUM": 34}` |

Uniform and Sensitivity are statistically indistinguishable at this point. The 10% degradation therefore is not caused primarily by the scaled per-layer budget profile.
