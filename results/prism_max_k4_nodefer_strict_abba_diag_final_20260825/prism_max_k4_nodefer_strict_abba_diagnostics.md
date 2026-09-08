# Prism-Max Re-prefill Grid

Per-UID repeat averages are used throughout. Tasks are never pooled.

Primary latency: response-ready. Logits-ready is retained as a phase metric.

Input-bundle exclusions (pre-applied before every run): `sst2-0, subj-0, trec-0, rte-0`.

Analysis-time exclusion manifest: `None` (no second-pass filtering when this is `None`).

## subj

| Budget | Method | UIDs×repeats | Accuracy | Response-ready mean / P95 (ms) | Logits-ready phase mean / P95 (ms) | SSD total (critical + selector) MiB/req | Prefetch stall | Actual keep | Selected MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k025 | ContiguousKV | 998×2 | 48.60% | 441.737 / 595.622 | 441.371 / 595.319 | 2.518 (2.518 + 0.000) | 49.04% | 24.93% | 60.156 |
| k025 | ProMixed | 998×2 | 82.46% | 440.296 / 507.599 | 440.038 / 507.303 | 0.454 (0.454 + 0.000) | 31.41% | 24.93% | 60.157 |
| k050 | ContiguousKV | 998×2 | 55.51% | 914.866 / 1202.225 | 914.475 / 1201.962 | 7.159 (7.159 + 0.000) | 62.99% | 49.95% | 120.531 |
| k050 | ProMixed | 998×2 | 85.47% | 808.040 / 1040.639 | 807.761 / 1040.300 | 1.609 (1.609 + 0.000) | 48.83% | 49.95% | 120.531 |

### Paired ProMixed deltas

Positive accuracy is better; negative latency, SSD, and stall are better.

| Budget | Baseline | UIDs | Δ accuracy pp [95% CI] | Δ response-ready ms [95% CI] | Δ logits-ready phase ms [95% CI] | Δ SSD MiB/req [95% CI] | Δ stall pp [95% CI] |
|---:|---|---:|---:|---:|---:|---:|---:|
| k025 | ContiguousKV | 998 | +33.868 [+30.361, +37.174] | -1.441 [-6.816, +3.842] | -1.334 [-6.747, +4.065] | -2.063 [-2.165, -1.960] | -17.626 [-17.996, -17.264] |
| k025 | IMPRESS | — | unavailable: missing method in this task/budget cell | — | — | — | — |
| k050 | ContiguousKV | 998 | +29.960 [+26.954, +33.066] | -106.826 [-116.147, -97.153] | -106.714 [-116.306, -97.052] | -5.550 [-5.659, -5.440] | -14.156 [-14.531, -13.767] |
| k050 | IMPRESS | — | unavailable: missing method in this task/budget cell | — | — | — | — |

### Cross-budget accuracy–response-ready Pareto

| Method | Pareto-optimal budgets |
|---|---|
| ContiguousKV | k025, k050 |
| IMPRESS | — |
| ProMixed | k025, k050 |

Joint method-budget frontier: ProMixed k025, ProMixed k050

## trec

| Budget | Method | UIDs×repeats | Accuracy | Response-ready mean / P95 (ms) | Logits-ready phase mean / P95 (ms) | SSD total (critical + selector) MiB/req | Prefetch stall | Actual keep | Selected MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k010 | ContiguousKV | 495×2 | 54.34% | 413.160 / 451.010 | 412.904 / 450.780 | 8.733 (8.733 + 0.000) | 57.06% | 9.96% | 26.906 |
| k010 | ProMixed | 495×2 | 64.04% | 249.701 / 288.625 | 249.455 / 288.373 | 0.347 (0.347 + 0.000) | 11.94% | 9.93% | 26.814 |
| k050 | ContiguousKV | 495×2 | 61.41% | 1236.500 / 1717.752 | 1236.072 / 1717.472 | 14.953 (14.953 + 0.000) | 67.05% | 50.12% | 135.406 |
| k050 | ProMixed | 495×2 | 59.39% | 901.905 / 1157.462 | 901.614 / 1157.177 | 2.402 (2.402 + 0.000) | 49.21% | 49.96% | 134.969 |

### Paired ProMixed deltas

Positive accuracy is better; negative latency, SSD, and stall are better.

| Budget | Baseline | UIDs | Δ accuracy pp [95% CI] | Δ response-ready ms [95% CI] | Δ logits-ready phase ms [95% CI] | Δ SSD MiB/req [95% CI] | Δ stall pp [95% CI] |
|---:|---|---:|---:|---:|---:|---:|---:|
| k010 | ContiguousKV | 495 | +9.697 [+5.051, +14.141] | -163.459 [-167.690, -159.185] | -163.449 [-167.641, -159.108] | -8.386 [-8.436, -8.335] | -45.124 [-45.528, -44.718] |
| k010 | IMPRESS | — | unavailable: missing method in this task/budget cell | — | — | — | — |
| k050 | ContiguousKV | 495 | -2.020 [-3.838, -0.404] | -334.595 [-354.904, -314.730] | -334.458 [-355.126, -314.978] | -12.551 [-12.658, -12.442] | -17.844 [-18.408, -17.265] |
| k050 | IMPRESS | — | unavailable: missing method in this task/budget cell | — | — | — | — |

### Cross-budget accuracy–response-ready Pareto

| Method | Pareto-optimal budgets |
|---|---|
| ContiguousKV | k010, k050 |
| IMPRESS | — |
| ProMixed | k010 |

Joint method-budget frontier: ProMixed k010
