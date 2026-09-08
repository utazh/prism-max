# Prism-Max Re-prefill Grid

Per-UID repeat averages are used throughout. Tasks are never pooled.

Exclusion manifest: `/home/panzihang/src/prism_max/configs/strict_eval_exclusions.json`

## trec

| Budget | Method | UIDs×repeats | Accuracy | Logits mean / P95 (ms) | SSD total (critical + selector) MiB/req | Prefetch stall | Actual keep | Selected MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| k010 | ContiguousKV | 32×2 | 53.12% | 531.520 / 592.332 | 28.934 (9.637 + 19.297) | 38.47% | 9.96% | 26.906 |
| k010 | IMPRESS | 32×2 | 50.00% | 2269.493 / 3870.715 | 34.680 (0.911 + 33.770) | 41.25% | 13.21% | 35.699 |
| k010 | ProMixed | 32×2 | 62.50% | 487.968 / 628.953 | 35.398 (1.026 + 34.373) | 7.27% | 9.93% | 26.815 |

### Paired ProMixed deltas

Positive accuracy is better; negative latency, SSD, and stall are better.

| Budget | Baseline | UIDs | Δ accuracy pp [95% CI] | Δ logits ms [95% CI] | Δ SSD MiB/req [95% CI] | Δ stall pp [95% CI] |
|---:|---|---:|---:|---:|---:|---:|
| k010 | ContiguousKV | 32 | +9.375 [-4.766, +26.641] | -43.552 [-66.553, -16.973] | +6.465 [+4.460, +8.181] | -31.200 [-32.749, -30.011] |
| k010 | IMPRESS | 32 | +12.500 [-4.766, +29.766] | -1781.525 [-2153.794, -1382.880] | +0.718 [-1.138, +2.417] | -33.983 [-38.907, -29.047] |

### Cross-budget accuracy–latency Pareto

| Method | Pareto-optimal budgets |
|---|---|
| ContiguousKV | k010 |
| IMPRESS | k010 |
| ProMixed | k010 |

Joint method-budget frontier: ProMixed k010
