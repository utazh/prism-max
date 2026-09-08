# Prism-Max Re-prefill Grid

Per-UID repeat averages are used throughout. Tasks are never pooled.

Exclusion manifest: `/home/panzihang/src/prism_max/configs/strict_eval_exclusions.json`

## trec

| Budget | Method | UIDs×repeats | Accuracy | Logits mean / P95 (ms) | SSD total (critical + selector) MiB/req | Prefetch stall | Actual keep | Selected MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| k010 | ContiguousKV | 127×2 | 55.12% | 464.868 / 520.544 | 9.144 (9.144 + 0.000) | 51.86% | 9.96% | 26.906 |
| k010 | ProMixed | 127×2 | 59.84% | 274.888 / 330.600 | 0.537 (0.537 + 0.000) | 12.29% | 9.93% | 26.814 |

### Paired ProMixed deltas

Positive accuracy is better; negative latency, SSD, and stall are better.

| Budget | Baseline | UIDs | Δ accuracy pp [95% CI] | Δ logits ms [95% CI] | Δ SSD MiB/req [95% CI] | Δ stall pp [95% CI] |
|---:|---|---:|---:|---:|---:|---:|
| k010 | ContiguousKV | 127 | +4.724 [-4.724, +14.173] | -189.980 [-197.061, -182.804] | -8.606 [-8.754, -8.459] | -39.571 [-40.619, -38.483] |
| k010 | IMPRESS | — | unavailable: missing method in this task/budget cell | — | — | — |

### Cross-budget accuracy–latency Pareto

| Method | Pareto-optimal budgets |
|---|---|
| ContiguousKV | k010 |
| IMPRESS | — |
| ProMixed | k010 |

Joint method-budget frontier: ProMixed k010
