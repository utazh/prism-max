# ProMixed isolated experiment report

Date: 2026-08-19 to 2026-08-20

## Scope and isolation

- Source inspected: `/home/panzihang/src/prism_datasetwise_logit_20260804`
- Isolated worktree: `/home/panzihang/src/prism_promixed`
- Baseline snapshot commit: `fb3b4af`
- ProMixed implementation commit: `5caaeef`
- The source experiment directory was not edited. Code, generated indexes,
  logs, and results remain on the server.
- The formal full grid, precision ablation, and repeat run were executed
  sequentially on GPU0. GPU1 was not used after the user selected single-GPU
  timing. Earlier concurrent trials are archived and excluded.

## Root-cause diagnosis

Qwen2.5-7B-Instruct has 28 query heads and 4 physical KV heads, hence seven
query heads per GQA group. The previous probe heads `0,1,2` all map to physical
KV group 0. Their apparent agreement therefore does not measure agreement
between GQA groups. This is especially damaging on TREC k=0.10:

- old Ours: 43.9516% accuracy, 266.485 ms mean TTFT;
- ContiguousKV: 55.0403% accuracy, 454.546 ms mean TTFT.

The old Ours errors are strongly class-skewed: only 16/113 NUM examples and
9/136 DESC examples are correct. Other identified latency issues were:

- cache-score maintenance executed before the recorded first-token boundary;
- ineffective predictive-period work (zero useful period jobs in the saved
  run);
- selector keys repeatedly read from storage;
- token-level selector scores copied back to CPU before block aggregation.

## Implemented method

### 1. GQA-aware hierarchical probing

- Probe query heads: `0,7,14,21`.
- Physical selector KV heads: `0,1,2,3`.
- Runtime validation requires exactly one query representative per physical
  GQA group.
- Every group produces an aligned block ranking. A fixed fraction of the exact
  block budget is allocated round-robin across group rankings, then the
  remainder is filled using normalized max/mean/vote utility. This prevents a
  minority group from being averaged away.

### 2. Uncertainty-adaptive reuse

For each selection leader, ProMixed records:

- mean pairwise Top-K Jaccard between physical GQA groups;
- normalized Top-K boundary margin;
- offline layer-sensitivity risk.

These form a bounded uncertainty score that selects a real P1/P2/P4/P8 reuse
horizon. The decoder schedules the next selection at the chosen layer rather
than merely reporting a dynamic period while continuing fixed-P8 execution.

Default policy:

```text
coverage_fraction = 0.50
margin_reference = 0.05
agreement_weight = 0.75
sensitivity_weight = 0.10
P1/P2/P4 thresholds = 0.90/0.82/0.68
```

### 3. Low-bit all-group selector index

Only probing keys are quantized. Final sparse K/V attention remains FP16
storage with BF16 model compute.

- symmetric signed INT4;
- per token, per physical KV head, group size 32;
- low-nibble-first packing;
- compressed index preloaded into pinned CPU memory before request timing;
- asynchronous transfer and GPU dequantization on the selector stream.

Index audit:

| Metric | Value |
|---|---:|
| FP16 source bytes | 549,756,928 |
| INT4 index bytes | 154,619,136 |
| Compression | 3.5556x |
| Aggregate relative RMSE | 0.094234 |
| Layer-0 cosine audit | 0.99395 |

The warmed smoke run reports zero selector disk bytes during requests.

### 4. Critical-path cleanup

- CKLFU cache-score updates are queued while layers load and flushed only
  after the first-token timestamp.
- Four-head token scores are summed into 16-token blocks on GPU; only block
  scores are transferred to CPU. This is mathematically equivalent to the old
  CPU block summation and reduces score transfer by about 16x.
- Ineffective predictive-period jobs are disabled; only the known adaptive
  horizon and next-layer speculative work remain.

## Verification

- Relevant suite: 75 tests passed, plus 15 subtests.
- INT4 pack/unpack, quantization, manifest geometry, preloading, and CPU
  dequantization are tested.
- GQA fair coverage, exact budgets, deterministic ties, uncertainty thresholds,
  and sensitivity behavior are tested.
- GPU-side block reduction is tested against token-side block aggregation.

## TREC k=0.10 calibration (same first 32 records)

| Method | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |
|---|---:|---:|---:|
| Old Ours | 43.750% | 257.752 | 288.232 |
| ContiguousKV | 56.250% | 441.409 | 464.700 |
| ProMixed default | 56.250% | 319.293 | 473.929 |
| ProMixed longer-reuse calibration | 59.375% | 386.856 | 497.049 |

ProMixed default recovers 12.5 percentage points over old Ours and matches
ContiguousKV accuracy, while reducing mean TTFT by 27.67% relative to
ContiguousKV. In paired outcomes it fixes six old-Ours errors and introduces
two regressions. The longer-reuse policy uses fewer selector calls
(5.84 vs. 7.72) but is slower because its asynchronous prefetch queue takes
longer; therefore the default policy is selected for the full evaluation.

Two attempted longer-reuse launches produced zero samples after unrelated
processes occupied their GPUs during model loading. Their OOM logs are retained
for audit and excluded from all statistics.

## Final TREC k=0.10 result (496 records)

Result:
`results/promixed_full_default_20260819/trec/k010_promixed_k4`

| Method | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |
|---|---:|---:|---:|
| Old Ours | 43.9516% | 266.485 | 291.294 |
| ContiguousKV | 55.0403% | 454.546 | 514.606 |
| **ProMixed default** | **63.9113%** | **290.363** | **355.905** |

At the same k=0.10 target budget (9.9253% effective for ProMixed), it:

- improves accuracy over old Ours by **19.9597 percentage points**;
- improves accuracy over ContiguousKV by **8.8710 percentage points**;
- reduces mean TTFT versus ContiguousKV by **36.1202%**;
- reduces P95 TTFT versus ContiguousKV by **30.8393%**;
- costs 8.9605% mean TTFT versus the much less accurate old Ours.

Paired correctness:

| Comparison | ProMixed fixes | ProMixed regressions | Exact McNemar p |
|---|---:|---:|---:|
| vs. old Ours | 122 | 23 | 1.73e-17 |
| vs. ContiguousKV | 93 | 49 | 2.78e-4 |

The improvement directly addresses the old class collapse:

| Label | Old Ours | ContiguousKV | ProMixed |
|---|---:|---:|---:|
| ABBR | 87.50% | 100.00% | 100.00% |
| DESC | 6.62% | 2.94% | 52.94% |
| ENTY | 88.17% | 91.40% | 67.74% |
| HUM | 90.77% | 90.77% | 86.15% |
| LOC | 55.56% | 60.49% | 70.37% |
| NUM | 14.16% | 60.18% | 53.98% |

The all-group policy trades some ENTY/HUM accuracy for large DESC/NUM gains,
while also improving LOC. Aggregate paired gains remain statistically
significant against both saved baselines.

Runtime audit for the final run:

- 7.792 selector calls/request and mean adaptive period 5.116;
- 11.65 ms selector load, 87.72 ms selector compute, 1.21 ms selector wait;
- 37.90 ms critical prefetch wait;
- 19.00 ms cache-score update, entirely after the TTFT timestamp;
- zero request-time selector disk bytes;
- 1.590% of payload-prefetch tensor tokens from disk (fraction `0.015897`).

## Cross-task safety screen

SST-2, SUBJ, and RTE were run in one process for 64 requests/task. The table
uses the strict 60-UID intersection with each saved baseline:

| Task | Old Ours | ContiguousKV | ProMixed |
|---|---:|---:|---:|
| SST-2 | 93.33% | 91.67% | 91.67% |
| SUBJ | 53.33% | 46.67% | 65.00% |
| RTE | 91.67% | 90.00% | 91.67% |

No catastrophic cross-task accuracy regression is observed. The combined
screen is not used for formal latency comparison because task ordering changes
the shared Pcache residency state; the final TREC run is the matched primary
performance result.

## Conclusion

The proposal succeeds on its motivating failure case. GQA-aware fair selection
does more than restore the 10.91-point old Ours-to-ContiguousKV accuracy gap:
it exceeds ContiguousKV by 8.87 points while retaining a 36.12% mean TTFT
advantage. The cost relative to the previous fast-but-inaccurate Ours is
23.88 ms mean TTFT.

## Full independent grid follow-up (2026-08-20)

### Protocol and headline

- Independent process per dataset/configuration; no pooled latency metric.
- Full samples: SST-2 868, SUBJ 999, TREC 496, and RTE 273.
- Exact 5%, 10%, 25%, and 50% layer-block budgets.
- One pass over 32 warmup samples, then complete label-continuation scoring.
- All formal runs serialized on GPU0. Each output was checked for the exact
  sample count, paired UIDs, zero selector fallbacks, and intended budget.

Against ContiguousKV, ProMixed reduces mean TTFT in 16/16 settings and P95 in
15/16. Accuracy improves in 14/16. The two decreases (SST-2 25%: -0.46 pp;
RTE 50%: -0.73 pp) are small and non-significant. SUBJ and TREC accuracy gains
are significant at every budget.

Entries below follow budgets 5% / 10% / 25% / 50%:

| Dataset | Accuracy delta (pp) | Mean TTFT reduction | P95 reduction |
|---|---|---|---|
| SST-2 | +2.76 / +1.38 / -0.46 / +0.23 | 8.92% / 17.57% / 12.36% / 17.22% | 2.48% / 12.16% / 10.48% / 19.26% |
| SUBJ | +11.41 / +23.62 / +36.24 / +23.12 | 11.21% / 19.96% / 15.54% / 25.54% | 4.19% / 12.69% / 9.33% / 18.26% |
| TREC | +7.46 / +8.87 / +10.89 / +10.69 | 42.99% / 43.79% / 35.63% / 28.55% | 36.57% / 37.85% / 30.44% / 40.13% |
| RTE | +3.30 / +2.93 / +1.83 / -0.73 | 5.91% / 14.50% / 19.42% / 35.14% | -25.39% / 8.26% / 15.66% / 33.77% |

The generated paired report contains all 16 absolute accuracies, mean/P95
latencies, exact McNemar p-values, per-request faster counts, SSD reductions,
and paired bootstrap intervals.

### Is the double win caused by mixed precision?

No. ContiguousKV and ProMixed both use BF16 model compute and FP16 final KV
storage/attention. INT4 is used only for the selector-key index; it affects
block ranking but never replaces final FP16 K/V values.

The controlled TREC 10% run keeps the GQA policy and other settings fixed while
removing the INT4 selector index:

| Method | Selector | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |
|---|---|---:|---:|---:|
| ContiguousKV | original FP16 path | 55.04% | 454.55 | 514.61 |
| ProMixed | FP16 selector | 67.74% | 381.95 | 437.76 |
| ProMixed | INT4 selector index | 63.91% | 255.51 | 319.84 |

FP16 ProMixed already beats ContiguousKV by 12.70 accuracy points (109
wrong-to-correct versus 46 correct-to-wrong, p=4.47e-7) and reduces mean TTFT
by 15.97%. INT4 then reduces mean TTFT another 33.10% relative to FP16, but
loses 3.83 accuracy points (26/45 paired flips, p=0.0319). INT4 is therefore a
speed/accuracy tradeoff, not the source of the accuracy gain.

Accuracy improves because the old query heads 0/1/2 all map to physical KV
head 0, while ProMixed representatives 0/7/14/21 cover all four Qwen GQA
groups. Fair group coverage, normalized fusion, and uncertainty-triggered
reselection preserve evidence omitted by the old selector.

TTFT falls mainly through the compact preloaded selector index and critical
path changes. On SST-2 5%, ContiguousKV averages 12.46 MB SSD reads/request,
including 11.71 MB selector keys, and 60.81 ms selector loading. ProMixed has
zero request-time selector disk bytes, 0.086 MB total SSD reads, and 8.45 ms
selector loading. GPU-side 16-token block reduction, known-period/value-ordered
prefetch, and post-TTFT cache-score maintenance provide additional reductions.
ProMixed actually spends more selector compute here, so cheaper INT4 arithmetic
alone cannot explain its lower TTFT.

### RTE 5% repeat and concurrency audit

The RTE 5% formal run improves mean TTFT but regresses P95. A separately stored
repeat has identical predictions/correctness for all 273 UIDs:

| Run | Accuracy | Mean TTFT (ms) | P95 TTFT (ms) |
|---|---:|---:|---:|
| ContiguousKV | 86.45% | 295.05 | 337.89 |
| ProMixed formal | 89.74% | 277.60 | 423.68 |
| ProMixed repeat | 89.74% | 288.33 | 386.94 |

Thus the RTE 5% P95 regression is a reproducible selector/prefetch long-tail
issue and the clearest remaining optimization target.

Three archived dual-GPU trials have predictions identical to their single-GPU
reruns, but mean TTFT inflation of 43.15% (SST-2 50%), 61.15% (SUBJ 5%), and
81.56% (SUBJ 10%). They carry `dual_gpu`/`dual_gpu_incomplete` suffixes and are
excluded. The canonical analyzer reads only GPU0-only directories.

### Full-grid artifacts

- Canonical data: `results/promixed_full_grid_20260819`
- Paired report: `results/promixed_full_grid_20260819/promixed_full_grid_report.md`
  and adjacent JSON.
- FP16 ablation: `results/promixed_ablation_20260820/trec/k010_promixed_fp16`
- RTE repeat: `results/promixed_repeat_20260820/rte/k005_promixed_k4`
- Figures: `results/promixed_full_grid_20260819/figures` (PNG and PDF).
