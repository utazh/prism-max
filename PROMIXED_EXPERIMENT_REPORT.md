# ProMixed isolated experiment report

Date: 2026-08-19

## Scope and isolation

- Source inspected: `/home/panzihang/src/prism_datasetwise_logit_20260804`
- Isolated worktree: `/home/panzihang/src/prism_promixed`
- Baseline snapshot commit: `fb3b4af`
- ProMixed implementation commit: `5caaeef`
- The source experiment directory was not edited. Code, generated indexes,
  logs, and results remain on the server.
- GPU runs use one experiment GPU and designate a different GPU as the
  reserve. Runs are refused when either is occupied at launch.

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

## Pending final run

Run the selected default policy on all 496 TREC records, then compare paired
accuracy and TTFT against the saved old Ours and ContiguousKV results. Add
cross-task checks only after the full TREC acceptance test.
