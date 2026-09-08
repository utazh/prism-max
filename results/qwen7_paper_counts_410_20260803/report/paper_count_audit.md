# Paper-count matrix audit

Status: **passed**

- Requests: 410 paired UIDs with counts {'sst2': 100, 'subj': 110, 'trec': 120, 'rte': 80}.
- IMPRESS calibration overlap: 0 of 16 excluded UIDs.
- GPU boundary snapshots: 24 on GPU 3.

| Run | Samples | Exact-budget mismatches | Prefetch submitted/completed | Failed | Cancelled |
|---|---:|---:|---:|---:|---:|
| k005_impress | 410 | 0 | 0/0 | 0 | 0 |
| k005_contiguouskv | 410 | 0 | 0/0 | 0 | 0 |
| k005_ours | 410 | 0 | 22548/22548 | 0 | 0 |
| k010_contiguouskv | 410 | 0 | 0/0 | 0 | 0 |
| k010_ours | 410 | 0 | 22550/22550 | 0 | 0 |
| k010_impress | 410 | 0 | 0/0 | 0 | 0 |
| k025_contiguouskv | 410 | 0 | 0/0 | 0 | 0 |
| k025_ours | 410 | 0 | 22550/22550 | 0 | 0 |
| k025_impress | 410 | 0 | 0/0 | 0 | 0 |
| k050_ours | 410 | 0 | 22550/22550 | 0 | 0 |
| k050_impress | 410 | 0 | 0/0 | 0 | 0 |
| k050_contiguouskv | 410 | 0 | 0/0 | 0 | 0 |
