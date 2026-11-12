# Where every number comes from

Split by what this repository can recompute and what is quoted for context. No number here
depends on a file that is not in the tree, except the competition images themselves.

## Reproducible inside this repository

| Number | System | Reproduce with | Expected |
|---|---|---|---|
| **val PQ 0.4591** · SQ 0.6891 · RQ 0.6662 · TP 933 · FP 474 · FN 461 | submitted student | notebook §8 — exact multi-annotator PQ over the 107 validation files after fresh inference | those six values |
| **1,279 rows over 177/180 test images** | submitted student | notebook §7 against the test images | `rows=1279 images=177` |
| **Submission passes 13/13 checks** | submitted student | `python tools/verify_submission.py submissions/submission_keep_s3_light.csv` | `13/13 checks pass` |
| **Reference SHA-256 `9ADF36A4…`** | submitted student | `sha256sum submissions/submission_keep_s3_light.csv` | the shipped CSV |
| **1,280 rows / 1,277 of 1,279 identical on re-inference** | submitted student | notebook §7 | `rows=1280`; the hash differs by one image, by construction |
| **26.9 img/s forward · 71 ms end-to-end** | submitted student | `python tools/bench_latency.py --n 30` and `python tools/bench_endtoend.py --n 30` (needs a CUDA GPU; the quoted numbers are an RTX 4070, batch 1, FP16) | 37.2 ms forward (26.9 img/s); 71 ms end-to-end (14.1 img/s) |
| **276 ms per image through the 8×1024 sliding window** | submitted student | `python tools/bench_latency.py --n 30`, third row | 276.4 ms (3.6 img/s) |

## Quoted for context, not recomputable from this tree

| Number | System | Why it is not recomputable here |
|---|---|---|
| val PQ 0.4773 | four-model ensemble (171 M) | needs the four member checkpoints and their probability caches, which are not shipped |
| val PQ 0.4599 · LB 0.37 | narrow student (64-wide) | the narrow student weight is shipped; its validation PQ needs the same evaluation over its own cached maps |
| 355 ms per image · 2.82 img/s | four-model ensemble (171 M) | needs the four member checkpoints, which are not shipped; the figure quotes the sum of four single passes |
| public LB 0.38 / 0.37 | all | the leaderboard is the organisers' artefact, not a local computation |
| 2.4% / 5.1% multi-partner at IoU>0 | submitted student | report Figure 4; computed from the development probability caches |

Deltas in the report (the KEEP rule at +0.0158 nested-CV PQ, the fine-tunes) come from
nested 5-fold cross-validation on the same 107 files with paired file-level bootstrap; the
tuning code is `keep_sweep.py`, and it needs the cached validation maps to re-run.

## Reading the paper

The submitted report has 4 pages. Table 1 has the three systems; Figure 2 measured throughput;
Figure 3 six zoomed validation examples; Figure 4 the IoU/Dice and overlap-partner
distributions; Table 2 the bootstrap
intervals for the rule and fine-tune effects.
