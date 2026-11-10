# Where every number comes from

Split by what this repository can recompute and what is quoted for context. No number here
depends on a file that is not in the tree, except the competition images themselves.

## Reproducible inside this repository

| Number | System | Reproduce with | Expected |
|---|---|---|---|
| **val PQ 0.4591** · SQ 0.6891 · RQ 0.6662 · TP 933 · FP 474 · FN 461 | S3 | notebook §8 — exact multi-annotator PQ over the 107 validation files after fresh inference | those six values |
| **1,279 rows over 177/180 test images** | S3 | notebook §7 against the test images | `rows=1279 images=177` |
| **Submission passes 13/13 checks** | S3 | `python tools/verify_submission.py submissions/submission_keep_s3_light.csv` | `13/13 checks pass` |
| **Reference SHA-256 `9ADF36A4…`** | S3 | `sha256sum submissions/submission_keep_s3_light.csv` | the shipped CSV |
| **1,280 rows / 1,277 of 1,279 identical on re-inference** | S3 | notebook §7 | `rows=1280`; the hash differs by one image, by construction |
| **107 GMAC, ~100 ms per image (FP16)** | S3 | notebook §5 runs the same single-pass path; timing varies by device | — |

## Quoted for context, not recomputable from this tree

| Number | System | Why it is not recomputable here |
|---|---|---|
| val PQ 0.4773 | S2, 171 M four-model ensemble | needs the four member checkpoints and their probability caches, which are not shipped |
| val PQ 0.4599 · LB 0.37 | S4, lean 64-wide student | the S4 weight is shipped; its validation PQ needs the same evaluation over its own cached maps |
| public LB 0.38 / 0.37 | all | the leaderboard is the organisers' artefact, not a local computation |
| 2.4% / 5.1% multi-partner at IoU>0 | S3 | report Figure 2; computed from the development probability caches |

Deltas in the report (the KEEP rule at +0.0158 nested-CV PQ, the fine-tunes) come from
nested 5-fold cross-validation on the same 107 files with paired file-level bootstrap; the
tuning code is `keep_sweep.py`, and it needs the cached validation maps to re-run.

## Reading the paper

`report/main.pdf` — 4 pages. Table 1 has the three operating points; Figure 2 the IoU/Dice
and one-to-many/many-to-one distributions; Figure 3 six zoomed validation examples;
Table 2 the bootstrap intervals for the rule and fine-tune effects.
