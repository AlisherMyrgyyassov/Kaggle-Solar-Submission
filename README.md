# Efficient Solar Filament Segmentation — submission repository

**4.28 M-parameter ConvNeXt-atto U-Net · 26.9 images/s · val PQ 0.4591 on 107 files · public LB 0.38**

Submitted to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(MAGFiLO / GONG H-α, IEEE BigData Cup). It reaches the same leaderboard score as a 171 M
teacher ensemble with **40× fewer parameters** and **9.5× higher throughput** per image.

![Six zoomed validation windows from the submitted model, with ground-truth and predicted contours.](docs/figures/hero.jpg)

*Gold: expert ground truth · Teal: prediction · Blue: a second annotator. Six windows of the
107-file validation split, drawn from the submitted model's own probability maps. (a,b) the
prediction follows thin barbs and a large curved body; (c,d) filaments it misses; (e,f) cases
where two annotators disagree and the prediction follows one. Each full-disk inset locates the crop.*

**[Pipeline notebook](notebooks/pipeline.ipynb)** ·
**[Compliance checklist](docs/COMPLIANCE.md)** ·
**[Where every number comes from](docs/RESULTS.md)**

The 4-page technical report is submitted with the entry and is not part of this repository.

## What was submitted

The entry is one CSV, produced by one checkpoint with one post-processing rule.

| Item | What it is |
|---|---|
| **Submission file** | `submissions/submission_keep_s3_light.csv` — 1,279 rows over 177 of the 180 test images (three images contain no filament). This is the exact file uploaded to Kaggle. sha256 `9adf36a4e28ae44368f59e644f26fa105e7456ed7d8f5ed07abf09adc92f8217` |
| **Checkpoint** | `weights/atto_dec128_full_randann_swa.pth` — ConvNeXt-atto U-Net, 128-wide decoder, 4.28 M parameters. sha256 `3ad0992aeb808f3892223367eaf84a4d1f307424eb5baefed31af87cf783973d` |
| **Post-processing** | the decoupled acceptance rule at `t_low=0.35, close=0, t_peak=0.99, a_min=100, mean=0.85`, applied to the averaged maps of eight dihedral views from a single 2048×2048 forward pass |

The command that regenerates the submission from these three is under *Reproduce the submitted
predictions without retraining* below; it reproduces 1,277 of the 1,279 rows byte for byte.

## Verify it yourself

```bash
pip install -r requirements.txt
# competition data under data/MAGFiLO_1.0_Kaggle_2026/ (see data/README.md),
# or point MAGFILO_DATA_DIR at it
jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline.ipynb
```

Expected output, in order: the file split (600 train / 107 validation), the model (4.28 M),
the submission CSV, the verifier, and the exact validation score —

```
rows=1280  images=177
13/13 checks pass
val PQ 0.4591 | SQ 0.6891 | RQ 0.6662 | TP 933 FP 474 FN 461
```

The notebook runs the whole pipeline — split, preprocessing, model, **optional training**,
inference, KEEP post-processing, submission and evaluation — in about two minutes on one GPU.
Set `TRAIN = True` in its first cell to train instead of loading the shipped weights.

## Results

| System | Params | Images/s | Val PQ (107 files) | Public LB |
|---|---|---|---|---|
| Teacher ensemble | 171.0 M | 2.8 | 0.4773 | 0.38 |
| **Submitted student** | **4.28 M** | **26.9** | **0.4591** | **0.38** |
| Narrow student (64-wide decoder) | 3.75 M | 33.1 | 0.4599 | 0.37 |

Throughput is a full 2048×2048 image per second, one RTX 4070, batch 1, FP16, 30 images
after warm-up, one forward pass per model (`tools/bench_latency.py`); the eight-view
augmentation used at submission applies to all three and is not included. End to end —
forward plus KEEP and RLE encoding — the submitted student takes 71 ms per image,
14 images/s (`tools/bench_endtoend.py`).

Under the official IoU>0.5 matching every instance has at most one partner by construction.
The informative count is at any overlap: at IoU>0, 2.4% of ground-truth instances and 5.1%
of predictions overlap more than one partner. The remaining error is under-detection rather
than fragmentation: 63% of false negatives are filaments the model never fires on.

## What is in the repository

| Path | Contents |
|---|---|
| `notebooks/pipeline.ipynb` | the whole pipeline, executed, training optional |
| `weights/` | the submitted student weight (16.4 MiB) and the narrow student weight (14.4 MiB) |
| `submissions/` | the submitted CSV, plus the narrow-student and ensemble CSVs as reference outputs |
| `predict.py`, `preprocessing.py` | single-pass + TTA inference; disk gate and KEEP primitives |
| `model_v2.py`, `presets.py` | ConvNeXt U-Net; one registry shared by evaluation and submission |
| `eval_pq_v2.py` | the exact multi-annotator Panoptic Quality used in the report |
| `train_v2.py` | training and fine-tuning entry point (the notebook drives it) |
| `tools/verify_submission.py` | 13-check submission CSV verifier |
| `tools/bench_latency.py`, `tools/bench_endtoend.py` | the timing commands behind the report's throughput figure (need a CUDA GPU) |
| `docs/COMPLIANCE.md` | each competition requirement mapped to the file that satisfies it |
| `docs/RESULTS.md` | every reported number and the command that reproduces it |

The tree carries only what reproduces the submission: inference, training, the exact metric, the tuning harness and the verifier. The v1 UltraUNet pipeline, the EdgeAttNet experiment and the older greedy evaluator were removed rather than shipped as dead code.

Only the two student checkpoints ship. The other configurations registered in `presets.py` are the
comparison points in Table 1 of the report (a ConvNeXt-tiny student, a ResNet-34d student, the
four-model teacher ensemble and its members); their weights are not part of this release, and
nothing in the reproduction path needs them.

## Setup

Python 3.11, pinned in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Download the competition data from Kaggle and place it under
`data/MAGFiLO_1.0_Kaggle_2026/` (see `data/README.md`). If it lives elsewhere:

```powershell
$env:MAGFILO_DATA_DIR = "D:\data\MAGFiLO_1.0_Kaggle_2026"
```

## Reproduce the submitted predictions without retraining

The notebook does this end to end. The equivalent CLI:

```powershell
python submit_keep.py --preset randann_student_swa --path full --tta 8 `
  --t_lo 0.35 --close 0 --t_hi 0.99 --min_area 100 --min_meanp 0.85 `
  --model_dir weights --half --out submissions/regenerated_submission_keep_s3_light.csv
python tools/verify_submission.py submissions/regenerated_submission_keep_s3_light.csv
```

Reference CSV `submissions/submission_keep_s3_light.csv`: 1,279 rows over 177/180 images
(three empty images are legitimate), SHA-256
`9ADF36A4E28AE44368F59E644F26FA105E7456ED7D8F5ED07ABF09ADC92F8217`, 13/13 checks passing.

Re-inference from the shipped weight reproduces **1,277 of those 1,279 rows byte-for-byte**
and the reported validation PQ (0.4591) exactly, in about a minute on one GPU. The difference
is one image: two contours shift by one pixel and a third boundary component passes the area
test, so the regenerated file has 1,280 rows and a different hash by construction. The stored
FP16 test cache is not a substitute — it differs from the submitted file on 31 rows across 22
images. Use CUDA with `--half`.

The nested-CV optimum was `min_area=300`; 100 and 300 are indistinguishable on validation
(0.45906 vs 0.45909), and the submitted CSV uses 100.

## Reproduce the narrow student point

```powershell
python submit_keep.py --preset atto_dec64_randann_swa --path full --tta 8 `
  --t_lo 0.30 --close 0 --t_hi 0.99 --min_area 100 --min_meanp 0.85 `
  --model_dir weights --half --out submissions/regenerated_submission_keep_s4_lean.csv
```

## Training

The submitted student is a 6-epoch fine-tune from the complete-GT dec128 checkpoint:
1024-px crops, batch 4, AdamW lr 3e-5, distillation from five teachers (MSE, weight 5),
`--mask_mode random_annotator`, SWA over the last 3 epochs, checkpoints selected by
full-image PQ under the deployed KEEP rule. The parent model was trained for 30 epochs at
lr 1e-4 on first-annotator targets. `train_v2.py` is the entry point; the notebook runs a
reduced version of it when `TRAIN = True`. The five teacher probability maps are not shipped
(they are larger than the model), so that section runs stage 1 only and says so.

## Evaluation

`eval_pq_v2.py` implements the exact competition metric: multi-annotator per-image ground
truth, all-pairs IoU>0.5 matching. `keep_sweep.py` runs nested 5-fold CV and paired
file-level bootstrap for the rule and ensemble choices quoted in the report.

## Data and code policy

Only the H-α competition images are used at inference: the solar-disk gate is computed from
each test image itself, and no external ground-truth metadata, test labels or test-time
adaptation is used. The four-model ensemble is reported for comparison only; only the
submitted student and narrow student weights are shipped. Per-requirement detail:
[`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).

## Licence

- **Code**: MIT — see `LICENSE`.
- **Weights**: fine-tuned from timm's ImageNet-pretrained `convnext_atto` backbone
  (Apache-2.0) and released under the same terms; the fine-tuned parameters are ours.
- **Data**: not redistributed; download from the competition page (see `data/README.md`).
