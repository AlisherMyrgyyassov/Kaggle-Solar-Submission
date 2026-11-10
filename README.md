# Efficient Solar Filament Segmentation — submission repository

Solution to the **Solar Filament Segmentation Challenge 2026** (MAGFiLO / GONG
H-alpha), IEEE BigData Cup.

**Submitted system (S3):** a 4.28 M-parameter ConvNeXt-atto U-Net trained with
knowledge distillation, randomized-annotator targets and SWA, deployed with
single-pass 2048-px inference, 8-view dihedral TTA and a decoupled KEEP rule.

| System | Params | MACs / image | Val PQ (107 files) | Public LB |
|---|---|---|---|---|
| S2 accuracy reference (4×1024 ensemble) | 171.0 M | 3.23 TMAC | 0.4773 | 0.38 |
| **S3 submitted student** | **4.28 M** | **107 GMAC** | **0.4591** | **0.38** |
| S4 lean frontier (64-wide decoder) | 3.75 M | 64.8 GMAC | 0.4599 | 0.37 |

At equal public LB, S3 uses **40× fewer parameters and ~30× fewer MACs** than
the accuracy ensemble. Under the official IoU>0.5 matching it produces zero
one-to-many / many-to-one relations.

## Repository layout

```
config.py, dataset.py, model_v2.py      data, model, training
model.py, model_edge.py                 EMA / auxiliary backbones
inference.py, predict.py                sliding-window + single-pass inference
preprocessing.py                        solar mask + KEEP postprocess primitives
presets.py, submit_keep.py              model registry + submission generator
cache_val_probs.py, keep_sweep.py       probability caches + nested-CV rule tuning
eval_pq.py, eval_pq_v2.py               v1 fast matcher + exact multi-annotator PQ
train_v2.py                             training / fine-tuning entry point
tools/verify_submission.py              13-check submission CSV verifier
notebooks/pipeline.ipynb                full pipeline: split, preprocessing, model, optional
                                        training, inference, KEEP, submission, evaluation
weights/                                submitted S3 + lean S4 checkpoints
submissions/                            reference S3/S4/S2 CSVs
report/                                 4-page technical report (LaTeX + PDF)
data/README.md                          expected competition-data layout
```

## Setup

Python 3.11:

```powershell
pip install -r requirements.txt
```

Download the competition data from Kaggle and place it under
`data/MAGFiLO_1.0_Kaggle_2026/` (see `data/README.md`). If it lives elsewhere:

```powershell
$env:MAGFILO_DATA_DIR = "D:\data\MAGFiLO_1.0_Kaggle_2026"
```

## Reproduce the submitted predictions (no retraining)

Open `notebooks/pipeline.ipynb` and run all cells. It loads
`weights/atto_dec128_full_randann_swa.pth`, runs the 180 test images through
single-pass 2048 + D4 TTA8, applies KEEP `0.35/0/0.99/100/0.85`, writes
`submissions/regenerated_submission_keep_s3_light.csv`, and verifies it.

Equivalent CLI:

```powershell
python submit_keep.py --preset randann_student_swa --path full --tta 8 `
  --t_lo 0.35 --close 0 --t_hi 0.99 --min_area 100 --min_meanp 0.85 `
  --model_dir weights --half --out submissions/regenerated_submission_keep_s3_light.csv
python tools/verify_submission.py submissions/regenerated_submission_keep_s3_light.csv
```

Reference CSV: `submissions/submission_keep_s3_light.csv`
- 1,279 rows, 177/180 images (three empty images are legitimate)
- SHA-256 `9ADF36A4E28AE44368F59E644F26FA105E7456ED7D8F5ED07ABF09ADC92F8217`
- 13/13 verifier checks pass

Re-inference from the released weight reproduces **1,277 of those 1,279 rows
byte-for-byte**, and the reported validation PQ (0.4591) exactly, in about a
minute on one GPU. The difference is a single image: two contours shift by one
pixel and a third boundary component passes the area test, so the regenerated
file has 1,280 rows — the hash differs by construction, not because the rule
changed. The stored FP16 test cache is not a substitute: it differs from the
submitted file on 31 rows across 22 images. Use CUDA + `--half`.

The nested-CV optimum was `min_area=300`; 100 and 300 are indistinguishable on
validation (0.45906 vs 0.45909), and the submitted CSV uses 100.

## Reproduce the lean S4 point

```powershell
python submit_keep.py --preset atto_dec64_randann_swa --path full --tta 8 `
  --t_lo 0.30 --close 0 --t_hi 0.99 --min_area 100 --min_meanp 0.85 `
  --model_dir weights --half --out submissions/regenerated_submission_keep_s4_lean.csv
```

## Training (for reference; weights are already included)

The submitted student is a 6-epoch fine-tune from the complete-GT dec128
checkpoint: 1024-px crops, batch 4, AdamW lr 3e-5, KD5 (MSE to five teachers),
`--mask_mode random_annotator`, SWA over the last 3 epochs, checkpoint selection
by full-image PQ under the deployed KEEP rule. The exact training entry point is
`train_v2.py`; the data split is 600 train / 107 validation files (85/15).

## Evaluation

`eval_pq_v2.py` implements the exact competition metric: multi-annotator
per-image ground truth and all-pairs IoU>0.5 matching. `keep_sweep.py` runs
nested 5-fold CV and paired bootstrap for rule/ensemble choices.

## Report

`report/main.pdf` is the 4-page technical report; `report/main.tex` and
`report/figures/` build it. The reported numbers are the verified S3/S2/S4
values above.

## Data / code policy

Only the H-alpha competition images are used at inference. No external
ground-truth metadata or test labels are used. The accuracy reference S2 is
reported for comparison; only the submitted S3 and lean S4 weights are shipped.

## Licence

- **Code**: MIT — see `LICENSE`.
- **Weights**: fine-tuned from timm's ImageNet-pretrained `convnext_atto`
  backbone (Apache-2.0) and released under the same terms; the fine-tuned
  parameters are ours.
- **Data**: not redistributed; download from the competition page
  (see `data/README.md`).
