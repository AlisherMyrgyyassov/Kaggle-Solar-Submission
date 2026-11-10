# Compliance with the final-submission requirements

Each requirement from the competition's **Final Submission**, **Open-Access Policy** and
**Evaluation** sections, and where this repository satisfies it. Nothing here needs a
credential, a private link or a manual download.

| Requirement | Where it is met | How to check |
|---|---|---|
| Public Git repository, reachable without access permission | this repository — public from competition close, and kept public until the winners are announced (Open-Access Policy) | open the URL in a private browser window |
| `requirements.txt` with every package and its version | [`requirements.txt`](../requirements.txt) | `pip install -r requirements.txt` |
| A Jupyter notebook illustrating the entire pipeline | [`notebooks/pipeline.ipynb`](../notebooks/pipeline.ipynb) — split, preprocessing, model, optional training, inference, post-processing, submission, evaluation; all cells executed | run it (README → *Verify it yourself*) |
| Model weights stored in the repository and loaded by the notebook, so results reproduce **without retraining** | [`weights/`](../weights) — the submitted weight (16.4 MiB) and the lean variant (14.4 MiB) | notebook §3 loads it; §7 regenerates the CSV |
| Technical report, 4 pages, single PDF, on the provided template | [`report/main.pdf`](../report/main.pdf) (4 pages) | `pdfinfo report/main.pdf` |
| Inference uses only the H-alpha images in the test set; no external ground-truth metadata | solar-disk gate computed from each test image; no external data, no test-time adaptation | notebook §2 and §5 |
| Source code reusable — licence | [`LICENSE`](../LICENSE) — MIT for the code; weights follow the Apache-2.0 lineage of the pretrained backbone | `LICENSE` |
| Competition data is not redistributed | [`data/README.md`](../data/README.md) documents the expected layout; nothing under `data/` is committed | `git ls-files data/` |
| Repository is self-contained: no additional file needed to reproduce the predictions | notebook + `requirements.txt` + `weights/` + the competition test images | run the notebook against the test images |
| Prizes: complete source, scripts, configuration needed to reproduce the results | the same tree; the training entry point is [`train_v2.py`](../train_v2.py) and the notebook can drive it (`TRAIN = True`) | notebook §4 |

## What is deliberately not here

- **Training-teacher probability maps** (KD targets): larger than the model itself, so the
  notebook's training section runs stage 1 (30-epoch pretraining) rather than the KD5
  fine-tune, and says so in its markdown.
- **Intermediate probability caches** used during development: the release ships the final
  weights and the evaluation code instead, so every number can be recomputed without them.
- **Optimiser / resume checkpoints**: not needed for inference.

## Numbers

Every reported number and the command that reproduces it: [`RESULTS.md`](RESULTS.md).
