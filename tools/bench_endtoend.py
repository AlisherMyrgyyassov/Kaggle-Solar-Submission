"""End-to-end latency for the submitted student: forward pass, then KEEP and RLE.

Model-only timing answers "how fast is the network". The notebook and the submission
script also pay for thresholding, connected components, the KEEP rule and RLE
encoding, so this measures the whole per-image path and breaks the cost down.

    python tools/bench_endtoend.py --n 30
"""
import argparse
import json
import os
import statistics
import sys
import time

import cv2
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import Config
from dataset import create_file_splits, load_and_parse_annotations
from predict import predict_prob_map_full
from preprocessing import component_stats, keep_mask_from_stats, kept_area_rles
from presets import load_checkpoint

_MODELS = os.path.join(_ROOT, "output", "models")
if not os.path.isdir(_MODELS):
    _MODELS = os.path.join(_ROOT, "weights")
MODEL_DIR = _MODELS

# S3's submitted rule
T_LO, T_HI, MIN_AREA, MIN_MEANP, CLOSE = 0.35, 0.99, 100, 0.85, 0


def timeit(fn, images, warmup=5):
    for im in images[:warmup]:
        fn(im)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out = []
    for im in images:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(im)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="atto_dec128_full_randann_swa")
    ap.add_argument("--out", default=None, help="optional JSON path for the results")
    args = ap.parse_args()

    Config.IN_CHANNELS = 3
    Config.NORMALIZE = True
    Config.HALF = torch.cuda.is_available()

    data = load_and_parse_annotations(Config.ANNOTATION_FILE)
    _, val_files = create_file_splits(data, Config.VAL_SPLIT, Config.SEED)
    stems = [os.path.splitext(f)[0] for f in val_files][:args.n]
    images = [cv2.imread(os.path.join(Config.TRAIN_IMG_DIR, s + ".jpeg"),
                         cv2.IMREAD_GRAYSCALE) for s in stems]

    model, spec, _ = load_checkpoint(args.tag, MODEL_DIR, Config.DEVICE,
                                     half=torch.cuda.is_available())
    Config.CROP_SIZE = spec["crop"]
    Config.MODEL_INPUT_SIZE = spec["model_size"]

    def forward(im):
        return predict_prob_map_full(im, model, Config, tta=1)

    def post(prob):
        labels, stats, area, meanp, pmax = component_stats(prob, T_LO, CLOSE)
        keep = keep_mask_from_stats(area, meanp, pmax, T_HI, MIN_AREA, MIN_MEANP)
        return kept_area_rles(labels, stats, keep, min_area=MIN_AREA)

    probs = [forward(im) for im in images]      # cached so post can be timed alone

    t_fwd = timeit(forward, images)
    t_post = timeit(post, probs)
    t_all = timeit(lambda im: post(forward(im)), images)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    res = dict(gpu=gpu, n=len(images), half=bool(Config.HALF),
               forward_ms=round(statistics.mean(t_fwd), 1),
               forward_sd=round(statistics.stdev(t_fwd), 1),
               post_ms=round(statistics.mean(t_post), 1),
               post_sd=round(statistics.stdev(t_post), 1),
               total_ms=round(statistics.mean(t_all), 1),
               total_median_ms=round(statistics.median(t_all), 1),
               total_sd=round(statistics.stdev(t_all), 1),
               fps=round(1000.0 / statistics.mean(t_all), 2))

    print(f"{gpu} | {res['n']} images | half={res['half']}")
    print(f"  forward only      {res['forward_ms']:7.1f} ms  (+- {res['forward_sd']:.1f})")
    print(f"  KEEP + RLE        {res['post_ms']:7.1f} ms  (+- {res['post_sd']:.1f})")
    print(f"  end to end        {res['total_ms']:7.1f} ms  "
          f"(median {res['total_median_ms']:.1f}, +- {res['total_sd']:.1f})")
    print(f"  throughput        {res['fps']:7.2f} images/s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
