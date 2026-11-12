"""Measure inference latency for the operating points present in this tree.

Model-only timing: the validation images are pre-loaded, so what is measured is the
forward pass (plus TTA/stitching where the path includes it), not disk I/O.

    python tools/bench_latency.py --n 30 [--out latency.json]

Works in both trees: checkpoints are read from output/models/ when present, otherwise
from weights/. Systems whose checkpoints are absent are skipped with a note, so the
ensemble row simply does not appear in a tree that ships only the student.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import Config
from dataset import create_file_splits, load_and_parse_annotations
from predict import predict_prob_map, predict_prob_map_full
from presets import PRESETS, load_checkpoint

_MODELS = os.path.join(_ROOT, "output", "models")
_URL = ""
if not os.path.isdir(_MODELS):
    _MODELS = os.path.join(_ROOT, "weights")
    _URL = " (see weights/README.md for the download link)"
MODEL_DIR = _MODELS


def have(tag):
    return bool(glob.glob(os.path.join(MODEL_DIR, tag + "*.pth")))


def time_calls(fn, images, warmup=5):
    for img in images[:warmup]:
        fn(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for img in images:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=None, help="optional JSON path for the results")
    args = ap.parse_args()

    Config.IN_CHANNELS = 3      # the submitted models take 3-channel replicated input
    Config.NORMALIZE = True

    data = load_and_parse_annotations(Config.ANNOTATION_FILE)
    _, val_files = create_file_splits(data, Config.VAL_SPLIT, Config.SEED)
    stems = [os.path.splitext(f)[0] for f in val_files][:args.n]
    images = [cv2.imread(os.path.join(Config.TRAIN_IMG_DIR, s + ".jpeg"), cv2.IMREAD_GRAYSCALE)
              for s in stems]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{len(images)} images | {gpu} | half={torch.cuda.is_available()} | "
          f"checkpoints {MODEL_DIR}{_URL}")

    results = []

    def bench(label, tag, kind, macs_gmac):
        if not have(tag):
            print(f"  {label:34s} skipped (no {tag}*.pth in this tree)")
            return
        model, spec, _ = load_checkpoint(tag, MODEL_DIR, Config.DEVICE,
                                         half=torch.cuda.is_available())
        Config.HALF = torch.cuda.is_available()
        Config.CROP_SIZE = spec["crop"]
        Config.MODEL_INPUT_SIZE = spec["model_size"]
        if kind == "full":
            fn = lambda im: predict_prob_map_full(im, model, Config, tta=1)
        elif kind == "tta8":
            fn = lambda im: predict_prob_map_full(im, model, Config, tta=8)
        else:
            fn = lambda im: predict_prob_map(im, model, Config)
        times = time_calls(fn, images)
        row = dict(label=label, tag=tag, path=kind, macs_gmac=macs_gmac,
                   mean_ms=round(statistics.mean(times), 1),
                   median_ms=round(statistics.median(times), 1),
                   sd_ms=round(statistics.stdev(times), 1) if len(times) > 1 else 0.0,
                   ips=round(1000.0 / statistics.mean(times), 2))
        results.append(row)
        print(f"  {label:34s} {row['mean_ms']:7.1f} ms  ({row['ips']:5.2f} img/s)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bench("S3 student, single pass", "atto_dec128_full_randann_swa", "full", 107.4)
    bench("S3 student, single pass + TTA8", "atto_dec128_full_randann_swa", "tta8", 107.4)
    bench("S3 student, 8x1024 patch path", "atto_dec128_full_randann_swa", "patch", 214.9)
    bench("S4 student (64-wide), single pass", "atto_dec64_full_randann_swa", "full", 64.8)

    members = PRESETS["ens4_oldfull"]
    if all(have(tag) for tag, _ in members):
        total = 0.0
        per_member = []
        for tag, _w in members:
            model, spec, _ = load_checkpoint(tag, MODEL_DIR, Config.DEVICE,
                                             half=torch.cuda.is_available())
            Config.HALF = torch.cuda.is_available()
            Config.CROP_SIZE = spec["crop"]
            Config.MODEL_INPUT_SIZE = spec["model_size"]
            times = time_calls(lambda im: predict_prob_map_full(im, model, Config, tta=1), images)
            per_member.append(dict(tag=tag, mean_ms=round(statistics.mean(times), 1)))
            total += statistics.mean(times)
            print(f"  member {tag:32s} {statistics.mean(times):7.1f} ms")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(dict(label="S2 ensemble, single pass (sum of 4)",
                            tag="+".join(t for t, _ in members), path="ensemble",
                            macs_gmac=3230.0, mean_ms=round(total, 1), median_ms=None,
                            sd_ms=None, ips=round(1000.0 / total, 2), members=per_member))
        print(f"  {'S2 ensemble, single pass (sum)':34s} {total:7.1f} ms")
    else:
        print("  S2 ensemble skipped (its four member checkpoints are not in this tree)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(n_images=len(images), gpu=gpu,
                           half=bool(torch.cuda.is_available()), rows=results), f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
