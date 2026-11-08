"""Generate a submission CSV with the decoupled KEEP rule.

    python submit_keep.py --preset ens4_oldfull --t_lo 0.35 --close 3 \
        --t_hi 0.97 --min_area 300 --min_meanp 0.75 \
        --out output\\submissions\\submission_keep_s1.csv

--path patch  : the original 8x1024 sliding window (`predict_prob_map`) — this is
                the path the val probability caches were built with, so it is the
                configuration the tuned rule was validated on.
--path full   : whole-image single forward (+ --tta 1/4/8), ~2.7x faster and
                -50% MACs for the same PQ (see predict.py). Only use it with a
                rule tuned on a matching `--pt full2048*` cache.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import csv

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from config import Config
from predict import predict_prob_map, predict_prob_map_full
from preprocessing import component_stats, keep_mask_from_stats, kept_area_rles
from presets import PRESETS, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ens4_oldfull")
    ap.add_argument("--members", default="",
                   help="Ad-hoc members 'tag:weight,tag:weight' (overrides --preset).")
    ap.add_argument("--path", default="patch", choices=["patch", "full"])
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--t_lo", type=float, default=0.35)
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--t_hi", type=float, default=0.97)
    ap.add_argument("--min_area", type=int, default=300)
    ap.add_argument("--min_meanp", type=float, default=0.75)
    ap.add_argument("--out", default="")
    ap.add_argument("--mask_dir", default="")
    ap.add_argument("--model_dir", default="",
                   help="Checkpoint directory (default: Config.MODEL_DIR).")
    ap.add_argument("--probs_dir", default="",
                   help="Cache/reuse mixed test probability maps here (float16 .npy per image).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="no CSV, stats only")
    args = ap.parse_args()

    config = Config
    config.IN_CHANNELS = 3
    config.NORMALIZE = True
    config.BATCH_SIZE = 8
    config.HALF = bool(args.half)

    members = ([tuple([p.split(":")[0], float(p.split(":")[1])]) for p in args.members.split(",")]
               if args.members else PRESETS[args.preset])
    wsum = sum(w for _, w in members)
    model_dir = args.model_dir or config.MODEL_DIR
    models = []
    for tag, w in members:
        m, spec, ck = load_checkpoint(tag, model_dir, config.DEVICE,
                                      half=args.half)
        models.append((tag, spec, m, w / wsum))
        print(f"loaded {tag} (w={w / wsum:.2f}, crop={spec['crop']}, "
              f"size={spec['model_size']}, epoch={ck.get('epoch')})")

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(config.TEST_IMG_DIR))
    if args.limit:
        stems = stems[:args.limit]
    mask_dir = args.mask_dir or os.path.join(config.OUTPUT_DIR, "test_masks_keep")
    os.makedirs(mask_dir, exist_ok=True)

    rows = []
    n_inst = 0
    if args.probs_dir:
        os.makedirs(args.probs_dir, exist_ok=True)
    for stem in tqdm(stems, desc=f"{args.preset}/{args.path}"):
        cache_path = (os.path.join(args.probs_dir, stem + ".npy")
                      if args.probs_dir else "")
        if cache_path and os.path.exists(cache_path):
            prob = np.load(cache_path).astype(np.float32)
        else:
            img = cv2.imread(os.path.join(config.TEST_IMG_DIR, stem + ".jpeg"),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = cv2.imread(os.path.join(config.TEST_IMG_DIR, stem + ".jpg"),
                                 cv2.IMREAD_GRAYSCALE)
            acc = None
            for _tag, spec, m, w in models:
                config.CROP_SIZE = spec["crop"]
                config.MODEL_INPUT_SIZE = spec["model_size"]
                if args.path == "patch":
                    p = predict_prob_map(img, m, config, tta=args.tta > 1)
                else:
                    p = predict_prob_map_full(img, m, config, tta=args.tta)
                acc = p * w if acc is None else acc + p * w
            prob = acc
            if cache_path:
                np.save(cache_path, prob.astype(np.float16))

        labels, stats, area, meanp, pmax = component_stats(prob, args.t_lo, args.close)
        keep = (keep_mask_from_stats(area, meanp, pmax, args.t_hi, args.min_area,
                                     args.min_meanp)
                if area.size else np.zeros(0, bool))
        rles = kept_area_rles(labels, stats, keep, min_area=args.min_area)

        binary = np.zeros_like(labels, dtype=np.uint8)
        for idx in np.flatnonzero(keep):
            binary[labels == int(idx) + 1] = 255
        cv2.imwrite(os.path.join(mask_dir, f"{stem}.png"), binary)

        for i, rle in enumerate(rles, start=1):
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": counts})
        n_inst += len(rles)

    dropped = np.zeros(0)
    print(f"{args.preset} / {args.path} tta={args.tta}: {n_inst} instances "
          f"across {len({r['filament_id'].rsplit('_', 1)[0] for r in rows})} images "
          f"({n_inst / max(1, len(stems)):.2f}/img)")
    print(f"rule: t_lo={args.t_lo} close={args.close} t_hi={args.t_hi} "
          f"min_area={args.min_area} min_meanp={args.min_meanp}")

    if args.dry_run:
        return
    out = args.out or os.path.join(config.SUBMISSION_DIR,
                                   f"submission_keep_{args.preset}_{args.path}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filament_id", "segmentation_rle"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
