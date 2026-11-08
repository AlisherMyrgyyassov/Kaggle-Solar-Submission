"""Cache validation probability maps for one inference path.

    python cache_val_probs.py --tags convnext_tiny_1024,convnext_small_1024 \
        --pt full2048_tta8 --tta 8 --half

Writes output/val_probs/<tag>_file_<pt>/<stem>.npy (float16, 2048x2048).
`--pt raw` is the original 8x1024 patch path (already cached for the old
models); every other value uses single-pass whole-image inference, optionally
with D4 TTA. Keeping the path in the directory name is what lets keep_sweep.py
tune the KEEP rule on exactly the probabilities a submission will use.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from dataset import create_file_splits, load_and_parse_annotations
from predict import predict_prob_map, predict_prob_map_full
from presets import MODEL_SPECS, PROB_DIR, load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma list of presets.MODEL_SPECS keys")
    ap.add_argument("--pt", default="full2048", help="probability-cache path tag")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0,
                   help="resize factor before the forward (a multi-scale view)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config = Config
    config.IN_CHANNELS = 3
    config.NORMALIZE = True
    config.BATCH_SIZE = 8
    config.HALF = bool(args.half)

    data = load_and_parse_annotations(config.ANNOTATION_FILE)
    _, val_files = create_file_splits(data, config.VAL_SPLIT, config.SEED)
    if args.limit:
        val_files = val_files[:args.limit]

    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        if tag not in MODEL_SPECS:
            raise SystemExit(f"unknown tag {tag}; known: {sorted(MODEL_SPECS)}")
        spec = MODEL_SPECS[tag]
        config.CROP_SIZE = spec["crop"]
        config.MODEL_INPUT_SIZE = spec["model_size"]
        out_dir = os.path.join(config.OUTPUT_DIR, "val_probs",
                               PROB_DIR[tag].format(pt=args.pt))
        os.makedirs(out_dir, exist_ok=True)
        todo = [f for f in val_files
                if not os.path.exists(os.path.join(out_dir, os.path.splitext(f)[0] + ".npy"))]
        if not todo:
            print(f"{tag}: cached ({args.pt})")
            continue
        model, _, ck = load_checkpoint(tag, config.MODEL_DIR, config.DEVICE, half=args.half)
        print(f"{tag}: path={args.pt} tta={args.tta} half={args.half} "
              f"epoch={ck.get('epoch')} best_pq={ck.get('best_pq', float('nan')):.4f} "
              f"-> {len(todo)} files")
        for f in tqdm(todo, desc=tag):
            img = cv2.imread(os.path.join(config.TRAIN_IMG_DIR, f), cv2.IMREAD_GRAYSCALE)
            if args.pt == "raw":
                prob = predict_prob_map(img, model, config, tta=args.tta > 1)
            else:
                prob = predict_prob_map_full(img, model, config, tta=args.tta, scale=args.scale)
            np.save(os.path.join(out_dir, os.path.splitext(f)[0] + ".npy"),
                    prob.astype(np.float16))
        del model
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
