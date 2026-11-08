"""Fast PQ evaluation using cached sliding-window probs + numpy bbox matching.

Loads cached probability maps (output/val_probs/{raw,ema}/*.npy), builds GT
instance label maps once, then sweeps threshold x min_area x close_kernel with
a fast numpy matcher (bbox overlap pre-filter) instead of pycocotools RLE IoU.
"""
import os
import csv
import json

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from config import Config
from dataset import (
    create_data_splits,
    load_and_parse_annotations,
    rasterize_instance_rles,
)
from preprocessing import label_instances

IOU_MATCH = 0.5


def unique_by_filename(annotation_data, image_ids):
    id_to_name = {img["id"]: img["file_name"] for img in annotation_data["images"]}
    seen = set()
    unique_ids, unique_files = [], []
    for img_id in image_ids:
        fname = id_to_name.get(img_id)
        if fname is None or fname in seen:
            continue
        seen.add(fname)
        unique_ids.append(img_id)
        unique_files.append(fname)
    return unique_ids, unique_files


def build_gt_maps(annotation_data, image_ids):
    info = {img["id"]: img for img in annotation_data["images"]}
    anns_by = {}
    for ann in annotation_data["annotations"]:
        anns_by.setdefault(ann["image_id"], []).append(ann)

    gt_maps = {}
    n_inst = 0
    for img_id in image_ids:
        img = info[img_id]
        stem = os.path.splitext(img["file_name"])[0]
        rles = rasterize_instance_rles(anns_by.get(img_id, []), img["height"], img["width"])
        h, w = img["height"], img["width"]
        gt_map = np.zeros((h, w), dtype=np.int32)
        areas = []
        for g, rle in enumerate(rles, start=1):
            m = mask_utils.decode(rle).astype(np.uint8)
            gt_map[m > 0] = g
            areas.append(int(m.sum()))
        gt_maps[stem] = (gt_map, areas)
        n_inst += len(rles)
    print(f"GT instances on val: {n_inst} across {len(gt_maps)} images")
    return gt_maps


def match_fast(gt_map, gt_areas, labels, stats, min_area):
    n_gt = len(gt_areas)
    pairs = []
    pred_areas = {}
    for lid in range(1, stats.shape[0]):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        pred_areas[lid] = area
        x, y = int(stats[lid, cv2.CC_STAT_LEFT]), int(stats[lid, cv2.CC_STAT_TOP])
        w, h = int(stats[lid, cv2.CC_STAT_WIDTH]), int(stats[lid, cv2.CC_STAT_HEIGHT])
        pred_mask = (labels[y:y + h, x:x + w] == lid)
        gt_roi = gt_map[y:y + h, x:x + w]
        cands = np.unique(gt_roi[pred_mask])
        cands = cands[cands > 0]
        for g in cands:
            gt_mask = (gt_roi == g)
            inter = int((pred_mask & gt_mask).sum())
            if inter == 0:
                continue
            iou = inter / (area + gt_areas[g - 1] - inter)
            if iou > IOU_MATCH:
                pairs.append((iou, lid, int(g)))

    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    iou_sum, tp = 0.0, 0
    for iou, lid, g in pairs:
        if lid in used_p or g in used_g:
            continue
        used_p.add(lid)
        used_g.add(g)
        iou_sum += iou
        tp += 1
    fp = len(pred_areas) - len(used_p)
    fn = n_gt - len(used_g)
    return iou_sum, tp, fp, fn


def pq_from_counts(iou_sum, tp, fp, fn):
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = (iou_sum / denom) if denom > 0 else 0.0
    sq = (iou_sum / tp) if tp > 0 else 0.0
    rq = (tp / denom) if denom > 0 else 0.0
    return pq, sq, rq


def run():
    config = Config
    data = load_and_parse_annotations(config.ANNOTATION_FILE)
    _, val_ids = create_data_splits(data, config.VAL_SPLIT, config.SEED)
    val_ids, val_files = unique_by_filename(data, val_ids)
    stems = [os.path.splitext(f)[0] for f in val_files]
    print(f"Val files: {len(stems)}")

    gt_maps = build_gt_maps(data, val_ids)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    min_areas = [200, 400, 600, 800, 1000, 1200, 1600]
    close_kernels = [0, 3, 5]
    checkpoints = ["raw", "ema"]

    rows = []
    for which in checkpoints:
        cache_dir = os.path.join(config.OUTPUT_DIR, "val_probs", which)
        if not os.path.isdir(cache_dir):
            print(f"Missing cache {cache_dir}, skip")
            continue
        for t in thresholds:
            for ck in close_kernels:
                labels_cache = {}
                for stem in stems:
                    prob = np.load(os.path.join(cache_dir, f"{stem}.npy")).astype(np.float32)
                    labels, stats = label_instances(prob, t, ck)
                    labels_cache[stem] = (labels, stats)
                for ma in min_areas:
                    iou_sum = tp = fp = fn = 0
                    for stem in stems:
                        gt_map, gt_areas = gt_maps[stem]
                        labels, stats = labels_cache[stem]
                        i, tt, fpp, fnn = match_fast(gt_map, gt_areas, labels, stats, ma)
                        iou_sum += i
                        tp += tt
                        fp += fpp
                        fn += fnn
                    pq, sq, rq = pq_from_counts(iou_sum, tp, fp, fn)
                    rows.append({
                        "checkpoint": which, "threshold": t, "min_area": ma,
                        "close_kernel": ck, "pq": pq, "sq": sq, "rq": rq,
                        "tp": tp, "fp": fp, "fn": fn, "iou_sum": iou_sum,
                    })
            print(f"{which} done thresholds up to {t}")

    rows.sort(key=lambda r: r["pq"], reverse=True)
    out = os.path.join(config.LOG_DIR, "pq_grid_v2.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    print(f"{'ckpt':<6} {'t':>6} {'area':>6} {'close':>5} {'PQ':>8} {'SQ':>8} {'RQ':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    for r in rows[:25]:
        print(f"{r['checkpoint']:<6} {r['threshold']:6.2f} {r['min_area']:6d} {r['close_kernel']:5d} "
              f"{r['pq']:8.4f} {r['sq']:8.4f} {r['rq']:8.4f} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}")

    with open(os.path.join(config.LOG_DIR, "pq_best_v2.json"), "w") as f:
        json.dump(rows[0], f, indent=2)
    print("\nBest:", rows[0])


if __name__ == "__main__":
    run()
