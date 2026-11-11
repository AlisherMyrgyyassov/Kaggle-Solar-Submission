"""Exact competition metric: multi-annotator GT + all-pairs matching.

Replicates the official self-evaluation notebook's `get_pq_score` semantics:
  - GT is grouped per annotator-image ("<annotator>-<image>"), every annotator's
    mask scored separately against the same prediction set.
  - TP = ALL (gt, pred) pairs with IoU > 0.5 (not greedy one-to-one).
  - FP = predictions that hit NO gt; FN = gts hit by NO prediction.
  - PQ = sum(IoU of TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not (os.path.isfile(os.path.join(_ROOT, "config.py"))
           and os.path.isfile(os.path.join(_ROOT, "dataset.py"))):
    _parent = os.path.dirname(_ROOT)
    if _parent == _ROOT:
        raise RuntimeError("Repository root not found above " + os.path.abspath(__file__))
    _ROOT = _parent
sys.path.insert(0, str(_ROOT))
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from dataset import rasterize_instance_rles

IOU_THR = 0.5


def build_multi_gt(data, val_files):
    """Return {stem: [list_of_gt_rles_per_annotator, ...]} — one entry per annotator record."""
    byfile = defaultdict(list)
    for img in data["images"]:
        byfile[img["file_name"]].append(img)

    anns_by = defaultdict(list)
    for a in data["annotations"]:
        anns_by[a["image_id"]].append(a)

    multi = defaultdict(list)
    for f in val_files:
        stem = os.path.splitext(f)[0]
        for img in byfile.get(f, []):
            rles = rasterize_instance_rles(anns_by.get(img["id"], []), img["height"], img["width"])
            multi[stem].append(rles)
    return multi


def exact_pq(multi_gt, pred_rles_by_stem):
    """Pooled PQ over all annotator-entries. pred_rles_by_stem: {stem: [rle_dicts]}."""
    tp_iou = []
    fp = 0
    fn = 0
    for stem, annotator_gt in multi_gt.items():
        preds = pred_rles_by_stem.get(stem, [])
        for gt_rles in annotator_gt:
            n_gt = len(gt_rles)
            n_pred = len(preds)
            if n_gt == 0:
                fp += n_pred
                continue
            if n_pred == 0:
                fn += n_gt
                continue
            # mask_utils.iou(dt, gt) -> (n_pred, n_gt)
            ious = mask_utils.iou(preds, gt_rles, [0] * n_gt)
            hit = ious > IOU_THR
            tp_iou.extend(float(ious[j, i]) for j in range(n_pred) for i in range(n_gt) if hit[j, i])
            fp += int((hit.sum(axis=1) == 0).sum())   # preds hitting no GT
            fn += int((hit.sum(axis=0) == 0).sum())   # GTs hit by no pred
    tp = len(tp_iou)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = (sum(tp_iou) / denom) if denom > 0 else 0.0
    sq = (sum(tp_iou) / tp) if tp > 0 else 0.0
    rq = (tp / denom) if denom > 0 else 0.0
    return pq, sq, rq, tp, fp, fn


def labeled_area_rles(labels, stats, min_area):
    """Return [(area, rle), ...] for components with area >= min_area."""
    h, w = labels.shape
    buf = np.zeros((h, w), dtype=np.uint8, order="F")
    out = []
    for lid in range(1, stats.shape[0]):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lid, cv2.CC_STAT_LEFT])
        y = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        buf[y:y + bh, x:x + bw] = (labels[y:y + bh, x:x + bw] == lid)
        out.append((area, mask_utils.encode(buf)))
        buf[y:y + bh, x:x + bw] = 0
    return out


def instances_to_rles(labels, stats, min_area):
    return [rle for _, rle in labeled_area_rles(labels, stats, min_area)]


def build_gt_maps(annotation_data, image_ids):
    """Per-image GT instance map (int32 label image) plus instance areas, keyed by stem.

    Used by training to score a validation pass without re-rasterising polygons.
    """
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
