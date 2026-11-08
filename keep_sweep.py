"""Tune and compare the decoupled KEEP rule with nested CV on the val split.

    python keep_sweep.py --presets ens4_old,ens4_oldfull,ens4_full,ens6 --pt raw
    python keep_sweep.py --presets atto_dec128 --pt full2048_tta8

For every preset it
  1. mixes the member probability maps,
  2. precomputes 8-connected components at each (t_lo, close),
  3. scores a grid of (t_hi, min_area, min_meanp) acceptance rules by pooled
     exact multi-annotator PQ,
  4. reports **nested** 5-fold OOF PQ (rule chosen on 4 folds, scored on the 5th)
     plus the rule that wins on the full val split, and
  5. paired file-level bootstrap between presets (OOF stats).

Nested CV is what makes a preset comparison honest: tuning the rule on the same
107 files it is scored on inflates every preset, and by different amounts.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import datetime
import hashlib
import json

import cv2
import numpy as np
from sklearn.model_selection import KFold

from config import Config
from dataset import create_file_splits, load_and_parse_annotations
from eval_pq_v2 import build_multi_gt
from presets import PRESETS, PROB_DIR
from preprocessing import component_stats, keep_mask_from_stats

VP = os.path.join(Config.OUTPUT_DIR, "val_probs")
FIXED = (0.35, 3, 0.0, 500, 0.0)          # the old coupled rule, for reference
SPARSE_R = 0.77                           # r fitted by the 0.39-LB notebook

T_LO = [0.25, 0.30, 0.35, 0.40]
CLOSE = [0, 3]
T_HI = [0.0, 0.90, 0.95, 0.97, 0.98, 0.99]
MIN_AREA = [100, 150, 200, 300, 400, 500]
MIN_MEANP = [0.0, 0.65, 0.70, 0.75, 0.80, 0.85]


def pq_from(summary):
    s_iou, tp, fp, fn = summary
    denom = tp + 0.5 * fp + 0.5 * fn
    return (s_iou / denom) if denom > 0 else 0.0


def pq_sparse(summary, r=SPARSE_R):
    """Expected PQ if each labelled filament survives to the test set with prob r."""
    s_iou, tp, fp, fn = summary
    denom = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return (r * s_iou / denom) if denom > 0 else 0.0


def mix_prob(preset, pt, stem):
    members = PRESETS[preset]
    wsum = sum(w for _, w in members)
    acc = None
    for tag, w in members:
        d = os.path.join(VP, PROB_DIR[tag].format(pt=pt))
        p = np.load(os.path.join(d, stem + ".npy")).astype(np.float32) * (w / wsum)
        acc = p if acc is None else acc + p
    return acc


def precompute(preset, pt, stems, multi_gt):
    """{stem: {t_lo: {close: (area, meanp, pmax, [(n_gt, hcp, iousp, hit), ...])}}}"""
    cache = {}
    for stem in stems:
        prob = mix_prob(preset, pt, stem)
        cache[stem] = {}
        for t_lo in T_LO:
            cache[stem][t_lo] = {}
            for close in CLOSE:
                labels, stats, area, meanp, pmax = component_stats(prob, t_lo, close)
                from pycocotools import mask as mask_utils
                from preprocessing import kept_area_rles
                all_rles = (kept_area_rles(labels, stats, np.ones(area.size, bool), min_area=0)
                            if area.size else [])
                annots = []
                for gt_rles in multi_gt[stem]:
                    n_gt = len(gt_rles)
                    if not all_rles or n_gt == 0:
                        annots.append((n_gt, None, None, None))
                        continue
                    ious = mask_utils.iou(all_rles, gt_rles, [0] * n_gt).astype(np.float64)
                    hit = ious > 0.5
                    hcp = hit.sum(axis=1)
                    iousp = np.where(hit, ious, 0.0).sum(axis=1)
                    annots.append((n_gt, hcp, iousp, hit))
                cache[stem][t_lo][close] = (area, meanp, pmax, annots)
        # free the probability map early; only component stats are kept
        del prob
    return cache


def score_rule(cache, stems, rule):
    t_lo, close, t_hi, min_area, min_meanp = rule
    s_iou = tp = fp = fn = 0.0
    for stem in stems:
        area, meanp, pmax, annots = cache[stem][t_lo][close]
        n_p = 0
        if area.size:
            keep = keep_mask_from_stats(area, meanp, pmax, t_hi, min_area, min_meanp)
            idx = np.flatnonzero(keep)
            n_p = int(idx.size)
        for n_gt, hcp, iousp, hit in annots:
            if n_gt == 0:
                fp += n_p
                continue
            if n_p == 0:
                fn += n_gt
                continue
            s_iou += float(iousp[idx].sum())
            tp += float(hcp[idx].sum())
            fp += float((hcp[idx] == 0).sum())
            fn += float((hit[idx].sum(axis=0) == 0).sum())
    return s_iou, float(tp), float(fp), float(fn)


def nested_cv(cache, stems, n_splits=5, seed=42):
    rules = [(a, b, c, d, e) for a in T_LO for b in CLOSE for c in T_HI
             for d in MIN_AREA for e in MIN_MEANP]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_stem = {s: np.zeros(4) for s in stems}
    picked = []
    for tr, va in kf.split(stems):
        tr_stems = [stems[i] for i in tr]
        va_stems = [stems[i] for i in va]
        best, best_pq = None, -1.0
        for rule in rules:
            pq = pq_from(score_rule(cache, tr_stems, rule))
            if pq > best_pq:
                best_pq, best = pq, rule
        picked.append(best)
        for stem in va_stems:
            per_stem[stem] = np.array(score_rule(cache, [stem], best))
    return per_stem, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="ens4_old,ens4_oldfull,ens4_full,ens6")
    ap.add_argument("--pt", default="raw")
    ap.add_argument("--nested", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting an existing --out file (default: refuse).")
    args = ap.parse_args()

    out = args.out or os.path.join(Config.LOG_DIR, f"keep_rules_{args.pt}.json")
    if os.path.exists(out) and not args.force:
        raise SystemExit(
            f"refusing to overwrite {out}; pass a distinct --out (recommended) or --force")

    def _sha1(path):
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    data = load_and_parse_annotations(Config.ANNOTATION_FILE)
    _, val_files = create_file_splits(data, Config.VAL_SPLIT, Config.SEED)
    stems = [os.path.splitext(f)[0] for f in val_files]
    multi_gt = build_multi_gt(data, val_files)

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    results = {
        "path_tag": args.pt,
        "nested_folds": args.nested,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gt": {
            "annotation_file": os.path.basename(Config.ANNOTATION_FILE),
            "size_bytes": os.path.getsize(Config.ANNOTATION_FILE),
            "sha1": _sha1(Config.ANNOTATION_FILE),
        },
        "presets": {},
    }
    oof = {}

    for preset in presets:
        print(f"\n=== {preset} ({args.pt}) ===")
        missing = []
        for tag, _w in PRESETS[preset]:
            d = os.path.join(VP, PROB_DIR[tag].format(pt=args.pt))
            missing += [s for s in stems if not os.path.exists(os.path.join(d, s + ".npy"))]
        if missing:
            print(f"  SKIP: {len(missing)} missing prob maps (e.g. {missing[0]})")
            continue
        cache = precompute(preset, args.pt, stems, multi_gt)

        ref_summ = score_rule(cache, stems, FIXED)
        ref = pq_from(ref_summ)
        print(f"  old coupled rule {FIXED}: PQ={ref:.4f} (PQ'={pq_sparse(ref_summ):.4f}) "
              f"TP={int(ref_summ[1])} FP={int(ref_summ[2])} FN={int(ref_summ[3])}")

        per_stem, picked = nested_cv(cache, stems, args.nested)
        oof[preset] = per_stem
        nested_pq = pq_from(np.array([per_stem[s] for s in stems]).sum(axis=0))
        nested_q = pq_sparse(np.array([per_stem[s] for s in stems]).sum(axis=0))

        best_rule, best_pq = None, -1.0
        for rule in [(a, b, c, d, e) for a in T_LO for b in CLOSE for c in T_HI
                     for d in MIN_AREA for e in MIN_MEANP]:
            pq = pq_from(score_rule(cache, stems, rule))
            if pq > best_pq:
                best_pq, best_rule = pq, rule
        summ = score_rule(cache, stems, best_rule)

        print(f"  nested OOF PQ          : {nested_pq:.4f} (PQ'={nested_q:.4f})")
        print(f"  full-val best rule     : PQ={best_pq:.4f} (PQ'={pq_sparse(summ):.4f}) "
              f"@ t_lo={best_rule[0]} close={best_rule[1]} t_hi={best_rule[2]} "
              f"ma={best_rule[3]} mp={best_rule[4]}")
        print(f"    TP={int(summ[1])} FP={int(summ[2])} FN={int(summ[3])}")
        print(f"  rules picked per fold  : {picked}")
        results["presets"][preset] = {
            "old_rule_pq": ref,
            "old_rule": dict(
                t_lo=FIXED[0], close=FIXED[1], t_hi=FIXED[2],
                min_area=FIXED[3], min_meanp=FIXED[4],
                counts=dict(sum_iou=ref_summ[0], tp=ref_summ[1],
                            fp=ref_summ[2], fn=ref_summ[3]),
            ),
            "nested_oof_pq": nested_pq,
            "nested_oof_pq_sparse": nested_q,
            "best_rule": dict(t_lo=best_rule[0], close=best_rule[1], t_hi=best_rule[2],
                              min_area=best_rule[3], min_meanp=best_rule[4]),
            "best_rule_fullval_pq": best_pq,
            "counts": dict(sum_iou=summ[0], tp=summ[1], fp=summ[2], fn=summ[3]),
            "fold_rules": [dict(t_lo=r[0], close=r[1], t_hi=r[2], min_area=r[3],
                                min_meanp=r[4]) for r in picked],
        }
        del cache

    # paired bootstrap between presets on the nested-OOF per-file stats
    names = [p for p in presets if p in oof]
    if len(names) > 1:
        base = names[0]
        rng = np.random.default_rng(11)
        print(f"\n=== paired bootstrap vs {base} (nested OOF, 4000 resamples) ===")
        for name in names[1:]:
            A = np.array([oof[base][s] for s in stems])
            B = np.array([oof[name][s] for s in stems])
            d = np.empty(4000)
            for i in range(4000):
                pick = rng.choice(len(stems), len(stems), replace=True)
                d[i] = pq_from(B[pick].sum(axis=0)) - pq_from(A[pick].sum(axis=0))
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = bool(lo > 0 or hi < 0)
            print(f"  {name:16s} - {base:16s} = {pq_from(B.sum(axis=0)) - pq_from(A.sum(axis=0)):+.4f} "
                  f"CI [{lo:+.4f}, {hi:+.4f}] {'SIG' if sig else ''}")
            results["presets"][name]["delta_vs_" + base] = {
                "delta": float(pq_from(B.sum(axis=0)) - pq_from(A.sum(axis=0))),
                "ci_lo": float(lo), "ci_hi": float(hi), "sig": sig,
            }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
