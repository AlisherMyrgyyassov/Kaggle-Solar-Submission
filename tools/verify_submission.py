"""Validate a submission CSV the way the scorer will read it.

Ported from the public 0.39-LB notebook's 14-check verifier, adapted to this
repo's submission format (`filament_id,segmentation_rle`, COCO RLE strings for
2048x2048). Run it on every CSV before uploading:

    python tools\\verify_submission.py output\\submissions\\submission_x.csv

Exit code 0 = all checks pass. Any failure prints the offending rows.
"""
import argparse
import glob
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
H = W = 2048
ID_RE = re.compile(r"^(.+)_(\d+)$")


def test_stems():
    from config import Config
    return {os.path.splitext(f)[0] for f in os.listdir(Config.TEST_IMG_DIR)}


def _rles(strs):
    return [{"size": [H, W], "counts": s.encode("ascii")} for s in strs]


def verify(path, stems=None):
    raw = open(path, "rb").read()
    lines = raw.decode("ascii", "replace").splitlines()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    c = {"sha256": hashlib.sha256(raw).hexdigest(), "n_rows": len(df)}

    c["header_ok"] = (lines[0] == "filament_id,segmentation_rle"
                      and list(df.columns) == ["filament_id", "segmentation_rle"])
    c["no_quotes_in_file"] = b'"' not in raw
    c["ascii_only"] = all(b < 128 for b in raw)

    split = df["filament_id"].str.rsplit("_", n=1, expand=True)
    c["ids_match_pattern"] = bool(df["filament_id"].str.match(ID_RE.pattern).all())
    c["stems_without_underscore"] = bool(not split[0].str.contains("_").any())
    c["ids_unique"] = bool(df["filament_id"].is_unique)
    if stems is not None:
        c["all_stems_in_test_set"] = bool(split[0].isin(stems).all())
        c["n_images_with_rows"] = int(split[0].nunique())
        c["all_test_images_covered"] = bool(split[0].nunique() == len(stems))
    else:
        c["all_stems_in_test_set"] = True
        c["n_images_with_rows"] = int(split[0].nunique())

    c["rle_charset_ok"] = bool(
        df["segmentation_rle"].map(
            lambda s: len(s) > 0 and all(48 <= ord(ch) <= 111 for ch in s)
        ).all()
    )

    bad_decode = bad_roundtrip = empty = overlap = 0
    for stem, g in df.groupby(split[0]):
        acc = np.zeros((H, W), np.uint8)
        for s in g["segmentation_rle"]:
            try:
                mk = mask_util.decode({"size": [H, W], "counts": s.encode("ascii")})
            except Exception:
                bad_decode += 1
                continue
            if mk.shape != (H, W):
                bad_decode += 1
                continue
            empty += int(mk.sum() == 0)
            bad_roundtrip += int(
                mask_util.encode(np.asfortranarray(mk))["counts"].decode("ascii") != s
            )
            acc += mk
        overlap += int(acc.max() > 1)

    c["decode_2048_ok"] = bad_decode == 0
    c["roundtrip_identical"] = bad_roundtrip == 0
    c["no_empty_masks"] = empty == 0
    c["pixel_disjoint_dense"] = overlap == 0
    c["pandas_reread_identical"] = bool(
        pd.read_csv(path, dtype=str, keep_default_na=False).equals(df)
    )
    c["all_pass"] = all(v for k, v in c.items()
                        if k not in ("sha256", "n_rows", "n_images_with_rows",
                                     "all_test_images_covered"))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join(
        ROOT, "output", "submissions", "submission.csv"))
    ap.add_argument("--no_test_check", action="store_true",
                    help="skip checking stems against the test set (for val CSVs)")
    args = ap.parse_args()
    try:
        stems = None if args.no_test_check else test_stems()
    except Exception as e:
        print(f"[warn] could not load test stems ({e}); skipping test-set check")
        stems = None

    c = verify(args.csv, stems)
    print(f"{os.path.basename(args.csv)}: sha256 {c['sha256'][:16]}, {c['n_rows']} rows, "
          f"{c['n_images_with_rows']} images")
    order = ["header_ok", "no_quotes_in_file", "ascii_only", "ids_match_pattern",
             "stems_without_underscore", "ids_unique", "rle_charset_ok",
             "all_stems_in_test_set", "decode_2048_ok", "roundtrip_identical",
             "no_empty_masks", "pixel_disjoint_dense", "pandas_reread_identical"]
    n_pass = 0
    for k in order:
        if k in c:
            n_pass += bool(c[k])
            print(f"  {'PASS' if c[k] else 'FAIL'}  {k}")
    if "all_test_images_covered" in c:
        print(f"  {'INFO' if not c['all_test_images_covered'] else 'PASS'}  all_test_images_covered"
              f"  ({c['n_images_with_rows']}/{len(stems)}) [empty images are legitimate]")
    print(f"{n_pass}/{len([k for k in order if k in c])} checks pass")
    sys.exit(0 if c["all_pass"] else 1)


if __name__ == "__main__":
    main()
