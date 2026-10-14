"""
Vima 1 tis epanekpaideysis: kovei ta prosopa gia to neo teleytaio epipedo tou
universalfakedetect.

Xrisi (apo ton fakelo tou repo):
    python retraining/make_crops.py crops_dir dff_dir own_dir [--train 800 --val 300 --test 400]

To DeepFakeFace xorizetai ana fakelo, wste kanena prosopo na min einai se dyo
splits: 00-69 train, 70-84 val, 85-99 test. To own_dir einai oi dikes mas
eikones (me_* pragmatikes, df_* face swaps), split "own". To kopsimo einai
idio me tis efarmogis (30% perithorio, PNG 256px), dhladh oti stelnei to
pipeline.py sto DeepSafe.
Grafei crops_dir/<split>/<label>/<onoma>.png kai crops_dir/index.csv.
"""

import argparse
import csv
import io
import os
import random
import sys

sys.path.insert(0, os.getcwd())
from PIL import Image

from app import config, face_matcher

SPLITS = {"train": range(0, 70), "val": range(70, 85), "test": range(85, 100)}


def crop(data):
    faces = face_matcher._extract_faces(data)
    if not faces:
        return None
    area = faces[0]["facial_area"]
    image = face_matcher._load_and_resize(data)
    side = max(area["w"], area["h"]) * (1 + config.FACE_CROP_MARGIN)
    cx, cy = area["x"] + area["w"] / 2, area["y"] + area["h"] / 2
    face = image.crop((int(cx - side / 2), int(cy - side / 2),
                       int(cx + side / 2), int(cy + side / 2)))
    face = face.resize((config.FACE_CROP_SIZE,) * 2, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    face.save(buf, format="PNG")
    return buf.getvalue()


def files_in(root, folders):
    out = []
    for f in folders:
        d = os.path.join(root, f"{f:02d}")
        if os.path.isdir(d):
            out += [os.path.join(d, n) for n in sorted(os.listdir(d))]
    return out


def main(a):
    jobs = []
    for split, folders in SPLITS.items():
        n = getattr(a, split)
        for label, sub, seed in ((0, "wiki", 101), (1, "insight", 202)):
            files = files_in(os.path.join(a.dff, sub), folders)
            random.Random(seed).shuffle(files)
            jobs += [(split, label, p) for p in files[:n]]
    for name in sorted(os.listdir(a.own)):
        if name.startswith(("me_", "df_")):
            jobs.append(("own", int(name.startswith("df_")), os.path.join(a.own, name)))

    index = os.path.join(a.out, "index.csv")
    done = set()
    if os.path.exists(index):
        done = {r["src"] for r in csv.DictReader(open(index, encoding="utf-8"))}
    fh = open(index, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, ["split", "label", "src", "crop"])
    if not done:
        w.writeheader()
    for i, (split, label, src) in enumerate(jobs):
        if src in done:
            continue
        data = crop(open(src, "rb").read())
        rel = ""
        if data:
            stem = os.path.splitext(os.path.relpath(src, a.dff if split != "own" else a.own))[0]
            rel = f"{split}/{label}/{stem.replace(os.sep, '_')}.png"
            path = os.path.join(a.out, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data)
        w.writerow({"split": split, "label": label, "src": src, "crop": rel})
        fh.flush()
        if i % 100 == 0:
            print(f"{i}/{len(jobs)}", flush=True)
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dff")
    ap.add_argument("own")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=400)
    main(ap.parse_args())
