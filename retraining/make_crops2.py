"""
Vima 1b: prosthetei pragmatika prosopa apo to FFHQ kai ypovathmismena
antigrafa ton train eikonon.

Xrisi (apo ton fakelo tou repo):
    python retraining/make_crops2.py crops_dir ffhq.parquet [--train 800 --val 200]

Grafei sto idio crops_dir/index.csv me to make_crops.py:
- Eikones FFHQ os pragmatikes (label 0): oi protes --train sto train, oi
  epomenes --val sto val, oi ypoloipes sto test. Idio kopsimo me tin efarmogi.
- Ena ypovathmismeno antigrafo gia tis mises train eikones (pragmatikes kai
  fake): resize x0.35-1, JPEG q35-90, kamia fora elafry blur. Split "train".
Xoris to FFHQ to neo montelo evgaze fake 4 apo tis 11 dikes mas pragmatikes.
"""

import argparse
import csv
import io
import os
import random
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import pyarrow.parquet as pq
from PIL import Image, ImageFilter

from make_crops import crop


def degrade(png, rnd):
    im = Image.open(io.BytesIO(png)).convert("RGB")
    s = rnd.uniform(0.35, 1.0)
    small = im.resize((max(32, int(256 * s)),) * 2, Image.Resampling.BILINEAR)
    im = small.resize((256, 256), Image.Resampling.BICUBIC)
    if rnd.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.0)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=rnd.randint(35, 90))
    out = io.BytesIO()
    Image.open(buf).save(out, "PNG")
    return out.getvalue()


def main(a):
    index = os.path.join(a.out, "index.csv")
    rows = list(csv.DictReader(open(index, encoding="utf-8")))
    done = {r["src"] for r in rows}
    fh = open(index, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, ["split", "label", "src", "crop"])

    def put(split, label, src, data, name):
        rel = ""
        if data:
            rel = f"{split}/{label}/{name}.png"
            path = os.path.join(a.out, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data)
        w.writerow({"split": split, "label": label, "src": src, "crop": rel})
        fh.flush()

    table = pq.read_table(a.parquet)
    col = [c for c in table.column_names if c.lower() in ("image", "img", "jpg", "png")][0]
    images = table.column(col).to_pylist()
    for i, img in enumerate(images):
        src = f"ffhq:{i}"
        if src in done:
            continue
        data = img["bytes"] if isinstance(img, dict) else img
        split = "train" if i < a.train else "val" if i < a.train + a.val else "test"
        put(split, 0, src, crop(data), f"ffhq_{i:05d}")
        if i % 100 == 0:
            print(f"ffhq {i}/{len(images)}", flush=True)

    rnd = random.Random(7)
    base = [r for r in csv.DictReader(open(index, encoding="utf-8"))
            if r["split"] == "train" and r["crop"] and not r["src"].startswith("aug:")]
    rnd.shuffle(base)
    for i, r in enumerate(base[:len(base) // 2]):
        src = "aug:" + r["src"]
        if src in done:
            continue
        data = degrade(open(os.path.join(a.out, r["crop"]), "rb").read(), rnd)
        name = "aug_" + os.path.splitext(os.path.basename(r["crop"]))[0]
        put("train", r["label"], src, data, name)
    fh.close()
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("parquet")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=200)
    main(ap.parse_args())
