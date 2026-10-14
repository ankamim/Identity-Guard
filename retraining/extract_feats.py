"""
Vima 2, trexei MESA sto container deepsafe-universalfakedetect.

Xrisi:
    python extract_feats.py /work/crops /work/feats.npz

Idio transform me to detector.py (CenterCrop 224 + kanonikopoiisi CLIP).
Apothikeyei ta 768-d CLIP features prin to teleytaio grammiko epipedo, mazi me
ta arxika varh tou fc. Synexizei apo ekei pou stamatise: ta features
apothikeyontai sto feats_partial.npz ana 200 eikones.
"""

import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("/app/model/universalfakedetect")
from models import get_model

root, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(os.cpu_count())
tf = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])
net = get_model("CLIP:ViT-L/14")
net.fc.load_state_dict(torch.load(
    "/app/model/universalfakedetect/pretrained_weights/fc_weights.pth", map_location="cpu"))
net.eval()

rows = [r for r in csv.DictReader(open(os.path.join(root, "index.csv"), encoding="utf-8")) if r["crop"]]
partial = out.replace(".npz", "_partial.npz")
cache = {}
if os.path.exists(partial):
    z = np.load(partial, allow_pickle=True)
    cache = dict(zip(z["crop"], z["feat"]))


def save(path):
    keys = list(cache)
    np.savez(path, crop=np.array(keys), feat=np.stack([cache[k] for k in keys]))


todo = [r for r in rows if r["crop"] not in cache]
print(f"{len(rows)} crops, {len(todo)} to do", flush=True)
t0 = time.time()
B = 16
with torch.no_grad():
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        x = torch.stack([tf(Image.open(os.path.join(root, r["crop"])).convert("RGB")) for r in chunk])
        f = net(x, return_feature=True).float().numpy()
        for r, v in zip(chunk, f):
            cache[r["crop"]] = v
        if (i // B) % 12 == 0:
            save(partial)
            done = i + len(chunk)
            print(f"{done}/{len(todo)}  {(time.time() - t0) / done:.2f}s/img", flush=True)
save(partial)

meta = {r["crop"]: r for r in rows}
keys = [r["crop"] for r in rows]
np.savez(out,
         crop=np.array(keys),
         split=np.array([meta[k]["split"] for k in keys]),
         label=np.array([int(meta[k]["label"]) for k in keys]),
         feat=np.stack([cache[k] for k in keys]),
         fc_w=net.fc.weight.detach().numpy(), fc_b=net.fc.bias.detach().numpy())
print("saved", out)
