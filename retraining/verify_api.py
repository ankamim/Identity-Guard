"""
Vima 5: elegxei ta egkatestimena varh mesa apo to pragmatiko /predict tou
DeepSafe, me to deepfake_client tis efarmogis. Sygkrinei tis pithanotites tou
API me aytes tou train_fc.py.

Xrisi (apo ton fakelo tou repo):
    python retraining/verify_api.py crops_dir feats.npz fc_faceswap.npz 0.24
"""

import asyncio
import os
import random
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

from app import config, deepfake_client

root, feats, fc, th = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
z, n = np.load(feats), np.load(fc)
off = dict(zip(z["crop"], 1 / (1 + np.exp(-(z["feat"].astype(float) @ n["weight"][0] + n["bias"][0])))))
lab = dict(zip(z["crop"], z["label"]))
own = sorted(k for k, s in zip(z["crop"], z["split"]) if s == "own")
test = [k for k, s in zip(z["crop"], z["split"]) if s == "test"]
random.Random(3).shuffle(test)
keys = own + test[:100]


async def main():
    config.DEEPSAFE_THRESHOLD = th
    res = await deepfake_client.analyse_batch({k: open(os.path.join(root, k), "rb").read() for k in keys})
    diffs, ok = [], 0
    for k in keys:
        p = res[k]["ensemble_score_is_fake"]
        diffs.append(abs(p - off[k]))
        right = (p >= th) == bool(lab[k])
        ok += right
        if k in own:
            print(f"{k:22} {'FAKE' if lab[k] else 'real'} api={p:.3f} offline={off[k]:.3f} "
                  f"verdict={res[k]['verdict']} {'OK' if right else 'WRONG'}")
    print(f"max |api-offline| = {max(diffs):.4f}, correct {ok}/{len(keys)}")


asyncio.run(main())
