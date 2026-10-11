"""
Vima 4, trexei MESA sto container universalfakedetect: metatrepei to
/work/fc_faceswap.npz se torch state_dict kai to vazei ekei pou to detector.py
fortonei to fc_weights.pth. Ta arxika varh kratiountai mia fora sto
/work/fc_weights_original.pth. To /work/fc_weights_faceswap.pth einai to
arxeio pou yparxei sto models/ tou repo.
"""

import os
import shutil

import numpy as np
import torch

P = "/app/model/universalfakedetect/pretrained_weights/fc_weights.pth"
z = np.load("/work/fc_faceswap.npz")
torch.save({"weight": torch.tensor(z["weight"]), "bias": torch.tensor(z["bias"])},
           "/work/fc_weights_faceswap.pth")
sd = torch.load("/work/fc_weights_faceswap.pth")
print({k: tuple(v.shape) for k, v in sd.items()}, "threshold", float(z["threshold"]))
if not os.path.exists("/work/fc_weights_original.pth"):
    shutil.copy(P, "/work/fc_weights_original.pth")
shutil.copy("/work/fc_weights_faceswap.pth", P)
print("installed", os.path.getsize(P), "bytes")
