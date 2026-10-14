"""
Vima 3: ekpaideyei neo grammiko epipedo pano sta CLIP features.

Xrisi:
    python retraining/train_fc.py feats.npz fc_faceswap.npz

Logistic regression (numpy, L2) sto train split. To L2 kai to threshold
dialegontai MONO sto val (ta perissotera fake me val flagged <= 8%, perithorio
kato apo to 10%). To test kai oi dikes mas eikones metrioyntai mia fora sto
telos, dipla sto arxiko fc. To neo epipedo doulevei sta raw features, ara
mpainei opos einai sto fc_weights.pth.
"""

import sys

import numpy as np

z = np.load(sys.argv[1])
X, y, split = z["feat"].astype(np.float64), z["label"], z["split"]
S = {s: split == s for s in ("train", "val", "test", "own")}
print({s: int(m.sum()) for s, m in S.items()})


def fit(Xt, yt, lam, steps=3000, lr=0.5):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Z = (Xt - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - yt
        w -= lr * (Z.T @ g / len(yt) + lam * w)
        b -= lr * g.mean()
    # h kanonikopoiisi mpainei mesa sta varh, gia na doulevoun sta raw features
    return w / sd, b - (w / sd) @ mu


def prob(w, b, m):
    return 1 / (1 + np.exp(-(X[m] @ w + b)))


def auc(p, t):
    o = np.argsort(-p)
    return np.trapezoid(np.r_[0, np.cumsum(t[o]) / t.sum()],
                        np.r_[0, np.cumsum(1 - t[o]) / (1 - t).sum()])


def rates(p, t, th):
    pred = p >= th
    return (pred == t).mean(), pred[t == 1].mean(), pred[t == 0].mean()


def pick_th(p, t, max_fpr=0.08):
    best = None
    for th in np.unique(p):
        a, rec, fpr = rates(p, t, th)
        if fpr <= max_fpr and (best is None or rec > best[1]):
            best = (th, rec)
    return best[0]


ffhq = np.char.find(z["crop"].astype(str), "ffhq_") >= 0
S["test_dff"] = S["test"] & ~ffhq
S["test_ffhq"] = S["test"] & ffhq


def report(name, w, b, th):
    for s in ("val", "test", "test_dff", "test_ffhq", "own"):
        if not S[s].any() or len(np.unique(y[S[s]])) < 2:
            if S[s].any():
                p = prob(w, b, S[s])
                print(f"  {name:10} {s:9} n={S[s].sum():4} real only: flagged="
                      f"{(p >= th).mean():.1%} ({int((p >= th).sum())}/{S[s].sum()})")
            continue
        p, t = prob(w, b, S[s]), y[S[s]]
        a, rec, fpr = rates(p, t, th)
        print(f"  {name:10} {s:9} n={len(t):4} AUC={auc(p, t):.3f} @ {th:.3f}: "
              f"acc={a:.1%} caught={rec:.1%} ({int((p >= th)[t == 1].sum())}/{int(t.sum())}) "
              f"flagged={fpr:.1%} ({int((p >= th)[t == 0].sum())}/{int((t == 0).sum())})")


print({s: int(m.sum()) for s, m in S.items()})
w0, b0 = z["fc_w"][0].astype(np.float64), float(z["fc_b"][0])
print("original fc, app threshold 0.06 and val-picked threshold")
report("orig", w0, b0, 0.06)
report("orig", w0, b0, pick_th(prob(w0, b0, S["val"]), y[S["val"]]))

best = None
for lam in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
    w, b = fit(X[S["train"]], y[S["train"]], lam)
    pv = prob(w, b, S["val"])
    th = pick_th(pv, y[S["val"]])
    rec = rates(pv, y[S["val"]], th)[1]
    print(f"lam={lam:g} val AUC={auc(pv, y[S['val']]):.3f} caught@<=8%={rec:.1%} th={th:.3f}")
    if best is None or rec > best[0]:
        best = (rec, lam, w, b, th)

_, lam, w, b, th = best
print(f"chosen lam={lam:g} th={th:.3f}")
report("new", w, b, th)
np.savez(sys.argv[2], weight=w[None, :].astype(np.float32), bias=np.array([b], np.float32), threshold=th)
