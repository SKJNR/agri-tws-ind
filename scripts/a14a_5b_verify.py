"""Exp 5b: verify anchor-anomaly quantities against team's earlier measurements.
Team's numbers to reproduce:
 - D-hat (mean of 6 anchor anomaly fields TWS-mu): std = 0.899, 62% of anchor variance
 - D-hat LOO error (predict anchor from mean of other anchors): ~0.8421
 - anchor-pair correlations r(h): 0.87/.78/.71/.66/.62/.59/.57
 - corr(D-hat, mu) = -0.77; corr(D-hat, trend-extrapolated field) = 0.52
Also: does the anchor anomaly correlate with ANY train month (ramp test)?
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, load_test, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)

dtr = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(dtr)
F = field_matrix(dtr, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)
lat_f = cells["lat"].values[full]; lon_f = cells["lon"].values[full]
cidx_full = {c: i for i, c in enumerate(zip(lat_f, lon_f))}

dte = load_test(cols=("time", "lat", "lon", "TWS_t", "TWS_t_masked"))
dte["unmask"] = ~dte["TWS_t_masked"].astype(bool)
tm = dte.groupby("t_abs")["unmask"].agg(["mean", "sum"])
anchors = sorted(tm[tm["mean"] > 0.9].index.tolist())

# anchor anomaly fields (UNDETRENDED: TWS - mu), full-history cells
B = {}
for ta in anchors:
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    ii = np.array([cidx_full[(la, lo)] for la, lo in zip(sub["lat"], sub["lon"].values)
                   if (la, lo) in cidx_full])
    yv = np.array([tws for (la, lo), tws in zip(zip(sub["lat"], sub["lon"].values),
                                                sub["TWS_t"].values) if (la, lo) in cidx_full])
    B[ta] = (ii, yv - mu[ii])

print("=== anchor anomaly (TWS - mu) stats, full-history cells ===")
allB = np.column_stack([np.full(C, np.nan) for _ in anchors])
for j, ta in enumerate(anchors):
    ii, b = B[ta]
    allB[ii, j] = b
    print(f"anchor {ta} (y={ta//12}, m={ta%12+1}): std={b.std():.4f} mean={b.mean():+.4f} n={len(ii)}")

# D-hat: mean over anchors of available cells
Dhat = np.nanmean(allB, axis=1)
ok = ~np.isnan(Dhat)
print(f"\nD-hat (mean of 6 anchor anomalies): std={Dhat[ok].std():.4f} (team: 0.899)")
# anchor variance explained by D-hat (pooled)
num = 0; den = 0
for j, ta in enumerate(anchors):
    m = ok & ~np.isnan(allB[:, j])
    num += ((allB[m, j] - Dhat[m]) ** 2).sum()
    den += (allB[m, j] ** 2).sum()
print(f"1 - resid/var (D-hat explains): {1 - num/den:.3f} (team: 0.62)")
print(f"corr(D-hat, mu) = {np.corrcoef(Dhat[ok], mu[ok])[0,1]:+.3f} (team: -0.77)")
trendex = beta * (24188 + 20 - t.mean())  # ~mid test era
print(f"corr(D-hat, trendex) = {np.corrcoef(Dhat[ok], trendex[ok])[0,1]:+.3f} (team: 0.52)")

# LOO D-hat error (undetrended)
print("\n=== D-hat LOO error (undetrended anchor anomaly) ===")
errs = []
for j, ta in enumerate(anchors):
    m = ~np.isnan(allB[:, j])
    others = np.nanmean(np.delete(allB, j, axis=1), axis=1)
    e = np.sqrt(np.nanmean((allB[m, j] - others[m]) ** 2))
    errs.append(e)
    print(f"anchor {ta}: LOO RMSE={e:.4f}  (target std {allB[m,j].std():.4f})")
print(f"mean LOO RMSE = {np.mean(errs):.4f} (team: 0.8421)")

# anchor-pair correlations
print("\n=== anchor-pair correlations (undetrended anomaly) ===")
for i in range(len(anchors)):
    for j in range(i + 1, len(anchors)):
        m = ~np.isnan(allB[:, i]) & ~np.isnan(allB[:, j])
        r = np.corrcoef(allB[m, i], allB[m, j])[0, 1]
        print(f"{anchors[i]} vs {anchors[j]} (gap {anchors[j]-anchors[i]:>2}): r={r:+.4f}")

# RAMP TEST: corr(anchor anomaly, train detrended anomaly at lag L back from 2015-08)
print("\n=== RAMP TEST: corr(anchor 2015-09, train anomaly field Xd[t]) ===")
ii, b = B[anchors[0]]
for back in [1, 2, 3, 6, 12, 24, 48, 96]:
    idx = T - 1 - back
    if idx < 0: break
    r = np.corrcoef(b, Xd[idx, ii])[0, 1]
    print(f"corr(anchor_2015-09, Xd[2015-{8-back if back<8 else '?'}] lag={back:>2}): {r:+.4f}")
# and consecutive train correlations for reference
print("\nreference: corr(Xd[-1], Xd[-1-back]) for back=1..6:",
      [f"{np.corrcoef(Xd[-1,ii], Xd[-1-back,ii])[0,1]:+.3f}" for back in range(1, 7)])

# fast-part check: remove D-hat from anchors, then corr of (anchor - Dhat) with train-end
print("\n=== after removing D-hat: corr(anchor_fast, Xd[-1]) ===")
for ta in anchors:
    ii, b = B[ta]
    fast = b - Dhat[ii]
    r = np.corrcoef(fast, Xd[-1, ii])[0, 1]
    print(f"anchor {ta} (gap {ta - months[-1]}): corr(fast_j, trainend)={r:+.4f} "
          f"fast_std={fast.std():.4f}")
# fast_j vs fast_k cross-corr (anchors, D-hat removed)
print("\nfast cross-corr after D-hat removal:")
for i in range(len(anchors)):
    for j in range(i + 1, len(anchors)):
        m = ~np.isnan(allB[:, i]) & ~np.isnan(allB[:, j])
        f1 = allB[m, i] - Dhat[m]; f2 = allB[m, j] - Dhat[m]
        print(f"{anchors[i]} vs {anchors[j]} (gap {anchors[j]-anchors[i]:>2}): "
              f"r={np.corrcoef(f1, f2)[0,1]:+.4f}")

# raw TWS distribution sanity: train vs test
print(f"\ntrain TWS std={X.std():.4f}; anchor TWS std={np.concatenate([b for _, b in B.values()]).std():.4f}")
print(f"train anomaly std={Xd.std():.4f}; anchor anomaly std={np.nanstd(allB):.4f}")
