"""Exp 6b: THE MONEY TEST — cov-field observation of the TWS state.

B2 in exp6 (multi-cov OLS reconstruction) failed on transfer — suspected
collinearity (SPEI_01/03/06/12 scores are MAs of each other) + regime shift.
Here:
1) per-mode SINGLE-cov reconstruction (1 parameter per mode, stable) and
   RIDGE multi-cov; honest 2002-2012 -> 2013-15 transfer; in-sample bound.
2) ANCHOR RECONSTRUCTION: predict each of the 6 test anchor anomaly fields
   (TWS - mu = D + fast) from the covariate deviation fields AT THAT MONTH,
   using per-mode loadings fit on train. K sweep. Compare vs D-hat LOO (0.837)
   and anchor anomaly std (~1.14). If covs observe D+fast, RMSE << 0.837.
3) per-mode r_k computed on train vs on the 6 anchors (field-level): transfer
   check of the loading structure.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, load_test, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)
COVS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]

df = load_train(cols=("time", "lat", "lon", "TWS_t") + tuple(COVS))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)
U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
V = Vt.T
S = U * s
Tm = V.shape[1]
lat_f = cells["lat"].values[full]; lon_f = cells["lon"].values[full]
cidx_full = {c: i for i, c in enumerate(zip(lat_f, lon_f))}

# cov deviation fields (train, full-history cells) - deviation from per-cell mean
Dv = {}
Fraw = {}
for c in COVS:
    Fc = field_matrix(df, c, cells, months)[:, full]
    Fraw[c] = Fc
    Dv[c] = Fc - Fc.mean(axis=0, keepdims=True)
Zs = {c: Dv[c] @ V for c in COVS}     # (T, Tm) cov scores

# ---------------- 1) honest reconstruction, single-cov + ridge ----------------
tr = months < 2013 * 12
te = ~tr
var_full_te = (Xd[te] ** 2).mean()
print("=== 1) reconstruction of detrended TWS anomaly from cov fields ===")
print(f"eval window 2013-15, target var={var_full_te:.4f} (std {np.sqrt(var_full_te):.4f})")
print(f"{'K':>4} " + " ".join(f"{c[:8]:>8}" for c in COVS) + f" {'ridge':>8} {'insample12':>10}")
for K in [10, 30, 50, 100, Tm]:
    row = [f"{K:>4}"]
    for c in COVS:
        b = (Zs[c][tr, :K] * S[tr, :K]).sum(0) / (Zs[c][tr, :K] ** 2).sum(0)
        pred = (Zs[c][te, :K] * b) @ V[:, :K].T
        r2 = 1 - ((Xd[te] - pred) ** 2).mean() / var_full_te
        row.append(f"{r2:>8.4f}")
    # ridge multi-cov per mode
    Ztr = np.column_stack([Zs[c][tr, :K] for c in COVS])
    Zte = np.column_stack([Zs[c][te, :K] for c in COVS])
    lam_r = 1e-3 * (Ztr ** 2).mean()
    W = np.linalg.solve(Ztr.T @ Ztr + lam_r * np.eye(5 * K), Ztr.T @ S[tr, :K])
    pred = (Zte @ W) @ V[:, :K].T
    r2r = 1 - ((Xd[te] - pred) ** 2).mean() / var_full_te
    row.append(f"{r2r:>8.4f}")
    # in-sample single best cov (SPEI_12) for reference
    b12 = (Zs["SPEI_12_t"][:, :K] * S[:, :K]).sum(0) / (Zs["SPEI_12_t"][:, :K] ** 2).sum(0)
    pred12 = (Zs["SPEI_12_t"][:, :K] * b12) @ V[:, :K].T
    r2i = 1 - ((Xd - pred12) ** 2).mean() / (Xd ** 2).mean()
    row.append(f"{r2i:>10.4f}")
    print(" ".join(row))

# ---------------- 2) ANCHOR RECONSTRUCTION ----------------
print("\n=== 2) reconstruct 6 test anchor fields from covs AT the anchor month ===")
dte = load_test(cols=("time", "lat", "lon", "TWS_t", "TWS_t_masked") + tuple(COVS))
dte["unmask"] = ~dte["TWS_t_masked"].astype(bool)
tms = dte.groupby("t_abs")["unmask"].mean()
anchors = sorted(tms[tms > 0.9].index.tolist())
# cov per-cell means (train) to build deviations at test months
cov_mean = {c: Fraw[c].mean(axis=0) for c in COVS}

anchor_data = []
for ta in anchors:
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    ii = np.array([cidx_full[(la, lo)] for la, lo in zip(sub["lat"].values, sub["lon"].values)
                   if (la, lo) in cidx_full])
    yv = np.array([v for (la, lo), v in zip(zip(sub["lat"].values, sub["lon"].values),
                                            sub["TWS_t"].values) if (la, lo) in cidx_full])
    anchor_data.append((ta, ii, yv - mu[ii]))   # anomaly vs train mu

# D-hat LOO reference
print("reference: D-hat LOO RMSE per anchor (undetrended):")
for j, (ta, ii, a_true) in enumerate(anchor_data):
    others = np.zeros(len(ii)); cnt = np.zeros(len(ii))
    for j2, (ta2, ii2, a2) in enumerate(anchor_data):
        if j2 == j: continue
        lut = dict(zip(ii2, a2))
        vals = np.array([lut.get(c, np.nan) for c in ii])
        m = ~np.isnan(vals)
        others[m] += vals[m]; cnt[m] += 1
    dh = others / np.maximum(cnt, 1)
    print(f"  anchor {ta}: Dhat_LOO={np.sqrt(np.nanmean((a_true-dh)**2)):.4f} std={a_true.std():.4f}")

# per-mode loadings fit on FULL train (all 138 months), single-cov
b_full = {c: (Zs[c] * S).sum(0) / (Zs[c] ** 2).sum(0) for c in COVS}
# ridge multi-cov full-train
K = Tm
Zall = np.column_stack([Zs[c] for c in COVS])
lam_r = 1e-3 * (Zall ** 2).mean()
Wfull = np.linalg.solve(Zall.T @ Zall + lam_r * np.eye(5 * Tm), Zall.T @ S)

print("\nreconstruction RMSE at anchors (target = anchor anomaly TWS-mu):")
print(f"{'anchor':>7} {'std':>6} " + " ".join(f"{c[:8]:>8}" for c in COVS) +
      f" {'ridge':>8} {'r12_K30':>8} {'r12_K50':>8}")
for j, (ta, ii, a_true) in enumerate(anchor_data):
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    # cov scores at anchor month
    zsc = {}
    for c in COVS:
        vals = np.full(C, np.nan)
        lut = dict(zip(zip(sub["lat"].values, sub["lon"].values), sub[c].values))
        for (la, lo), idx in zip(zip(lat_f, lon_f), range(C)):
            pass
        # faster: build via arrays
        key = {(la, lo) for la, lo in zip(sub["lat"].values, sub["lon"].values)}
        arr = sub[c].values
        lookup = dict(zip(zip(sub["lat"].values, sub["lon"].values), arr))
        v = np.array([lookup.get((la, lo), np.nan) for la, lo in zip(lat_f, lon_f)])
        v = v - cov_mean[c]
        zsc[c] = V.T @ np.nan_to_num(v)   # scores (Tm,)
    row = [f"{ta:>7}", f"{a_true.std():>6.3f}"]
    for c in COVS:
        pred = (zsc[c] * b_full[c]) @ V.T
        row.append(f"{np.sqrt(np.mean((pred[ii]-a_true)**2)):>8.4f}")
    zr = np.concatenate([zsc[c] for c in COVS])
    pred = (Wfull.T @ zr) @ V.T
    row.append(f"{np.sqrt(np.mean((pred[ii]-a_true)**2)):>8.4f}")
    for Kc in [30, 50]:
        zc = zsc["SPEI_12_t"].copy(); zc[Kc:] = 0
        b = b_full["SPEI_12_t"].copy(); b[Kc:] = 0
        pred = (zc * b) @ V.T
        row.append(f"{np.sqrt(np.mean((pred[ii]-a_true)**2)):>8.4f}")
    print(" ".join(row))

# combined cov prediction of anchor: weighted average of single-cov preds
print("\ncombined (equal-weight avg of 5 single-cov preds, K=Tm) per anchor:")
for j, (ta, ii, a_true) in enumerate(anchor_data):
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    lookup = {c: dict(zip(zip(sub["lat"].values, sub["lon"].values), sub[c].values)) for c in COVS}
    preds = []
    for c in COVS:
        v = np.array([lookup[c].get((la, lo), np.nan) for la, lo in zip(lat_f, lon_f)])
        v = np.nan_to_num(v - cov_mean[c])
        z = V.T @ v
        preds.append((z * b_full[c]) @ V.T)
    pavg = np.mean(preds, axis=0)
    print(f"  anchor {ta}: avg5 RMSE={np.sqrt(np.mean((pavg[ii]-a_true)**2)):.4f} "
          f"corr={np.corrcoef(pavg[ii], a_true)[0,1]:.4f}")

# ---------------- 3) transfer of per-mode r_k to test era ----------------
print("\n=== 3) per-mode corr at anchors: cov scores vs anchor-anomaly scores ===")
# pooled over 6 anchors: corr of (z_k, s_k^anchor) across the 6x-cell samples
print(f"{'mode':>4} {'train_r(S12)':>12} {'anchor_r(S12)':>13} {'anchor_r(SOIL)':>14}")
for k in [0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 19, 29, 49]:
    zz12, zzsoil, ss = [], [], []
    for ta, ii, a_true in anchor_data:
        sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
        lut12 = dict(zip(zip(sub["lat"].values, sub["lon"].values), sub["SPEI_12_t"].values))
        lutso = dict(zip(zip(sub["lat"].values, sub["lon"].values), sub["SOIL_MOISTURE_t"].values))
        for idx, val in zip(ii, a_true):
            la, lo = lat_f[idx], lon_f[idx]
            if (la, lo) in lut12:
                zz12.append(lut12[(la, lo)] - cov_mean["SPEI_12_t"][idx])
                zzsoil.append(lutso[(la, lo)] - cov_mean["SOIL_MOISTURE_t"][idx])
                ss.append(val)
    zz12 = np.array(zz12); zzsoil = np.array(zzsoil); ss = np.array(ss)
    # project onto mode k
    vk = V[:, k]
    z12k = zz12 @ vk[ii] / 1.0   # not exactly a projection score; approximate via cell values
    zk12 = (zz12 * vk[ii]); sk = (ss * vk[ii])
    r12 = np.corrcoef(zk12, sk)[0, 1]
    zso = (zzsoil * vk[ii])
    rso = np.corrcoef(zso, sk)[0, 1]
    r_tr = np.corrcoef(Zs["SPEI_12_t"][:, k], S[:, k])[0, 1]
    print(f"{k+1:>4} {r_tr:>12.3f} {r12:>13.3f} {rso:>14.3f}")
print("done")
