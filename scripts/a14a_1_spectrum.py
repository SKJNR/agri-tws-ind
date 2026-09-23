"""Exp 1: Eigenvalue spectrum of the detrended train anomaly field.
Questions: exact rank gap? noise floor sigma^2? flat tail (iid noise) or decaying tail?
Conventions: X = (T=138 months, C=full-history cells) detrended anomaly field.
Eigenvalues v_k = s_k^2 / (C*(T-1))  -> sum_k v_k = average per-cell sample variance.
For iid per-cell noise sigma^2, the beyond-rank eigenvalues cluster at v ~= sigma^2
(MP spread +-2*sqrt(T/C)*sigma ~ +-4.7%).
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)

df = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(df)
print(f"months={len(months)} cells={len(cells)} rows={len(df)}")
F = field_matrix(df, "TWS_t", cells, months)  # (T, C) with NaN
nan_per_cell = np.isnan(F).sum(axis=0)
full = nan_per_cell == 0
print(f"full-history cells: {full.sum()} / {len(cells)}  (hist of nan counts: "
      f"{np.bincount(nan_per_cell[nan_per_cell>0])[:10]} nonzero-nan cells...)")
print(f"missing (cell,month) entries overall: {np.isnan(F).sum()}")

X = F[:, full]  # (T, Cf)
T, C = X.shape
t = months.astype(float)

# ---- per-cell mean and OLS detrend on calendar time (per cell, full history) ----
mu = X.mean(axis=0)
A = X - mu  # anomaly
tc = t - t.mean()
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xdet = A - np.outer(tc, beta)  # detrended anomaly

# sanity: residual per-cell mean should be ~0
print("\nper-cell mean of detrended |max| =", np.abs(Xdet.mean(axis=0)).max())

var_anom = A.var(axis=0, ddof=1).mean()
var_det = Xdet.var(axis=0, ddof=1).mean()
var_trend = (beta ** 2).mean() * (tc ** 2).mean()
print(f"avg per-cell variance: anomaly={var_anom:.4f}, detrended={var_det:.4f}, "
      f"trend share={1 - var_det/var_anom:.4f}")
print(f"per-cell slope std={beta.std():.5f}/month  ({beta.std()*12:.4f}/yr)")

# ---- SVD of detrended field ----
U, s, Vt = np.linalg.svd(Xdet, full_matrices=False)
v = s ** 2 / (C * (T - 1))          # per-cell variance units
cum = np.cumsum(v) / v.sum()
print(f"\nsum v_k (=avg per-cell detrended var) = {v.sum():.4f}")
print("\n=== eigenvalue spectrum (per-cell variance units) ===")
print(f"{'k':>4} {'v_k':>12} {'v_k/v_1':>9} {'cum%':>8} {'gap v_k/v_k+1':>13}")
for k in list(range(1, 31)) + list(range(32, 139, 2)):
    gap = v[k-1] / v[k] if k < T else np.nan
    print(f"{k:>4} {v[k-1]:>12.6f} {v[k-1]/v[0]:>9.5f} {100*cum[k-1]:>8.3f} {gap:>13.4f}")

# ---- tail analysis: where does the flat floor start? ----
print("\n=== tail flatness ===")
for K in [50, 80, 90, 100, 105, 110, 115, 120, 125, 130]:
    tail = v[K:]
    if len(tail) > 2:
        print(f"K={K:>3}: tail mean={tail.mean():.6f} std={tail.std():.6f} "
              f"cv={tail.std()/tail.mean():.4f}  v_K/tailmean={v[K-1]/tail.mean():.4f}")

# biggest log-ratio drop
lr = np.log(v[:-1]) - np.log(v[1:])
order = np.argsort(-lr)[:10]
print("\nlargest log-eigenvalue drops at k -> k+1:",
      [(int(i+1), round(float(lr[i]), 3)) for i in order])

# ---- noise sigma estimate from tail ----
K_star = 100
tail = v[K_star:]
sig2 = tail.mean()
print(f"\n--- noise estimates ---")
print(f"tail mean (K*={K_star}): sigma^2={sig2:.5f} sigma={np.sqrt(sig2):.5f}")
print(f"noise share of detrended variance: {sig2*T/ (v.sum()* (T-1)):.4f} "
      f"(= {T*sig2:.4f} / {v.sum()*(T-1):.4f} total)")
print(f"noise share (simple): {(T-K_star)*sig2 / v.sum()/(T-1)*T:.4f}")
# per-cell residual variance after removing top-K factors
Xhat = (U[:, :K_star] * s[:K_star]) @ Vt[:K_star]
R = Xdet - Xhat
print(f"per-cell residual variance (after top-{K_star}): mean={R.var(axis=0,ddof=1).mean():.5f} "
      f"std={R.var(axis=0,ddof=1).std():.5f}")
print(f"per-cell residual std mean: {np.sqrt(R.var(axis=0,ddof=1)).mean():.5f}")

# distribution of per-cell total variance
pv = Xdet.var(axis=0, ddof=1)
print(f"\nper-cell detrended variance: min={pv.min():.4f} p5={np.percentile(pv,5):.4f} "
      f"med={np.median(pv):.4f} p95={np.percentile(pv,95):.4f} max={pv.max():.4f}")
np.save("/home/z/my-project/scripts/a14a_cache_spec.npy",
        {"v": v, "cum": cum, "sig2": sig2, "beta": beta, "mu": mu,
         "full_mask": full, "cells_lat": cells["lat"].values, "cells_lon": cells["lon"].values},
        allow_pickle=True)
print("\nsaved cache a14a_cache_spec.npy")
