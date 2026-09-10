"""Exp 8b: k=0 floor with the correct SLOW/FAST split.

The 2013-15 honest window contains a regime drift (trend misfit, std~0.79) that
is NOT covariate-observable and NOT in the train-fit AR coefficients. But for
k=0 rows (TWS_t observed exactly) the drift at t is KNOWN and persists.
Models (honest: infra from 2002-2012, eval on 18 consecutive 2013-15 pairs):
 H2: per-mode c_k s_k(t): c_k = 1 (persist) for slow modes k<=Ks, phi_k else.
 H2b: H2 + cov(t+1) observation of FAST modes only (d_k z_k(t+1), d_k train-fit).
 ORACLE: per-mode a_k (and + cov) fit ON the eval window -> in-sample floor.
Sweep Ks. Reference: persistence 0.6989.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)
COVS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]

df = load_train(cols=("time", "lat", "lon", "TWS_t") + tuple(COVS))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float)

tr = months < 2013 * 12
Xtr = X[tr]; ttr = t[tr]; tctr = ttr - ttr.mean()
mu_h = Xtr.mean(axis=0)
Atr = Xtr - mu_h
beta_h = (Atr * tctr[:, None]).sum(axis=0) / (tctr ** 2).sum()
Xdtr = Atr - np.outer(tctr, beta_h)
Uh, sh, Vth = np.linalg.svd(Xdtr, full_matrices=False)
Vh = Vth.T
Sh = Uh * sh
K = Vh.shape[1]
iph = np.where(np.diff(months[tr]) == 1)[0]
phi_h = np.zeros(K)
for k in range(K):
    phi_h[k] = (Sh[iph, k] * Sh[iph + 1, k]).sum() / (Sh[iph, k] ** 2).sum()

# cov deviations vs train-window means, scores in Vh
Zall = {}
for c in COVS:
    Fc = field_matrix(df, c, cells, months)[:, full]
    Dv = Fc - Fc[tr].mean(axis=0, keepdims=True)
    Zall[c] = Dv @ Vh

def anom(i):
    return X[i] - mu_h - beta_h * (t[i] - ttr.mean())

ev = [(i, i + 1) for i in range(len(months) - 1)
      if months[i + 1] - months[i] == 1 and not tr[i] and not tr[i + 1]]
x_cur = np.stack([anom(i) for i, _ in ev])
y_true = np.stack([anom(j) for _, j in ev])
Znxt = {c: np.stack([Zall[c][j] for _, j in ev]) for c in COVS}
scur = x_cur @ Vh
var_ev = (y_true ** 2).mean()
print(f"eval pairs={len(ev)}, target std={np.sqrt(var_ev):.4f}")
print(f"persistence RMSE = {np.sqrt(((x_cur - y_true)**2).mean()):.4f}")

# d_k: cov(t+1) loading per mode from train window (response = fast residual)
Ztr_nxt = {c: Zall[c][iph + 1] for c in COVS}
resid_tr = Sh[iph + 1] - phi_h[None, :] * Sh[iph]
D = {}
for c in COVS:
    d = np.zeros(K)
    for k in range(K):
        d[k] = (Ztr_nxt[c][:, k] * resid_tr[:, k]).sum() / (Ztr_nxt[c][:, k] ** 2).sum()
    D[c] = d

print("\n=== H2 (split propagation, no covs) ===")
for Ks in [0, 3, 5, 8, 12, 20]:
    c_k = phi_h.copy(); c_k[:Ks] = 1.0
    pred = (scur * c_k) @ Vh.T
    print(f"  Ks={Ks:>2}: RMSE = {np.sqrt(((pred - y_true)**2).mean()):.4f}")

print("\n=== H2b (split + cov(t+1) obs of fast modes k>Ks) ===")
for Ks in [3, 5, 8, 12]:
    c_k = phi_h.copy(); c_k[:Ks] = 1.0
    best = None
    for c in COVS:
        d_eff = D[c].copy(); d_eff[:Ks] = 0.0
        pred = (scur * c_k + Znxt[c] * d_eff) @ Vh.T
        r = np.sqrt(((pred - y_true) ** 2).mean())
        best = (c, r) if best is None or r < best[1] else best
    # sum over covs (equal weights on fast modes)
    dsum = sum(D[c] for c in COVS) / len(COVS)
    d_eff = dsum.copy(); d_eff[:Ks] = 0.0
    pred = (scur * c_k + sum(Znxt[c] for c in COVS) * 0.0 + (np.stack([Znxt[c] for c in COVS]).sum(0)) * 0.0) @ Vh.T  # placeholder
    # proper: average of per-cov preds
    preds = [(scur * c_k + Znxt[c] * np.where(np.arange(K) >= Ks, D[c], 0.0)) @ Vh.T for c in COVS]
    pavg = np.mean(preds, axis=0)
    r_avg = np.sqrt(((pavg - y_true) ** 2).mean())
    print(f"  Ks={Ks:>2}: best single ({best[0][:9]}) = {best[1]:.4f};  avg5 = {r_avg:.4f}")

print("\n=== ORACLE (fit per-mode on eval window — in-sample floor) ===")
for Ks in [0, 5]:
    # oracle a_k on eval
    a_or = np.zeros(K)
    for k in range(K):
        a_or[k] = (scur[:, k] * (y_true @ Vh)[:, k]).sum() / (scur[:, k] ** 2).sum()
    pred = (scur * a_or) @ Vh.T
    print(f"  oracle per-mode (no cov): RMSE = {np.sqrt(((pred - y_true)**2).mean()):.4f}")
    # oracle with covs
    sy = y_true @ Vh
    best = None
    for c in COVS:
        r2best = -1
        for k in range(K):
            Xd_ = np.column_stack([scur[:, k], Znxt[c][:, k]])
            cf = np.linalg.lstsq(Xd_, sy[:, k], rcond=None)[0]
            predk = Xd_ @ cf
            r2best += 0
        # vectorized per-mode via normal equations
        num = (Znxt[c] * sy).sum(0) * (scur ** 2).sum(0) - (scur * sy).sum(0) * (scur * Znxt[c]).sum(0)
        den = (scur ** 2).sum(0) * (Znxt[c] ** 2).sum(0) - (scur * Znxt[c]).sum(0) ** 2
        a_k = num / den
        b_num = (scur * sy).sum(0) * (Znxt[c] ** 2).sum(0) - (Znxt[c] * sy).sum(0) * (scur * Znxt[c]).sum(0)
        b_k = b_num / den
        pred = (scur * a_k + Znxt[c] * b_k) @ Vh.T
        r = np.sqrt(((pred - y_true) ** 2).mean())
        best = (c, r) if best is None or r < best[1] else best
    print(f"  oracle per-mode + best cov ({best[0][:9]}): RMSE = {best[1]:.4f}")
    break
print("done")
