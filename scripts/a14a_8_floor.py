"""Exp 8: THE FLOOR TEST — how good can predictions get with per-mode
(field-level) covariate observation?  Honest: fit 2002-2012, eval 2013-15.

Tasks:
 T0 persistence (anomaly) — reference.
 T1 k=0 floor: predict TWS(t+1) from EXACT TWS anomaly(t) + covs(t+1):
    per-mode s_k(t+1) ~ a_k s_k(t) + sum_c b_kc z_kc(t+1).  (a_k free)
 T1b same without covs(t+1) (pure propagate).
 T2 masked-with-covs(t+1) floor: per-mode from covs(t+1) only.
 T3 masked-no-covs(t+1): state from covs(t) (per-mode), propagate 1 month.
 T4 horizon sweep h=1..7: state from covs(t) propagated h months.
 Cov-check: r_k for DETRENDED cov fields (is slow-mode observation trend-driven?)
 Test-era adjustment: T1/T2/T3 recomputed 'slow-blind' (cov obs of modes 1-5
 zeroed; slow/D handled separately with D-error 0.42 in quadrature).
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
t = months.astype(float); tc = t - t.mean()
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)
U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
V = Vt.T
S = U * s
Tm = V.shape[1]
ip = np.where(np.diff(months) == 1)[0]

# cov deviation (mean-removed) and DETRENDED cov fields + scores
Zs, Zs_dt = {}, {}
for c in COVS:
    Fc = field_matrix(df, c, cells, months)[:, full]
    Dv = Fc - Fc.mean(axis=0, keepdims=True)
    Zs[c] = Dv @ V
    bc = (Dv * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
    Ddt = Dv - np.outer(tc, bc)
    Zs_dt[c] = Ddt @ V

# cov-check: r_k mean-removed vs detrended cov
print("=== cov-check: r_k (cov score vs TWS detrended score) ===")
print(f"{'mode':>4} {'S12_meanrem':>11} {'S12_detrended':>13} {'SOIL_meanrem':>12} {'SOIL_detr':>10}")
for k in [0, 1, 2, 3, 4, 5, 7, 9, 11, 19, 49]:
    r1 = np.corrcoef(Zs["SPEI_12_t"][:, k], S[:, k])[0, 1]
    r2 = np.corrcoef(Zs_dt["SPEI_12_t"][:, k], S[:, k])[0, 1]
    r3 = np.corrcoef(Zs["SOIL_MOISTURE_t"][:, k], S[:, k])[0, 1]
    r4 = np.corrcoef(Zs_dt["SOIL_MOISTURE_t"][:, k], S[:, k])[0, 1]
    print(f"{k+1:>4} {r1:>11.3f} {r2:>13.3f} {r3:>12.3f} {r4:>10.3f}")

# ---------------- honest setup ----------------
tr = months < 2013 * 12
# train-window infrastructure
Xtr = X[tr]; ttr = t[tr]; tctr = ttr - ttr.mean()
mu_h = Xtr.mean(axis=0)
Atr = Xtr - mu_h
beta_h = (Atr * tctr[:, None]).sum(axis=0) / (tctr ** 2).sum()
Xdtr = Atr - np.outer(tctr, beta_h)
Uh, sh, Vth = np.linalg.svd(Xdtr, full_matrices=False)
Vh = Vth.T
Sh = Uh * sh
iph = np.where(np.diff(months[tr]) == 1)[0]
K = 136
# per-mode phi (train window, consecutive pairs)
phi_h = np.zeros(Vh.shape[1])
for k in range(Vh.shape[1]):
    phi_h[k] = (Sh[iph, k] * Sh[iph + 1, k]).sum() / (Sh[iph, k] ** 2).sum()

# cov scores in Vh basis (train window)
Zh = {}
for c in COVS:
    Fc = field_matrix(df, c, cells, months)[:, full]
    Dv = Fc[tr] - Fc[tr].mean(axis=0, keepdims=True)
    Zh[c] = Dv @ Vh
# full-series cov deviations against TRAIN-window cov means
Zall = {}
for c in COVS:
    Fc = field_matrix(df, c, cells, months)[:, full]
    Dv = Fc - Fc[tr].mean(axis=0, keepdims=True)
    Zall[c] = Dv @ Vh

# eval pairs: (tau, tau+1) both in 2013-15
ev = [(i, i + 1) for i in range(len(months) - 1)
      if months[i + 1] - months[i] == 1 and not tr[i] and not tr[i + 1]]
tau_idx = np.array([a for a, b in ev]); nxt_idx = np.array([b for a, b in ev])
print(f"\n eval pairs: {len(ev)} (2013-15 consecutive months)")

def anom(i):   # honest anomaly vs train-window mu/beta
    return X[i] - mu_h - beta_h * (t[i] - ttr.mean())

y_true = np.stack([anom(j) for j in nxt_idx])          # (n_ev, C)
x_cur = np.stack([anom(i) for i in tau_idx])
Zcur = {c: np.stack([Zall[c][i] for i in tau_idx]) for c in COVS}     # covs at t
Znxt = {c: np.stack([Zall[c][j] for j in nxt_idx]) for c in COVS}     # covs at t+1

# scores of current anomaly in Vh basis
scur = x_cur @ Vh
var_ev = (y_true ** 2).mean()
print(f" target (2013-15 anomaly) std = {np.sqrt(var_ev):.4f}")

def recon(scores, basis=Vh):
    return scores @ basis.T

# ---- T0 persistence ----
rmse = lambda p: np.sqrt(((p - y_true) ** 2).mean())
print(f"\n T0 persistence:            RMSE = {rmse(x_cur):.4f}")

# ---- per-mode regression helper ----
def fitpermode(blocks_tr, resp_tr):
    """blocks_tr: list of (n,K) feature blocks; per-mode LSQ. Returns (K, nblocks)."""
    K = resp_tr.shape[1]; nb = len(blocks_tr)
    X = np.column_stack(blocks_tr)
    Wc = np.zeros((K, nb))
    for k in range(K):
        cols = [b * K + k for b in range(nb)]
        Wc[k] = np.linalg.lstsq(X[:, cols], resp_tr[:, k], rcond=None)[0]
    return Wc

def applypermode(blocks_ev, Wc):
    K, nb = Wc.shape
    X = np.column_stack(blocks_ev)
    out = np.zeros((X.shape[0], K))
    for k in range(K):
        cols = [b * K + k for b in range(nb)]
        out[:, k] = X[:, cols] @ Wc[k]
    return out

# train-window feature rows (consecutive pairs)
s_tr_cur = Sh[iph]
s_tr_nxt = Sh[iph + 1]
Ztr_cur = {c: Zh[c][iph] for c in COVS}
Ztr_nxt = {c: Zh[c][iph + 1] for c in COVS}

# ---- T1: per-mode a_k s_k(t) + covs(t+1) ----
W1 = fitpermode([s_tr_cur] + [Ztr_nxt[c] for c in COVS], s_tr_nxt)
pred = recon(applypermode([scur] + [Znxt[c] for c in COVS], W1))
print(f" T1 k=0 (TWS_t + covs(t+1)): RMSE = {rmse(pred):.4f}   "
      f"R2 = {1 - ((pred-y_true)**2).mean()/var_ev:.4f}")
# slow-blind variant: zero cov coefs for modes 1-5
W1sb = W1.copy(); W1sb[:5, 1:] = 0.0
pred_sb = recon(applypermode([scur] + [Znxt[c] for c in COVS], W1sb))
print(f"    slow-blind (cov obs of PC1-5 off): RMSE = {rmse(pred_sb):.4f}"
      f"  -> test-era-adjusted (add 0.42 D-err): "
      f"{np.sqrt(((pred_sb-y_true)**2).mean() + 0.42**2):.4f}")

# ---- T1b: propagate only ----
W1b = fitpermode([s_tr_cur], s_tr_nxt)
pred = recon(applypermode([scur], W1b))
print(f" T1b k=0 (TWS_t only):      RMSE = {rmse(pred):.4f}")

# ---- T2: covs(t+1) only ----
W2 = fitpermode([Ztr_nxt[c] for c in COVS], s_tr_nxt)
pred = recon(applypermode([Znxt[c] for c in COVS], W2))
print(f" T2 masked+covs(t+1):       RMSE = {rmse(pred):.4f}   "
      f"R2 = {1 - ((pred-y_true)**2).mean()/var_ev:.4f}")
W2sb = W2.copy(); W2sb[:5, :] = 0.0
pred_sb = recon(applypermode([Znxt[c] for c in COVS], W2sb))
print(f"    slow-blind: RMSE = {rmse(pred_sb):.4f}"
      f"  -> test-era-adjusted: {np.sqrt(((pred_sb-y_true)**2).mean() + 0.42**2):.4f}")
# single-cov
for c in ["SPEI_12_t", "SOIL_MOISTURE_t"]:
    W2c = fitpermode([Ztr_nxt[c]], s_tr_nxt)
    pred = recon(applypermode([Znxt[c]], W2c))
    print(f"    T2 single {c}: RMSE = {rmse(pred):.4f}")

# ---- T3: state from covs(t), propagate 1 month ----
W3 = fitpermode([Ztr_cur[c] for c in COVS], s_tr_cur)
s_hat = applypermode([Zcur[c] for c in COVS], W3)
pred = recon(s_hat * phi_h[None, :])
print(f" T3 masked, 1-mo from covs(t): RMSE = {rmse(pred):.4f}")
W3sb = W3.copy(); W3sb[:5, :] = 0.0
s_hat_sb = applypermode([Zcur[c] for c in COVS], W3sb)
pred_sb = recon(s_hat_sb * phi_h[None, :])
print(f"    slow-blind: RMSE = {rmse(pred_sb):.4f}"
      f"  -> test-era-adjusted: {np.sqrt(((pred_sb-y_true)**2).mean() + 0.42**2):.4f}")

# ---- T4: horizon sweep (state from covs(t), propagate h) ----
print("\n T4 horizon sweep (masked rows; state from covs(t) propagated h months):")
print(f" {'h':>2} {'full':>8} {'slowblind':>9} {'+Derr0.42':>10}")
for h in range(1, 8):
    evh = [(i, i + h) for i in range(len(months) - h)
           if months[i + h] - months[i] == h and not tr[i] and not tr[i + h]]
    if not evh: continue
    xs = np.stack([anom(i) for i, _ in evh]) @ Vh
    ys = np.stack([anom(j) for _, j in evh])
    Zs_ev = {c: np.stack([Zall[c][i] for i, _ in evh]) for c in COVS}
    s_hat = applypermode([Zs_ev[c] for c in COVS], W3)
    pred = recon(s_hat * (phi_h[None, :] ** h))
    r_full = np.sqrt(((pred - ys) ** 2).mean())
    s_hat_sb = applypermode([Zs_ev[c] for c in COVS], W3sb)
    pred_sb = recon(s_hat_sb * (phi_h[None, :] ** h))
    r_sb = np.sqrt(((pred_sb - ys) ** 2).mean())
    print(f" {h:>2} {r_full:>8.4f} {r_sb:>9.4f} {np.sqrt(r_sb**2 + 0.42**2):>10.4f}")

# ---- innovation observability (train-fit, honest) ----
print("\n innovation observability: per-mode R2 of innov on covs at same month")
innov_tr = Sh[iph + 1] - phi_h[None, :] * Sh[iph]
Ztr_innov = {c: Zh[c][iph + 1] for c in COVS}
WI = fitpermode([Ztr_innov[c] for c in COVS], innov_tr)
Inov_hat = applypermode([Ztr_innov[c] for c in COVS], WI)
r2 = 1 - ((innov_tr - Inov_hat) ** 2).sum() / (innov_tr ** 2).sum()
w_var = (innov_tr ** 2).sum(axis=0)
r2w = 1 - ((innov_tr - Inov_hat) ** 2).sum(0) / (innov_tr ** 2).sum(0)
print(f"  overall innovation R2 (5 covs, modes 1-136, var-weighted): {r2:.4f}")
print(f"  per-mode innovation R2 (k=1,5,10,20,50,100): "
      f"{[round(float(r2w[k]), 3) for k in [0, 4, 9, 19, 49, 99]]}")
cumw = np.cumsum(w_var) / w_var.sum()
print(f"  innovation var cumsum at k=10/20/50: {cumw[9]:.3f}/{cumw[19]:.3f}/{cumw[49]:.3f}")
print("done")
