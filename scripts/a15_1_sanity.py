"""a15_1_sanity.py — infrastructure sanity + era diagnostics before building predictors.

Checks (vs a14a_8 floor-test numbers where applicable):
  A. per-year anomaly std (fit-window detrending) — is the val era elevated? in/out subspace?
  B. val-era phi vs fit-era phi (top modes) — dynamics transfer.
  C. per-mode cov->state corr r_k: fit window vs val window (nonstationarity check).
  D. ONE-STEP-ALIGNMENT: predict val anomaly(t+1) from anomaly(t) on the 18 consecutive
     val pairs (a14a_8 protocol):
       T0  persistence                       [ref 0.6989]
       T1n per-mode phi propagation, NO r    [ref 0.9429]
       T1r per-mode phi propagation + r(t)
       T1k ModeKF 1-step (fast+d split, VarD prior) + r(t)
       T1z T1k + cov update at t+1           [a14a_8 T1 ref 0.9215 (unshrunk LS)]
"""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/z/my-project/scripts')
from a15_common import load_train, FactorCore, ModeKF, estimate_VarD, rmse, COVS

t0 = time.time()
OUT = '/home/z/my-project/download/a15_1_sanity.txt'
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

train = load_train()
n_cells = int(train['cc'].max()) + 1
fit = train[train['time'].dt.year <= 2012]
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]
P(f"train {len(train):,} rows | fit {len(fit):,} | val {len(val):,}  [{time.time()-t0:.0f}s]")

core = FactorCore(fit, n_cells, K=100, rec_edges=(2010, 2013))
V, Vx, phi, q = core.V, core.Vx, core.phi, core.q
full, nfull = core.full, core.nfull
P(f"core built [{time.time()-t0:.0f}s]")

# ---------------- val structures ----------------
val_months = np.sort(val['t_abs'].unique())
vm2i = {int(m): i for i, m in enumerate(val_months)}
Tval = len(val_months)
Xv = np.full((Tval, n_cells), np.nan, dtype=np.float32)
mi = val['t_abs'].map(vm2i).values
Xv[mi, val['cc'].values] = val['TWS_t'].values
# anomaly vs fit-window mu/beta (honest extrapolation)
TDv = val_months[:, None].astype(np.float64) - core.tbar_c[None, :]
Av = Xv.astype(np.float64) - core.mu_c[None, :] - TDv * core.beta_c[None, :]
# per-cov val fields + z scores (era-mean removed)
vmonths, vfields = core.era_cov(val)
vz = core.zscores(vfields, vmonths)

# ---------------- A. per-year anomaly std ----------------
P("\n=== A. per-year detrended-anomaly std (fit-window mu/beta) ===")
tr_months = core.months
trF = np.full((len(tr_months), n_cells), np.nan, dtype=np.float32)
ymi = {int(v): i for i, v in enumerate(np.sort(fit['ym'].unique()))}
trF[fit['ym'].map(ymi).values, fit['cc'].values] = fit['TWS_t'].values
TDtr = tr_months[:, None].astype(np.float64) - core.tbar_c[None, :]
Atr = trF.astype(np.float64) - core.mu_c[None, :] - TDtr * core.beta_c[None, :]
allyr = np.concatenate([tr_months // 12, val_months // 12])
allA = np.vstack([Atr, Av])
for yr in range(2002, 2016):
    m = allyr == yr
    if m.sum() == 0:
        continue
    P(f"  {yr}: std={np.nanstd(allA[m]):.4f}  months={int(m.sum())}")
# in/out subspace split of the val anomaly (per-cell variance units)
gv = Av[:, full]
gv = np.where(np.isfinite(gv), gv, 0.0)
fv = gv @ V
rv = gv - fv @ V.T
var_tot = np.nanvar(Av[:, full])
P(f"val anomaly per-cell var: total={var_tot:.4f} (std {np.sqrt(var_tot):.4f}) | "
  f"in-sub(K=100)={np.var(fv@V.T)/1:.4f} out-of-sub={np.var(rv):.4f} (std {np.sqrt(np.var(rv)):.4f})")
P(f"fit anomaly per-cell var: total={np.nanvar(Atr[:, full]):.4f}")

# val extra-slow: mean of val anomaly over all val months (era mean = slow-ish level)
era_mean = np.nanmean(Av[:, full], axis=0)
P(f"val era-mean anomaly: std={np.std(era_mean):.4f}  in-sub={np.std(era_mean @ V @ V.T):.4f} "
  f"out-of-sub={np.std(era_mean - (era_mean @ V) @ V.T):.4f}")

# ---------------- B. phi transfer ----------------
P("\n=== B. phi transfer (fit <=2012 vs val 2013-15 consecutive pairs) ===")
ipv = np.where(np.diff(val_months) == 1)[0]
fv_pairs = fv[ipv]
num = (fv_pairs * fv[ipv + 1]).sum(axis=0)
den = (fv_pairs ** 2).sum(axis=0)
phi_val = num / np.maximum(den, 1e-9)
P("  mode: fit_phi  val_phi   Vx")
for k in list(range(10)) + [14, 19, 29, 49, 99]:
    P(f"  {k+1:4d}: {phi[k]:+.3f}  {phi_val[k]:+.3f}  {Vx[k]:9.1f}")
P(f"  var-weighted mean phi (fit): {(phi*Vx).sum()/Vx.sum():.3f}  (val): {(phi_val*Vx).sum()/Vx.sum():.3f}")

# ---------------- C. cov->state corr transfer ----------------
P("\n=== C. per-mode corr(z_jk, f_k): fit window (recency) vs val ===")
w = core.wrec
J = len(COVS)
zf_val = np.stack([vz[int(m)] for m in val_months])        # (Tval, J, K)
fz_val = fv
for j, c in enumerate(COVS):
    zf = core.Zj[j]                                        # (Tfit, K)
    wz = w[:, None]
    Cf = (wz * zf * core.S).sum(0) / (wz * core.S ** 2).sum(0)
    Vz = (wz * zf ** 2).sum(0) / w.sum()
    Vf = (wz * core.S ** 2).sum(0) / w.sum()
    r_fit = Cf * np.sqrt(Vf / Vz)
    num = (zf_val[:, j, :] * fz_val).sum(0)
    den = np.sqrt((zf_val[:, j, :] ** 2).sum(0) * (fz_val ** 2).sum(0))
    r_val = num / np.maximum(den, 1e-9)
    P(f"  {c:16s} r_fit(top10)={np.round(r_fit[:10], 2)}")
    P(f"  {'':16s} r_val(top10)={np.round(r_val[:10], 2)}")
    P(f"  {'':16s} var-wt r: fit={np.sum(r_fit[:50]*Vx[:50])/Vx[:50].sum():+.3f} "
      f"val={np.sum(r_val[:50]*Vx[:50])/Vx[:50].sum():+.3f}  (modes 1-50)")

# ---------------- D. one-step-ahead on the 18 val consecutive pairs ----------------
P("\n=== D. one-step-ahead (a14a_8 protocol, 18 val consecutive pairs) ===")
ev = ipv
tau, nxt = ev, ev + 1
y_true = Av[nxt][:, full]
y_true = np.where(np.isfinite(y_true), y_true, np.nan)
g_cur = Av[tau][:, full]
g_cur = np.where(np.isfinite(g_cur), g_cur, 0.0)
f_cur = g_cur @ V
r_cur = g_cur - f_cur @ V.T
var_ev = np.nanmean(y_true ** 2)
P(f"eval pairs={len(ev)}  target std={np.sqrt(var_ev):.4f}  [a14a_8 ref 1.0842]")

def ev_rmse(pred_full):
    d = pred_full - y_true
    return float(np.sqrt(np.nanmean(d ** 2)))

# T0 persistence
P(f"T0  persistence:                 {ev_rmse(g_cur):.4f}   [ref 0.6989]")
# T1n no r
P(f"T1n per-mode phi, NO r:          {ev_rmse((f_cur * phi[None, :]) @ V.T):.4f}   [ref 0.9429]")
# T1r with r
P(f"T1r per-mode phi + r(t):         {ev_rmse((f_cur * phi[None, :]) @ V.T + r_cur):.4f}")
# T1k ModeKF (fast+d split) + r; VarD from val anchors (pure-trend Dtil)
val_anchor_months = [m for m in val_months if (m % 12 + 1) in {1, 6, 7, 9, 11, 12}]
YA = []
for a in val_anchor_months:
    i = vm2i[int(a)]
    g = Av[i, full]
    g = np.where(np.isfinite(g), g, 0.0)
    YA.append(V.T @ g)
YA = np.stack(YA)
gaps = [(i, i + 1, int(val_months[i + 1] - val_months[i]))
        for i in range(len(val_anchor_months) - 1)]
VarD = estimate_VarD(YA, gaps, phi, Vx, smooth=9)
alpha1 = (VarD + phi * Vx) / (VarD + Vx)
P(f"VarD (val, pure-trend Dtil): per-cell std-equiv={np.sqrt(VarD.sum()/nfull):.4f}; "
  f"sqrt(VarD) top10={np.round(np.sqrt(VarD[:10]), 1)}")
P(f"alpha_1mo top10={np.round(alpha1[:10], 3)}  mid(20-30)={np.round(alpha1[19:30], 2)}  "
  f"tail(90-100)={np.round(alpha1[90:100], 2)}")

kf = ModeKF(phi, Vx, q, core.H, core.Rcov, VarD)
pred_k = np.zeros_like(f_cur)
for e in range(len(ev)):
    kf.reset()
    y = f_cur[e]
    kf.anchor_update(y)
    kf.step(1)
    m1 = val_months[nxt[e]]
    if int(m1) in vz:
        kf.cov_update(vz[int(m1)])
    pred_k[e] = kf.mean_var()[0]
P(f"T1k ModeKF(fast+d)+r:            {ev_rmse(pred_k @ V.T + r_cur):.4f}  [with cov(t+1) update]")

# separate: no-cov version
pred_nc = np.zeros_like(f_cur)
for e in range(len(ev)):
    kf.reset()
    kf.anchor_update(f_cur[e])
    kf.step(1)
    pred_nc[e] = kf.mean_var()[0]
P(f"T1k-no-cov:                      {ev_rmse(pred_nc @ V.T + r_cur):.4f}")

# direct per-mode joint regression: f(t+1) ~ a f(t) + sum_j B_j z_j(t+1), recency ridge
ia, ib = core.ip, core.ip + 1
wtr = core.wrec[ia]
J = len(COVS)
Xrows = np.concatenate([core.S[ia][:, None, :]] + [core.Zj[j][ib][:, None, :] for j in range(J)], axis=1)  # (npairs, 1+J, K)
Yrows = core.S[ib]  # (npairs, K)
pred_dir = np.zeros_like(f_cur)
pred_dir_nc = np.zeros_like(f_cur)
lamb_ridge = 0.3
for k in range(core.K):
    Xk = Xrows[:, :, k]                                # (npairs, 1+J)
    sw = np.sqrt(wtr)[:, None]
    Xs = Xk * sw
    XtX = Xs.T @ Xs
    Xty = Xs.T @ (Yrows[:, k] * np.sqrt(wtr))
    XtX[np.arange(1 + J), np.arange(1 + J)] += lamb_ridge * np.trace(XtX) / (1 + J)
    coef = np.linalg.solve(XtX, Xty)
    Xev = np.column_stack([f_cur[:, k]] + [zf_val[nxt, j, k] for j in range(J)])
    pred_dir[:, k] = Xev @ coef
    pred_dir_nc[:, k] = coef[0] * f_cur[:, k]
P(f"T1d direct joint regr (ridge .3): {ev_rmse(pred_dir @ V.T + r_cur):.4f}  "
  f"(no-cov: {ev_rmse(pred_dir_nc @ V.T + r_cur):.4f})")

# innovation observability honest check on val (var-weighted over modes 1-50)
innov_val = fv[nxt] - fv[tau] * phi[None, :]
for j, c in enumerate(COVS):
    zz = zf_val[nxt][:, j, :50]
    ii = innov_val[:, :50]
    rj = np.sum((zz * ii).sum(0) * Vx[:50]) / np.sqrt(
        np.sum((zz ** 2).sum(0) * Vx[:50]) * np.sum((ii ** 2).sum(0) * Vx[:50]))
    P(f"  innov-obs {c:16s} var-wt r = {rj:+.3f}")
P(f"1-step R2 (T1z vs persistence): {1 - ev_rmse(pred_k @ V.T + r_cur)**2/var_ev:.4f} "
  f"vs {1 - ev_rmse(g_cur)**2/var_ev:.4f}")

with open(OUT, 'w') as f:
    f.write('\n'.join(log) + '\n')
P(f"\nsaved {OUT}  [{time.time()-t0:.0f}s]")
