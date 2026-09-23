"""
UNIFIED VALIDATION HARNESS: all approaches on IDENTICAL rows.

Val rows: block structure, anchors 2013-01..2015-02 [::2], k=0..6,
requiring anchor TWS present (same filter as run_B).
Models evaluated:
  k=0:  persistence | global AR(1) | per-cell AR(1) | LGBM-A | blends
  k>=1: trees (B1+B2) | level-Kalman | per-cell AR-decay | blends
"""
import sys
sys.path.insert(0, '/home/z/my-project/scripts')
import pandas as pd
import numpy as np
import lightgbm as lgb
from pipeline import GridContext, load_train, COVARS

DATA = '/home/z/my-project/data'
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
VAL_START = pd.Timestamp('2013-01-01')

train = load_train()
g = GridContext(train, clim_cutoff=VAL_START)
n_cells = g.n_cells
n_months = g.n_months
print(f"Context ready: {n_cells} cells, {n_months} months", flush=True)

# ---------- standard val rows ----------
rows_mask = (g.row_time >= np.datetime64(VAL_START))
times_w = sorted(g.train.loc[rows_mask, 'time'].unique())
anchor_times = times_w[::2]
vidx_list, vk_list = [], []
for t in anchor_times:
    t = pd.Timestamp(t)
    a_midx = g.ym_to_midx[t.year * 100 + t.month]
    L_anchor = g.M_tws[a_midx]
    for k in range(7):
        tot = t.year * 12 + (t.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in g.ym_to_midx:
            continue
        sel = np.where(rows_mask & (g.row_ym == tym))[0]
        if len(sel) == 0:
            continue
        # require anchor present for these cells
        ac = g.row_cell[sel]
        ok = ~np.isnan(L_anchor[ac])
        sel = sel[ok]
        if len(sel) == 0:
            continue
        vidx_list.append(sel)
        vk_list.append(np.full(len(sel), k, dtype=np.int8))
vidx = np.concatenate(vidx_list)
vk = np.concatenate(vk_list)
y = g.row_target[vidx]
print(f"Unified val rows: {len(vidx):,} | k dist: {np.bincount(vk.astype(int)).tolist()}", flush=True)

w = np.array([K_WEIGHTS.get(int(k), 1) for k in vk], dtype=np.float64)

def wrmse(p):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))

def per_k(p):
    return {k: float(np.sqrt(np.mean((y[vk == k] - p[vk == k]) ** 2))) for k in range(7) if (vk == k).sum()}

results = {}

# ---------- k=0 models ----------
m0 = vk == 0
i0 = vidx[m0]
y0 = y[m0]
L_t = g.M_tws[g.row_midx[i0], g.row_cell[i0]]
results['k0_persist'] = (m0, L_t.copy())

# per-cell AR(1) level-space (params from context pre-2013)
pre = g.train[g.row_time < np.datetime64(VAL_START)]
gp = pre.groupby('cell_code')
n_ = gp.size().astype(float).reindex(range(n_cells)).fillna(0)
mu_c = gp['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype(np.float32)
sx_ = gp['TWS_t'].sum().reindex(range(n_cells)).fillna(0)
sy_ = gp['target'].sum().reindex(range(n_cells)).fillna(0)
sxx_ = gp.apply(lambda d: (d['TWS_t'] ** 2).sum(), include_groups=False).reindex(range(n_cells)).fillna(0)
sxy_ = gp.apply(lambda d: (d['TWS_t'] * d['target']).sum(), include_groups=False).reindex(range(n_cells)).fillna(0)
scxx = sxx_ - n_ * mu_c**2
scxy = sxy_ - (sx_ * sy_) / n_
phi_raw = (scxy / scxx.replace(0, np.nan)).clip(0.2, 0.99)
lam = n_ / (n_ + 50.0)
PHI_G = 0.77
phi_c = (lam * phi_raw.fillna(PHI_G) + (1 - lam) * PHI_G).values.astype(np.float32)
q_raw = pre.groupby('cell_code').apply(lambda d: np.nanvar(d['target'] - PHI_G * d['TWS_t'] - (1-PHI_G)*d['TWS_t'].mean()), include_groups=False).reindex(range(n_cells))
Q_G = float(np.var(pre['target'] - PHI_G * pre['TWS_t']))
lam_q = n_ / (n_ + 50.0)
q_c = (lam_q * q_raw.fillna(Q_G).clip(0.01, 2.0) + (1 - lam_q) * Q_G).values.astype(np.float32)

cc0 = g.row_cell[i0]
p_ar1 = mu_c[cc0] + phi_c[cc0] * (L_t - mu_c[cc0])
# global AR(1): pooled
b_g = np.polyfit(g.M_tws[g.row_midx[np.where(g.row_time < np.datetime64(VAL_START))[0]], g.row_cell[np.where(g.row_time < np.datetime64(VAL_START))[0]]], g.row_target[np.where(g.row_time < np.datetime64(VAL_START))[0]], 1)
# simpler: use pre arrays
pre_idx = np.where(g.row_time < np.datetime64(VAL_START))[0]
xpre = g.M_tws[g.row_midx[pre_idx], g.row_cell[pre_idx]]
YPRE = g.row_target[pre_idx]
slope_g, icept_g = np.polyfit(xpre, YPRE, 1)
p_ar1g = slope_g * L_t + icept_g

# LGBM A (retrain fast: 300 trees like run_A)
Xa_tr = g.features_A(pre_idx)
mA = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=80,
                       min_child_samples=100, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1, max_bin=127)
mA.fit(Xa_tr, YPRE)
Xa0 = g.features_A(i0)
p_A = mA.predict(Xa0)
del Xa_tr
print("\n=== k=0 (n={:,}) ===".format(m0.sum()))
print(f"persistence:     {np.sqrt(np.mean((y0-L_t)**2)):.4f}")
print(f"global AR(1):    {np.sqrt(np.mean((y0-p_ar1g)**2)):.4f}")
print(f"per-cell AR(1):  {np.sqrt(np.mean((y0-p_ar1)**2)):.4f}")
print(f"LGBM-A:          {np.sqrt(np.mean((y0-p_A)**2)):.4f}")
for wgt in [0.3, 0.5, 0.7]:
    pb = wgt*p_A + (1-wgt)*p_ar1
    print(f"blend A*{wgt}+AR1: {np.sqrt(np.mean((y0-pb)**2)):.4f}")
pb0 = 0.5*p_A + 0.5*p_ar1

# ---------- k>=1: Kalman level-space ----------
cov_mat5 = g.cov_mat[:, :5]
A = np.column_stack([cov_mat5[pre_idx], np.ones(len(pre_idx))])
bLv = g.M_tws[g.row_midx[pre_idx], g.row_cell[pre_idx]]
coef_obs = np.linalg.solve(A.T @ A + 10.0 * np.eye(6), A.T @ bLv)
M_z = np.full((n_months, n_cells), np.nan, dtype=np.float32)
z_all = cov_mat5 @ coef_obs[:5] + coef_obs[5]
M_z[g.train["midx"].values, g.row_cell] = z_all
cov_ok = ~np.isnan(cov_mat5).any(axis=1)
M_zok = np.zeros((n_months, n_cells), dtype=bool)
M_zok[g.train["midx"].values, g.row_cell] = cov_ok
# honest R
m_fit = g.row_time < np.datetime64('2010-01-01')
m_ev = (g.row_time >= np.datetime64('2010-01-01')) & (g.row_time < np.datetime64(VAL_START))
Af = np.column_stack([cov_mat5[np.where(m_fit)[0]], np.ones(m_fit.sum())])
bf = g.M_tws[g.row_midx[np.where(m_fit)[0]], g.row_cell[np.where(m_fit)[0]]]
ch = np.linalg.solve(Af.T @ Af + 10*np.eye(6), Af.T @ bf)
Ae = np.column_stack([cov_mat5[np.where(m_ev)[0]], np.ones(m_ev.sum())])
be = g.M_tws[g.row_midx[np.where(m_ev)[0]], g.row_cell[np.where(m_ev)[0]]]
R_h = float(np.var(be - Ae @ ch))
print(f"\nobs honest R: {R_h:.4f}", flush=True)

# Kalman over all anchors, record preds at each k
kalman_pred = np.full(len(vidx), np.nan, dtype=np.float32)
vi_pos = {(int(r), int(k)): i for i, (r, k) in enumerate(zip(vidx, vk))}
for t in anchor_times:
    t = pd.Timestamp(t)
    a_midx = g.ym_to_midx[t.year * 100 + t.month]
    L_anchor = g.M_tws[a_midx].copy()
    xh = np.where(np.isnan(L_anchor), 0.0, L_anchor).astype(np.float32)
    P = np.where(~np.isnan(L_anchor), 0.0, 1e6).astype(np.float32)
    for k in range(1, 7):
        tot = t.year * 12 + (t.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in g.ym_to_midx:
            continue
        t_midx = g.ym_to_midx[tym]
        xh = mu_c + phi_c * (xh - mu_c)
        P = phi_c ** 2 * P + q_c
        z = M_z[t_midx]
        ok = M_zok[t_midx] & np.isfinite(z) & (P < 1e5)
        K = np.where(ok, P / (P + R_h), 0.0).astype(np.float32)
        xh = np.where(ok, xh + K * (z - xh), xh)
        P = np.where(ok, (1 - K) * P, P)
        sel = np.where(rows_mask & (g.row_ym == tym))[0]
        if len(sel) == 0:
            continue
        cc = g.row_cell[sel]
        ok2 = ~np.isnan(L_anchor[cc])
        sel = sel[ok2]; cc = cc[ok2]
        pred = mu_c[cc] + phi_c[cc] * (xh[cc] - mu_c[cc])
        for r, pr in zip(sel, pred):
            pos = vi_pos.get((int(r), k))
            if pos is not None:
                kalman_pred[pos] = pr
valid_k = vk >= 1
print(f"\n=== k>=1 Kalman ===")
p_full = kalman_pred.copy(); p_full[m0] = pb0; print(f"system (k=0 blend + kalman): {wrmse(p_full):.4f}")
pk_k = per_k(kalman_pred)
print("per-k: " + " ".join(f"k={k}:{v:.3f}" for k, v in pk_k.items()))

# ---------- k>=1: trees (B1+B2 trained pre-2013 — reuse modelB1_val/modelB2_val) ----------
mB1 = lgb.Booster(model_file='/home/z/my-project/scripts/modelB1_val.txt')
mB2 = lgb.Booster(model_file='/home/z/my-project/scripts/modelB2_val.txt')
with open('/home/z/my-project/scripts/modelB_coefs.json') as f:
    import json
    coefs_B = {int(k): np.array(v) for k, v in json.load(f).items()}

# build tree features for val rows k>=1
tree_pred = np.full(len(vidx), np.nan, dtype=np.float32)
Xparts, positions = [], []
for k in range(1, 7):
    msk = valid_k & (vk == k)
    if msk.sum() == 0:
        continue
    idxs = vidx[msk]
    amap, has, atws, ssum, complete = g.anchor_info(idxs, k)
    okm = has & complete & ~np.isnan(atws)
    idxs = idxs[okm]
    pos = np.where(msk)[0][okm]
    Xk, link, _ = g.features_B(idxs, k, g.row_cell[idxs], g.row_midx[idxs], amap[okm], atws[okm], ssum[okm])
    p1 = mB1.predict(Xk)
    s1 = link.astype(np.float64) @ coefs_B[k]
    p2 = s1 + mB2.predict(Xk)
    tree_pred[pos] = 0.5 * p1 + 0.5 * p2
tmask = valid_k & ~np.isnan(tree_pred)
print(f"\n=== k>=1 trees (n={tmask.sum():,}) ===")
pk_t = {k: float(np.sqrt(np.mean((y[(vk==k) & ~np.isnan(tree_pred)] - tree_pred[(vk==k) & ~np.isnan(tree_pred)])**2))) for k in range(1,7) if ((vk==k) & ~np.isnan(tree_pred)).sum()}
print("per-k: " + " ".join(f"k={k}:{v:.3f}" for k, v in pk_t.items()))

# ---------- blends on common valid rows ----------
both = valid_k & ~np.isnan(tree_pred) & ~np.isnan(kalman_pred)
yb = y[both]; tb = tree_pred[both]; kb_ = kalman_pred[both]; wb = w[both]
print(f"\n=== blends on common k>=1 rows (n={both.sum():,}) ===")
wsys = np.array([K_WEIGHTS.get(int(k), 1) for k in vk[both]], dtype=float)
for wgt in [0.0, 0.3, 0.5, 0.7, 1.0]:
    ens = wgt*tb + (1-wgt)*kb_
    print(f"trees*{wgt:.1f} + kalman*{1-wgt:.1f}: {np.sqrt(np.average((yb-ens)**2, weights=wsys)):.4f}")

np.savez('/home/z/my-project/scripts/unified_val.npz', vidx=vidx, vk=vk, y=y,
         kalman=kalman_pred, tree=tree_pred, p_ar1=np.where(m0, p_ar1, np.nan),
         p_ar1g=np.where(m0, p_ar1g, np.nan), p_A=np.where(m0, p_A, np.nan))
print("\nSaved unified_val.npz", flush=True)
