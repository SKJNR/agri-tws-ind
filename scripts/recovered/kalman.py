"""
KALMAN FILTER for masked TWS reconstruction + prediction.

State model (per cell c):
  x_t = anomaly vs per-cell-month climatology
  x_{t+1} = phi_c * x_t + w_t,   w ~ N(0, q_c)
Observation (per month, global coefficients, pooled ridge):
  z_t = b1*S01_an + b2*S03_an + b3*S06_an + b4*S12_an + b5*SM_an = x_t + v, v ~ N(0, r)
Anchor: x at anchor month known exactly (P=0).
Target: TWS_{t+1} = clim_{c,m(t+1)} + phi_c * x_t (+ E[w] = 0)

Vectorized over cells; iterates month by month.
Validation: block structure on 2013+ (same as run_B for comparability).
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
VAL_START = pd.Timestamp('2013-01-01')
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target'] + COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = int(train['cc'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
train['m'] = train['time'].dt.month.astype('int8')
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
train['midx'] = ym_idx.astype('int32')
n_months = len(ym_codes)
ym_to_midx = {int(v): i for i, v in enumerate(ym_codes)}
print(f"Loaded {len(train):,}", flush=True)

pre_mask = (train['time'] < VAL_START).values
pre = train[pre_mask]

# ---------- parameter estimation ----------
# per-cell-month climatology (pre-2013)
def clim_of(col):
    g = pre.groupby(['cc', 'm'])[col].mean()
    a = np.full((n_cells, 13), np.nan, dtype=np.float32)
    for (cc, m), v in g.items():
        a[cc, m] = v
    gm = pre.groupby('m')[col].mean()
    fb = np.array([gm.get(m, 0.0) for m in range(13)], dtype=np.float32)
    return np.where(np.isnan(a), fb[None, :], a)

clim = clim_of('TWS_t')
cov_clim = {c: clim_of(c) for c in COVS}

# anomalies
train['x'] = train['TWS_t'].values - clim[train['cc'].values, train['m'].values]
cov_an = np.column_stack([train[c].values - cov_clim[c][train['cc'].values, train['m'].values] for c in COVS])
train['xt'] = train['target'].values - clim[train['cc'].values, ((train['m'].values % 12) + 1)]

# phi per cell (shrunk toward 0.77), q per cell (shrunk toward global)
g = pre.groupby('cc')
n_ = g.size().astype(float).reindex(range(n_cells)).fillna(0)
sx_ = g['TWS_t'].sum().reindex(range(n_cells)).fillna(0)
sy_ = g['target'].sum().reindex(range(n_cells)).fillna(0)
sxx_ = g.apply(lambda d: (d['TWS_t'] ** 2).sum(), include_groups=False).reindex(range(n_cells)).fillna(0)
sxy_ = g.apply(lambda d: (d['TWS_t'] * d['target']).sum(), include_groups=False).reindex(range(n_cells)).fillna(0)
den = (n_ * sxx_ - sx_ ** 2).replace(0, np.nan)
phi_raw = ((n_ * sxy_ - sx_ * sy_) / den)
# innovation var per cell
q_raw = pre.groupby('cc').apply(
    lambda d: np.nanvar(d['target'] - 0.77 * d['TWS_t']), include_groups=False)
q_raw = q_raw.reindex(range(n_cells))
PHI_G = 0.77
Q_G = float(np.nanvar(pre['target'] - PHI_G * pre['TWS_t']))

# shrinkage both toward global (heavy, since noise dominates)
lam_phi = n_ / (n_ + 60.0)   # effective shrink: ~138 obs -> weight 0.7
phi_c = (lam_phi * phi_raw.fillna(PHI_G).clip(0.2, 0.99) + (1 - lam_phi) * PHI_G).values.astype(np.float32)
lam_q = n_ / (n_ + 60.0)
q_c = (lam_q * q_raw.fillna(Q_G).clip(0.01, 2.0) + (1 - lam_q) * Q_G).values.astype(np.float32)
print(f"phi_c: mean={phi_c.mean():.3f} | q_c: mean={q_c.mean():.4f} (global q={Q_G:.4f})", flush=True)

# observation model: pooled ridge x_t ~ cov anomalies (global coefs)
A = np.column_stack([cov_an[pre_mask], np.ones(pre_mask.sum())])
b = train['x'].values[pre_mask]
AtA = A.T @ A + 10.0 * np.eye(6)
coef_obs = np.linalg.solve(AtA, A.T @ b)
resid = b - A @ coef_obs
R_G = float(np.var(resid))
print(f"obs coefs: {coef_obs[:5].round(3)} r_var={R_G:.4f}", flush=True)

# monthly matrices
M_x = np.full((n_months, n_cells), np.nan, dtype=np.float32)  # true anomaly (train)
M_tws = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_z = np.full((n_months, n_cells), np.nan, dtype=np.float32)  # pseudo-observation
M_x[ym_idx, train['cc'].values] = train['x'].values
M_tws[ym_idx, train['cc'].values] = train['TWS_t'].values
z_all = cov_an @ coef_obs[:5] + coef_obs[5]
M_z[ym_idx, train['cc'].values] = z_all
# covariate availability per month-cell (all covs present?)
cov_ok = ~np.isnan(cov_an).any(axis=1)
M_zok = np.zeros((n_months, n_cells), dtype=bool)
M_zok[ym_idx, train['cc'].values] = cov_ok

row_cc = train['cc'].values
row_ym = train['ym'].values
row_midx = ym_idx
row_target = train['target'].values
row_xtgt = train['xt'].values
row_t1m = ((train['time'].dt.month.values % 12) + 1).astype('int8')


def kalman_chain(anchor_midx, t_midx, cc_mask_all, x_anchor):
    """Run Kalman from anchor month to t month for all cells.
    Returns x_hat, P at month t. Vectorized over cells.
    Cells missing at anchor: state 0 (climatology) with huge variance."""
    xh = np.where(np.isnan(x_anchor), 0.0, x_anchor).astype(np.float32)
    P = np.where(cc_mask_all & ~np.isnan(x_anchor), 0.0, 1e6).astype(np.float32)
    for mi in range(anchor_midx + 1, t_midx + 1):
        # predict
        xh = phi_c * xh
        P = phi_c ** 2 * P + q_c
        # update (only where observation available)
        z = M_z[mi]
        ok = M_zok[mi] & np.isfinite(z) & (P < 1e5)
        K = np.where(ok, P / (P + R_G), 0.0).astype(np.float32)
        xh = np.where(ok, xh + K * (z - xh), xh)
        P = np.where(ok, (1 - K) * P, P)
    return xh, P


# ---------- VALIDATION: block structure (same anchors as run_B) ----------
rows_mask = (train['time'] >= VAL_START).values
times_w = sorted(train.loc[rows_mask, 'time'].unique())
anchor_times = times_w[::2]

# for each (anchor, k): the row month = anchor+k; predict its target
y_true, y_pred, k_list = [], [], []
for t in anchor_times:
    t = pd.Timestamp(t)
    a_midx = ym_to_midx[t.year * 100 + t.month]
    # anchor anomalies per cell
    x_anchor = M_x[a_midx].copy()
    for k in range(1, 7):
        tot = t.year * 12 + (t.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in ym_to_midx:
            continue
        t_midx = ym_to_midx[tym]
        sel = np.where(rows_mask & (row_ym == tym))[0]
        if len(sel) == 0:
            continue
        cc = row_cc[sel]
        # run kalman only for needed cells (but vectorized for all, then subset)
        xh, P = kalman_chain(a_midx, t_midx, np.ones(n_cells, bool), x_anchor)
        # target prediction: clim_{t+1} + phi*x_t
        t1m = row_t1m[sel]
        pred = clim[cc, t1m] + phi_c[cc] * xh[cc]
        y_true.append(row_target[sel])
        y_pred.append(pred)
        k_list.append(np.full(len(sel), k))
        # also track: simple decay prediction for reference
y = np.concatenate(y_true); p = np.concatenate(y_pred); kk = np.concatenate(k_list)
w = np.array([K_WEIGHTS.get(int(k), 1) for k in kk], dtype=np.float64)
print(f"\n=== KALMAN (k=1..6, block val 2013+) ===")
print(f"weighted RMSE: {np.sqrt(np.average((y - p) ** 2, weights=w)):.4f}")
for k in range(1, 7):
    m = kk == k
    if m.sum():
        print(f"  k={k}: {np.sqrt(np.mean((y[m]-p[m])**2)):.4f} (n={m.sum():,})")
print("(run_B trees on same val: B12 weighted 0.7846, per-k ~0.68-0.98)")
np.savez('/home/z/my-project/scripts/kalman_val.npz', y=y, p=p, k=kk)
print("Saved.", flush=True)
