#!/usr/bin/env python3
"""
THE BREAKTHROUGH EXPERIMENT: can we predict the innovation field eta?

eta(c,t) is 97% spatially smooth (F3). Covariates observe the fast state at
r~0.5. Our Kalman extracts them via a GLOBAL SCALAR observation model
(H=0.24-0.31). Hypothesis: a spatially-DENOISED, LOCALLY-fitted regression
on covariate innovations recovers eta far better.

Tests, all on train (leak-free):
  T1: eta ~ raw cov innovations, global linear           (Kalman-equivalent)
  T2: eta ~ SMOOTHED cov innovations, global linear      (denoising gain)
  T3: eta ~ SMOOTHED cov innovations + levels, per-block (localization gain)
  T4: oracle check — eta ~ smoothed eta itself (what R^2 is achievable at all
      given spatial smoothing ceiling)
Reports implied masked-RMSE math for each R^2 level.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

DATA = '/home/z/my-project/data'
rng = np.random.default_rng(11)
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1
def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
train['cc'] = ckey(train['lat'], train['lon'])

grid = train[['cc', 'lat', 'lon']].drop_duplicates('cc').set_index('cc').sort_index()
mu_c = train.groupby('cc')['TWS_t'].mean()
lat = grid['lat'].values; lon = grid['lon'].values
n = len(grid)

# ---- sparse gaussian smoother (sigma = 2 deg) ----
pts = np.column_stack([lat, lon])
tree = cKDTree(pts)
pairs = tree.query_pairs(r=6.0, output_type='ndarray')
dist = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
sigma_s = 2.0
w = np.exp(-dist**2 / (2 * sigma_s**2))
# add self
ii = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
jj = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
ww = np.concatenate([w, w, np.ones(n)])
Sm = coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
rs = np.asarray(Sm.sum(axis=1)).ravel()
Sm = coo_matrix((ww / rs[ii], (ii, jj)), shape=(n, n)).tocsr()

# ---- pivots ----
piv = train.pivot_table(index='cc', columns='t_abs', values='TWS_t').reindex(grid.index)
anom = piv.sub(mu_c.reindex(grid.index), axis=0)
t_cols = anom.columns.values
cov_pivs = {c: train.pivot_table(index='cc', columns='t_abs', values=c).reindex(grid.index) for c in COVS}

# per-cell detrend of anomaly (remove linear trend)
t_norm = t_cols.astype(float)
X = np.vstack([np.ones_like(t_norm), t_norm - t_norm.mean()]).T
A = anom.values
trend = np.zeros((n, 2))
for c in range(n):
    y = A[c]; m = ~np.isnan(y)
    if m.sum() > 24:
        trend[c] = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
trend_field = X @ trend.T  # (months, cells)
y_dt = A - trend_field.T   # detrended anomaly (cells, months)
print(f"detrended anomaly: std {np.nanstd(y_dt):.3f}")

PHI = 0.82

# ---- gather transitions ----
trans = []
for i in range(len(t_cols) - 1):
    if t_cols[i + 1] - t_cols[i] == 1:
        trans.append(i)
print(f"consecutive transitions available: {len(trans)}")
sample = rng.choice(trans, size=min(90, len(trans)), replace=False)

# ---- build innovation matrices ----
eta_list = []
Xcov_raw = []     # raw cov innovations
Xcov_sm = []      # smoothed cov innovations
Xcov_sm_lvl = []  # smoothed cov innovations + levels
block_id = []     # spatial block per cell (10x10 deg)
blk = (np.floor(lat / 10).astype(int) * 1000 + np.floor(lon / 10).astype(int))
blk_codes, blk_idx = np.unique(blk, return_inverse=True)
print(f"spatial blocks (10x10 deg): {len(blk_codes)}")

for i in sample:
    t0, t1 = t_cols[i], t_cols[i + 1]
    y0 = y_dt[:, i]; y1 = y_dt[:, i + 1]
    cov0 = np.column_stack([cov_pivs[c][t0].values if t0 in cov_pivs[c].columns else np.nan for c in COVS])
    cov1 = np.column_stack([cov_pivs[c][t1].values if t1 in cov_pivs[c].columns else np.nan for c in COVS])
    m = ~np.isnan(y0) & ~np.isnan(y1) & ~np.isnan(cov0).any(1) & ~np.isnan(cov1).any(1)
    if m.sum() < 8000:
        continue
    eta = y1[m] - PHI * y0[m]
    # remove cross-cell mean of eta (global level not the target here)
    eta = eta - eta.mean()
    # raw cov innovations
    cinn_raw = cov1[m] - PHI * cov0[m]
    # smoothed cov fields then innovations
    c0_sm = np.column_stack([Sm @ np.nan_to_num(cov_pivs[c][t0].values) for c in COVS])[m]
    c1_sm = np.column_stack([Sm @ np.nan_to_num(cov_pivs[c][t1].values) for c in COVS])[m]
    cinn_sm = c1_sm - PHI * c0_sm
    # full feature: innovations + levels of smoothed covs at t1
    feat = np.column_stack([cinn_sm, c1_sm])
    eta_list.append(eta)
    Xcov_raw.append(cinn_raw - cinn_raw.mean(0))
    Xcov_sm.append(cinn_sm - cinn_sm.mean(0))
    Xcov_sm_lvl.append(feat - feat.mean(0))
    block_id.append(blk_idx[m])

eta_all = np.concatenate(eta_list)
Xraw_all = np.vstack(Xcov_raw)
Xsm_all = np.vstack(Xcov_sm)
XsmL_all = np.vstack(Xcov_sm_lvl)
blk_all = np.concatenate(block_id)
N = len(eta_all)
print(f"pooled eta samples: {N:,}; eta std {eta_all.std():.3f}")

def pooled_r2(Xmat, y, groups=None, n_splits=6, return_fitted=False):
    """R^2 with by-month-block splits (no leakage across time); optional per-group fits."""
    # month-block split: each transition is a block; use contiguous groups
    month_block = np.concatenate([np.full(len(e), k) for k, e in enumerate(eta_list)])
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y)
    for tr, te in gkf.split(Xmat, y, groups=month_block):
        if groups is None:
            mdl = Ridge(alpha=1.0).fit(Xmat[tr], y[tr])
            preds[te] = mdl.predict(Xmat[te])
        else:
            # per-group (spatial block) ridge, pooled where too few
            preds[te] = 0
            for g in np.unique(groups[te]):
                mte = te[groups[te] == g]
                mtr = tr[groups[tr] == g]
                if mtr.sum() < 200:
                    mtr = tr
                mdl = Ridge(alpha=1.0).fit(Xmat[mtr], y[mtr])
                preds[mte] = mdl.predict(Xmat[mte])
    r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2

print("\n===== T1: eta ~ RAW cov innovations, global (Kalman-equivalent) =====")
r2_t1 = pooled_r2(Xraw_all, eta_all)
print(f"  pooled out-of-sample R^2 = {r2_t1:.3f}")

print("\n===== T2: eta ~ SMOOTHED cov innovations, global =====")
r2_t2 = pooled_r2(Xsm_all, eta_all)
print(f"  pooled out-of-sample R^2 = {r2_t2:.3f}")

print("\n===== T3: eta ~ SMOOTHED cov innovations + levels, per-block =====")
r2_t3 = pooled_r2(XsmL_all, eta_all, groups=blk_all)
print(f"  pooled out-of-sample R^2 = {r2_t3:.3f}")

print("\n===== T3b: eta ~ SMOOTHED innovations only, per-block =====")
r2_t3b = pooled_r2(Xsm_all, eta_all, groups=blk_all)
print(f"  pooled out-of-sample R^2 = {r2_t3b:.3f}")

# ---- T4 oracle: spatial smoothing ceiling on eta itself ----
print("\n===== T4: oracle — smoothed eta vs eta (spatial smoothing ceiling) =====")
sm_r2s = []
for k, i in enumerate(sample[:20]):
    t0, t1 = t_cols[i], t_cols[i + 1]
    y0 = y_dt[:, i]; y1 = y_dt[:, i + 1]
    m = ~np.isnan(y0) & ~np.isnan(y1)
    eta_f = np.full(n, np.nan)
    eta_f[m] = y1[m] - PHI * y0[m]
    eta_sm = Sm @ np.nan_to_num(eta_f)
    ok = m
    r2m = 1 - np.nansum((eta_f[ok] - eta_sm[ok]) ** 2) / np.nansum((eta_f[ok] - np.nanmean(eta_f[ok])) ** 2)
    sm_r2s.append(r2m)
print(f"  smoothing-explained variance of eta fields: mean R^2 = {np.mean(sm_r2s):.3f}")

# ---- implied gains ----
print("\n===== IMPLIED MASKED-RMSE MATH (h->infinity, D-tilde 0.45 & 0.35) =====")
var_f = 0.50; phi = PHI
q = var_f * (1 - phi**2)
for r2 in [0.0, r2_t1, r2_t2, r2_t3, 0.4, 0.5, 0.6]:
    q_eff = q * (1 - r2)
    fast_inf = np.sqrt(q_eff / (1 - phi**2))  # unconditional error with info each month
    for dtil in [0.45, 0.35]:
        masked = np.sqrt(dtil**2 + fast_inf**2)
        lb = np.sqrt(0.6652 * masked**2 + 0.3348 * 0.640**2)
        print(f"  eta R^2={r2:.3f}, Dtil={dtil}: fast~{fast_inf:.3f} -> masked {masked:.3f} -> LB {lb:.4f}")
    print()
