"""
MODEL B v2: State-space reconstruction of masked TWS.

Design (per cell, per masked chain):
  State: x_t = TWS anomaly (vs per-cell-month climatology)
  Transition: x_{m} = phi * x_{m-1} + w  (phi per cell, shrunk)
  Observations at each month m: SPEI_01_an, SOIL_M_an  (+ SPEI_03/06/12 as level obs at t)
  Anchor: x_anchor known EXACTLY (variance 0)

  Estimate x_t via learned-weight combination of:
    - decayed anchor anomaly: phi^(t-anchor) * anom_anchor
    - per-month observation anomalies with AR-decayed weights
  Then target = clim_{t+1} + phi * x_t (+ direct SPEI level correction)

Implementation: linear model with engineered features (Kalman-equivalent for
linear-Gaussian case with learned global weights per k):
  anom_target = a_k * anom_anchor
              + b_k * sum_j phi^(t-j) * SPEI_01_an(j)      for j in (anchor, t]
              + c_k * sum_j phi^(t-j) * SOIL_M_an(j)
              + d_k * SPEI_12_an(t) + e_k * SPEI_06_an(t) + f_k * SPEI_03_an(t)
  fit per k on pre-2013, validate 2013+.
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
VAL_START = pd.Timestamp('2013-01-01')
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
COVARS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVARS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target'] + COVARS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = int(train['cc'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
train['m'] = train['time'].dt.month.astype('int8')
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
train['midx'] = ym_idx.astype('int32')
n_months = len(ym_codes)
ym_to_midx = {int(v): i for i, v in enumerate(ym_codes)}
print(f"Loaded {len(train):,} rows", flush=True)

# ---------- climatologies (pre-2013) for TWS and covariates ----------
pre = train[train['time'] < VAL_START]


def make_clim(col):
    g = pre.groupby(['cc', 'm'])[col].mean()
    arr = np.full((n_cells, 13), np.nan, dtype=np.float32)
    for (cc, m), v in g.items():
        arr[cc, m] = v
    gm = pre.groupby('m')[col].mean()
    fb = np.array([gm.get(m, 0.0) for m in range(13)], dtype=np.float32)
    return np.where(np.isnan(arr), fb[None, :], arr)


tws_clim = make_clim('TWS_t')
cov_clim = {c: make_clim(c) for c in COVARS}

# monthly anomaly matrices (months x cells)
M_anom = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_tws = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_cov_anom = {c: np.full((n_months, n_cells), np.nan, dtype=np.float32) for c in COVARS}
M_tws[ym_idx, train['cc'].values] = train['TWS_t'].values
for i, (cc, m) in enumerate(zip(train['cc'].values, train['m'].values)):
    pass
# vectorized climatology attach
train['tws_anom'] = train['TWS_t'].values - tws_clim[train['cc'].values, train['m'].values]
M_anom[ym_idx, train['cc'].values] = train['tws_anom'].values
for c in COVARS:
    av = train[c].values - cov_clim[c][train['cc'].values, train['m'].values]
    M_cov_anom[c][ym_idx, train['cc'].values] = av

# ---------- per-cell phi (shrunk, pre-2013) ----------
g = pre.groupby('cc')
n_obs = g.size().astype(float)
sx = g['TWS_t'].sum(); sy = g['target'].sum()
sxx = g.apply(lambda d: (d['TWS_t'] ** 2).sum(), include_groups=False)
sxy = g.apply(lambda d: (d['TWS_t'] * d['target']).sum(), include_groups=False)
n = n_obs.reindex(range(n_cells)).fillna(0)
sx = sx.reindex(range(n_cells)).fillna(0); sy = sy.reindex(range(n_cells)).fillna(0)
sxx = sxx.reindex(range(n_cells)).fillna(0); sxy = sxy.reindex(range(n_cells)).fillna(0)
den = (n * sxx - sx.pow(2)).replace(0, np.nan)
phi_raw = ((n * sxy - sx * sy) / den).values
cells = train[['cc', 'lat', 'lon']].drop_duplicates().sort_values('cc')
lat_arr = cells['lat'].values
band = np.clip(np.round(lat_arr / 6) * 6, -84, 84)
phi_shrunk = np.full(n_cells, 0.76, dtype=np.float32)
for b in np.unique(band):
    msk = band == b
    ph = phi_raw[msk]
    ok = ~np.isnan(ph)
    if ok.sum() >= 5:
        v_within = np.var(ph[ok])
        se2 = np.mean((1 - ph[ok] ** 2) ** 2 / np.maximum(n.values[msk][ok], 2))
        tau2 = max(v_within - se2, 0.01)
        w = tau2 / (tau2 + se2)
        phi_shrunk[msk] = w * np.nanmean(ph) + (1 - w) * 0.76
phi_shrunk = np.clip(phi_shrunk, 0.2, 0.995)
print(f"phi: mean={phi_shrunk.mean():.3f}", flush=True)

# ---------- build examples: for each row (val or train), features for its k ----------
row_cc = train['cc'].values
row_midx = ym_idx
row_year = train['time'].dt.year.values
row_month = train['time'].dt.month.values
row_ym = train['ym'].values
row_t1m = ((train['time'].dt.month.values % 12) + 1).astype('int8')
row_target_anom = train['target'].values - tws_clim[row_cc, row_t1m]

# cumsum matrices for AR-weighted observation sums:
# feature per month j: phi^(months between j and t) * obs_anom(j)
# sum over j in (anchor, t]. Since phi varies per cell, exact per-cell weighting is heavy;
# approximate with global mean phi for weights, keep per-cell phi for anchor decay.
PHI_G = 0.75
W = {}
for c in ['SPEI_01_t', 'SOIL_MOISTURE_t']:
    V = M_cov_anom[c].copy()
    nanmask = np.isnan(V)
    # weight month j by PHI_G^(rm - j): build via convolution over month axis
    VW = np.where(nanmask, 0, V) * 0.0
    # cumulative trick: weighted sum over last k months with geometric weights
    # S_k(t) = sum_{j=t-k+1..t} PHI^(t-j) * V(j) = V(t) + PHI * S_{k-1}(t-1)
    S = np.zeros((n_months, n_cells), dtype=np.float32)
    CNT = np.zeros((n_months, n_cells), dtype=np.int32)
    Vz = np.where(nanmask, 0, V)
    Cz = (~nanmask).astype(np.int32)
    for i in range(1, n_months):
        S[i] = Vz[i] + PHI_G * S[i - 1]
        CNT[i] = Cz[i] + CNT[i - 1]  # not exact for weights but fine for completeness check
    W[c] = (S, Cz)

def anchor_midx(ym_arr, k):
    tot = (ym_arr // 100) * 12 + (ym_arr % 100) - 1 - k
    aym = (tot // 12) * 100 + tot % 12 + 1
    return np.array([ym_to_midx.get(int(v), -1) for v in aym], dtype=np.int32)


def build_feats(idxs, k):
    """Features for rows idxs at lag k (months since anchor)."""
    cc = row_cc[idxs]
    rm = row_midx[idxs]
    amap = anchor_midx(row_ym[idxs], k)
    has = amap >= 0
    amap_s = np.where(has, amap, 0)
    anchor_anom = M_anom[amap_s, cc]
    ok = has & ~np.isnan(anchor_anom)
    out_idx = idxs[ok]
    cc = cc[ok]; rm = rm[ok]; amap_ok = amap_s[ok]; anchor_anom = anchor_anom[ok]
    # anchor decayed: phi^(k+1) * anchor_anom (for target at t+1, anchor at t-k)
    f_decay = (phi_shrunk[cc] ** (k + 1)) * anchor_anom
    # AR-weighted observation sums (already includes month t's observation):
    s_spei, cz_spei = W['SPEI_01_t']
    s_soil, cz_soil = W['SOIL_MOISTURE_t']
    # sums from anchor+1..t: S(t) - PHI^? * S(anchor)... need sum over (anchor, t]:
    # S(t) = sum_{j<=t} PHI^(t-j)*V(j) (running). Sum over (anchor,t] with weights PHI^(t-j):
    # = S(t) - PHI^(t-anchor) * S(anchor)
    f_spei = s_spei[rm, cc] - (PHI_G ** k) * s_spei[amap_ok, cc]
    f_soil = s_soil[rm, cc] - (PHI_G ** k) * s_soil[amap_ok, cc]
    # level observations at t (anomalies)
    f_s12 = M_cov_anom['SPEI_12_t'][rm, cc]
    f_s06 = M_cov_anom['SPEI_06_t'][rm, cc]
    f_s03 = M_cov_anom['SPEI_03_t'][rm, cc]
    f_sm = M_cov_anom['SOIL_MOISTURE_t'][rm, cc]
    X = np.column_stack([f_decay, f_spei, f_soil, f_s12, f_s06, f_s03, f_sm]).astype(np.float32)
    y = row_target_anom[out_idx]
    return X, y, out_idx


# ---------- fit per-k on pre-2013, validate 2013+ ----------
print("\n=== STATE-SPACE RECONSTRUCTION: fit pre-2013, validate 2013+ ===", flush=True)
pre_mask = (train['time'] < VAL_START).values
base = np.where(pre_mask)[0]
rows_mask = ~pre_mask
times_w = sorted(train.loc[rows_mask, 'time'].unique())
anchor_times = times_w[::2]

results = {}
val_y, val_p, val_k = [], [], []
for k in range(1, 7):
    Xk, yk, _ = build_feats(base, k)
    coef, *_ = np.linalg.lstsq(np.column_stack([Xk, np.ones(len(Xk))]).astype(np.float64),
                               yk.astype(np.float64), rcond=None)
    # validation rows for this k
    idxs_list = []
    for t in anchor_times:
        t = pd.Timestamp(t)
        tot = t.year * 12 + (t.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in ym_to_midx:
            continue
        sel = np.where(rows_mask & (row_ym == tym))[0]
        if len(sel):
            idxs_list.append(sel)
    if not idxs_list:
        continue
    vidx = np.concatenate(idxs_list)
    Xv, yv, _ = build_feats(vidx, k)
    pv = np.column_stack([Xv, np.ones(len(Xv))]).astype(np.float64) @ coef
    val_y.append(yv); val_p.append(pv); val_k.append(np.full(len(yv), k))
    results[k] = coef

y_all = np.concatenate(val_y); p_all = np.concatenate(val_p); k_all = np.concatenate(val_k)
w = np.array([K_WEIGHTS.get(int(k), 1) for k in k_all], dtype=np.float64)
print(f"weighted RMSE (anomaly space, k=1..6): {np.sqrt(np.average((y_all-p_all)**2, weights=w)):.4f}")
print(f"anomaly std: {np.std(y_all):.4f}")
for k in range(1, 7):
    m = k_all == k
    if m.sum():
        print(f"  k={k}: {np.sqrt(np.mean((y_all[m]-p_all[m])**2)):.4f} (n={m.sum():,})")
print(f"\ncoef (k=3): decay={results[3][0]:.3f} spei_w={results[3][1]:.3f} soil_w={results[3][2]:.3f} "
      f"s12={results[3][3]:.3f} s06={results[3][4]:.3f} s03={results[3][5]:.3f} sm={results[3][6]:.3f}")
np.save('/home/z/my-project/scripts/bv2_coefs.npy', np.array([results[k] for k in range(1, 7)]))
print("Saved.", flush=True)
