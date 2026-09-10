"""
k0_lgbm_test.py — the clean k=0 model shootout we never ran.

k=0 rows = TWS_t VISIBLE (33.5% of test). Current linear model CV RMSE 0.6280.
This tests whether gradient boosting with richer features (neighbor TWS, spatial
context, era features) beats the linear model on the honest 2013-15 val window.

Features (all available at prediction time on test k=0 rows):
  base  : TWS_t, covs(t), covs(t+1) [what linear uses]
  +spatial: 3x3/5x5 box means of TWS_t field (from visible cells same month),
            box means of cov fields
  +cell : lat, lon, mu_c, clim covs, beta_c trend, t_abs
  +hist : TWS_t lag -1/-2/-3/-6/-12 (from train history; on test, only some lags exist
            before the test window starts — CAREFUL: only use lags that exist in-test)

Protocol: fit 2002-2012, eval 2013-15 unmasked rows (same rows as k0_smoothing_test).
"""
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

print("Loading...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1

fit = train[train['time'].dt.year <= 2012].copy()
val  = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()

test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
cal_mask_frac = test_raw.assign(m=test_raw['TWS_t_masked'].astype(bool), cm=test_raw['time'].dt.month).groupby('cm')['m'].mean()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5

# grid
lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)
cc_of = np.full((n_cells, 2), -1, dtype=np.int32)
for (la, lo), grp in train.groupby(['lat','lon']):
    cc = int(grp['cc'].iloc[0]); cc_grid[lat_i[la], lon_i[lo]] = cc; cc_of[cc] = (lat_i[la], lon_i[lo])
mask_g = cc_grid >= 0
def to_grid(v):
    g = np.full((NI, NJ), np.nan, dtype=np.float32); g[mask_g] = v[cc_grid[mask_g]]; return g
def from_grid(g):
    out = np.full(n_cells, np.nan, dtype=np.float32); out[cc_grid[mask_g]] = g[mask_g]; return out
def shift(g, di, dj):
    gg = np.roll(g, dj, axis=1)
    if di > 0: gg = np.vstack([np.full((di, NJ), np.nan, np.float32), gg[:-di]])
    if di < 0: gg = np.vstack([gg[-di:], np.full((-di, NJ), np.nan, np.float32)])
    return gg
def box(g, r):
    acc = np.zeros_like(g, dtype=np.float64); cnt = np.zeros_like(g, dtype=np.int32)
    for di in range(-r, r+1):
        for dj in range(-r, r+1):
            gg = shift(g, di, dj); ok = np.isfinite(gg)
            acc[ok] += gg[ok]; cnt[ok] += 1
    return np.where(cnt > 0, acc/np.maximum(cnt, 1), np.nan).astype(np.float32)

# infra
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
lalo = np.zeros((n_cells, 2), dtype=np.float32)
for cc in range(n_cells):
    i, j = cc_of[cc]; lalo[cc] = (lats[i], lons[j])
# trend slope
yms_fit = np.sort(fit['ym'].unique())
Fm = np.full((len(yms_fit), n_cells), np.nan, dtype=np.float32)
ym_to_i = {int(v):i for i,v in enumerate(yms_fit)}
Fm[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_fit], dtype=np.float64)
F64 = Fm.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0).astype(np.float32)

# per-month TWS field cache (train) for spatial features
def month_tws_field(df, m):
    sel = df['t_abs'] == m
    f = np.full(n_cells, np.nan, dtype=np.float32)
    f[df['cc'].values[sel.values]] = df['TWS_t'].values[sel.values]
    return f

def build_features(df, is_val):
    """returns X DataFrame aligned to df rows (k=0-eligible rows only handled by caller)"""
    n = len(df)
    cc = df['cc'].values; ta = df['t_abs'].values
    feats = {}
    feats['tws_dev'] = df['TWS_t'].values - mu_c[cc]
    for j, c in enumerate(COVS):
        feats[f'{c}_dev'] = df[c].values - clim[cc, j]
    # covs t+1 via merge
    nxt = df[['cc','t_abs']].copy(); nxt['t_abs'] = nxt['t_abs']+1
    key = df[['cc','t_abs']].copy()
    covn = df[['cc','t_abs']+COVS].copy()
    covn['t_abs'] = covn['t_abs'] + 1
    covn.columns = ['cc','t_abs'] + [c+'_nxt' for c in COVS]
    mg = key.merge(covn, on=['cc','t_abs'], how='left')
    for j, c in enumerate(COVS):
        feats[f'{c}_nxt_dev'] = mg[c+'_nxt'].values - clim[cc, j]
    # spatial: box means of TWS_t field and cov field for this month
    print(f"  building spatial features for {df['t_abs'].nunique()} months...", flush=True)
    tws_box3 = np.full(n, np.nan, dtype=np.float32); tws_box5 = np.full(n, np.nan, dtype=np.float32)
    sm_box3  = np.full(n, np.nan, dtype=np.float32)
    for m in np.sort(df['t_abs'].unique()):
        selm = (ta == m)
        f = month_tws_field(df, m)
        b3 = from_grid(box(to_grid(f), 1)); b5 = from_grid(box(to_grid(f), 2))
        # soil moisture field
        sm = np.full(n_cells, np.nan, dtype=np.float32)
        smv = df[COVS[-1]].values
        sm[cc[selm]] = smv[selm]
        sb3 = from_grid(box(to_grid(sm), 1))
        idx = np.where(selm)[0]
        tws_box3[idx] = b3[cc[idx]] - mu_c[cc[idx]]
        tws_box5[idx] = b5[cc[idx]] - mu_c[cc[idx]]
        sm_box3[idx] = sb3[cc[idx]] - clim[cc[idx], COVS.index('SOIL_MOISTURE_t')]
    feats['tws_box3'] = tws_box3; feats['tws_box5'] = tws_box5; feats['sm_box3'] = sm_box3
    # cell context
    feats['lat'] = lalo[cc, 0]; feats['lon'] = lalo[cc, 1]
    feats['mu_c'] = mu_c[cc]; feats['beta_c'] = beta_c[cc]
    feats['t_abs'] = ta.astype(np.float32)
    return pd.DataFrame(feats)

# ---- fit rows: all train k=0-eligible (TWS_t present) with target ----
print("Building FIT features...", flush=True)
ftr = fit[fit['target'].notna() & fit['TWS_t'].notna()].copy()
Xf_ = build_features(ftr, False)
yf_ = (ftr['target'].values - mu_c[ftr['cc'].values]).astype(np.float32)
yr_w = np.where(ftr['t_abs'].values//12 <= 2006, 1.0, np.where(ftr['t_abs'].values//12 <= 2009, 2.0, 3.0)).astype(np.float32)

print("Building VAL features...", flush=True)
vtr = val[val['target'].notna() & val['TWS_t'].notna() & (~val['masked'])].copy()
Xv_ = build_features(vtr, True)
yv_ = (vtr['target'].values - mu_c[vtr['cc'].values]).astype(np.float32)

base_cols = ['tws_dev'] + [f'{c}_dev' for c in COVS] + [f'{c}_nxt_dev' for c in COVS]
sp_cols  = ['tws_box3','tws_box5','sm_box3']
ctx_cols = ['lat','lon','mu_c','beta_c','t_abs']

def eval_masked_rows(yhat_dev, cc):
    pred = yhat_dev + mu_c[cc]
    return float(np.sqrt(np.mean((pred - vtr['target'].values.astype(np.float64))**2)))

# ---- reference: linear (reproduce ~0.628) ----
Xl = np.column_stack([Xf_[c].values for c in base_cols] + [np.ones(len(Xf_))]).astype(np.float32)
Xl = np.nan_to_num(Xl, nan=0.0)
sw = np.sqrt(yr_w)
coefL = np.linalg.solve((Xl*sw[:,None]).T@(Xl*sw[:,None]) + 1e-3*np.eye(Xl.shape[1]), (Xl*sw[:,None]).T@(yf_*sw))
Xv_l = np.nan_to_num(np.column_stack([Xv_[c].values for c in base_cols] + [np.ones(len(Xv_))]).astype(np.float32), nan=0.0)
r_lin = eval_masked_rows(Xv_l @ coefL, vtr['cc'].values)
print(f"\nlinear baseline (recomputed on this row set): {r_lin:.4f}  (n={len(vtr):,})")

# ---- LGBM: base only ----
def run_lgb(cols, tag, params=None):
    Xa = Xf_[cols].values.astype(np.float32)
    Xb = Xv_[cols].values.astype(np.float32)
    p = dict(objective='l2', n_estimators=600, learning_rate=0.05, num_leaves=63,
             min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
             n_jobs=8, verbose=-1)
    if params: p.update(params)
    m = lgb.LGBMRegressor(**p)
    m.fit(Xa, yf_, sample_weight=yr_w)
    yhat = m.predict(Xb)
    r = eval_masked_rows(yhat, vtr['cc'].values)
    print(f"  LGBM [{tag}]: {r:.4f}  ({r-r_lin:+.4f})")
    return r, yhat

print("\n=== LGBM k=0 shootout (fit 2002-2012, eval 2013-15 unmasked) ===")
r1, _ = run_lgb(base_cols, 'base only')
r2, _ = run_lgb(base_cols + sp_cols, 'base+spatial')
r3, _ = run_lgb(base_cols + sp_cols + ctx_cols, 'base+spatial+ctx')
r4, yhat_full = run_lgb(base_cols + sp_cols + ctx_cols, 'full, deeper', params=dict(n_estimators=1200, learning_rate=0.03, num_leaves=127, min_child_samples=100))

# ---- blend LGBM + linear ----
print("\n=== blend linear + LGBM ===")
for a in [0.3, 0.5, 0.7]:
    yb = (1-a)*Xv_l @ coefL + a*np.nan_to_num(yhat_full)
    r = eval_masked_rows(yb, vtr['cc'].values)
    print(f"  a_lgbm={a}: {r:.4f}  ({r-r_lin:+.4f})")

print("\nDone.", flush=True)
