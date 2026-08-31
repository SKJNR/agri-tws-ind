"""
k0_stack_test.py — final k=0 stack: linear-dual + LGBM(base+spatial) blend + sigma-1.5 smoothing.
Quick CV verification, reusing the dual linear spec (full/reduced) from submission_v5.py.
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

lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)
for (la, lo), grp in train.groupby(['lat','lon']):
    cc_grid[lat_i[la], lon_i[lo]] = int(grp['cc'].iloc[0])
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
def gaussW(sig, r=4):
    return {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*sig*sig)))
            for di in range(-r,r+1) for dj in range(-r,r+1)
            if np.exp(-(di*di+dj*dj)/(2*sig*sig)) > 0.01}
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

def dual_linear(train_df, target_df):
    """returns predict function for target_df rows"""
    Lk = train_df[['cc','t_abs','TWS_t','target']+COVS].copy()
    Lk['t_next'] = Lk['t_abs'] + 1
    Rk = train_df[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
    mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
    mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
    has_nxt = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
    ccm = mgk['cc'].values; yr = mgk['t_abs'].values // 12
    Xf = np.column_stack([
        mgk['TWS_t'].values - mu_c[ccm],
        *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
        *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
        np.ones(len(mgk))
    ]).astype(np.float32)
    yA = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
    Xf = np.nan_to_num(Xf, nan=0.0)
    w_rec = np.where(yr <= 2006, 1.0, np.where(yr <= 2009, 2.0, 3.0)).astype(np.float32)
    sw = np.sqrt(w_rec)
    A_f = Xf[has_nxt]*sw[has_nxt,None]
    coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[has_nxt]*sw[has_nxt]))
    colsR = [0,1,2,3,4,5,11]
    A_r = Xf[:, colsR]*sw[:,None]
    coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))

    Lt = target_df[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
    Rt = target_df[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
    mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
    has_nxt_t = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
    cct = mgt['cc'].values
    Xt = np.column_stack([
        mgt['TWS_t'].values - mu_c[cct],
        *[mgt[c].values - clim[cct, j] for j, c in enumerate(COVS)],
        *[mgt[c+'_nxt'].values - clim[cct, j] for j, c in enumerate(COVS)],
        np.ones(len(mgt))
    ]).astype(np.float32)
    Xt = np.nan_to_num(Xt, nan=0.0)
    k0 = np.full(len(mgt), np.nan, dtype=np.float64)
    tw_ok = np.isfinite(mgt['TWS_t'].values)
    f_full = has_nxt_t & tw_ok; f_red = (~has_nxt_t) & tw_ok
    k0[f_full] = mu_c[cct[f_full]] + Xt[f_full] @ coefF
    k0[f_red]  = mu_c[cct[f_red]] + Xt[f_red][:, colsR] @ coefR
    bad = tw_ok & np.isnan(k0)
    if bad.any(): k0[bad] = mu_c[cct][bad]
    return k0, cct

def spatial_feats(df):
    n = len(df); cc = df['cc'].values; ta = df['t_abs'].values
    out = np.full((n, 3), np.nan, dtype=np.float32)
    for m in np.sort(df['t_abs'].unique()):
        selm = np.where(ta == m)[0]
        f = np.full(n_cells, np.nan, dtype=np.float32); f[cc[selm]] = df['TWS_t'].values[selm]
        sm = np.full(n_cells, np.nan, dtype=np.float32); sm[cc[selm]] = df[COVS[-1]].values[selm]
        b3 = from_grid(box(to_grid(f), 1)); b5 = from_grid(box(to_grid(f), 2)); sb3 = from_grid(box(to_grid(sm), 1))
        j = COVS.index('SOIL_MOISTURE_t')
        out[selm, 0] = b3[cc[selm]] - mu_c[cc[selm]]
        out[selm, 1] = b5[cc[selm]] - mu_c[cc[selm]]
        out[selm, 2] = sb3[cc[selm]] - clim[cc[selm], j]
    return out

# ---- val rows (unmasked, k=0-eligible) ----
vtr = val[val['target'].notna() & val['TWS_t'].notna() & (~val['masked'])].copy()
tv = vtr['target'].values.astype(np.float64)
vcc = vtr['cc'].values; vta = vtr['t_abs'].values

# 1) dual linear
lin, _ = dual_linear(fit, vtr)
ok = np.isfinite(lin)
r_lin = float(np.sqrt(np.mean((lin[ok]-tv[ok])**2)))
print(f"dual linear: {r_lin:.4f}")

# 2) LGBM base+spatial
ftr = fit[fit['target'].notna() & fit['TWS_t'].notna()].copy()
print("building spatial feats (fit)...", flush=True)
SPf = spatial_feats(ftr)
SPv = spatial_feats(vtr)
base_cols = ['TWS_t'] + COVS
Xf_l = np.column_stack([ftr[c].values - (mu_c[ftr['cc'].values] if c=='TWS_t' else clim[ftr['cc'].values, COVS.index(c)]) for c in base_cols] + [ftr[c+'_nxt'].values if False else np.zeros(len(ftr)) for c in []]).astype(np.float32)
# simpler: raw dev features incl covs nxt via merge
nxtf = ftr[['cc','t_abs']].copy(); nxtf['t_abs'] += 1
covn = ftr[['cc','t_abs']+COVS].copy(); covn['t_abs'] += 1
covn.columns = ['cc','t_abs'] + [c+'_nxt' for c in COVS]
mgf = ftr[['cc','t_abs']].merge(covn, on=['cc','t_abs'], how='left')
nxtv = vtr[['cc','t_abs']].copy(); nxtv['t_abs'] += 1
covnv = vtr[['cc','t_abs']+COVS].copy(); covnv['t_abs'] += 1
covnv.columns = ['cc','t_abs'] + [c+'_nxt' for c in COVS]
mgv = vtr[['cc','t_abs']].merge(covnv, on=['cc','t_abs'], how='left')
Xf_l = np.column_stack(
    [ftr['TWS_t'].values - mu_c[ftr['cc'].values]] +
    [ftr[c].values - clim[ftr['cc'].values, j] for j, c in enumerate(COVS)] +
    [mgf[c+'_nxt'].values - clim[ftr['cc'].values, j] for j, c in enumerate(COVS)] +
    [SPf]).astype(np.float32)
Xv_l = np.column_stack(
    [vtr['TWS_t'].values - mu_c[vcc]] +
    [vtr[c].values - clim[vcc, j] for j, c in enumerate(COVS)] +
    [mgv[c+'_nxt'].values - clim[vcc, j] for j, c in enumerate(COVS)] +
    [SPv]).astype(np.float32)
yf_ = (ftr['target'].values - mu_c[ftr['cc'].values]).astype(np.float32)
yr_w = np.where(ftr['t_abs'].values//12 <= 2006, 1.0, np.where(ftr['t_abs'].values//12 <= 2009, 2.0, 3.0)).astype(np.float32)
m = lgb.LGBMRegressor(objective='l2', n_estimators=600, learning_rate=0.05, num_leaves=63,
                      min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      n_jobs=8, verbose=-1)
m.fit(Xf_l, yf_, sample_weight=yr_w)
lgb_pred = m.predict(Xv_l) + mu_c[vcc]
r_lgb = float(np.sqrt(np.mean((lgb_pred-tv)**2)))
print(f"LGBM base+spatial: {r_lgb:.4f}")

# 3) blends
best = (r_lin, None)
for a in [0.3, 0.5, 0.7]:
    yb = (1-a)*lin + a*lgb_pred
    r = float(np.sqrt(np.mean((yb-tv)**2)))
    if r < best[0]: best = (r, a)
    print(f"blend a={a}: {r:.4f}")
a_star = best[1] if best[1] else 0.0
yb = (1-a_star)*lin + a_star*lgb_pred

# 4) + smoothing sigma 1.5 on best blend
def smooth_rows(pred, sigma):
    Wt = gaussW(sigma)
    out = pred.copy()
    for mm in np.unique(vta):
        selm = np.where((vta == mm) & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[vcc[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), Wt))
        out[selm] = sm[vcc[selm]]
    return out
for sig in [1.0, 1.5, 2.0]:
    ps = smooth_rows(yb, sig)
    okk = np.isfinite(ps)
    r = float(np.sqrt(np.mean((ps[okk]-tv[okk])**2)))
    print(f"blend(a*={a_star}) + smooth s={sig}: {r:.4f}  (vs dual-linear {r_lin:.4f}: {r-r_lin:+.4f})")

print("\nDone.", flush=True)
