"""
SUBMISSION V6: v5b architecture + upgraded k=0 stack + kernel/ensemble variants.

CV-validated components (sparse-anchor protocol + honest 2013-15 window):
  masked rows: W-pooled Kalman (box r=2) + Gaussian sigma-2.0 smoothing   [v5b, LB 0.7050]
  k=0 rows   : dual-linear + LGBM(base+spatial) blend 0.5 + sigma-1.0 smoothing
               (k0_stack_test.py: 0.6339 -> 0.6266 on k=0 rows)

Variants:
  v6a: v5b masked + upgraded k=0 stack                       (main bet)
  v6b: v6a but masked smoothing sigma=2.5                    (kernel-width readout)
  v6c: masked = avg(phi .74 Wpool gau2.0, phi .80 Wpool gau2.0) + upgraded k=0  (ensemble)
"""
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------------- load train ----------------
print("Loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_train = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((len(yms), n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_train[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# ---------------- grid ----------------
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
def gaussW(sig, r=4):
    return {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*sig*sig)))
            for di in range(-r,r+1) for dj in range(-r,r+1)
            if np.exp(-(di*di+dj*dj)/(2*sig*sig)) > 0.01}
BOX2 = {(di,dj): 1.0 for di in range(-2,3) for dj in range(-2,3)}
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)
GAU2 = gaussW(2.0); GAU25 = gaussW(2.5)

# ---------------- load test ----------------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
assert (test['cc'] >= 0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values

# ---------------- cov fields + anchors (same as v5) ----------------
Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W_raw = {int(m): cov_field[m] - S for m in all_m}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W_raw.items()}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"anchors: {anchors}")
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
Xs, ys = [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
w1, w2, w3 = map(float, w_)
print(f"D-tilde weights: {w1:.3f}/{w2:.3f}/{w3:.3f}")
def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

# ---------------- Kalman (W-pooled) ----------------
LAM_F = 0.84
def calibrate(Wd):
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a)
        ok = np.isfinite(Wd[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(Wd[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(Wd[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    return c_/(LAM_F*var_f), max(varz - c_*c_/(LAM_F*var_f), 1e-4), var_f

def kalman_predict(phi_f, Wd):
    H, R, var_f = calibrate(Wd)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for a in anchors:
        fa = AF[a] - Dtil(a)
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        sel0 = np.where((ta == a) & msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[cc_t[sel0]] + Dtil(a+1)[cc_t[sel0]] + phi_f*x[cc_t[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in Wd:
                wv = Wd[m]; okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((ta == m) & msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in Wd:
                wv = Wd[tm]; okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[cc_t[sel]] + Dtil(tm)[cc_t[sel]] + x2[cc_t[sel]]
    return pred

def smooth_rows(pred, rows, Wt):
    out = pred.copy()
    for m in np.unique(ta[rows & np.isfinite(pred)]):
        selm = np.where((ta == m) & rows & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[cc_t[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), Wt))
        out[selm] = sm[cc_t[selm]]
    return out

# ---------------- k=0: dual linear + LGBM(base+spatial) blend ----------------
print("k=0 dual linear...", flush=True)
Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
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
A_f = Xf[has_nxt_tr]*sw[has_nxt_tr,None]
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[has_nxt_tr]*sw[has_nxt_tr]))
colsR = [0,1,2,3,4,5,11]
A_r = Xf[:, colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))

Lt = test[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = test[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
Xt = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct],
    *[mgt[c].values - clim[cct, j] for j, c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j, c in enumerate(COVS)],
    np.ones(len(mgt))
]).astype(np.float32)
Xt = np.nan_to_num(Xt, nan=0.0)
k0_lin = np.full(len(test), np.nan, dtype=np.float64)
tw_ok = np.isfinite(mgt['TWS_t'].values)
f_full = has_nxt_te & tw_ok; f_red = (~has_nxt_te) & tw_ok
k0_lin[f_full] = mu_c[cct[f_full]] + Xt[f_full] @ coefF
k0_lin[f_red]  = mu_c[cct[f_red]] + Xt[f_red][:, colsR] @ coefR
bad = tw_ok & np.isnan(k0_lin)
if bad.any(): k0_lin[bad] = mu_c[cct][bad]

print("k=0 LGBM (base+spatial)...", flush=True)
def spatial_feats_test():
    n = len(test); out = np.full((n, 3), np.nan, dtype=np.float32)
    for m in np.sort(test['t_abs'].unique()):
        selm = np.where((ta == m) & np.isfinite(test['TWS_t'].values))[0]
        f = np.full(n_cells, np.nan, dtype=np.float32); f[cc_t[selm]] = test['TWS_t'].values[selm]
        sm = np.full(n_cells, np.nan, dtype=np.float32); sm[cc_t[selm]] = test[COVS[-1]].values[selm]
        b3 = from_grid(kpool(to_grid(f), {(0,0):1,(1,0):1,(-1,0):1,(0,1):1,(0,-1):1,(1,1):1,(1,-1):1,(-1,1):1,(-1,-1):1}))
        b5 = from_grid(kpool(to_grid(f), BOX2))
        sb3 = from_grid(kpool(to_grid(sm), {(0,0):1,(1,0):1,(-1,0):1,(0,1):1,(0,-1):1,(1,1):1,(1,-1):1,(-1,1):1,(-1,-1):1}))
        j = COVS.index('SOIL_MOISTURE_t')
        allm = np.where(ta == m)[0]
        out[allm, 0] = b3[cc_t[allm]] - mu_c[cc_t[allm]]
        out[allm, 1] = b5[cc_t[allm]] - mu_c[cc_t[allm]]
        out[allm, 2] = sb3[cc_t[allm]] - clim[cc_t[allm], j]
    return out
SPt = spatial_feats_test()
# fit LGBM on full train with matching features
ftr = train[train['target'].notna() & train['TWS_t'].notna()].copy()
SPf = np.full((len(ftr), 3), np.nan, dtype=np.float32)
fcc = ftr['cc'].values; fta_ = ftr['t_abs'].values
for m in np.sort(ftr['t_abs'].unique()):
    selm = np.where(fta_ == m)[0]
    f = np.full(n_cells, np.nan, dtype=np.float32); f[fcc[selm]] = ftr['TWS_t'].values[selm]
    sm = np.full(n_cells, np.nan, dtype=np.float32); sm[fcc[selm]] = ftr[COVS[-1]].values[selm]
    b3 = from_grid(kpool(to_grid(f), {(0,0):1,(1,0):1,(-1,0):1,(0,1):1,(0,-1):1,(1,1):1,(1,-1):1,(-1,1):1,(-1,-1):1}))
    b5 = from_grid(kpool(to_grid(f), BOX2))
    sb3 = from_grid(kpool(to_grid(sm), {(0,0):1,(1,0):1,(-1,0):1,(0,1):1,(0,-1):1,(1,1):1,(1,-1):1,(-1,1):1,(-1,-1):1}))
    j = COVS.index('SOIL_MOISTURE_t')
    SPf[selm, 0] = b3[fcc[selm]] - mu_c[fcc[selm]]
    SPf[selm, 1] = b5[fcc[selm]] - mu_c[fcc[selm]]
    SPf[selm, 2] = sb3[fcc[selm]] - clim[fcc[selm], j]
nxtf = ftr[['cc','t_abs']].copy(); nxtf['t_abs'] += 1
covn = ftr[['cc','t_abs']+COVS].copy(); covn['t_abs'] += 1
covn.columns = ['cc','t_abs'] + [c+'_nxt' for c in COVS]
mgf = ftr[['cc','t_abs']].merge(covn, on=['cc','t_abs'], how='left')
Xf_lgbm = np.column_stack(
    [ftr['TWS_t'].values - mu_c[fcc]] +
    [ftr[c].values - clim[fcc, j] for j, c in enumerate(COVS)] +
    [mgf[c+'_nxt'].values - clim[fcc, j] for j, c in enumerate(COVS)] +
    [SPf]).astype(np.float32)
yf_lgbm = (ftr['target'].values - mu_c[fcc]).astype(np.float32)
yr_w = np.where(ftr['t_abs'].values//12 <= 2006, 1.0, np.where(ftr['t_abs'].values//12 <= 2009, 2.0, 3.0)).astype(np.float32)
mlgb = lgb.LGBMRegressor(objective='l2', n_estimators=600, learning_rate=0.05, num_leaves=63,
                         min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         n_jobs=8, verbose=-1)
mlgb.fit(Xf_lgbm, yf_lgbm, sample_weight=yr_w)
nxtv = test[['cc','t_abs']].copy(); nxtv['t_abs'] += 1
covnv = test[['cc','t_abs']+COVS].copy(); covnv['t_abs'] += 1
covnv.columns = ['cc','t_abs'] + [c+'_nxt' for c in COVS]
mgv = test[['cc','t_abs']].merge(covnv, on=['cc','t_abs'], how='left')
Xt_lgbm = np.column_stack(
    [test['TWS_t'].values - mu_c[cc_t]] +
    [test[c].values - clim[cc_t, j] for j, c in enumerate(COVS)] +
    [mgv[c+'_nxt'].values - clim[cc_t, j] for j, c in enumerate(COVS)] +
    [SPt]).astype(np.float32)
k0_lgbm_dev = mlgb.predict(Xt_lgbm)
k0_lgbm = k0_lgbm_dev + mu_c[cc_t]

k0_blend = 0.5*k0_lin + 0.5*np.where(np.isfinite(k0_lgbm), k0_lgbm, k0_lin)
GAU10 = gaussW(1.0)
k0_final = smooth_rows(np.where(np.isfinite(k0_blend), k0_blend, k0_lin), ~msk & tw_ok, GAU10)
k0_final = np.where(np.isfinite(k0_final), k0_final, k0_lin)

# ---------------- assemble ----------------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm = ~msk

def assemble(masked_pred, tag, masked_Wt):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_final[unm]
    pred[msk] = masked_pred[msk]
    pred = np.where(np.isnan(pred), mu_c[cc_t], pred)
    pred = smooth_rows(pred, msk, masked_Wt)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} nan={int(out['Target'].isna().sum())}", flush=True)

print("\n--- v6a: v5b masked + upgraded k=0 ---")
p74 = kalman_predict(0.74, W_pool)
assemble(p74, 'v6a', GAU2)
print("--- v6b: sigma 2.5 masked smoothing ---")
assemble(p74, 'v6b', GAU25)
print("--- v6c: phi-ensemble masked + upgraded k=0 ---")
p80 = kalman_predict(0.80, W_pool)
p_ens = 0.5*p74 + 0.5*p80
assemble(p_ens, 'v6c', GAU2)

# sanity
for v in ['v6a','v6b','v6c']:
    f = pd.read_csv(f'{DL}/submission_{v}.csv')
    assert len(f) == 280961 and (f['ID'].values == sub['ID'].values).all() and f['Target'].notna().all() and np.isfinite(f['Target']).all()
    print(f"{v} verified: 280,961 rows, IDs match, no NaN, range [{f['Target'].min():.3f}, {f['Target'].max():.3f}]")
print("\nDone.")
