"""
SUBMISSION V10 — the combination that was never built.

Board state (RMSE, lower = better):
  v6c = 0.703041139 (BEST, rank 46): W-pool(box2) Kalman phi-ens{.74,.80} + gau2.0 masked
        smoothing + k0 dual-linear/LGBM(base+spatial) blend + gau1.0 k0 smoothing
  v8a = 0.70702259              : v2b-core + obs-PC-dn + percell H/R + k0 two-comp blend
  The v8 line FORGOT the v5/v6 spatial stack; the v6 line never got the v8 axes.
  v10 grafts v8's validated axes onto the v6c base.

Variants (all include v9's Dhat bugfix: 7 zero-anchor cells -> Dhat=0 fill):
  ctrl : exact v6c rebuild (offline ONLY — diffed vs submitted v6c CSV for fidelity)
  v10a : v6c + masked obs = dn(W_pool)  (obs-PC-denoise axis; global H/R on raw pooled)
  v10b : v6c + k0 = v8 two-comp D-aware blend (+gau1.0 smoothing)   (k0 axis)
  v10c : v6c + both axes
"""
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------------- load train (v6 verbatim) ----------------
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

# PC-denoise basis (v9 verbatim): top-200 right sing. vecs of detrended centered anomaly
full = ~np.isnan(F).any(axis=0)
TDm = t_abs_train[:, None] - tbar_c[None, :]
A_dt = F64 - mu_c[None, :] - TDm*beta_c[None, :]
A_dtf = A_dt[:, full]; A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(A_dtf, full_matrices=False)
V = Vt[:200].T.astype(np.float32)
def dn(field):
    out = field.copy(); x = field[full]; okx = np.isfinite(x)
    out[full] = np.where(okx, V@(V.T@np.where(okx, x, 0.0)), x)
    return out

# ---------------- grid utils (v6 verbatim) ----------------
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
GAU2 = gaussW(2.0); GAU10 = gaussW(1.0)

# ---------------- load test (v6 verbatim) ----------------
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

# ---------------- cov fields + anchors (v6 verbatim) ----------------
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
W_pool_dn = {m: dn(v) for m, v in W_pool.items()}          # v10 axis: PC-denoised pooled obs

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"anchors: {anchors}")
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

# Dhat + v9 BUGFIX (7 zero-anchor cells -> 0 fill instead of NaN)
Dhat_raw = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
n_zero_anchor = int((~np.isfinite(Dhat_raw)).sum())
Dhat = np.where(np.isfinite(Dhat_raw), Dhat_raw, 0.0).astype(np.float32)
print(f"zero-anchor cells filled: {n_zero_anchor}")
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

# ---------------- Kalman (v6 verbatim; obs dict is a parameter) ----------------
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

# ---------------- k=0 model A: v6 stack (verbatim) ----------------
print("k=0 A: v6 dual linear...", flush=True)
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

print("k=0 A: v6 LGBM (base+spatial)...", flush=True)
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

k0_blendA = 0.5*k0_lin + 0.5*np.where(np.isfinite(k0_lgbm), k0_lgbm, k0_lin)
k0_A = smooth_rows(np.where(np.isfinite(k0_blendA), k0_blendA, k0_lin), ~msk & tw_ok, GAU10)
k0_A = np.where(np.isfinite(k0_A), k0_A, k0_lin)

# ---------------- k=0 model B: v8 two-comp D-aware blend (v9 verbatim + bugfixed Dhat) ----------------
print("k=0 B: v8 two-comp + LGBM blend...", flush=True)
def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr, dtype=np.float64)-tbar_c[cc_arr])*beta_c[cc_arr]).astype(np.float32)
slow0 = trendex_vec(mgk['t_abs'].values, ccm); slow1 = trendex_vec(mgk['t_next'].values, ccm)
Xlin = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm] - slow0,
    *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
yB = (mgk['target'].values - mu_c[ccm] - slow1).astype(np.float32)
Xlin = np.nan_to_num(Xlin, nan=0.0)
w_recB = np.where(yr <= 2006, 1.0, np.where(yr <= 2009, 1.5, 2.0)).astype(np.float32)
swB = np.sqrt(w_recB); selF = has_nxt_tr
A_fB = Xlin[selF]*swB[selF,None]
coefF_B = np.linalg.solve(A_fB.T@A_fB + 1e-3*np.eye(12), A_fB.T@(yB[selF]*swB[selF]))
A_rB = Xlin[:, colsR]*swB[:,None]
coefR_B = np.linalg.solve(A_rB.T@A_rB + 1e-3*np.eye(7), A_rB.T@(yB*swB))
print(f"k0-B linear coef[FAST_est]={coefF_B[0]:.4f}")

lat_cc = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_arr = lat_cc['lat'].values.astype(np.float32); lon_arr = lat_cc['lon'].values.astype(np.float32)
mon_tr = (mgk['t_abs'].values % 12) + 1
XLB = np.column_stack([Xlin[:,0], slow1, Xlin[:,1:11],
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm], lat_arr[ccm], lon_arr[ccm]]).astype(np.float32)
okY = np.isfinite(yB)
params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=8)
ds  = lgb.Dataset(XLB[selF], label=yB[selF], weight=w_recB[selF])
dsr = lgb.Dataset(XLB[~selF & okY], label=yB[~selF & okY], weight=w_recB[~selF & okY])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, dsr, num_boost_round=400)
print("k0-B LGBM trained.")

Dt0 = np.zeros(len(mgt), dtype=np.float32); Dt1 = np.zeros(len(mgt), dtype=np.float32)
uniq_m = np.unique(mgt['t_abs'].values)
Dcache = {int(m): (Dtil(int(m)), Dtil(int(m)+1)) for m in uniq_m}
for i, (m, tn) in enumerate(zip(mgt['t_abs'].values, mgt['t_next'].values)):
    d0, d1 = Dcache[int(m)]
    Dt0[i] = d0[cct[i]]; Dt1[i] = d1[cct[i]]
Xv = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct] - Dt0,
    *[mgt[c].values - clim[cct, j] for j, c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j, c in enumerate(COVS)],
    np.ones(len(mgt))]).astype(np.float32)
Xv = np.nan_to_num(Xv, nan=0.0)
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
mon_te = (mgt['t_abs'].values % 12) + 1
XLv = np.column_stack([Xv[:,0], Dt1, Xv[:,1:11],
    has_nxt_te.astype(np.float32),
    np.sin(2*np.pi*mon_te/12), np.cos(2*np.pi*mon_te/12),
    beta_c[cct], mu_c[cct], lat_arr[cct], lon_arr[cct]]).astype(np.float32)

k0_B_lin = np.full(len(test), np.nan, dtype=np.float64)
k0_B_lin[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + Xv[use_full] @ coefF_B
k0_B_lin[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + Xv[use_red][:, colsR] @ coefR_B
k0_B_lgb = np.full(len(test), np.nan, dtype=np.float64)
k0_B_lgb[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + bst.predict(XLv[use_full])
k0_B_lgb[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + bst_r.predict(XLv[use_red])
k0_blendB = 0.5*np.nan_to_num(k0_B_lin) + 0.5*np.nan_to_num(k0_B_lgb)
badB = (~msk) & np.isnan(k0_blendB)
if badB.any(): k0_blendB[badB] = mu_c[cc_t][badB]
k0_B = smooth_rows(np.where(np.isfinite(k0_blendB), k0_blendB, mu_c[cc_t]), ~msk & tw_ok, GAU10)
k0_B = np.where(np.isfinite(k0_B), k0_B, k0_blendB)
print(f"k0-B built; bad-fallback rows={int(badB.sum())}")

# ---------------- assemble ----------------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm = ~msk

def assemble(masked_pred, k0_pred, tag, masked_Wt):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_pred[unm]
    pred[msk] = masked_pred[msk]
    pred = np.where(np.isnan(pred), mu_c[cc_t], pred)
    pred = smooth_rows(pred, msk, masked_Wt)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} nan={int(out['Target'].isna().sum())}", flush=True)
    return pred

# ---- masked-row predictors ----
print("\nmasked Kalman runs...", flush=True)
p74_raw = kalman_predict(0.74, W_pool)          # v6c core
p80_raw = kalman_predict(0.80, W_pool)          # v6c core
p74_dn  = kalman_predict(0.74, W_pool_dn)       # v10 obs-dn axis (global H/R on raw pooled: v9-style)
p80_dn  = kalman_predict(0.80, W_pool_dn)
p_ens_raw = 0.5*p74_raw + 0.5*p80_raw           # v6c masked
p_ens_dn  = 0.5*p74_dn  + 0.5*p80_dn            # v10a/c masked

# ---- CONTROL: exact v6c rebuild (offline fidelity check vs submitted CSV) ----
print("\n--- ctrl: exact v6c rebuild (offline check) ---", flush=True)
pred_ctrl = np.empty(len(test), dtype=np.float64)
pred_ctrl[unm] = k0_A[unm]
pred_ctrl[msk] = p_ens_raw[msk]
pred_ctrl = np.where(np.isnan(pred_ctrl), mu_c[cc_t], pred_ctrl)
pred_ctrl = smooth_rows(pred_ctrl, msk, GAU2)
try:
    disk = pd.read_csv(f'{DL}/submission_v6c.csv')['Target'].values.astype(np.float64)
    d = pred_ctrl - disk
    print(f"FIDELITY vs submitted v6c: corr={np.corrcoef(pred_ctrl, disk)[0,1]:.6f} "
          f"mean|d|={np.abs(d).mean():.6f} max|d|={np.abs(d).max():.6f} "
          f"exact-match-rows={int((np.abs(d)<1e-6).sum())}/{len(d)}")
except Exception as e:
    print(f"FIDELITY check skipped: {e}")

# ---- variants ----
print("\n--- v10a: v6c + obs-PC-dn (masked axis) ---", flush=True)
assemble(p_ens_dn, k0_A, 'v10a', GAU2)
print("--- v10b: v6c + v8 two-comp k0 blend ---", flush=True)
assemble(p_ens_raw, k0_B, 'v10b', GAU2)
print("--- v10c: both axes ---", flush=True)
assemble(p_ens_dn, k0_B, 'v10c', GAU2)

# ---------------- verification ----------------
print("\n=== verification ===", flush=True)
base = pd.read_csv(f'{DL}/submission_v6c.csv')['Target'].values
for tag in ['v10a','v10b','v10c']:
    o = pd.read_csv(f'{DL}/submission_{tag}.csv')
    assert len(o) == 280961, tag
    assert (o['ID'].values == sub['ID'].values).all(), tag
    assert o['Target'].notna().all() and np.isfinite(o['Target']).all(), tag
    d = o['Target'].values - base
    dm = d[msk]; dk = d[unm]
    print(f"{tag}: corr vs v6c={np.corrcoef(o['Target'].values, base)[0,1]:.5f} "
          f"mean|d|={np.abs(d).mean():.4f} | masked-row |d|={np.abs(dm).mean():.4f} "
          f"k0-row |d|={np.abs(dk).mean():.4f} | range [{o['Target'].min():.3f}, {o['Target'].max():.3f}]")
print("\nDONE.")
