"""
SUBMISSION V9 — clean A/B isolation of v8a's two axes + zero-bug fix.

v8a (LB 0.70702259) changed TWO things vs v2b (LB 0.713681445):
  masked axis: v2b-core Kalman -> + obs-PC-denoise + per-cell H/R   (LOO: -0.0070 masked)
  k0 axis:     v1-era linear   -> two-comp linear + LGBM blend      (CV: -0.058, LB ~-0.006)
v9a/v9b complete the 2x2 factorial so the LB tells us exactly which axis delivered:
  v9a = masked [obs-dn + per-cell H/R]  + k0 [v2b v1-era linear]   <- masked-axis isolate
  v9b = masked [v2b core]               + k0 [v8 two-comp blend]   <- k0-axis isolate
Bugfix vs v8: Dhat was NaN for 7 zero-anchor cells -> 9 unmasked rows got literal 0.0
  and 7 masked rows fell back to mu_c. Now Dhat=0 fill (D-tilde = W2*S + W3*trendex there).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
PHI = 0.74

# ================= train infrastructure =================
print("loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique()); T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_tr = np.array([(int(v)//100)*12+(int(v)%100)-1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:,None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None,:]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where(sxx>100, np.nansum(td*F64,axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t)-tbar_c)*beta_c).astype(np.float32)
def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr,dtype=np.float64)-tbar_c[cc_arr])*beta_c[cc_arr]).astype(np.float32)

full = ~np.isnan(F).any(axis=0)
TDm = t_abs_tr[:,None]-tbar_c[None,:]
A_dt = F64 - mu_c[None,:] - TDm*beta_c[None,:]
A_dtf = A_dt[:,full]; A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
_,_,Vt = np.linalg.svd(A_dtf, full_matrices=False)
V = Vt[:200].T.astype(np.float32)
def dn(field):
    out = field.copy(); x = field[full]; okx = np.isfinite(x)
    out[full] = np.where(okx, V@(V.T@np.where(okx,x,0.0)), x)
    return out

Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1)&np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# per-cell H/R on full train (same as v8)
cov_est_tr = np.full(len(train), np.nan, dtype=np.float32)
Zok = np.isfinite(Z).all(axis=1)
cov_est_tr[Zok] = np.column_stack([Z[Zok], np.ones(Zok.sum())]) @ coef
tmp = pd.DataFrame({'cc':train['cc'].values,'cov':cov_est_tr,'tws':yv,
                    'ta':train['t_abs'].values})
tmp['A'] = tmp['tws'] - mu_c[tmp['cc'].values] - trendex_vec(tmp['ta'].values, tmp['cc'].values)
gmean = tmp.groupby('cc')['cov'].mean().reindex(range(n_cells)).fillna(0).values
tmp['Wd'] = tmp['cov'] - gmean[tmp['cc'].values]
tok = tmp.dropna()
ci = tok['cc'].values.astype(np.int64)
Wv = tok['Wd'].values.astype(np.float64); Av = tok['A'].values.astype(np.float64)
n_c = np.bincount(ci, minlength=n_cells)
sW = np.bincount(ci, weights=Wv, minlength=n_cells); sA = np.bincount(ci, weights=Av, minlength=n_cells)
sWW = np.bincount(ci, weights=Wv*Wv, minlength=n_cells); sAA = np.bincount(ci, weights=Av*Av, minlength=n_cells)
sWA = np.bincount(ci, weights=Wv*Av, minlength=n_cells)
with np.errstate(invalid='ignore', divide='ignore'):
    varW_c = sWW/n_c-(sW/n_c)**2; varA_c = sAA/n_c-(sA/n_c)**2; covWA_c = sWA/n_c-(sW/n_c)*(sA/n_c)
    Hc_raw = covWA_c/(LAM_F*varA_c); Rc_raw = varW_c - Hc_raw**2*(LAM_F*varA_c)
valid_c = (n_c>=100)&(varA_c>1e-6)&(varW_c>1e-6)&(Rc_raw>0)
Hg_tr = float(np.nanmean(np.where(valid_c,Hc_raw,np.nan))); Rg_tr = float(np.nanmean(np.where(valid_c,Rc_raw,np.nan)))
print(f"per-cell fit: Hg={Hg_tr:.4f} Rg={Rg_tr:.4f} valid={valid_c.sum()}")

# ================= test =================
print("loading test...", flush=True)
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'],test['lon'])]
assert (test['cc']>=0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values

Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta==m)&okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W = {int(m): cov_field[m]-S for m in all_m}
Wdn = {int(m): dn(W[m]) for m in all_m}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac<0.01].index)
AF = {}
for a in anchors:
    sel = (ta==a)&(~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
print(f"anchors: {anchors}")

# D-tilde weights LOO-fit on the 6 test anchors
Xs, ys = [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b!=a]), axis=0)
    tx = trendex(a)
    ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])&np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok],S[ok],tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + 1e-3*np.eye(3), np.vstack(Xs).T@np.concatenate(ys))
W1,W2,W3 = map(float, w_)
print(f"D-tilde weights: {W1:.3f}/{W2:.3f}/{W3:.3f}")
Dhat_raw = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
n_zero_anchor = int((~np.isfinite(Dhat_raw)).sum())
Dhat = np.where(np.isfinite(Dhat_raw), Dhat_raw, 0.0).astype(np.float32)   # BUGFIX: 7 zero-anchor cells
print(f"zero-anchor cells filled: {n_zero_anchor}")
def Dtil(t): return (W1*Dhat + W2*S + W3*trendex(t)).astype(np.float32)

# global H/R calibration (raw W at anchors) + per-cell arrays
cs, zs, vfs = [], [], []
for a in anchors:
    dj = Dtil(a); w_ = W[a]
    ok = np.isfinite(w_)&np.isfinite(AF[a])&np.isfinite(dj)
    cs.append(np.cov(w_[ok],(AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok])); vfs.append(np.nanvar(AF[a]-dj))
c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
H = c_/(LAM_F*var_f); R = max(varz-c_*c_/(LAM_F*var_f),1e-4)
print(f"calib: H={H:.4f} R={R:.4f} var_f={var_f:.4f}")
rH = np.clip(Hc_raw/Hg_tr, 0.3, 3.0); rR = np.clip(Rc_raw/Rg_tr, 0.3, 3.0)
Hc = np.where(valid_c, H*(0.5+0.5*rH), H).astype(np.float32)
Rc = np.where(valid_c, np.maximum(R*(0.5+0.5*rR),1e-4), R).astype(np.float32)

# ================= k=0 model A: v8 two-comp linear + LGBM (unchanged) =================
print("\nbuilding k=0 models...", flush=True)
Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values; yr = mgk['t_abs'].values//12
slow0 = trendex_vec(mgk['t_abs'].values, ccm); slow1 = trendex_vec(mgk['t_next'].values, ccm)
Xlin = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm] - slow0,
    *[mgk[c].values - clim[ccm, j] for j,c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j,c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
yA = (mgk['target'].values - mu_c[ccm] - slow1).astype(np.float32)
Xlin = np.nan_to_num(Xlin, nan=0.0)
w_rec = np.where(yr<=2006, 1.0, np.where(yr<=2009, 1.5, 2.0)).astype(np.float32)
sw = np.sqrt(w_rec); selF = has_nxt_tr
A_f = Xlin[selF]*sw[selF,None]
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
colsR = [0,1,2,3,4,5,11]
A_r = Xlin[:,colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))
print(f"k0-v8 linear coef[FAST_est]={coefF[0]:.4f}")

import lightgbm as lgb
lat_cc = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_arr = lat_cc['lat'].values.astype(np.float32); lon_arr = lat_cc['lon'].values.astype(np.float32)
mon_tr = (mgk['t_abs'].values % 12) + 1
XL = np.column_stack([Xlin[:,0], slow1, Xlin[:,1:11],
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm], lat_arr[ccm], lon_arr[ccm]]).astype(np.float32)
params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=4)
okY = np.isfinite(yA)
ds  = lgb.Dataset(XL[selF], label=yA[selF], weight=w_rec[selF])
dsr = lgb.Dataset(XL[~selF & okY], label=yA[~selF & okY], weight=w_rec[~selF & okY])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, dsr, num_boost_round=400)
print("k0-v8 LGBM trained.")

# ---- test k=0 features (v8) ----
Lt = test[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = test[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
Dt0 = np.zeros(len(mgt), dtype=np.float32); Dt1 = np.zeros(len(mgt), dtype=np.float32)
uniq_m = np.unique(mgt['t_abs'].values)
Dcache = {int(m): (Dtil(int(m)), Dtil(int(m)+1)) for m in uniq_m}
for i,(m,tn) in enumerate(zip(mgt['t_abs'].values, mgt['t_next'].values)):
    d0, d1 = Dcache[int(m)]
    Dt0[i] = d0[cct[i]]; Dt1[i] = d1[cct[i]]
Xv = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct] - Dt0,
    *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
    np.ones(len(mgt))]).astype(np.float32)
Xv = np.nan_to_num(Xv, nan=0.0)
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
mon_te = (mgt['t_abs'].values % 12) + 1
XLv = np.column_stack([Xv[:,0], Dt1, Xv[:,1:11],
    has_nxt_te.astype(np.float32),
    np.sin(2*np.pi*mon_te/12), np.cos(2*np.pi*mon_te/12),
    beta_c[cct], mu_c[cct], lat_arr[cct], lon_arr[cct]]).astype(np.float32)

k0_v8_lin = np.full(len(test), np.nan, dtype=np.float64)
k0_v8_lin[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + Xv[use_full] @ coefF
k0_v8_lin[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + Xv[use_red][:,colsR] @ coefR
k0_v8_lgb = np.full(len(test), np.nan, dtype=np.float64)
k0_v8_lgb[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + bst.predict(XLv[use_full])
k0_v8_lgb[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + bst_r.predict(XLv[use_red])
k0_v8_blend = 0.5*np.nan_to_num(k0_v8_lin) + 0.5*np.nan_to_num(k0_v8_lgb)
# fallback for rows with no model output (should be none after Dhat fix)
bad = (~msk) & np.isnan(k0_v8_blend)
if bad.any(): k0_v8_blend[bad] = mu_c[cc_t][bad]
print(f"k0-v8 blend built; bad-fallback rows={int(bad.sum())}")

# ================= k=0 model B: v2b/v1-era linear (VERBATIM from submission_v2.py) =================
Xf2 = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm],
    *[mgk[c].values - clim[ccm, j] for j,c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j,c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
yA2 = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
Xf2 = np.nan_to_num(Xf2, nan=0.0)
w_rec2 = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
sw2 = np.sqrt(w_rec2)
A_f2 = Xf2[selF]*sw2[selF,None]
coefF2 = np.linalg.solve(A_f2.T@A_f2 + 1e-3*np.eye(12), A_f2.T@(yA2[selF]*sw2[selF]))
A_r2 = Xf2[:, colsR]*sw2[:,None]
coefR2 = np.linalg.solve(A_r2.T@A_r2 + 1e-3*np.eye(7), A_r2.T@(yA2*sw2))
print(f"k0-v2b linear coef[persistence]={coefF2[0]:.4f}")

Xt2 = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct],
    *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
    np.ones(len(mgt))]).astype(np.float32)
Xt2 = np.nan_to_num(Xt2, nan=0.0)
k0_v2b = np.full(len(test), np.nan, dtype=np.float64)
k0_v2b[use_full] = mu_c[cct[use_full]] + Xt2[use_full] @ coefF2
k0_v2b[use_red]  = mu_c[cct[use_red]]  + Xt2[use_red][:, colsR] @ coefR2
bad2 = (~msk) & np.isnan(k0_v2b)
if bad2.any(): k0_v2b[bad2] = mu_c[cct][bad2]
print(f"k0-v2b built; bad-fallback rows={int(bad2.sum())}")

# ================= masked-row prediction =================
print("\nmasked-row Kalman...", flush=True)
q = LAM_F*var_f*(1-PHI**2); P0 = LAM_F*(1-LAM_F)*var_f
anchors_arr = np.array(anchors)

def masked_predict(obs_dn=True, percell=True):
    H_ = Hc if percell else np.full(n_cells, H, np.float32)
    R_ = Rc if percell else np.full(n_cells, R, np.float32)
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for m in sorted(set(ta[msk].tolist())):
        tm = m + 1
        sel = np.where((ta==m)&msk)[0]
        if len(sel)==0: continue
        i = int(anchors_arr[np.searchsorted(anchors_arr, m, side='right')-1])
        x = np.where(np.isfinite(AF[i]), LAM_F*np.nan_to_num(AF[i]-Dtil(i)), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        for mm in range(i+1, tm+1):
            x = PHI*x; P = PHI**2*P + q
            if mm in W:
                w_ = Wdn[mm] if obs_dn else W[mm]
                okw = np.isfinite(w_)
                K = np.where(okw, P*H_/(H_*H_*P+R_), 0).astype(np.float32)
                x = np.where(okw, x+K*(np.nan_to_num(w_)-H_*x), x)
                P = np.where(okw, (1-K*H_)*P, P)
        pred[sel] = mu_c[cc_t[sel]] + Dtil(tm)[cc_t[sel]] + x[cc_t[sel]]
    return pred

# ================= assemble =================
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm = ~msk
def assemble(masked_pred, k0_pred, tag):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_pred[unm]
    pred[msk] = masked_pred[msk]
    pred = np.where(np.isnan(pred), mu_c[cc_t], pred)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    ok = np.isfinite(out['Target'].values).all()
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} NaN-free={ok}", flush=True)

print("\n--- v9a: masked [obs-dn + percell] | k0 [v2b v1-era linear] ---")
mp_a = masked_predict(obs_dn=True, percell=True)
assemble(mp_a, k0_v2b, 'v9a')
print("--- v9b: masked [v2b core] | k0 [v8 two-comp blend] ---")
mp_b = masked_predict(obs_dn=False, percell=False)
assemble(mp_b, k0_v8_blend, 'v9b')

# verification + comparison vs v8a/v2b on disk
v8a = pd.read_csv(f'{DL}/submission_v8a.csv')['Target'].values
for tag in ['v9a','v9b']:
    o = pd.read_csv(f'{DL}/submission_{tag}.csv')
    assert len(o)==280961, tag
    assert (o['ID'].values == sub['ID'].values).all(), tag
    assert o['Target'].notna().all(), tag
    d = o['Target'].values - v8a
    print(f"{tag}: corr vs v8a={np.corrcoef(o['Target'].values, v8a)[0,1]:.4f} "
          f"mean|d|={np.abs(d).mean():.4f} | k0-row diff={np.abs(d[unm]).mean():.4f} "
          f"masked-row diff={np.abs(d[msk]).mean():.4f}")
print("\nDONE.")
