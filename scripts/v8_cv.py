"""
V8 CV — consolidated validation on the honest 2013-15 window.
Decides the final v8 submission configuration.

Part A (k=0 rows):
  A1: baselines (v4-linear, two-comp D-hat, two-comp+covs)   [reproduce agent 2-a]
  A2: two-comp+covs phi sweep (ship form: D-tilde slow)
  A3: LightGBM with decomposed features vs linear two-comp
Part B (masked rows):
  B1: v2b core baseline
  B2: shrink sweep {0.80,0.85,0.90,0.95}
  B3: per-cell H/R (agent form + era-scaled form) at shrink 0.85
  B4: backward pass arbitration
Part C: overall val RMSE for candidate v8 configs
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1

fit = train[train['time'].dt.year <= 2012].copy()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()

test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
test_raw['masked'] = test_raw['TWS_t_masked'].astype(bool)
cal_mask_frac = test_raw.groupby(test_raw['time'].dt.month)['masked'].mean()
val['masked'] = val['time'].dt.month.map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values; val_ta = val['t_abs'].values; val_msk = val['masked'].values

# ---------------- infra on fit ----------------
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

yms_all = np.sort(train['ym'].unique()); ym_to_i = {int(v):i for i,v in enumerate(yms_all)}
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12+(int(v)%100)-1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:,None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None,:]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx>100)&(ok_t.sum(axis=0)>=24), np.nansum(td*F64,axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return (np.float64(t)-tbar_c)*beta_c
def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr,dtype=np.float64)-tbar_c[cc_arr])*beta_c[cc_arr]).astype(np.float32)

# per-cell H/R source quantities on fit era:
#   W_c(t) = cov_est - per-cell temporal mean;  A_c(t) = TWS - mu - trend  (detrended anomaly)
cov_est_fit = np.full(len(fit), np.nan, dtype=np.float32)
Zfit_ok = np.isfinite(Z).all(axis=1)
cov_est_fit[Zfit_ok] = np.column_stack([Z[Zfit_ok], np.ones(Zfit_ok.sum())]) @ coef
tmp = pd.DataFrame({'cc':fit['cc'].values, 't_abs':fit['t_abs'].values,
                    'cov':cov_est_fit, 'tws':fit['TWS_t'].values})
tmp['A'] = tmp['tws'] - mu_c[tmp['cc'].values] - trendex_vec(tmp['t_abs'].values, tmp['cc'].values)
gmean = tmp.groupby('cc')['cov'].mean().reindex(range(n_cells)).fillna(0).values
tmp['Wd'] = tmp['cov'] - gmean[tmp['cc'].values]
tmp_ok = tmp.dropna()
# vectorized per-cell cov/var via accumulation
cc_idx = tmp_ok['cc'].values.astype(np.int64)
Wd_v = tmp_ok['Wd'].values.astype(np.float64)
A_v  = tmp_ok['A'].values.astype(np.float64)
n_c   = np.bincount(cc_idx, minlength=n_cells)
sW    = np.bincount(cc_idx, weights=Wd_v, minlength=n_cells)
sA    = np.bincount(cc_idx, weights=A_v, minlength=n_cells)
sWW   = np.bincount(cc_idx, weights=Wd_v*Wd_v, minlength=n_cells)
sAA   = np.bincount(cc_idx, weights=A_v*A_v, minlength=n_cells)
sWA   = np.bincount(cc_idx, weights=Wd_v*A_v, minlength=n_cells)
with np.errstate(invalid='ignore', divide='ignore'):
    varW_c = sWW/n_c - (sW/n_c)**2
    varA_c = sAA/n_c - (sA/n_c)**2
    covWA_c = sWA/n_c - (sW/n_c)*(sA/n_c)
    Hc_raw = covWA_c/(LAM_F*varA_c)          # train-era per-cell H
    Rc_raw = varW_c - Hc_raw**2*(LAM_F*varA_c)
valid_c = (n_c >= 60) & (varA_c > 1e-6) & (varW_c > 1e-6) & (Rc_raw > 0)
Hc_raw = np.where(valid_c, Hc_raw, np.nan)
Rc_raw = np.where(valid_c, Rc_raw, np.nan)
Hg_fit = float(np.nanmean(Hc_raw)); Rg_fit = float(np.nanmean(Rc_raw))
print(f"[per-cell fit] H global={Hg_fit:.4f} R global={Rg_fit:.4f}; cells valid={valid_c.sum()}")

# ---------------- val cov fields / anchors / D-tilde ----------------
Zv = val[COVS].values.astype('float32'); okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta==m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
S = np.nanmean(np.array([cov_field[m] for m in sorted(cov_field)]), axis=0)
W = {int(m): cov_field[m]-S for m in cov_field}

mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v<0.5].index)
AF = {}
for a in anchors_v:
    sel = (val_ta==a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)

Xs, ys = [], []
for a in anchors_v:
    dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b!=a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])&np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + 1e-3*np.eye(3), np.vstack(Xs).T@np.concatenate(ys))
w1,w2,w3 = map(float, w_)
print(f"D-tilde LOO weights: {w1:.3f}/{w2:.3f}/{w3:.3f}")
def D_trendex(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

def rmse(p, msk):
    ok = msk & np.isfinite(p) & np.isfinite(val_target)
    return float(np.sqrt(np.mean((p[ok]-val_target[ok])**2))), int(ok.sum())

# ================= PART A: k=0 models =================
print("\n=== A: k=0 (unmasked val rows) ===", flush=True)
k0_sel = ~val_msk
tws_v = val_tws_visible.values.astype(np.float32)

# ---- fit two-comp model on fit era (slow = trend) ----
fs = fit.sort_values(['cc','t_abs']); g = fs.groupby('cc')
fs['t_prev'] = g.TWS_t.shift(1); fs['ta_prev'] = g.t_abs.shift(1)
cp = fs[(fs.t_abs - fs.ta_prev)==1].copy()
Xfast = cp.TWS_t.values - mu_c[cp.cc.values] - trendex_vec(cp.ta_prev.values, cp.cc.values)
phi_fit = float(np.sum(Xfast*(cp.target.values-mu_c[cp.cc.values]-trendex_vec(cp.t_abs.values,cp.cc.values)))/np.sum(Xfast*Xfast))
print(f"train-fit phi (slow=trend): {phi_fit:.4f}")

# ---- build k=0 design matrices (shared by linear & LGBM) ----
Lk = fit[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = fit[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values; yr = mgk['t_abs'].values//12
slow0 = trendex_vec(mgk['t_abs'].values, ccm)
slow1 = trendex_vec(mgk['t_next'].values, ccm)
Xlin = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm] - slow0,       # FAST_est (slow=trend in train)
    *[mgk[c].values - clim[ccm, j] for j,c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j,c in enumerate(COVS)],
    np.ones(len(mgk))
]).astype(np.float32)
yA = (mgk['target'].values - mu_c[ccm] - slow1).astype(np.float32)   # predict residual beyond slow(t+1)
Xlin = np.nan_to_num(Xlin, nan=0.0)
w_rec = np.where(yr<=2006, 1.0, np.where(yr<=2009, 1.5, 2.0)).astype(np.float32)
sw = np.sqrt(w_rec); selF = has_nxt_tr
A_f = Xlin[selF]*sw[selF,None]
# linear: fix phi coefficient at 1 on FAST_est? No: fit freely (includes phi implicitly)
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
colsR = [0,1,2,3,4,5,11]
A_r = Xlin[:,colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))
print(f"linear coef[FAST_est] (full model) = {coefF[0]:.4f}  (this is the effective phi)")

# ---- val: apply with slow = D-tilde ----
Lt = val[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = val[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
Dt0 = np.zeros(len(mgt), dtype=np.float32); Dt1 = np.zeros(len(mgt), dtype=np.float32)
for i,(m,tn) in enumerate(zip(mgt['t_abs'].values, mgt['t_next'].values)):
    d0 = D_trendex(int(m)); d1 = D_trendex(int(tn))
    Dt0[i] = d0[cct[i]]; Dt1[i] = d1[cct[i]]
Xv = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct] - Dt0,
    *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
    np.ones(len(mgt))
]).astype(np.float32)
Xv = np.nan_to_num(Xv, nan=0.0)
tw_ok = np.isfinite(mgt['TWS_t'].values)

p_lin = np.full(len(val), np.nan)
use_full = (~val_msk) & has_nxt_te & tw_ok
use_red  = (~val_msk) & (~has_nxt_te) & tw_ok
p_lin[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + Xv[use_full] @ coefF
p_lin[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + Xv[use_red][:,colsR] @ coefR
r, n = rmse(p_lin, k0_sel); print(f"A1 two-comp+covs linear (free coef): RMSE={r:.4f} (n={n})")

# fixed-phi variant: force coef[0]=phi, refit others
for phi_fix in [phi_fit, 0.74, 0.80]:
    yres = yA - phi_fix*Xlin[:,0]
    B = Xlin[:,1:]; Bw = B*sw[:,None]
    cc_ = np.linalg.solve(Bw.T@Bw + 1e-3*np.eye(11), Bw.T@(yres*sw))
    yres_r = yA - phi_fix*Xlin[:,0]
    Br = Xlin[:,1:6]; Brw = Br*sw[:,None]
    cc_r = np.linalg.solve(Brw.T@Brw + 1e-3*np.eye(5), Brw.T@(yres_r*sw))
    p2 = np.full(len(val), np.nan)
    p2[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + phi_fix*Xv[use_full,0] + Xv[use_full,1:] @ cc_
    p2[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + phi_fix*Xv[use_red,0] + Xv[use_red,1:6] @ cc_r
    r, _ = rmse(p2, k0_sel); print(f"A2 fixed phi={phi_fix:.2f}: RMSE={r:.4f}")

# ---- A3: LightGBM with decomposed features ----
print("\nA3: LightGBM (decomposed features)...", flush=True)
import lightgbm as lgb
lat_cc = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_arr = lat_cc['lat'].values.astype(np.float32); lon_arr = lat_cc['lon'].values.astype(np.float32)
mon_tr = (mgk['t_abs'].values % 12) + 1
XL = np.column_stack([
    Xlin[:,0],                                   # FAST_est
    slow1,                                       # slow(t+1)
    Xlin[:,1:11],                                # covs t & t+1 anomalies
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm], lat_arr[ccm], lon_arr[ccm]
]).astype(np.float32)
XL = np.nan_to_num(XL, nan=0.0)
mon_te = (mgt['t_abs'].values % 12) + 1
XLv = np.column_stack([
    Xv[:,0], Dt1,
    Xv[:,1:11],
    has_nxt_te.astype(np.float32),
    np.sin(2*np.pi*mon_te/12), np.cos(2*np.pi*mon_te/12),
    beta_c[cct], mu_c[cct], lat_arr[cct], lon_arr[cct]
]).astype(np.float32)
XLv = np.nan_to_num(XLv, nan=0.0)

params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0,
              num_threads=4)
ds = lgb.Dataset(XL[selF], label=yA[selF], weight=w_rec[selF])
ds_r = lgb.Dataset(XL[~selF & np.isfinite(yA)], label=yA[~selF & np.isfinite(yA)], weight=w_rec[~selF & np.isfinite(yA)])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, ds_r, num_boost_round=400)
p_lgb = np.full(len(val), np.nan)
p_lgb[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + bst.predict(XLv[use_full])
selr = (~val_msk) & (~has_nxt_te) & tw_ok
p_lgb[selr] = mu_c[cct[selr]] + Dt1[selr] + bst_r.predict(XLv[selr])
r, _ = rmse(p_lgb, k0_sel); print(f"A3 LGBM decomposed: RMSE={r:.4f}")
# blend
for wgt in [0.25, 0.5, 0.75]:
    p = wgt*np.nan_to_num(p_lgb,nan=0.0) + (1-wgt)*np.nan_to_num(p_lin,nan=0.0)
    r, _ = rmse(p, k0_sel); print(f"   blend {wgt:.2f}*LGBM: RMSE={r:.4f}")

# ================= PART B: masked rows =================
print("\n=== B: masked rows ===", flush=True)
def calibrate(Dfun, anchors):
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dfun(a)
        ok = np.isfinite(W[a])&np.isfinite(AF[a])&np.isfinite(dj)
        cs.append(np.cov(W[a][ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(W[a][ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    vx = LAM_F*var_f
    return c_/vx, max(varz - c_*c_/vx, 1e-4), var_f

H_g, R_g, var_f = calibrate(D_trendex, anchors_v)
print(f"global H={H_g:.4f} R={R_g:.4f} var_f={var_f:.4f}")

def kalman_predict(phi_f, Dfun, shrink=1.0, Hc=None, Rc=None, use_bwd=False):
    H_ = H_g if Hc is None else Hc
    R_ = R_g if Rc is None else Rc
    Hs = H_ if np.isscalar(H_) else None
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(val), np.nan, dtype=np.float64)
    anchors_arr = np.array(anchors_v)
    for ai, a in enumerate(anchors_v):
        dja = Dfun(a); fa = AF[a] - dja
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        sel0 = np.where((val_ta==a)&val_msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[val_cc[sel0]] + Dfun(a+1)[val_cc[sel0]] + shrink*phi_f*x[val_cc[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in W:
                wv = W[m]; okw = np.isfinite(wv)
                if Hs is not None:
                    K = np.where(okw, P*Hs/(Hs*Hs*P+R_), 0).astype(np.float32)
                    x = np.where(okw, x+K*(np.nan_to_num(wv)-Hs*x), x)
                    P = np.where(okw, (1-K*Hs)*P, P)
                else:
                    Hcc = H_; Rcc = R_
                    K = np.where(okw, P*Hcc/(Hcc*Hcc*P+Rcc), 0).astype(np.float32)
                    x = np.where(okw, x+K*(np.nan_to_num(wv)-Hcc*x), x)
                    P = np.where(okw, (1-K*Hcc)*P, P)
            sel = np.where((val_ta==m)&val_msk)[0]
            if len(sel)==0: continue
            tm = m+1
            x2 = phi_f*x; P2 = phi_f**2*P + q
            if tm in W:
                wv = W[tm]; okw = np.isfinite(wv)
                if Hs is not None:
                    K2 = np.where(okw, P2*Hs/(Hs*Hs*P2+R_), 0).astype(np.float32)
                    x2 = np.where(okw, x2+K2*(np.nan_to_num(wv)-Hs*x2), x2)
                else:
                    Hcc = H_; Rcc = R_
                    K2 = np.where(okw, P2*Hcc/(Hcc*Hcc*P2+Rcc), 0).astype(np.float32)
                    x2 = np.where(okw, x2+K2*(np.nan_to_num(wv)-Hcc*x2), x2)
            xfin = x2
            if use_bwd:
                later = anchors_arr[anchors_arr > m]
                if len(later):
                    kb = int(later[0])
                    xb = np.where(np.isfinite(AF[kb]-Dfun(kb)), LAM_F*np.nan_to_num(AF[kb]-Dfun(kb)), 0.0).astype(np.float32)
                    Pb = np.where(np.isfinite(AF[kb]), P0, var_f).astype(np.float32)
                    for m2 in range(kb-1, tm-1, -1):
                        xb = phi_f*xb; Pb = phi_f**2*Pb + q
                        if m2 in W:
                            wv = W[m2]; okw = np.isfinite(wv)
                            Kb = np.where(okw, Pb*H_g/(H_g*H_g*Pb+R_g), 0).astype(np.float32)
                            xb = np.where(okw, xb+Kb*(np.nan_to_num(wv)-H_g*xb), xb)
                            Pb = np.where(okw, (1-Kb*H_g)*Pb, Pb)
                    wf = (1.0/np.maximum(P2,1e-6))/(1.0/np.maximum(P2,1e-6)+1.0/np.maximum(Pb,1e-6))
                    xfin = wf*x2 + (1-wf)*xb
            pred[sel] = mu_c[val_cc[sel]] + Dfun(tm)[val_cc[sel]] + shrink*xfin[val_cc[sel]]
    return pred

base = kalman_predict(0.74, D_trendex)
r0, n0 = rmse(base, val_msk); print(f"B1 v2b core: RMSE={r0:.4f} (n={n0})")
for sh in [0.80, 0.85, 0.90, 0.95]:
    p = kalman_predict(0.74, D_trendex, shrink=sh)
    r, _ = rmse(p, val_msk); print(f"B2 shrink={sh:.2f}: RMSE={r:.4f}")

# B3a: per-cell H/R — agent form (Hc mixed 50/50 with val-global)
ratio_H = np.clip(Hc_raw/Hg_fit, 0.3, 3.0); ratio_R = np.clip(Rc_raw/Rg_fit, 0.3, 3.0)
Hc_a = np.where(valid_c, 0.5*H_g + 0.5*ratio_H*H_g, H_g).astype(np.float32)
Rc_a = np.where(valid_c, np.maximum(0.5*R_g + 0.5*ratio_R*R_g, 1e-4), R_g).astype(np.float32)
p = kalman_predict(0.74, D_trendex, shrink=0.85, Hc=Hc_a, Rc=Rc_a)
r, _ = rmse(p, val_msk); print(f"B3a per-cell H/R (agent form) + shrink 0.85: RMSE={r:.4f}")

# B3b: era-scaled form (ratio shrunk 50% toward 1)
Hc_b = np.where(valid_c, H_g*(0.5 + 0.5*ratio_H), H_g).astype(np.float32)
Rc_b = np.where(valid_c, np.maximum(R_g*(0.5 + 0.5*ratio_R), 1e-4), R_g).astype(np.float32)
p = kalman_predict(0.74, D_trendex, shrink=0.85, Hc=Hc_b, Rc=Rc_b)
r, _ = rmse(p, val_msk); print(f"B3b per-cell H/R (era-scaled) + shrink 0.85: RMSE={r:.4f}")

# B4: backward pass (with shrink 0.85, global H/R)
p = kalman_predict(0.74, D_trendex, shrink=0.85, use_bwd=True)
r, _ = rmse(p, val_msk); print(f"B4 shrink 0.85 + backward pass: RMSE={r:.4f}")

# ================= PART C: overall =================
print("\n=== C: overall val RMSE (all rows) ===", flush=True)
best_masked = kalman_predict(0.74, D_trendex, shrink=0.85)   # placeholder; refine per results above
combos = {
  'current (v4-lin k0 + v2b masked)': (p_lin*0 + np.nan, base),   # replaced below
}
cur = np.where(val_msk, base, p_lin*0 + np.nan)
# current: v4-linear k0 — refit quickly for reference (features w/o slow decomp)
X_cur = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm],
    *[mgk[c].values - clim[ccm, j] for j,c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j,c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
X_cur = np.nan_to_num(X_cur, nan=0.0)
y_cur = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
cF = np.linalg.solve((X_cur[selF]*sw[selF,None]).T@(X_cur[selF]*sw[selF,None]) + 1e-3*np.eye(12),
                     (X_cur[selF]*sw[selF,None]).T@(y_cur[selF]*sw[selF]))
cR = np.linalg.solve((X_cur[:,colsR]*sw[:,None]).T@(X_cur[:,colsR]*sw[:,None]) + 1e-3*np.eye(7),
                     (X_cur[:,colsR]*sw[:,None]).T@(y_cur*sw))
Xv_cur = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct],
    *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
    np.ones(len(mgt))]).astype(np.float32)
Xv_cur = np.nan_to_num(Xv_cur, nan=0.0)
p_cur = np.full(len(val), np.nan)
p_cur[use_full] = mu_c[cct[use_full]] + Xv_cur[use_full] @ cF
p_cur[use_red]  = mu_c[cct[use_red]]  + Xv_cur[use_red][:,colsR] @ cR
r_cur, _ = rmse(p_cur, k0_sel); print(f"ref: v4-linear k0 (no decomp): RMSE={r_cur:.4f}")

for name, (pk, pm) in {
    'v2b-equivalent (v4-lin k0 + core masked)': (p_cur, base),
    'v8 candidate A (two-comp k0 + shrink)':     (p_lin, best_masked),
    'v8 candidate B (LGBM k0 + shrink)':         (p_lgb, best_masked),
}.items():
    p = np.where(val_msk, pm, pk)
    okA = np.isfinite(p)&np.isfinite(val_target)
    print(f"{name}: overall={float(np.sqrt(np.mean((p[okA]-val_target[okA])**2))):.4f}")

print("\nDONE v8_cv.")
