"""
15-b K0 AUDIT — Mission B (honest test-like sparse-anchor k=0 harness) + Mission D (LGBM rounds)
+ Mission A k0-side constants (recency weights, phi, blend weight, LGBM hyperparams).

Design (fixes the diagnosed CV inflation): the 2013-15 val window has ~13 anchor months
(~6/yr) while TEST has 6 anchors in 40 months. We rebuild D-hat/D-tilde from only 6
test-like anchor months (3 deterministic draws) and re-measure every k=0 model.
Evaluations:
  (i)  ALL k0 val rows, per anchor-set D-tilde  (continuity with v8_cv numbers)
  (ii) SUBSET of k0 rows at the 6 sparse-anchor months: dense-D-tilde vs sparse-D-tilde
       head-to-head on identical rows (isolates pure D-tilde quality effect).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
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
del test_raw

# ---------------- infra on fit era ----------------
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

# ---------------- val cov fields / anchors ----------------
Zv = val[COVS].values.astype('float32'); okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
val_tai = val['t_abs'].values
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_tai==m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
S = np.nanmean(np.array([cov_field[m] for m in sorted(cov_field)]), axis=0)

mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v<0.5].index)
print(f"val anchor months (dense, n={len(anchors_v)}): {anchors_v}")
AF = {}
for a in anchors_v:
    sel = (val_ta==a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c

SPARSE = {
 'spA': [2013*12+0, 2013*12+5, 2013*12+10, 2014*12+5, 2014*12+10, 2015*12+5],   # 2013-01,06,11 2014-06,11 2015-06
 'spB': [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6],     # 2013-01,07 2014-01,09 2015-01,07
 'spC': [2013*12+0, 2013*12+5, 2013*12+11, 2014*12+10, 2015*12+0, 2015*12+5],   # 2013-01,06,12 2014-11 2015-01,06
}
for nm, ss in SPARSE.items():
    assert all(s in anchors_v for s in ss), (nm, ss)

def build_dtilde(anchor_set):
    """returns D_tilde(t) function + Dhat for a given anchor month set"""
    Dh = np.nanmean(np.array([AF[a] for a in anchor_set]), axis=0)
    Xs, ys = [], []
    for a in anchor_set:
        dloo = np.nanmean(np.array([AF[b] for b in anchor_set if b!=a]), axis=0)
        tx = trendex(a).astype(np.float32)
        ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])&np.isfinite(tx)
        Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
    w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + 1e-3*np.eye(3), np.vstack(Xs).T@np.concatenate(ys))
    cache = {}
    def Dt(t):
        t = int(t)
        if t not in cache:
            cache[t] = (w_[0]*Dh + w_[1]*S + w_[2]*trendex(t)).astype(np.float32)
        return cache[t]
    return Dt, Dh, w_

SETS = {'dense': anchors_v}
SETS.update({k: v for k, v in SPARSE.items()})
DT = {}; DH = {}; WW = {}
for nm, ss in SETS.items():
    DT[nm], DH[nm], WW[nm] = build_dtilde(ss)
    print(f"  {nm}: D-tilde weights = {WW[nm].round(3)}")

# D-tilde error proxy vs dense D-hat (best static D estimate) over val anchor months
Dh_dense = DH['dense']
for nm in SETS:
    diffs = []
    for a in anchors_v:
        if nm=='dense':
            dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b!=a]), axis=0)
            d = (WW[nm][0]*dloo + WW[nm][1]*S + WW[nm][2]*trendex(a)).astype(np.float32)
        else:
            d = DT[nm](a)
        ok = np.isfinite(d)&np.isfinite(Dh_dense)
        diffs.append(d[ok]-Dh_dense[ok])
    diffs = np.concatenate(diffs)
    print(f"  D-tilde[{nm}] deviation from dense D-hat: std={np.std(diffs):.4f}")

def rmse(p, msk):
    ok = msk & np.isfinite(p) & np.isfinite(val_target)
    return float(np.sqrt(np.mean((p[ok]-val_target[ok])**2))), int(ok.sum())

# ================= train-side k=0 design matrices (shared) =================
Lk = fit[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = fit[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
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
colsR = [0,1,2,3,4,5,11]
selF = has_nxt_tr

def fit_linear(recency):
    w_rec = np.where(yr<=2006, recency[0], np.where(yr<=2009, recency[1], recency[2])).astype(np.float32)
    sw = np.sqrt(w_rec)
    A_f = Xlin[selF]*sw[selF,None]
    cF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
    A_r = Xlin[:,colsR]*sw[:,None]
    cR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))
    return cF, cR

def fit_linear_nodecomp(recency):
    """v1/v4-style: no slow decomposition (target = mu + resid)"""
    w_rec = np.where(yr<=2006, recency[0], np.where(yr<=2009, recency[1], recency[2])).astype(np.float32)
    sw = np.sqrt(w_rec)
    Xn = np.column_stack([
        mgk['TWS_t'].values - mu_c[ccm],
        *[mgk[c].values - clim[ccm, j] for j,c in enumerate(COVS)],
        *[mgk[c+'_nxt'].values - clim[ccm, j] for j,c in enumerate(COVS)],
        np.ones(len(mgk))]).astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0)
    yn = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
    A_f = Xn[selF]*sw[selF,None]
    cF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yn[selF]*sw[selF]))
    A_r = Xn[:,colsR]*sw[:,None]
    cR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yn*sw))
    return cF, cR, Xn

REC = {'rec1x1x1': (1.0,1.0,1.0), 'rec1_1.5_2 (ship)': (1.0,1.5,2.0), 'rec1_2_3': (1.0,2.0,3.0)}
LIN = {}
for nm, rc in REC.items(): LIN[nm] = fit_linear(rc)
cF_nd, cR_nd, Xn = fit_linear_nodecomp((1.0,1.5,2.0))   # v4-linear ship recency

# fixed-phi two-comp variants (refit covariates given phi)
def fit_fixed_phi(phi):
    yres = yA - phi*Xlin[:,0]
    w_rec = np.where(yr<=2006, 1.0, np.where(yr<=2009, 1.5, 2.0)).astype(np.float32)
    sw2 = np.sqrt(w_rec)
    B = Xlin[:,1:]; Bw = B*sw2[:,None]
    cc_ = np.linalg.solve(Bw.T@Bw + 1e-3*np.eye(11), Bw.T@(yres*sw2))
    Br = Xlin[:,1:6]; Brw = Br*sw2[:,None]
    cc_r = np.linalg.solve(Brw.T@Brw + 1e-3*np.eye(5), Brw.T@(yres*sw2))
    return cc_, cc_r
FIXPHI = {0.68: fit_fixed_phi(0.68), 0.74: fit_fixed_phi(0.74)}

# ---------------- val-side feature builder per anchor set ----------------
Lt = val[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = val[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~val_msk) & has_nxt_te & tw_ok
use_red  = (~val_msk) & (~has_nxt_te) & tw_ok
k0_sel = ~val_msk

def build_val_X(setname):
    Dt = DT[setname]
    Dt0 = np.zeros(len(mgt), dtype=np.float32); Dt1 = np.zeros(len(mgt), dtype=np.float32)
    uniq = np.unique(mgt['t_abs'].values)
    Dcache = {int(m): (Dt(int(m)), Dt(int(m)+1)) for m in uniq}
    for i,(m,tn) in enumerate(zip(mgt['t_abs'].values, mgt['t_next'].values)):
        d0, d1 = Dcache[int(m)]
        Dt0[i] = d0[cct[i]]; Dt1[i] = d1[cct[i]]
    Xv = np.column_stack([
        mgt['TWS_t'].values - mu_c[cct] - Dt0,
        *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
        *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
        np.ones(len(mgt))]).astype(np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0)
    return Xv, Dt0, Dt1

XV = {}; DT0 = {}; DT1 = {}
for nm in SETS:
    XV[nm], DT0[nm], DT1[nm] = build_val_X(nm)

# v4-linear val features (no decomp)
Xv_nd = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct],
    *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
    np.ones(len(mgt))]).astype(np.float32)
Xv_nd = np.nan_to_num(Xv_nd, nan=0.0)

p_v4lin = np.full(len(val), np.nan)
p_v4lin[use_full] = mu_c[cct[use_full]] + Xv_nd[use_full] @ cF_nd
p_v4lin[use_red]  = mu_c[cct[use_red]]  + Xv_nd[use_red][:,colsR] @ cR_nd
r, n = rmse(p_v4lin, k0_sel); print(f"\nv4-linear (D-free reference, ship recency): RMSE={r:.4f} (n={n})")

def pred_twocomp(setname, recname='rec1_1.5_2 (ship)', phi=None):
    cF, cR = LIN[recname]
    Xv = XV[setname]; Dt1 = DT1[setname]
    p = np.full(len(val), np.nan)
    if phi is None:
        p[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + Xv[use_full] @ cF
        p[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + Xv[use_red][:,colsR] @ cR
    else:
        cc_, cc_r = FIXPHI[phi]
        p[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + phi*Xv[use_full,0] + Xv[use_full,1:] @ cc_
        p[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + phi*Xv[use_red,0] + Xv[use_red,1:6] @ cc_r
    return p

# ================= LGBM =================
import lightgbm as lgb
lat_cc = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_arr = lat_cc['lat'].values.astype(np.float32); lon_arr = lat_cc['lon'].values.astype(np.float32)
mon_tr = (mgk['t_abs'].values % 12) + 1
XL = np.column_stack([Xlin[:,0], slow1, Xlin[:,1:11],
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm], lat_arr[ccm], lon_arr[ccm]]).astype(np.float32)
XL = np.nan_to_num(XL, nan=0.0)
mon_te = (mgt['t_abs'].values % 12) + 1
def build_XLv(setname):
    return np.nan_to_num(np.column_stack([XV[setname][:,0], DT1[setname], XV[setname][:,1:11],
        has_nxt_te.astype(np.float32),
        np.sin(2*np.pi*mon_te/12), np.cos(2*np.pi*mon_te/12),
        beta_c[cct], mu_c[cct], lat_arr[cct], lon_arr[cct]]).astype(np.float32), nan=0.0)
XLv = {nm: build_XLv(nm) for nm in SETS}

w_rec_tr = np.where(yr<=2006, 1.0, np.where(yr<=2009, 1.5, 2.0)).astype(np.float32)
BASE = dict(objective='regression', learning_rate=0.05, num_leaves=63,
            min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=2)
print("\ntraining baseline LGBM (500 rounds)...", flush=True)
import os
ds  = lgb.Dataset(XL[selF], label=yA[selF], weight=w_rec_tr[selF])
dsr = lgb.Dataset(XL[~selF & np.isfinite(yA)], label=yA[~selF & np.isfinite(yA)], weight=w_rec_tr[~selF & np.isfinite(yA)])
bst = lgb.train(BASE, ds, num_boost_round=500)
bst_r = lgb.train(BASE, dsr, num_boost_round=400)
bst.save_model('/home/z/my-project/scripts/audit_15b_lgbm_base.txt')
bst_r.save_model('/home/z/my-project/scripts/audit_15b_lgbm_red.txt')

def pred_lgbm(setname, num_it=500):
    p = np.full(len(val), np.nan)
    p[use_full] = mu_c[cct[use_full]] + DT1[setname][use_full] + bst.predict(XLv[setname][use_full], num_iteration=num_it)
    p[use_red]  = mu_c[cct[use_red]]  + DT1[setname][use_red]  + bst_r.predict(XLv[setname][use_red], num_iteration=min(num_it,400))
    return p

# ================= MISSION D: learning curve =================
print("\n=== MISSION D: LGBM learning curve (val k0 rows) ===", flush=True)
print(f"{'rounds':>7} | {'dense(all)':>11} | {'spB(all)':>9} | {'spB(subset)':>11}")
subB = k0_sel & np.isin(val_ta, SPARSE['spB'])
for ni in [50, 100, 200, 300, 400, 500]:
    pd_ = pred_lgbm('dense', ni); rd_, nd_ = rmse(pd_, k0_sel)
    pb_ = pred_lgbm('spB', ni); rb_, nb_ = rmse(pb_, k0_sel)
    sb_, _ = rmse(pb_, subB)
    print(f"{ni:>7} | {rd_:>11.4f} | {rb_:>9.4f} | {sb_:>11.4f}")

# ================= MISSION B: model comparison, dense vs sparse =================
print("\n=== MISSION B: k=0 models, ALL k0 rows, per anchor set ===", flush=True)
res = {}
for nm in ['dense','spA','spB','spC']:
    p2 = pred_twocomp(nm); r2, _ = rmse(p2, k0_sel)
    pl = pred_lgbm(nm); rl, _ = rmse(pl, k0_sel)
    pb = 0.5*np.nan_to_num(pl)+0.5*np.nan_to_num(p2); rb, _ = rmse(pb, k0_sel)
    res[nm] = (r2, rl, rb)
    print(f"{nm:>6}: two-comp={r2:.4f}  LGBM={rl:.4f}  blend.5={rb:.4f}")
print(f"{'v4lin':>6}: {r:.4f}  (D-free, constant across sets)")
print(f"\nCV-promised gain (dense): v4lin - blend.5 = {r - res['dense'][2]:+.4f}")
for nm in ['spA','spB','spC']:
    print(f"honest gain ({nm}, all rows): {r - res[nm][2]:+.4f}  (survival {(r-res[nm][2])/max(r-res['dense'][2],1e-9)*100:.0f}%)")

print("\n=== MISSION B (ii): subset at sparse-anchor months, dense vs sparse D-tilde ===", flush=True)
for nm in ['spA','spB','spC']:
    sub = k0_sel & np.isin(val_ta, SPARSE[nm])
    rv, _ = rmse(p_v4lin, sub)
    p2d = pred_twocomp('dense'); p2s = pred_twocomp(nm)
    pld = pred_lgbm('dense'); pls = pred_lgbm(nm)
    r2d, _ = rmse(p2d, sub); r2s, _ = rmse(p2s, sub)
    rld, _ = rmse(pld, sub); rls, _ = rmse(pls, sub)
    rbd, _ = rmse(0.5*np.nan_to_num(pld)+0.5*np.nan_to_num(p2d), sub)
    rbs, _ = rmse(0.5*np.nan_to_num(pls)+0.5*np.nan_to_num(p2s), sub)
    print(f"{nm}: n={int(sub.sum())} | v4lin={rv:.4f} | 2comp dense={r2d:.4f} sparse={r2s:.4f} | "
          f"LGBM dense={rld:.4f} sparse={rls:.4f} | blend dense={rbd:.4f} sparse={rbs:.4f}")

# ================= MISSION A: k0 constants =================
print("\n=== MISSION A: k0 constants sensitivity ===", flush=True)
print("--- two-comp phi (fixed vs free), dense + spB all rows ---")
print(f"free coef (ship, coef[0]={LIN['rec1_1.5_2 (ship)'][0][0]:.3f}): dense={res['dense'][0]:.4f} spB={res['spB'][0]:.4f}")
for phi in [0.68, 0.74]:
    pdn = pred_twocomp('dense', phi=phi); rdn, _ = rmse(pdn, k0_sel)
    psb = pred_twocomp('spB', phi=phi); rsb, _ = rmse(psb, k0_sel)
    print(f"fixed phi={phi:.2f}: dense={rdn:.4f} spB={rsb:.4f}")
print("--- recency weights (two-comp linear, dense + spB) ---")
for rcname in REC:
    pdn = pred_twocomp('dense', recname=rcname); rdn, _ = rmse(pdn, k0_sel)
    psb = pred_twocomp('spB', recname=rcname); rsb, _ = rmse(psb, k0_sel)
    print(f"{rcname}: dense={rdn:.4f} spB={rsb:.4f}")
print("--- k0 blend weight (sparse honest setting, spB all rows) ---")
p2 = pred_twocomp('spB'); pl = pred_lgbm('spB')
for wgt in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
    pb = wgt*np.nan_to_num(pl)+(1-wgt)*np.nan_to_num(p2); rb, _ = rmse(pb, k0_sel)
    print(f"blend {wgt:.2f}*LGBM: spB={rb:.4f}")

print("\n--- LGBM hyperparam sweep (500 rounds, dense + spB all rows) ---", flush=True)
CONFIGS = {
 'lr.03': {**BASE, 'learning_rate':0.03},
 'lr.08': {**BASE, 'learning_rate':0.08},
 'leaves31': {**BASE, 'num_leaves':31},
 'leaves127': {**BASE, 'num_leaves':127},
 'mc200': {**BASE, 'min_child_samples':200},
 'mc1000': {**BASE, 'min_child_samples':1000},
 'ff1.0': {**BASE, 'feature_fraction':1.0},
 'bag1.0': {**BASE, 'bagging_fraction':1.0, 'bagging_freq':0},
}
if os.environ.get('K0SWEEP'):
    only = os.environ.get('CFG')
    items = [(k,v) for k,v in CONFIGS.items() if (only is None or only==k)]
    with open('/home/z/my-project/scripts/audit_15b_lgbm_sweep.txt','a') as fh:
        for nm, params in items:
            m = lgb.train(params, ds, num_boost_round=500)
            pred_red = bst_r.predict(XLv['dense'][use_red])   # reduced model shared across configs
            p = np.full(len(val), np.nan)
            p[use_full] = mu_c[cct[use_full]] + DT1['dense'][use_full] + m.predict(XLv['dense'][use_full])
            p[use_red] = mu_c[cct[use_red]] + DT1['dense'][use_red] + pred_red
            rd_, _ = rmse(p, k0_sel)
            p = np.full(len(val), np.nan)
            p[use_full] = mu_c[cct[use_full]] + DT1['spB'][use_full] + m.predict(XLv['spB'][use_full])
            p[use_red] = mu_c[cct[use_red]] + DT1['spB'][use_red] + pred_red
            rs_, _ = rmse(p, k0_sel)
            line = f"{nm}: dense={rd_:.4f} spB={rs_:.4f}"
            print(line, flush=True); fh.write(line+"\n"); fh.flush()
else:
    print("(sweep skipped - set K0SWEEP=1 [CFG=name] to run)")

print("\nDONE audit_15b_k0.")
