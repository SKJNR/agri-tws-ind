"""
fresh_eyes_val.py — fresh-eyes review experiments on the honest 2013-15 val window.

E1: k=0 (unmasked-row) model comparison:
    (a) v4-style linear model (train-fit, covs t + t+1, recency weights)
    (b) single-coef persistence, train-fit slope
    (c) single-coef persistence, val-optimal slope (diagnostic bound)
    (d) two-component D-aware: pred = mu + Dhat + phi*(TWS - mu - Dhat), phi fit on train era
    (e) two-component + cov(t+1) correction
E2: masked-row Kalman (v2b config): global shrink sweep on FAST component
E3: masked-row: value of an oracle TWS observation 1 month before target (bounds
    the pool-cell / spatial-interp opportunity on real test)
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
val_cc = val['cc'].values
val_ta = val['t_abs'].values
val_msk = val['masked'].values

# ---- infra on fit ----
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

yms_all = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms_all)}
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None,:]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx>100)&(ok_t.sum(axis=0)>=24), np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return (np.float64(t) - tbar_c) * beta_c
def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr, dtype=np.float64) - tbar_c[cc_arr]) * beta_c[cc_arr]).astype(np.float32)

# ---- val cov fields ----
Zv = val[COVS].values.astype('float32'); okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
S = np.nanmean(np.array([cov_field[m] for m in sorted(cov_field)]), axis=0)
W = {int(m): cov_field[m] - S for m in cov_field}

mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v < 0.5].index)
AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)

# D-tilde trendex weights (LOO on val anchors)
Xs, ys = [], []
for a in anchors_v:
    dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b != a]), axis=0)
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

# ================= E1: k=0 models =================
print("\n=== E1: k=0 (unmasked val rows) ===")
k0_sel = ~val_msk
# (b) single-coef persistence, slope fit on FIT era calendar-consecutive pairs
fs = fit.sort_values(['cc','t_abs'])
g = fs.groupby('cc')
fs['t_prev'] = g.TWS_t.shift(1); fs['ta_prev'] = g.t_abs.shift(1)
cp = fs[(fs.t_abs - fs.ta_prev)==1]
slope_train = float(np.sum((cp.TWS_t-cp.t_prev)*(cp.target-cp.t_prev))/np.sum((cp.TWS_t-cp.t_prev)**2))
print(f"train-fit 1-month persistence slope: {slope_train:.4f}")
tws_v = val_tws_visible.values  # TWS at month m for unmasked rows
p_b = mu_c[val_cc] + slope_train*(tws_v - mu_c[val_cc])
r, n = rmse(p_b, k0_sel); print(f"(b) persistence train-slope {slope_train:.3f}: RMSE={r:.4f} (n={n})")

# (c) val-optimal slope (diagnostic)
best=(9,0)
for a in np.arange(0.70, 0.95, 0.01):
    p = mu_c[val_cc] + a*(tws_v - mu_c[val_cc])
    r,_ = rmse(p, k0_sel)
    if r < best[0]: best = (r, a)
print(f"(c) val-optimal single slope: {best[1]:.2f} -> RMSE={best[0]:.4f}  (gap vs train slope = misspec cost)")

# (d) two-component D-aware, phi fit on FIT era with trend as slow comp
Xf = cp.TWS_t.values - mu_c[cp.cc.values] - trendex_vec(cp.ta_prev.values, cp.cc.values)
Yf = cp.target.values - mu_c[cp.cc.values] - trendex_vec(cp.t_abs.values, cp.cc.values)
phi_fit = float(np.sum(Xf*Yf)/np.sum(Xf*Xf))
print(f"train-fit two-component phi (slow=trend): {phi_fit:.4f}")
tws_v32 = tws_v.astype(np.float32)
for phi in [phi_fit, 0.74, 0.80, 0.85]:
    p = mu_c[val_cc] + Dhat[val_cc] + phi*(tws_v32 - mu_c[val_cc] - Dhat[val_cc])
    r, n = rmse(p, k0_sel); print(f"(d) two-comp phi={phi:.2f}: RMSE={r:.4f} (n={n})")

# (a) v4-style linear model
Lk = fit[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = fit[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values; yr = mgk['t_abs'].values//12
Xf2 = np.column_stack([mgk['TWS_t'].values - mu_c[ccm],
    *[mgk[c].values - clim[ccm, j] for j,c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j,c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
yA = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
Xf2 = np.nan_to_num(Xf2, nan=0.0)
w_rec = np.where(yr<=2006, 1.0, np.where(yr<=2009, 1.5, 2.0)).astype(np.float32)
sw = np.sqrt(w_rec); selF = has_nxt
A_f = Xf2[selF]*sw[selF,None]
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
colsR = [0,1,2,3,4,5,11]
A_r = Xf2[:,colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))

Lt = val[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = val[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_t = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
Xt = np.column_stack([mgt['TWS_t'].values - mu_c[cct],
    *[mgt[c].values - clim[cct, j] for j,c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j,c in enumerate(COVS)],
    np.ones(len(mgt))]).astype(np.float32)
Xt = np.nan_to_num(Xt, nan=0.0)
p_a = np.full(len(val), np.nan)
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~val_msk) & has_nxt_t & tw_ok
use_red  = (~val_msk) & (~has_nxt_t) & tw_ok
p_a[use_full] = mu_c[cct[use_full]] + Xt[use_full] @ coefF
p_a[use_red]  = mu_c[cct[use_red]]  + Xt[use_red][:, colsR] @ coefR
r, n = rmse(p_a, k0_sel); print(f"(a) v4 linear (covs t+t+1):    RMSE={r:.4f} (n={n})")

# (e) two-component + cov(t+1) correction: fit on FIT era residuals
resid = yA - (phi_fit*(Xf2[:,0] - trendex_vec(mgk['t_abs'].values, ccm)) + trendex_vec(mgk['t_abs'].values, ccm))
# features: cov anomalies t and t+1
Xcov = Xf2[:, 1:11]
ccoef = np.linalg.solve((Xcov*sw[:,None]).T@(Xcov*sw[:,None]) + 1e-3*np.eye(10), (Xcov*sw[:,None]).T@(resid*sw))
# val: two-comp + cov correction
Xcov_v = Xt[:, 1:11]
p_e = mu_c[val_cc] + Dhat[val_cc] + phi_fit*(tws_v32 - mu_c[val_cc] - Dhat[val_cc]) + np.nan_to_num(Xcov_v, nan=0.0) @ ccoef
r, n = rmse(p_e, k0_sel); print(f"(e) two-comp + covs(t,t+1):     RMSE={r:.4f} (n={n})")

# blend of (a) and (d)
p_d = mu_c[val_cc] + Dhat[val_cc] + phi_fit*(tws_v32 - mu_c[val_cc] - Dhat[val_cc])
for wgt in [0.3, 0.5, 0.7]:
    p = wgt*np.nan_to_num(p_d, nan=0.0) + (1-wgt)*np.nan_to_num(p_a, nan=0.0)
    r,_ = rmse(p, k0_sel); print(f"    blend {wgt:.1f}*two-comp + {1-wgt:.1f}*linear: RMSE={r:.4f}")

# ================= E2: masked Kalman + shrink sweep =================
print("\n=== E2: masked rows, v2b-config Kalman, FAST shrink sweep ===")
def calibrate(Dfun):
    cs, zs, vfs = [], [], []
    for a in anchors_v:
        dj = Dfun(a)
        ok = np.isfinite(W[a])&np.isfinite(AF[a])&np.isfinite(dj)
        cs.append(np.cov(W[a][ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(W[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    vx = LAM_F*var_f
    return c_/vx, max(varz - c_*c_/vx, 1e-4), var_f

def kalman_predict(phi_f, Dfun, shrink=1.0):
    H, R, var_f = calibrate(Dfun)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(val), np.nan, dtype=np.float64)
    for a in anchors_v:
        dja = Dfun(a); fa = AF[a] - dja
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        sel0 = np.where((val_ta == a) & val_msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[val_cc[sel0]] + Dfun(a+1)[val_cc[sel0]] + phi_f*x[val_cc[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in W:
                wv = W[m]; okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((val_ta == m) & val_msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in W:
                wv = W[tm]; okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[val_cc[sel]] + Dfun(tm)[val_cc[sel]] + shrink*x2[val_cc[sel]]
    return pred

base = kalman_predict(0.74, D_trendex)
r0, n0 = rmse(base, val_msk); print(f"baseline (shrink=1.0): RMSE={r0:.4f} (n={n0})")
for sh in [0.85, 0.90, 0.95, 1.05]:
    p = kalman_predict(0.74, D_trendex, shrink=sh)
    r, _ = rmse(p, val_msk); print(f"shrink={sh:.2f}: RMSE={r:.4f}")

# ================= E3: oracle fresh TWS obs value =================
print("\n=== E3: masked rows + ORACLE TWS obs at row month m (1 month before target) ===")
# for each masked val row: pred = mu + D(tm) + phi*(TWS(m) - mu - D(m))
tv = val_tws_visible.values.astype(np.float32)
Dm  = {int(m): D_trendex(int(m)) for m in np.unique(val_ta)}
Dp1 = {int(m): D_trendex(int(m)+1) for m in np.unique(val_ta)}
p3 = np.full(len(val), np.nan)
for i in range(len(val)):
    if not val_msk[i] or not np.isfinite(tv[i]): continue
    m = int(val_ta[i])
    p3[i] = mu_c[val_cc[i]] + Dp1[m][val_cc[i]] + 0.74*(tv[i] - mu_c[val_cc[i]] - Dm[m][val_cc[i]])
r, n = rmse(p3, val_msk & np.isfinite(p3)); print(f"oracle 1-month TWS obs: RMSE={r:.4f} (n={n})  vs baseline {r0:.4f}")

# also: 2-months-before oracle (distance 2)
p4 = np.full(len(val), np.nan)
# for masked rows at month m, find TWS at m-1 if that row unmasked in val
key = {(int(val_ta[i]), int(val_cc[i])): tv[i] for i in range(len(val)) if np.isfinite(tv[i])}
for i in range(len(val)):
    if not val_msk[i]: continue
    m = int(val_ta[i]); c = int(val_cc[i])
    tvm1 = key.get((m-1, c), np.nan)
    if not np.isfinite(tvm1): continue
    p4[i] = mu_c[c] + Dp1[m][c] + 0.74*0.74*(tvm1 - mu_c[c] - Dm[m-1][c])
r, n = rmse(p4, val_msk & np.isfinite(p4)); print(f"oracle 2-month TWS obs (decay 0.74^2): RMSE={r:.4f} (n={n})")

print("\nDone.")

# ================= E2b: per-bucket shrink (months since anchor) =================
print("\n=== E2b: optimal shrink by months-since-anchor bucket ===")
def kalman_predict_buckets(phi_f, Dfun):
    """return per-row x2, k (months since anchor), D(tm), cc"""
    H, R, var_f = calibrate(Dfun)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    xs = np.full(len(val), np.nan); ks = np.full(len(val), -1)
    Dtm = np.full(len(val), np.nan)
    for a in anchors_v:
        dja = Dfun(a); fa = AF[a] - dja
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        sel0 = np.where((val_ta == a) & val_msk)[0]
        if len(sel0):
            xs[sel0] = phi_f*x[val_cc[sel0]]; ks[sel0] = 0; Dtm[sel0] = Dfun(a+1)[val_cc[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in W:
                wv = W[m]; okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((val_ta == m) & val_msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in W:
                wv = W[tm]; okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            xs[sel] = x2[val_cc[sel]]; ks[sel] = k; Dtm[sel] = Dfun(tm)[val_cc[sel]]
    return xs, ks, Dtm

xs, ks, Dtm = kalman_predict_buckets(0.74, D_trendex)
ok = val_msk & np.isfinite(xs)
base_pred = mu_c[val_cc] + Dtm + xs
for kb in [1,2,3,4,5,6,7,8]:
    s = ok & (ks==kb)
    if s.sum()==0: continue
    # sweep shrink for this bucket
    best = (9,1)
    for sh in np.arange(0.5, 1.15, 0.05):
        p = mu_c[val_cc[s]] + Dtm[s] + sh*xs[s]
        r = float(np.sqrt(np.mean((p-val_target[s])**2)))
        if r<best[0]: best=(r,sh)
    r0b = float(np.sqrt(np.mean((base_pred[s]-val_target[s])**2)))
    print(f"k={kb}: n={int(s.sum()):6d}  baseline={r0b:.4f}  best shrink={best[1]:.2f} -> {best[0]:.4f}")

# ================= E1f: two-comp k=0 with D-tilde instead of raw Dhat =================
print("\n=== E1f: k=0 two-comp with D-tilde (ship-ready form) ===")
Dt0 = {int(m): D_trendex(int(m)) for m in np.unique(val_ta)}
Dt1 = {int(m): D_trendex(int(m)+1) for m in np.unique(val_ta)}
p_f = np.full(len(val), np.nan)
for i in range(len(val)):
    if val_msk[i] or not np.isfinite(tv[i]): continue
    m = int(val_ta[i])
    p_f[i] = mu_c[val_cc[i]] + Dt1[m][val_cc[i]] + phi_fit*(tv[i] - mu_c[val_cc[i]] - Dt0[m][val_cc[i]])
r, n = rmse(p_f, k0_sel); print(f"(f) two-comp D-tilde phi={phi_fit:.2f}: RMSE={r:.4f} (n={n})")

# ================= E4: combined overall val RMSE =================
print("\n=== E4: combined val RMSE (all rows) ===")
cur = np.where(val_msk, base_pred, p_a)   # current: v4 k=0 + v2b masked
new = np.where(val_msk, mu_c[val_cc]+Dtm+0.85*xs, p_e)  # new: E1e k=0 + shrink masked
okA = np.isfinite(cur)&np.isfinite(new)&np.isfinite(val_target)
print(f"current: {float(np.sqrt(np.mean((cur[okA]-val_target[okA])**2))):.4f}")
print(f"upgraded: {float(np.sqrt(np.mean((new[okA]-val_target[okA])**2))):.4f}")
