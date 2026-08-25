"""
SUBMISSION V2: trend-augmented two-component model.

Generator (acf_resolution.py):
  TRAIN: TWS = mu_c + SLOW_c(t) + FAST_c(t) + noise
         SLOW: per-cell near-linear trend (24.9% var, = PC1, noiseless extrapolation)
         FAST: mean-reverting (detrended lag-1 ACF 0.68, negative by lag 24)
  TEST:  D(c,t) ~= w1*D-hat + w2*S + w3*trendex(t)

Variants: v2a phi=0.80 | v2b phi=0.74 | v2c phi=0.70 (all trendex D-tilde)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------------- load train ----------------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_train = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

# ---- per-cell OLS slope on calendar time (SLOW component) ----
F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_train[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxy = np.nansum(td * F64, axis=0)
sxx = np.nansum(td * td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), sxy / np.where(sxx > 0, sxx, 1), 0.0)
beta_c = beta_c.astype(np.float32)
print(f"beta_c: std={beta_c.std():.5f}/mo, mean={beta_c.mean():+.6f}, zeroed cells={int((beta_c==0).sum())}")

def trendex(t):
    """per-cell trend anomaly at calendar time t (relative to cell's own train mean)"""
    return (np.float64(t) - tbar_c) * beta_c

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
ta = test['t_abs'].values
cc_t = test['cc'].values
msk = test['masked'].values

# ---------------- global cov regression ----------------
Z = train[COVS].values.astype('float32')
yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

Zt_raw = test[COVS].values.astype('float32')
okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m}

# ---------------- anchors ----------------
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)

# ---------------- LOO weight fit for D-tilde ----------------
# target: F_a ; regressors: D-hat_{-a}, S, trendex(t_a)
Xs, ys = [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]]))
    ys.append(AF[a][ok])
Xw = np.vstack(Xs); yw = np.concatenate(ys)
w_ = np.linalg.solve(Xw.T@Xw + np.array([1e-3, 1e-3, 1e-3]), Xw.T@yw)
w1, w2, w3 = float(w_[0]), float(w_[1]), float(w_[2])
print(f"D-tilde weights (LOO fit): D-hat={w1:.3f}, S={w2:.3f}, trendex={w3:.3f}")

# compare LOO RMSE: old combo (0.676 D-hat + 0.490 S) vs new (w1,w2,w3)
rmse_old, rmse_new, rmse_dhat, rmse_trend = [], [], [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    tgt = AF[a][ok]
    rmse_dhat.append(np.sqrt(np.mean((dloo[ok]-tgt)**2)))
    rmse_old.append(np.sqrt(np.mean((0.676*dloo[ok]+0.490*S[ok]-tgt)**2)))
    rmse_new.append(np.sqrt(np.mean((w1*dloo[ok]+w2*S[ok]+w3*tx[ok]-tgt)**2)))
    rmse_trend.append(np.sqrt(np.mean((tx[ok]-tgt)**2)))
print(f"LOO RMSE at anchors: D-hat only={np.mean(rmse_dhat):.4f} | old combo={np.mean(rmse_old):.4f} | "
      f"new combo={np.mean(rmse_new):.4f} | trendex only={np.mean(rmse_trend):.4f}")

def Dtil(t):
    return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

# ---------------- Kalman ----------------
LAM_F = 0.84
def calibrate():
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a)
        ok = np.isfinite(W[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W[a][ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

H, R, var_f = calibrate()
print(f"Kalman: H={H:.3f} R={R:.4f} var_f={var_f:.4f}")

def kalman_predict(phi_f):
    q = LAM_F*var_f*(1-phi_f**2)
    P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for a in anchors:
        dja = Dtil(a)
        fa = AF[a] - dja
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        x0 = x.copy()
        sel0 = np.where((ta == a) & msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[cc_t[sel0]] + Dtil(a+1)[cc_t[sel0]] + phi_f*x0[cc_t[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x
            P = phi_f**2*P + q
            if m in W:
                wv = W[m]
                okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((ta == m) & msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in W:
                wv = W[tm]
                okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[cc_t[sel]] + Dtil(tm)[cc_t[sel]] + x2[cc_t[sel]]
    return pred

# ---------------- k=0 model (unchanged from V1) ----------------
Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy()
Lk['t_next'] = Lk['t_abs'] + 1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values
yr = mgk['t_abs'].values // 12
Xf = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm],
    *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))
]).astype(np.float32)
yA = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
Xf = np.nan_to_num(Xf, nan=0.0)
w_rec = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
sw = np.sqrt(w_rec)
selF = has_nxt_tr
A_f = Xf[selF]*sw[selF,None]
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
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
k0 = np.full(len(test), np.nan, dtype=np.float64)
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
k0[use_full] = mu_c[cct[use_full]] + Xt[use_full] @ coefF
k0[use_red]  = mu_c[cct[use_red]]  + Xt[use_red][:, colsR] @ coefR
bad = (~msk) & np.isnan(k0)
if bad.any(): k0[bad] = mu_c[cct][bad]

# ---------------- assemble ----------------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm = ~msk

def assemble(masked_pred, tag):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0[unm]
    pred[msk] = masked_pred[msk]
    pred = np.where(np.isnan(pred), mu_c[cc_t], pred)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} "
          f"(unm std={pred[unm].std():.3f}, masked std={pred[msk].std():.3f})", flush=True)

print("\n--- v2a: phi_f=0.80, trendex D-tilde ---")
assemble(kalman_predict(0.80), 'v2a')
print("--- v2b: phi_f=0.74 ---")
assemble(kalman_predict(0.74), 'v2b')
print("--- v2c: phi_f=0.70 ---")
assemble(kalman_predict(0.70), 'v2c')
print("\nDone. v2a/v2b/v2c saved.")
