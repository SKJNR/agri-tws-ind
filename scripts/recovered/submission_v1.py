"""
SUBMISSION V1: two-component generator model (cross-AI Rounds 3-5).

Test-era generator: TWS = mu_c + D(c,t) + f(c,t) + noise
  D : quasi-static spatial offset (std~0.90, ~51% of anchor variance), drifts ~2yr scale
  f : fast component, phi_f in [0.74,0.85], noise fraction lambda_f~0.84
  covs: monthly deviation W_m tracks (F - D-tilde) at r~0.45; static part S overlaps D

Variants:
  V1a: pure empirical decay  pred = mu + r(h)*(anchor - mu)     [no covs, no D split]
  V1b: two-component Kalman, phi_f=0.80, GLOBAL D-tilde (anchor mean + cov static combo)
  V1c: two-component Kalman, phi_f=0.85, ERA-INTERPOLATED D-tilde
k=0 rows (all variants): linear model TWS_t + covs(t) + covs(t+1) in anomaly space,
                          recency-weighted; reduced model when covs(t+1) unavailable
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
RH = {1:0.874, 2:0.781, 3:0.712, 4:0.660, 5:0.623, 6:0.595, 7:0.574}

# ---------------- load train ----------------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
mu_c = train.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

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

# ---------------- global cov regression (raw) ----------------
Z = train[COVS].values.astype('float32')
yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# ---------------- cov fields, S, W ----------------
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

# ---------------- anchors, D-tilde ----------------
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
# LOO combo weights: F_j ~ wD * D-hat_{-j} + wS * S
Xs, ys = [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a])
    Xs.append(np.column_stack([dloo[ok], S[ok]])); ys.append(AF[a][ok])
Xw = np.vstack(Xs); yw = np.concatenate(ys)
w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(2), Xw.T@yw)
wD, wS = float(w_[0]), float(w_[1])
early = [a for a in anchors if a < 2017*12]
late  = [a for a in anchors if a >= 2018*12]
D_early = np.nanmean(np.array([AF[a] for a in early]), axis=0)
D_late  = np.nanmean(np.array([AF[a] for a in late]), axis=0)
c_early, c_late = float(np.mean(early)), float(np.mean(late))
Dtil_glb   = wD*Dhat + wS*S
Dtil_early = wD*D_early + wS*S
Dtil_late  = wD*D_late + wS*S
def D_glb(t): return Dtil_glb
def D_int(t):
    u = np.clip((t - c_early)/(c_late - c_early), 0.0, 1.0)
    return (1-u)*Dtil_early + u*Dtil_late
print(f"anchors(t_abs)={anchors} | combo wD={wD:.3f} wS={wS:.3f} | era centers {c_early:.0f}->{c_late:.0f}")

# ---------------- Kalman observation calibration ----------------
LAM_F = 0.84
def calibrate(Dfun):
    cs, zs, vf = [], [], []
    for a in anchors:
        dj = Dfun(a)
        ok = np.isfinite(W[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        z = W[a][ok]; y = (AF[a]-dj)[ok]
        cs.append(np.cov(z, y)[0,1]); zs.append(np.var(z))
    c = float(np.mean(cs)); varz = float(np.mean(zs))
    var_f = float(np.mean([np.nanvar(AF[a]-Dfun(a)) for a in anchors]))
    var_x = LAM_F*var_f
    H = c/var_x
    R = varz - c*c/var_x
    return H, R, var_f

# ---------------- two-component Kalman ----------------
def kalman_predict(phi_f, Dfun):
    H, R, var_f = calibrate(Dfun)
    q = LAM_F*var_f*(1-phi_f**2)
    P0 = LAM_F*(1-LAM_F)*var_f
    print(f"  [kalman] H={H:.3f} R={R:.4f} var_f={var_f:.4f}")
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for a in anchors:
        dja = Dfun(a)
        x = np.where(np.isfinite(AF[a]-dja), LAM_F*(AF[a]-dja), 0.0).astype(np.float32)
        x = np.nan_to_num(x)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        x0 = x.copy()
        # masked rows AT the anchor month itself (k=0 but masked)
        sel0 = np.where((ta == a) & msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[cc_t[sel0]] + Dfun(a+1)[cc_t[sel0]] + phi_f*x0[cc_t[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f * x
            P = phi_f**2 * P + q
            if m in W:
                wv = W[m]
                okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P + R), 0.0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((ta == m) & msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f * x
            if tm in W:
                wv = W[tm]
                okw = np.isfinite(wv)
                P2 = phi_f**2 * P + q
                K2 = np.where(okw, P2*H/(H*H*P2 + R), 0.0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[cc_t[sel]] + Dfun(tm)[cc_t[sel]] + x2[cc_t[sel]]
    return pred

# ---------------- V1a: pure empirical decay ----------------
def decay_predict():
    pred = np.full(len(test), np.nan, dtype=np.float64)
    anchors_arr = np.array(anchors)
    row_anchor = anchors_arr[np.searchsorted(anchors_arr, ta, side='right')-1]
    for a in anchors:
        sel = np.where((row_anchor == a) & msk)[0]
        if len(sel) == 0: continue
        h = (ta[sel] + 1 - a).astype(int)
        coef_h = np.array([RH.get(int(v), RH[7]) for v in h], dtype=np.float32)
        anom = AF[a][cc_t[sel]]
        pred[sel] = mu_c[cc_t[sel]] + coef_h*np.nan_to_num(anom)
    return pred

# ---------------- k=0 linear models (anomaly space, recency-weighted) ----------------
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
# honest sanity: fit <=2012, eval >=2013 (rows with covs_nxt)
fitm = selF & (yr <= 2012); evam = selF & (yr >= 2013)
Af2 = Xf[fitm]*sw[fitm,None]
cf2 = np.linalg.solve(Af2.T@Af2 + 1e-3*np.eye(12), Af2.T@(yA[fitm]*sw[fitm]))
rmse_k0 = float(np.sqrt(np.mean((Xf[evam]@cf2 - yA[evam])**2)))
print(f"[k=0] honest val RMSE (fit<=2012, eval>=2013, with covs(t+1)): {rmse_k0:.4f}")

# apply to test unmasked rows
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
n_k0nan = int((~msk & np.isnan(k0)).sum())
if n_k0nan: k0[~msk & np.isnan(k0)] = mu_c[cct][~msk & np.isnan(k0)]
print(f"[k=0] unmasked rows: full model {int(use_full.sum())}, reduced {int(use_red.sum())}, fallback {n_k0nan}")

# ---------------- assemble ----------------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
assert len(sub) == len(test)
unm = ~msk

def assemble(masked_pred, tag):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0[unm]
    pred[msk] = masked_pred[msk]
    nan_left = int(np.isnan(pred).sum())
    if nan_left:
        fill = np.where(np.isnan(pred), mu_c[cc_t], pred)
        pred = fill
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} "
          f"(unm std={pred[unm].std():.3f}, masked std={pred[msk].std():.3f}, nan_filled={nan_left})", flush=True)

print("\n--- V1a: empirical decay ---")
assemble(decay_predict(), 'v1a')
print("--- V1b: two-component Kalman phi=0.80, global D-tilde ---")
assemble(kalman_predict(0.80, D_glb), 'v1b')
print("--- V1c: two-component Kalman phi=0.85, era-interpolated D-tilde ---")
assemble(kalman_predict(0.85, D_int), 'v1c')
print("\nDone. V1a/V1b/V1c saved.")
