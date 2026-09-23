"""
SUBMISSION V4: validated structural upgrades over v2 (harness: masked-row 0.7424 -> 0.7185).

Upgrades vs v2b:
  1. PC-denoising of anchor init AND cov observations (K=200 basis from detrended train field)
  2. Backward pass from the NEXT anchor for masked months between anchors (precision blend)
  3. D-tilde weights (0.70, 0.45, 0.073) from E7 grid
Variants: v4a full (phi=0.74) | v4b no-bwd (isolate bwd on real LB) | v4c full, phi=0.80
k=0 rows: unchanged from v2 (linear, covs(t+1), recency-weighted).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
W1, W2, W3 = 0.70, 0.45, 0.073

# ---------------- train ----------------
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
t_abs_tr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where(sxx > 100, np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

full = ~np.isnan(F).any(axis=0)
TD = t_abs_tr[:, None] - tbar_c[None, :]
A_dt = F64 - mu_c[None, :] - TD * beta_c[None, :]
A_dtf = A_dt[:, full]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(A_dtf, full_matrices=False)
V = Vt[:200].T.astype(np.float32)
def dn(field):
    out = field.copy()
    x = field[full]
    okx = np.isfinite(x)
    out[full] = np.where(okx, V @ (V.T @ np.where(okx, x, 0.0)), x)
    return out

# ---------------- test ----------------
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

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
def Dtil(t):
    return (W1*Dhat + W2*S + W3*trendex(t)).astype(np.float32)

# calibration (all anchors)
cs, zs, vfs = [], [], []
for a in anchors:
    dj = Dtil(a)
    w_ = dn(W[a])
    ok = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
    cs.append(np.cov(w_[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok]))
    vfs.append(np.nanvar(AF[a]-dj))
c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
H = c_/(LAM_F*var_f); R = max(varz - c_*c_/(LAM_F*var_f), 1e-4)
print(f"calib: H={H:.3f} R={R:.4f} var_f={var_f:.4f}")

def fwd(i, tm, phi):
    """forward Kalman from anchor i to target month tm (inclusive of obs at tm)"""
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = dn(AF[i] - Dtil(i))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    for m in range(i+1, tm+1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = dn(W[m])
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

def bwd(k, tm, phi):
    """backward Kalman from later anchor k down to target month tm (inclusive of obs at tm)"""
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = dn(AF[k] - Dtil(k))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[k]), P0, var_f).astype(np.float32)
    for m in range(k-1, tm-1, -1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = dn(W[m])
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

def masked_predict(phi, use_bwd):
    pred = np.full(len(test), np.nan, dtype=np.float64)
    anchors_arr = np.array(anchors)
    masked_months = sorted(set(ta[msk].tolist()))
    for m in masked_months:
        tm = m + 1
        sel = np.where((ta == m) & msk)[0]
        if len(sel) == 0: continue
        # last anchor before m
        i = int(anchors_arr[np.searchsorted(anchors_arr, m, side='right')-1])
        xf, Pf = fwd(i, tm, phi)
        later = anchors_arr[anchors_arr > m]
        if use_bwd and len(later):
            k = int(later[0])
            xb, Pb = bwd(k, tm, phi)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        dT = Dtil(tm)
        pred[sel] = mu_c[cc_t[sel]] + dT[cc_t[sel]] + x[cc_t[sel]]
    return pred

# ---------------- k=0 model (unchanged from v2) ----------------
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

print("\n--- v4a: full (phi=0.74, denoise, bwd) ---")
assemble(masked_predict(0.74, True), 'v4a')
print("--- v4b: no backward pass ---")
assemble(masked_predict(0.74, False), 'v4b')
print("--- v4c: full, phi=0.80 ---")
assemble(masked_predict(0.80, True), 'v4c')
print("\nDone. v4a/v4b/v4c saved.")
