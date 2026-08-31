"""cv_v4_diagnostic.py — Case A vs Case B: why did v4a (LB 0.7155) lose to v2b (LB 0.7137)
when the anchor-LOO harness said v4 was better (0.7424 -> 0.7185)?

Hypothesis: the val window has DENSER anchors (14 in 2.5yr, gaps 1-5mo) than the real test
(6 anchors over 3.25yr, gaps ~6mo). The backward pass gains value when the next anchor is
CLOSE; with test's long gaps it gains little and the PC-denoise may hurt.

Test: run v2b and v4 configs on the SAME val masked rows under two anchor structures:
  dense  : all 14 calendar anchors (as cv_lb_correlation.py used)  -> CV optimistic for bwd
  sparse : 6 anchors spaced ~6 months (2013-01, 2013-07, 2014-01, 2014-07, 2015-01, 2015-07)
           -> mimics real test anchor density

If v4 beats v2b on dense but loses on sparse -> anchor density is the mechanism;
the sparse CV is the right harness for v4-family decisions going forward.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
K_PCS = 200

# ---------------- load train, split ----------------
print("Loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms_all = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms_all)}

fit = train[train['time'].dt.year <= 2012]
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values
val_ta = val['t_abs'].values

# calendar-month masking from test
test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
test_raw['masked'] = test_raw['TWS_t_masked'].astype(bool)
cal_mask_frac = test_raw.groupby(test_raw['time'].dt.month)['masked'].mean()
val['cal_mon'] = val['time'].dt.month
val_msk = (val['cal_mon'].map(cal_mask_frac).values > 0.5)
val_tws_visible = val['TWS_t'].values.astype(np.float32)   # NOTE: we do NOT nan-out; visibility governed by val_msk
print(f"val: {len(val):,} rows, masked {val_msk.sum():,}")

# ---------------- fit infrastructure (2002-2012 only) ----------------
print("Fitting infrastructure on 2002-2012...", flush=True)
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

Z = fit[COVS].values.astype('float32')
yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# fit-period field matrix + trend + PC basis
F = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
fit_ii = fit['ym'].map(ym_to_i).values
F[fit_ii, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
fit_months = np.array([ym_to_i[int(v)] for v in np.sort(fit['ym'].unique())])

F64 = F[fit_months].astype(np.float64)      # fit months only
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[fit_months][:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxy = np.nansum(td*F64, axis=0)
sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), sxy/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# PC basis from fit detrended fields (full-history cells)
A_dt = F64 - mu_c[None, :] - (t_abs_yms[fit_months][:, None] - tbar_c[None, :]) * beta_c[None, :]
full = ~np.isnan(F64).any(axis=0)
A_dtf = A_dt[:, full]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(A_dtf, full_matrices=False)
V = Vt[:K_PCS].T.astype(np.float32)
print(f"PC basis: K={K_PCS}, full-history cells={full.sum()}, explains "
      f"{(np.sum(Vt[:K_PCS]**2)/np.sum(Vt**2)*100):.1f}% of fit detrended var")
def dn(field):
    out = field.copy()
    x = field[full]
    okx = np.isfinite(x)
    out[full] = np.where(okx, V @ (V.T @ np.where(okx, x, 0.0)), x)
    return out

# ---------------- val cov fields, S, W ----------------
Zv = val[COVS].values.astype('float32')
okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m_v}

# ---------------- two anchor structures ----------------
mfrac_v = pd.Series(val_msk).groupby(val_ta).mean()
dense_anchors = sorted(int(v) for v in mfrac_v[mfrac_v < 0.5].index)
# sparse set: ~6mo spacing, ONLY months that actually exist in val (train has month gaps, e.g. Jul 2014 missing)
sparse_anchors = [m for m in [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6] if m in set(val_ta.tolist())]
print(f"\ndense anchors ({len(dense_anchors)}): {dense_anchors}")
print(f"sparse anchors ({len(sparse_anchors)}): {sparse_anchors}")
print(f"gaps: {np.diff(sparse_anchors).tolist()}")

# ---------------- evaluation harness ----------------
def run_config(phi, use_dn, use_bwd, anchor_set, label):
    anchors = list(anchor_set)
    # anchor fields from VISIBLE val TWS at anchor months
    AF = {}
    for a in anchors:
        sel = (val_ta == a) & (~val_msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[val_cc[sel]] = val_tws_visible[sel]
        AF[a] = fa - mu_c
    Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    # LOO weights for D-tilde
    Xs, ys = [], []
    for a in anchors:
        dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
        tx = trendex(a).astype(np.float32)
        ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
        Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    w_ = np.linalg.solve(Xw.T@Xw + np.array([1e-3,1e-3,1e-3]), Xw.T@yw)
    w1, w2, w3 = float(w_[0]), float(w_[1]), float(w_[2])
    def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

    # calibration
    cs, zs, vfs = [], [], []
    for a in anchors:
        if a not in W: continue
        dj = Dtil(a)
        wobs = dn(W[a]) if use_dn else W[a]
        ok = np.isfinite(wobs) & np.isfinite(AF[a]) & np.isfinite(dj)
        if ok.sum() < 100: continue
        cs.append(np.cov(wobs[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(wobs[ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(LAM_F*var_f); R = max(varz - c_*c_/(LAM_F*var_f), 1e-4)

    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    obs = (lambda x: dn(x)) if use_dn else (lambda x: x)

    def fwd(i, tm):
        f0 = obs(AF[i] - Dtil(i))
        x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        for m in range(i+1, tm+1):
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = obs(W[m]); okw = np.isfinite(w_)
                Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
                P = np.where(okw, (1-Kg*H)*P, P)
        return x, P

    def bwd(k, tm):
        f0 = obs(AF[k] - Dtil(k))
        x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[k]), P0, var_f).astype(np.float32)
        for m in range(k-1, tm-1, -1):
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = obs(W[m]); okw = np.isfinite(w_)
                Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
                P = np.where(okw, (1-Kg*H)*P, P)
        return x, P

    anchors_arr = np.array(anchors)
    pred = np.full(len(val), np.nan, dtype=np.float64)
    for m in sorted(set(val_ta[val_msk].tolist())):
        tm = m + 1
        sel = np.where((val_ta == m) & val_msk)[0]
        if len(sel) == 0: continue
        i = int(anchors_arr[np.searchsorted(anchors_arr, m, side='right')-1])
        if i > m: i = int(anchors_arr[anchors_arr <= m][-1]) if (anchors_arr <= m).any() else None
        if i is None: continue
        xf, Pf = fwd(i, tm)
        later = anchors_arr[anchors_arr > m]
        if use_bwd and len(later):
            k = int(later[0])
            xb, Pb = bwd(k, tm)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        dT = Dtil(tm)
        pred[sel] = mu_c[val_cc[sel]] + dT[val_cc[sel]] + x[val_cc[sel]]

    mask = val_msk & np.isfinite(pred) & np.isfinite(val_target)
    rmse = float(np.sqrt(np.mean((pred[mask]-val_target[mask])**2)))
    print(f"  [{label}] w=({w1:.3f},{w2:.3f},{w3:.3f}) H={H:.3f} R={R:.4f} | rows={mask.sum()} RMSE={rmse:.4f}")
    return rmse

# ---------------- run the diagnostic ----------------
results = {}
print("\n=== DENSE anchors (14, gaps 1-5mo) — CV as previously run ===")
results[('dense','v2b')]  = run_config(0.74, False, False, dense_anchors, 'v2b  phi=.74 no-dn no-bwd')
results[('dense','v4b')]  = run_config(0.74, True,  False, dense_anchors, 'v4b  phi=.74 dn     no-bwd')
results[('dense','v4a')]  = run_config(0.74, True,  True,  dense_anchors, 'v4a  phi=.74 dn     bwd')
results[('dense','v4c')]  = run_config(0.80, True,  True,  dense_anchors, 'v4c  phi=.80 dn     bwd')

print("\n=== SPARSE anchors (6, gaps 6mo) — mimics real test anchor density ===")
results[('sparse','v2b')] = run_config(0.74, False, False, sparse_anchors, 'v2b  phi=.74 no-dn no-bwd')
results[('sparse','v4b')] = run_config(0.74, True,  False, sparse_anchors, 'v4b  phi=.74 dn     no-bwd')
results[('sparse','v4a')] = run_config(0.74, True,  True,  sparse_anchors, 'v4a  phi=.74 dn     bwd')
results[('sparse','v4c')] = run_config(0.80, True,  True,  sparse_anchors, 'v4c  phi=.80 dn     bwd')

# ---------------- verdict ----------------
print("\n=== VERDICT ===")
print(f"{'config':<8}{'dense CV':<12}{'sparse CV':<12}{'LB':<10}")
LB = {'v2b': 0.7137, 'v4a': 0.7155, 'v4b': None, 'v4c': 0.7144}
for cfg in ['v2b','v4b','v4a','v4c']:
    d = results[('dense',cfg)]; s = results[('sparse',cfg)]
    lb = LB[cfg] if LB[cfg] else float('nan')
    print(f"{cfg:<8}{d:<12.4f}{s:<12.4f}{lb:<10.4f}")

dense_order = sorted(['v2b','v4a','v4c'], key=lambda c: results[('dense',c)])
sparse_order = sorted(['v2b','v4a','v4c'], key=lambda c: results[('sparse',c)])
lb_order = sorted(['v2b','v4a','v4c'], key=lambda c: LB[c])
print(f"\ndense ranks:  {dense_order}")
print(f"sparse ranks: {sparse_order}")
print(f"LB ranks:     {lb_order}")
if sparse_order == lb_order and dense_order != lb_order:
    print("\n==> ANCHOR DENSITY IS THE MECHANISM. Sparse CV replicates LB order; use sparse CV for v4-family decisions.")
elif dense_order == lb_order and sparse_order != lb_order:
    print("\n==> DENSE CV replicates LB order (unexpected).")
elif sparse_order == lb_order and dense_order == lb_order:
    print("\n==> Both CVs replicate LB order. CV verdict intact.")
else:
    print("\n==> NEITHER CV replicates LB order. CV verdict broken — fix before further submissions.")
