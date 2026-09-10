"""V14 K0 LANE — val head-to-head on h=1 (k0-analog) rows: structured Kalman h=1 path
vs the k0-B LGBM stack (the LB-proven recipe), and their blends.

WHY: public k0 share = 46,939/109,222 = 0.430 EXACT. Implied public k0 RMSE:
  v10b (static Dcache) ~0.600 -> v12b (era Dcache) ~0.587.
  Top-10 public 0.667 needs k0 ~0.505 with masked fixed, or k0 ~0.55 + masked ~0.72.
  MOHAR (0.5596 total) implies k0 ~0.45-0.50 AND masked ~0.60-0.63 (both lanes structural).

PROTOCOL (honest, from v12_attribution — Spearman 1.0 vs LB on 5 variants):
  fit = train 2002-2012 (all infra + LGBM training), val = 2013-2015 with the test's
  calendar-month masking simulated. k0-analog rows = val rows at val-ANCHOR months with
  TWS_t visible (row-wise problem identical to real k0 rows: TWS observed, predict t+1).
  Rows where t+1 is itself a val anchor month are EXCLUDED (target would be visible in
  the anchor fields = leakage; on the real test this never happens — audited).

METHODS (all scored on identical rows):
  M0  persistence: pred = TWS_obs
  M1  Kalman h=1 (frozen v10b H/R, phi-ens {0.74,0.80}, W(t+1) obs update)
      Dcache combos: (init,pred) in {static,era}^2
  M1d M1 + PC-denoise (K=200) of the anchor fast-field init
      [val UNDERESTIMATES this: val anchor fields ~50-65% visible vs 100% on test]
  M2  k0-B two-comp stack (linear+LGBM 50/50), trained on fit, Dcache static / era
      [M2-era = the v12b recipe analog = the number to beat]
  M3  blends w*M1best + (1-w)*M2best, w grid
Scoring: RMSE raw + GAU10-smoothed (test applies GAU10 to k0 rows).
Secondary set: ALL visible val rows (h=1 everywhere, ~2x rows, excludes t+1-anchor rows).

PRE-REGISTERED:
  - M2-era >= M2-static expected (v12b's LB win replicating on val). If NOT, the val
    harness cannot rank k0 variants -> distrust it for k0 iteration.
  - Build v14 tonight iff best method beats M2-era by >= 0.004 (smoothed, primary set).
"""
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
K_DN = 200
TAU = 12.0

# ---------------- load train (v12_attribution verbatim) ----------------
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

fit = train[train['time'].dt.year <= 2012].copy()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
print(f"fit {len(fit):,} | val {len(val):,}", flush=True)

# ---------------- masking pattern from test ----------------
test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
test_raw['masked'] = test_raw['TWS_t_masked'].astype(bool)
test_raw['cal_mon'] = test_raw['time'].dt.month
cal_mask_frac = test_raw.groupby('cal_mon')['masked'].mean()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values; val_ta = val['t_abs'].values; val_msk = val['masked'].values

# ---------------- infra on fit only (v12_attribution verbatim) ----------------
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
sxy = np.nansum(td*F64, axis=0); sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), sxy/np.where(sxx > 0, sxx, 1), 0).astype('float32')
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype('float32')

# grid utils
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
BOX2 = {(di,dj): 1.0 for di in range(-2,3) for dj in range(-2,3)}
def gaussW(sig, r=4):
    return {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*sig*sig)))
            for di in range(-r,r+1) for dj in range(-r,r+1)
            if np.exp(-(di*di+dj*dj)/(2*sig*sig)) > 0.01}
GAU10 = gaussW(1.0)
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

# ---------------- val cov fields, S, W_pool (v12_attribution verbatim) ----------------
Zv = val[COVS].values.astype('float32')
okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W_raw = {int(m): cov_field[m] - S for m in all_m_v}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W_raw.items()}

# ---------------- val anchors + AF (v12_attribution verbatim) ----------------
mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v < 0.5].index)
print(f"val anchors ({len(anchors_v)}): {anchors_v}")
AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c

Dhat_raw = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)
Dhat = np.where(np.isfinite(Dhat_raw), Dhat_raw, 0.0).astype('float32')

def dhat_era(t, excl=None):
    others = [b for b in anchors_v if b != excl]
    ws = np.array([np.exp(-abs(t-b)/TAU) for b in others], dtype=np.float64)
    stack = np.array([AF[b] for b in others], dtype=np.float64)
    ok = np.isfinite(stack)
    wmat = np.where(ok, ws[:, None], 0.0)
    num = np.nansum(np.where(ok, stack, 0.0)*wmat, axis=0)
    den = wmat.sum(axis=0)
    d = np.where(den > 1e-9, num/np.maximum(den, 1e-9), np.nan)
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)

def dhat_static(excl=None):
    others = [b for b in anchors_v if b != excl]
    d = np.nanmean(np.array([AF[b] for b in others]), axis=0)
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)

def fit_weights(dhatfun):
    Xs, ys = [], []
    for a in anchors_v:
        d_ = dhatfun(a, excl=a); tx = trendex(a)
        ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
        Xs.append(np.column_stack([d_[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
    w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
    return tuple(map(float, w_))
w1s, w2s, w3s = fit_weights(lambda t, excl: dhat_static(excl))
w1e, w2e, w3e = fit_weights(lambda t, excl: dhat_era(t, excl=excl))
print(f"static weights: {w1s:.3f}/{w2s:.3f}/{w3s:.3f} | era weights: {w1e:.3f}/{w2e:.3f}/{w3e:.3f}")

def Dtil_static(t): return (w1s*dhat_static() + w2s*S + w3s*trendex(t)).astype(np.float32)
def Dtil_era(t):    return (w1e*dhat_era(t) + w2e*S + w3e*trendex(t)).astype(np.float32)

# ---------------- frozen H/R: v10b recipe = static-full at anchors ----------------
def calibrate():
    cs, zs, vfs = [], [], []
    for a in anchors_v:
        dj = Dtil_static(a)
        ok = np.isfinite(W_pool[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W_pool[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W_pool[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    return c_/(LAM_F*var_f), max(varz - c_*c_/(LAM_F*var_f), 1e-4), var_f
H, R, var_f = calibrate()
q = LAM_F*var_f*(1-0.74**2); P0 = LAM_F*(1-LAM_F)*var_f
print(f"frozen calibration: H={H:.4f} R={R:.4f} var_f={var_f:.4f}")

# ---------------- D-cache: Dtil fields for every needed month ----------------
months_needed = sorted(set([int(m) for m in np.unique(val_ta)]) | set([int(m)+1 for m in np.unique(val_ta)]))
Dcache = {}
for dc, fn in [('static', Dtil_static), ('era', Dtil_era)]:
    for m in months_needed:
        Dcache[(dc, m)] = fn(m)
print(f"D-cache built: {len(Dcache)} fields")

# ---------------- PC basis from fit detrended anomalies (for init-denoise) ----------------
print("building PC basis (fit era)...", flush=True)
fit_ok = np.isfinite(Fmat)
full_cells = fit_ok.all(axis=0)
A_dt = F64 - mu_c[None, :] - (t_abs_yms[:, None] - tbar_c[None, :]) * beta_c[None, :]
A_dtf = A_dt[:, full_cells]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
U, S_, Vt = np.linalg.svd(A_dtf, full_matrices=False)
VK = Vt[:K_DN].T.astype('float32')
print(f"  full cells: {int(full_cells.sum())} | PC1-200 var: {(S_[:200]**2).sum()/(S_**2).sum()*100:.1f}%")
def denoise(field):
    out = field.copy()
    x = field[full_cells]
    okx = np.isfinite(x)
    xz = np.where(okx, x, 0.0)
    proj = VK @ (VK.T @ xz)
    out[full_cells] = np.where(okx, proj, x)
    return out
# denoised x0 cache at anchor months: LAM_F * denoise(AF[a] - Dtil_dc(a))
dn_cache = {}
for a in anchors_v:
    for dc in ['static', 'era']:
        fld = AF[a] - Dcache[(dc, a)]
        dn_cache[(a, dc)] = np.where(np.isfinite(fld), LAM_F*denoise(fld), np.nan).astype(np.float32)

# ---------------- k0-B two-comp stack: trained on fit (honest) ----------------
print("k0-B: training two-comp + LGBM on fit...", flush=True)
Lk = fit[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = fit[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values; yr = mgk['t_abs'].values // 12
colsR = [0,1,2,3,4,5,11]

def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr, dtype=np.float64)-tbar_c[cc_arr])*beta_c[cc_arr]).astype('float32')
slow0 = trendex_vec(mgk['t_abs'].values, ccm); slow1 = trendex_vec(mgk['t_next'].values, ccm)
Xlin = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm] - slow0,
    *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))]).astype('float32')
yB = (mgk['target'].values - mu_c[ccm] - slow1).astype('float32')
Xlin = np.nan_to_num(Xlin, nan=0.0)
w_recB = np.where(yr <= 2006, 1.0, np.where(yr <= 2009, 1.5, 2.0)).astype('float32')
swB = np.sqrt(w_recB); selF = has_nxt_tr
A_fB = Xlin[selF]*swB[selF,None]
coefF_B = np.linalg.solve(A_fB.T@A_fB + 1e-3*np.eye(12), A_fB.T@(yB[selF]*swB[selF]))
A_rB = Xlin[:, colsR]*swB[:,None]
coefR_B = np.linalg.solve(A_rB.T@A_rB + 1e-3*np.eye(7), A_rB.T@(yB*swB))

lat_cc = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_arr = lat_cc['lat'].values.astype('float32'); lon_arr = lat_cc['lon'].values.astype('float32')
mon_tr = (mgk['t_abs'].values % 12) + 1
XLB = np.column_stack([Xlin[:,0], slow1, Xlin[:,1:11],
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm], lat_arr[ccm], lon_arr[ccm]]).astype('float32')
okY = np.isfinite(yB)
params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=8)
ds  = lgb.Dataset(XLB[selF], label=yB[selF], weight=w_recB[selF])
dsr = lgb.Dataset(XLB[~selF & okY], label=yB[~selF & okY], weight=w_recB[~selF & okY])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, dsr, num_boost_round=400)
print("k0-B LGBM trained (fit era only — honest).")

# ---------------- per-month cov arrays for t+1 features ----------------
covm = {}
for m in np.sort(val['t_abs'].unique()):
    arr = np.full((n_cells, 5), np.nan, dtype='float32')
    sub = val[val['t_abs'] == m]
    arr[sub['cc'].values] = sub[COVS].values.astype('float32')
    covm[int(m)] = arr

# ---------------- k0-analog row sets ----------------
anchor_set = set(anchors_v)
tws_arr = val['TWS_t'].values
tgt_finite = np.isfinite(val_target)
tws_finite = np.isfinite(tws_arr)
next_is_anchor = np.array([(int(m)+1) in anchor_set for m in val_ta])
sel_prim = np.array([m in anchor_set for m in val_ta]) & (~val_msk) & tgt_finite & tws_finite & (~next_is_anchor)
sel_sec = (~val_msk) & tgt_finite & tws_finite & (~next_is_anchor)
print(f"\nk0-analog rows: PRIMARY (anchor months) {int(sel_prim.sum()):,} | SECONDARY (all visible) {int(sel_sec.sum()):,}")
print(f"  (leak-excluded rows [t+1 = anchor]: {int(next_is_anchor.sum()):,})")

# ---------------- methods (month-wise vectorized) ----------------
def m1_kalman(idx, dc_init, dc_pred, use_denoise=False):
    ta_i = val_ta[idx]; cc_i = val_cc[idx]; tws_i = tws_arr[idx]
    out = np.zeros(int(idx.sum()))
    for phi in [0.74, 0.80]:
        P1 = phi**2*P0 + q; K1 = P1*H/(H*H*P1 + R)
        pred = np.zeros(len(ta_i))
        for a in np.unique(ta_i):
            selm = ta_i == a
            cc_m = cc_i[selm]
            dt0 = Dcache[(dc_init, int(a))][cc_m]
            x0 = LAM_F*(tws_i[selm] - mu_c[cc_m] - dt0)
            if use_denoise:
                dnf = dn_cache.get((int(a), dc_init))
                if dnf is not None:
                    xd = dnf[cc_m]
                    x0 = np.where(np.isfinite(xd), xd, x0)
            x1 = phi*x0
            wv = W_pool.get(int(a)+1)
            if wv is not None:
                w = wv[cc_m]; okw = np.isfinite(w)
                x1 = np.where(okw, x1 + K1*(w - H*x1), x1)
            dt1 = Dcache[(dc_pred, int(a)+1)][cc_m]
            pred[selm] = mu_c[cc_m] + dt1 + x1
        out += pred/2.0
    return out

def m2_stack(idx, dc):
    ta_i = val_ta[idx]; cc_i = val_cc[idx]; tws_i = tws_arr[idx]
    n = int(idx.sum())
    dt0 = np.zeros(n, dtype=np.float32); dt1 = np.zeros(n, dtype=np.float32)
    cov_a = np.nan_to_num(val[COVS].values.astype('float32')[idx] - clim[cc_i])
    cov_n = np.zeros((n, 5), dtype=np.float32); has_n = np.zeros(n, dtype=np.float32)
    for a in np.unique(ta_i):
        selm = ta_i == a
        cc_m = cc_i[selm]
        dt0[selm] = Dcache[(dc, int(a))][cc_m]
        dt1[selm] = Dcache[(dc, int(a)+1)][cc_m]
        nx = covm.get(int(a)+1)
        if nx is not None:
            nv = nx[cc_m]
            okn = np.isfinite(nv).all(axis=1)
            cov_n[selm] = np.where(okn[:, None], nv - clim[cc_m], 0.0)
            has_n[selm] = okn.astype(np.float32)
    Xv = np.column_stack([tws_i - mu_c[cc_i] - dt0, cov_a, cov_n, np.ones(n)]).astype('float32')
    mon = (ta_i % 12) + 1
    XLv = np.column_stack([Xv[:,0], dt1, Xv[:,1:11], has_n,
        np.sin(2*np.pi*mon/12), np.cos(2*np.pi*mon/12),
        beta_c[cc_i], mu_c[cc_i], lat_arr[cc_i], lon_arr[cc_i]]).astype('float32')
    lin = mu_c[cc_i] + dt1 + Xv @ coefF_B
    lgbm_p = mu_c[cc_i] + dt1 + bst.predict(XLv)
    return lin, lgbm_p, 0.5*lin + 0.5*lgbm_p

def gau10_smooth(idx, preds):
    out = preds.copy()
    ta_i = val_ta[idx]; cc_i = val_cc[idx]
    for m in np.unique(ta_i):
        selm = ta_i == m
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[cc_i[selm]] = preds[selm].astype('float32')
        sm = from_grid(kpool(to_grid(cm), GAU10))
        out[selm] = sm[cc_i[selm]]
    return out

def rmse(p, t): return float(np.sqrt(np.mean((p - t)**2)))

# ---------------- run all methods ----------------
def run_set(idx, label):
    print(f"\n=== {label} (n={int(idx.sum()):,}) ===", flush=True)
    tgt = val_target[idx]
    ta_i = val_ta[idx]
    tws_i = tws_arr[idx]
    results = {}

    results['M0 persistence'] = tws_i.astype(np.float64)

    for dc_i, dc_p in [('static','static'), ('era','era'), ('static','era'), ('era','static')]:
        results[f'M1 kalman {dc_i}->{dc_p}'] = m1_kalman(idx, dc_i, dc_p)
    results['M1d kalman era->era +dnInit'] = m1_kalman(idx, 'era', 'era', use_denoise=True)

    lin_s, lgb_s, blend_s = m2_stack(idx, 'static')
    lin_e, lgb_e, blend_e = m2_stack(idx, 'era')
    results['M2 stack static blend'] = blend_s
    results['M2 stack era blend'] = blend_e
    results['M2 stack era lin-only'] = lin_e
    results['M2 stack era lgb-only'] = lgb_e

    m1_names = [k for k in results if k.startswith('M1')]
    m1_best_name = min(m1_names, key=lambda k: rmse(results[k], tgt))
    m1_best = results[m1_best_name]
    for w in [0.25, 0.40, 0.50, 0.60, 0.75]:
        results[f'M3 blend {int(w*100)}pct M1'] = w*m1_best + (1-w)*blend_e

    print(f"  [best M1 by raw RMSE: {m1_best_name}]")
    print(f"  {'method':<32} {'raw':>8} {'GAU10':>8}")
    scored = {}
    for k, p in results.items():
        ps = gau10_smooth(idx, p)
        r_raw, r_sm = rmse(p, tgt), rmse(ps, tgt)
        scored[k] = (r_raw, r_sm)
        print(f"  {k:<32} {r_raw:8.4f} {r_sm:8.4f}", flush=True)

    yr_i = ta_i // 12
    e13 = yr_i <= 2013; l15 = yr_i >= 2015
    msg = []
    for k in [m1_best_name, 'M2 stack era blend', 'M3 blend 50pct M1']:
        ps = gau10_smooth(idx, results[k])
        msg.append(f"{k}: 13={rmse(ps[e13],tgt[e13]):.4f} 15={rmse(ps[l15],tgt[l15]):.4f}")
    print("  era breakdown (GAU10): " + " | ".join(msg))
    return scored

prim = run_set(sel_prim, "PRIMARY: k0-analog rows (visible at val anchor months)")
# NOTE: sel_sec == sel_prim verified (n and all RMSEs identical): under the calendar
# masking pattern, visible val rows exist ONLY at anchor months -> no separate set.

# ---------------- PHASE 2: refined k0 grids (lin/lgb weight x kalman blend) ----------------
print("\n=== PHASE 2: refined k0 grids (PRIMARY) ===", flush=True)
idx = sel_prim
tgt = val_target[idx]
lin_e, lgb_e, _ = m2_stack(idx, 'era')
k_ss = m1_kalman(idx, 'static', 'static')
k_ee = m1_kalman(idx, 'era', 'era')
grids = {}
for w_lgb in [0.50, 0.60, 0.75, 1.00]:
    stack = w_lgb*lgb_e + (1-w_lgb)*lin_e
    tag = f'stack lgb{int(w_lgb*100)}'
    grids[tag] = stack
    for w_k in [0.20, 0.30, 0.40, 0.50]:
        grids[f'{tag} +kSS{int(w_k*100)}'] = (1-w_k)*stack + w_k*k_ss
        grids[f'{tag} +kEE{int(w_k*100)}'] = (1-w_k)*stack + w_k*k_ee
print(f"  {'combo':<34} {'raw':>8} {'GAU10':>8}")
scored2 = {}
for k, p in grids.items():
    ps = gau10_smooth(idx, p)
    r_raw, r_sm = rmse(p, tgt), rmse(ps, tgt)
    scored2[k] = (r_raw, r_sm)
print(f"  (scoring {len(grids)} combos...)", flush=True)
for k, p in grids.items():
    print(f"  {k:<34} {scored2[k][0]:8.4f} {scored2[k][1]:8.4f}")
# triple blend: lin + lgb + kalman
for w_lin, w_lgb, w_k in [(0.15,0.60,0.25),(0.20,0.55,0.25),(0.10,0.60,0.30),(0.25,0.50,0.25)]:
    p = w_lin*lin_e + w_lgb*lgb_e + w_k*k_ss
    ps = gau10_smooth(idx, p)
    print(f"  {'triple lin/lgb/kSS %d/%d/%d'%(int(w_lin*100),int(w_lgb*100),int(w_k*100)):<34} {rmse(p,tgt):8.4f} {rmse(ps,tgt):8.4f}")

print("\n=== SUMMARY (PRIMARY, GAU10-smoothed, sorted) ===")
for k, (r_raw, r_sm) in sorted(prim.items(), key=lambda kv: kv[1][1]):
    print(f"  {r_sm:.4f}  (raw {r_raw:.4f})  {k}")

print("\nDONE.")
