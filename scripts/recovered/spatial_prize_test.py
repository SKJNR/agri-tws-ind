"""
spatial_prize_test.py — quantify the untapped SPATIAL error pool on the trusted
sparse-anchor CV protocol (the one that correctly ranked v2b > v4a/v4c on LB).

Question this answers: our masked-RMSE floor ~0.69 (sparse CV, v2b) decomposes into
target noise (~0.456) + D-tilde error + fast-state error. The fast field is
near rank-50 (neighbor residual corr 0.97) yet our Kalman is per-cell.
How much do we recover by borrowing strength across neighboring cells?

Tests (sparse CV, masked rows only):
  T0  v2b baseline reproduction (phi=0.74, D-trendex, cov obs)     expect ~0.6945
  T1  residual spatial autocorrelation (diag: is error smooth or noisy?)
  T2  neighbor-pooled anchor init: x0 = lam*(w*AF[c] + (1-w)*pool_3x3/5x5(AF))
  T3  spatially smoothed D-hat (3x3 / 5x5) inside D-trendex
  T4  spatially smoothed cov-obs fields W[m] (3x3 / 5x5)
  T5  post-hoc prediction smoothing (3x3 / 5x5 box on prediction field)
  T6  best combo + CV-ensemble (v2b + v1c predictions averaged)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

# ---------------- load train ----------------
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
val  = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
print(f"fit {len(fit):,} / val {len(val):,}", flush=True)

# masking pattern from test
test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
cal_mask_frac = test_raw.assign(m=test_raw['TWS_t_masked'].astype(bool), cm=test_raw['time'].dt.month).groupby('cm')['m'].mean()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values
val_ta = val['t_abs'].values
val_msk = val['masked'].values

# ---------------- grid mapping for spatial ops ----------------
lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)   # grid -> cc
cc_of = np.full((n_cells, 2), -1, dtype=np.int32) # cc -> (i,j)
for (la, lo), grp in train.groupby(['lat','lon']):
    cc = int(grp['cc'].iloc[0]); i, j = lat_i[la], lon_i[lo]
    cc_grid[i, j] = cc; cc_of[cc] = (i, j)
print(f"grid {NI}x{NJ}, cells {n_cells}, filled {int((cc_grid>=0).sum())}")

def to_grid(v_cell, fill=np.nan):
    g = np.full((NI, NJ), fill, dtype=np.float32)
    m = cc_grid >= 0
    g[m] = v_cell[cc_grid[m]]
    return g

def from_grid(g):
    out = np.full(n_cells, np.nan, dtype=np.float32)
    m = cc_grid >= 0
    out[cc_grid[m]] = g[m]
    return out

def box_pool(g, r=1):
    """valid-neighbor box mean, radius r (3x3 if r=1), wrap on lon."""
    acc = np.zeros_like(g); cnt = np.zeros_like(g, dtype=np.int32)
    for di in range(-r, r+1):
        for dj in range(-r, r+1):
            gg = np.roll(g, dj, axis=1)              # wrap lon
            if di > 0: gg = np.vstack([np.full((di, NJ), np.nan, np.float32), gg[:-di]])
            if di < 0: gg = np.vstack([gg[-di:], np.full((-di, NJ), np.nan, np.float32)])
            ok = np.isfinite(gg)
            acc[ok] += gg[ok]; cnt[ok] += 1
    return np.where(cnt > 0, acc/np.maximum(cnt, 1), np.nan)

# ---------------- fit infrastructure (2002-2012 only) ----------------
print("Fitting infra...", flush=True)
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')

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
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# ---------------- val cov fields, S, W ----------------
print("Building val cov fields...", flush=True)
Zv = val[COVS].values.astype('float32'); okv = np.isfinite(Zv).all(axis=1)
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

# ---------------- SPARSE anchors (trusted protocol) ----------------
cand = [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6]
anchors_v = [m for m in cand if m in set(val_ta.tolist())]
print(f"sparse anchors: {anchors_v}  gaps {np.diff(anchors_v).tolist()}")

AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)

# D-trendex LOO weights
Xs, ys = [], []
for a in anchors_v:
    dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b != a]), axis=0)
    tx = trendex(a)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
w1, w2, w3 = map(float, w_)
print(f"D-tilde LOO weights: {w1:.3f}/{w2:.3f}/{w3:.3f}")

def rmse_masked(pred):
    ok = np.isfinite(pred) & val_msk
    return float(np.sqrt(np.mean((pred[ok]-val_target[ok])**2))), int(ok.sum())

# ---------------- T0: v2b baseline ----------------
def calibrate(Dfun):
    cs, zs, vfs = [], [], []
    for a in anchors_v:
        dj = Dfun(a)
        ok = np.isfinite(W[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

def kalman_predict(phi_f, Dfun, init_mode='plain', W_pool=None):
    """init_mode: plain | pool3:w | pool5:w  (w = weight on self cell)
       W_pool: radius r for spatially pooling cov-obs fields (None = off)"""
    H, R, var_f = calibrate(Dfun)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    Wloc = {m: (from_grid(box_pool(to_grid(W[m]), W_pool)) if W_pool else W[m]) for m in W}
    pred = np.full(len(val), np.nan, dtype=np.float64)
    for a in anchors_v:
        if init_mode.startswith('pool'):
            rr = 1 if init_mode.startswith('pool3') else 2
            wself = float(init_mode.split(':')[1])
            pooled = from_grid(box_pool(to_grid(AF[a]), rr))
            fa = wself*AF[a] + (1-wself)*pooled
        else:
            fa = AF[a]
        dja = Dfun(a)
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        x0 = x.copy()
        sel0 = np.where((val_ta == a) & val_msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[val_cc[sel0]] + Dfun(a+1)[val_cc[sel0]] + phi_f*x0[val_cc[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in Wloc:
                wv = Wloc[m]; okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((val_ta == m) & val_msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in Wloc:
                wv = Wloc[tm]; okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[val_cc[sel]] + Dfun(tm)[val_cc[sel]] + x2[val_cc[sel]]
    return pred

D_tx = lambda t: (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

print("\n=== T0: v2b baseline (sparse CV, masked rows) ===")
p0 = kalman_predict(0.74, D_tx)
r0, n0 = rmse_masked(p0)
print(f"v2b baseline: RMSE {r0:.4f}  (n={n0:,}; expect ~0.6945)")

# ---------------- T1: residual spatial autocorrelation ----------------
print("\n=== T1: residual spatial autocorrelation ===")
res = p0 - val_target
for m in np.unique(val_ta[val_msk]):
    selm = np.where((val_ta == m) & val_msk & np.isfinite(p0))[0]
    if len(selm) < 500: continue
    rm = np.full(n_cells, np.nan); rm[val_cc[selm]] = res[selm]
    g = to_grid(rm)
    gn = box_pool(g, 1)
    ok = np.isfinite(g) & np.isfinite(gn)
    if m == list(np.unique(val_ta[val_msk]))[2]:
        c1 = np.corrcoef(g[ok], gn[ok])[0,1]
print(f"residual vs 3x3-pooled residual corr (sample month): {c1:.3f}  (>0.5 = smooth/structural error; <0.2 = noisy error, pooling helps)")
nns = []
for m in np.unique(val_ta[val_msk]):
    selm = np.where((val_ta == m) & val_msk & np.isfinite(p0))[0]
    if len(selm) < 500: continue
    rm = np.full(n_cells, np.nan); rm[val_cc[selm]] = res[selm]
    g = to_grid(rm)
    ge = np.roll(g, 1, axis=1)  # east neighbor
    ok = np.isfinite(g) & np.isfinite(ge)
    nns.append(np.corrcoef(g[ok], ge[ok])[0,1])
print(f"east-neighbor residual corr across masked months: mean {np.mean(nns):.3f}")

# ---------------- T2: neighbor-pooled init sweep ----------------
print("\n=== T2: neighbor-pooled anchor init ===")
for mode in ['pool3:1.0','pool3:0.8','pool3:0.6','pool3:0.4','pool3:0.0','pool5:0.8','pool5:0.6','pool5:0.4']:
    p = kalman_predict(0.74, D_tx, init_mode=mode)
    r, _ = rmse_masked(p)
    print(f"  init {mode:12s}: RMSE {r:.4f}  ({r-r0:+.4f})")

# ---------------- T3: smoothed D-hat ----------------
print("\n=== T3: spatially smoothed D-hat inside D-trendex ===")
for r in [1, 2]:
    Dh_s = from_grid(box_pool(to_grid(Dhat), r))
    Ds = lambda t, Dh_s=Dh_s: (w1*Dh_s + w2*S + w3*trendex(t)).astype(np.float32)
    p = kalman_predict(0.74, Ds)
    r, _ = rmse_masked(p)
    print(f"  D-hat pool r={r}: RMSE {r:.4f}  ({r-r0:+.4f})")

# ---------------- T4: pooled cov obs ----------------
print("\n=== T4: spatially pooled cov-obs fields W ===")
for r in [1, 2]:
    p = kalman_predict(0.74, D_tx, W_pool=r)
    r, _ = rmse_masked(p)
    print(f"  W pool r={r}: RMSE {r:.4f}  ({r-r0:+.4f})")

# ---------------- T6: best-combo probe + phi re-check ----------------
print("\n=== T5: post-hoc prediction smoothing ===")
def smooth_pred(pred, r):
    out = pred.copy()
    for m in np.unique(val_ta[(val_msk) & np.isfinite(pred)]):
        selm = np.where((val_ta == m) & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan); cm[val_cc[selm]] = pred[selm]
        sm = from_grid(box_pool(to_grid(cm), r))
        out[selm] = sm[val_cc[selm]]
    return out
for r in [1, 2]:
    r_, _ = rmse_masked(smooth_pred(p0, r))
    print(f"  pred smooth r={r}: RMSE {r_:.4f}  ({r_-r0:+.4f})")

print("\n=== T6: combos ===")
for combo in [('pool3:0.6', 1, 1), ('pool3:0.6', None, 1), ('pool3:0.6', 1, None)]:
    im, wr, dr = combo
    Df = D_tx
    if dr:
        Dh_s = from_grid(box_pool(to_grid(Dhat), dr)); Df = lambda t: (w1*Dh_s + w2*S + w3*trendex(t)).astype(np.float32)
    p = kalman_predict(0.74, Df, init_mode=im, W_pool=wr)
    r, _ = rmse_masked(p)
    print(f"  init {im} + Wpool {wr} + Dpool {dr}: RMSE {r:.4f}  ({r-r0:+.4f})")

print("\nDone.", flush=True)
