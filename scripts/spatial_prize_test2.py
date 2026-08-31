"""
spatial_prize_test2.py — refine the smoothing win + stack effects.

T5 showed: box 5x5 post-hoc prediction smoothing = -0.0182 on sparse-CV masked rows.
This script:
  S1 kernel sweep: box r=1..3, Gaussian sigma=1.5..3.0, LS-fitted 5x5 kernel
  S2 stacking: W-pool + best smoothing (redundancy check)
  S3 per-month stability of the gain (is it broad-based or 1-2 months?)
  S4 per-k (months-since-anchor) breakdown
  S5 bias-transfer probe: fit per-cell bias on 2013-14 masked residuals (using
     only anchors 2013-2014), apply to 2015 masked rows -> does it help?
     (if yes, submission-time bias correction from CV is a real extra channel)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

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

# grid
lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)
for (la, lo), grp in train.groupby(['lat','lon']):
    cc_grid[lat_i[la], lon_i[lo]] = int(grp['cc'].iloc[0])
mask_g = cc_grid >= 0

def to_grid(v_cell):
    g = np.full((NI, NJ), np.nan, dtype=np.float32); g[mask_g] = v_cell[cc_grid[mask_g]]; return g
def from_grid(g):
    out = np.full(n_cells, np.nan, dtype=np.float32); out[cc_grid[mask_g]] = g[mask_g]; return out

def shift(g, di, dj):
    gg = np.roll(g, dj, axis=1)
    if di > 0: gg = np.vstack([np.full((di, NJ), np.nan, np.float32), gg[:-di]])
    if di < 0: gg = np.vstack([gg[-di:], np.full((-di, NJ), np.nan, np.float32)])
    return gg

def kernel_pool(g, Wt):
    """Wt: dict (di,dj)->weight. valid-weighted mean."""
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

def boxW(r):
    return {(di,dj):1.0 for di in range(-r,r+1) for dj in range(-r,r+1)}
def gaussW(sig, r=4):
    Wt = {}
    for di in range(-r,r+1):
        for dj in range(-r,r+1):
            w = np.exp(-(di*di+dj*dj)/(2*sig*sig))
            if w > 0.01: Wt[(di,dj)] = w
    return Wt

# infra
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

print("Building val cov fields...", flush=True)
Zv = val[COVS].values.astype('float32'); okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype='float32')
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m_v}

cand = [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6]
anchors_v = [m for m in cand if m in set(val_ta.tolist())]
AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)
Xs, ys = [], []
for a in anchors_v:
    dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b != a]), axis=0)
    tx = trendex(a)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
w1, w2, w3 = map(float, w_)
D_tx = lambda t: (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

def rmse_masked(pred):
    ok = np.isfinite(pred) & val_msk
    return float(np.sqrt(np.mean((pred[ok]-val_target[ok])**2)))

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

def kalman_predict(phi_f, Dfun, W_pool=None):
    H, R, var_f = calibrate(Dfun)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    Wloc = {m: (from_grid(kernel_pool(to_grid(W[m]), boxW(W_pool))) if W_pool else W[m]) for m in W}
    pred = np.full(len(val), np.nan, dtype=np.float64)
    for a in anchors_v:
        fa = AF[a]; dja = Dfun(a)
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        sel0 = np.where((val_ta == a) & val_msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[val_cc[sel0]] + Dfun(a+1)[val_cc[sel0]] + phi_f*x[val_cc[sel0]]
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

def smooth_pred(pred, Wt):
    out = pred.copy()
    for m in np.unique(val_ta[(val_msk) & np.isfinite(pred)]):
        selm = np.where((val_ta == m) & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float64); cm[val_cc[selm]] = pred[selm]
        sm = from_grid(kernel_pool(to_grid(cm.astype(np.float32)), Wt))
        out[selm] = sm[val_cc[selm]]
    return out

p0 = kalman_predict(0.74, D_tx)
r0 = rmse_masked(p0)
print(f"\nbaseline v2b: {r0:.4f}")

print("\n=== S1: kernel sweep ===")
best = (None, r0)
for name, Wt in [('box r=1', boxW(1)), ('box r=2', boxW(2)), ('box r=3', boxW(3)),
                 ('gau 1.5', gaussW(1.5)), ('gau 2.0', gaussW(2.0)), ('gau 2.5', gaussW(2.5)), ('gau 3.0', gaussW(3.0))]:
    r = rmse_masked(smooth_pred(p0, Wt))
    tag = ''
    if r < best[1]: best = (name, r); tag = ' *'
    print(f"  {name:10s}: {r:.4f} ({r-r0:+.4f}){tag}")
print(f"best kernel: {best[0]} -> {best[1]:.4f}")

print("\n=== S2: stack with W-pool (r=2 box) ===")
pw = kalman_predict(0.74, D_tx, W_pool=2)
rw = rmse_masked(pw)
print(f"  Wpool only: {rw:.4f} ({rw-r0:+.4f})")
rws = rmse_masked(smooth_pred(pw, boxW(2)))
print(f"  Wpool + box r=2 smooth: {rws:.4f} ({rws-r0:+.4f})")

print("\n=== S3: per-month gain stability (box r=2) ===")
ps = smooth_pred(p0, boxW(2))
ms = sorted(np.unique(val_ta[val_msk]))
for m in ms:
    ok = (val_ta == m) & val_msk & np.isfinite(ps)
    if ok.sum() < 100: continue
    r_base = np.sqrt(np.mean((p0[ok]-val_target[ok])**2)); r_sm = np.sqrt(np.mean((ps[ok]-val_target[ok])**2))
    print(f"  m={int(m)}: n={int(ok.sum()):6d}  base {r_base:.4f} -> smooth {r_sm:.4f}  ({r_sm-r_base:+.4f})")

print("\n=== S4: per-k breakdown (months since last anchor) ===")
anchors_arr = np.array(anchors_v)
row_anchor = anchors_arr[np.searchsorted(anchors_arr, val_ta, side='right')-1]
kdist = val_ta - row_anchor
for k in range(0, 9):
    ok = np.isfinite(ps) & val_msk & (kdist == k)
    if ok.sum() < 100: continue
    r_base = np.sqrt(np.mean((p0[ok]-val_target[ok])**2)); r_sm = np.sqrt(np.mean((ps[ok]-val_target[ok])**2))
    print(f"  k={k}: n={int(ok.sum()):6d}  base {r_base:.4f} -> smooth {r_sm:.4f}  ({r_sm-r_base:+.4f})")

print("\n=== S5: bias-transfer probe ===")
# fit per-cell bias on 2013-14 residuals only, evaluate on 2015
early_msk = val_msk & (val_ta < 2015*12)
late_msk = val_msk & (val_ta >= 2015*12)
res = ps - val_target   # use smoothed preds (post-fix residual)
bias = np.full(n_cells, np.nan, dtype=np.float64)
acc = np.zeros(n_cells); cnt = np.zeros(n_cells)
np.add.at(acc, val_cc[early_msk & np.isfinite(res)], res[early_msk & np.isfinite(res)])
np.add.at(cnt, val_cc[early_msk & np.isfinite(res)], 1)
bias[cnt > 0] = acc[cnt > 0]/cnt[cnt > 0]
# shrink bias by factor s (avoid overfit to noise): sweep s
bs_grid = to_grid(bias.astype(np.float32))
bs_s = from_grid(kernel_pool(bs_grid, gaussW(2.0)))  # smooth the bias field too
for s in [0.0, 0.3, 0.5, 0.7, 1.0]:
    pcor = ps.copy()
    okl = late_msk & np.isfinite(ps)
    pcor[okl] -= s*np.nan_to_num(bs_s[val_cc[okl]])
    ok = late_msk & np.isfinite(pcor)
    r = float(np.sqrt(np.mean((pcor[ok]-val_target[ok])**2)))
    r_ref = float(np.sqrt(np.mean((ps[ok]-val_target[ok])**2)))
    print(f"  bias-corr s={s:.1f}: 2015 RMSE {r:.4f} (ref {r_ref:.4f}, {r-r_ref:+.4f})")

print("\nDone.", flush=True)
