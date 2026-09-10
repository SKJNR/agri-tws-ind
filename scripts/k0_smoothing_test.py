"""
k0_smoothing_test.py — should we submit v5c? Test the k=0-row smoothing channel
on the trusted sparse-anchor CV (v5c = v5b + Gaussian smoothing of UNMASKED rows).

The masked-row smoothing was validated (-0.019 CV). But k=0 rows are different:
their predictions are anchored on the TRUE TWS_t of that cell (no per-cell
estimation noise to remove). Smoothing only helps if the k=0 residual
(target - k0_pred) has a spatially SMOOTH component (model bias) that
outweighs the added distortion.

Also: sigma sensitivity for k=0, and total-score impact math
(public LB = all rows: ~33.5% k=0 + 66.5% masked).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

print("Loading...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1

fit = train[train['time'].dt.year <= 2012].copy()
val  = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()

test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
cal_mask_frac = test_raw.assign(m=test_raw['TWS_t_masked'].astype(bool), cm=test_raw['time'].dt.month).groupby('cm')['m'].mean()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values
val_ta = val['t_abs'].values
val_msk = val['masked'].values
# for k=0 eval, TWS_t stays VISIBLE on unmasked rows (that's the point)

# grid
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
def gaussW(sig, r=4):
    return {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*sig*sig)))
            for di in range(-r,r+1) for dj in range(-r,r+1)
            if np.exp(-(di*di+dj*dj)/(2*sig*sig)) > 0.01}
def kernel_pool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

# ---- k=0 model fit on fit data only (same spec as submission_v5.py) ----
print("Fitting k=0 model on 2002-2012...", flush=True)
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

Lk = fit[['cc','t_abs','TWS_t','target']+COVS].copy()
Lk['t_next'] = Lk['t_abs'] + 1
Rk = fit[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
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
w_rec = np.where(yr <= 2006, 1.0, np.where(yr <= 2009, 2.0, 3.0)).astype(np.float32)
sw = np.sqrt(w_rec)
A_f = Xf[has_nxt]*sw[has_nxt,None]
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[has_nxt]*sw[has_nxt]))
colsR = [0,1,2,3,4,5,11]
A_r = Xf[:, colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))

# ---- apply to val unmasked rows ----
Lv = val[['cc','t_abs','TWS_t']+COVS+['target']].copy(); Lv['t_next'] = Lv['t_abs']+1
Rv = val[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgv = Lv.merge(Rv, on=['cc','t_next'], how='left')
has_nxt_v = mgv[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccv = mgv['cc'].values
Xv = np.column_stack([
    mgv['TWS_t'].values - mu_c[ccv],
    *[mgv[c].values - clim[ccv, j] for j, c in enumerate(COVS)],
    *[mgv[c+'_nxt'].values - clim[ccv, j] for j, c in enumerate(COVS)],
    np.ones(len(mgv))
]).astype(np.float32)
Xv = np.nan_to_num(Xv, nan=0.0)
tv = mgv['target'].values.astype(np.float64)
k0 = np.full(len(mgv), np.nan, dtype=np.float64)
tw_ok = np.isfinite(mgv['TWS_t'].values)
f1 = (~val_msk) & has_nxt_v & tw_ok
f2 = (~val_msk) & (~has_nxt_v) & tw_ok
k0[f1] = mu_c[ccv[f1]] + Xv[f1] @ coefF
k0[f2] = mu_c[ccv[f2]] + Xv[f2][:, colsR] @ coefR
bad = (~val_msk) & np.isnan(k0)
if bad.any(): k0[bad] = mu_c[ccv][bad]

ok0 = (~val_msk) & np.isfinite(k0) & np.isfinite(tv)
r_k0 = float(np.sqrt(np.mean((k0[ok0]-tv[ok0])**2)))
print(f"\nk=0 rows: n={int(ok0.sum()):,}  baseline RMSE {r_k0:.4f}")

# ---- residual spatial structure ----
res0 = k0 - tv
rm = np.full(n_cells, np.nan, dtype=np.float64)
np.maximum
acc = np.zeros(n_cells); cnt = np.zeros(n_cells)
np.add.at(acc, val_cc[ok0], res0[ok0]); np.add.at(cnt, val_cc[ok0], 1)
rm[cnt>0] = acc[cnt>0]/cnt[cnt>0]
g = to_grid(rm.astype(np.float32)); gn = kernel_pool(g, gaussW(2.0))
okg = np.isfinite(g) & np.isfinite(gn)
print(f"k=0 residual vs 3x3-ish pooled corr: {np.corrcoef(g[okg], gn[okg])[0,1]:.3f}")

# ---- smoothing sweep on k=0 rows ----
print("\n=== k=0 smoothing sweep (apply to predictions, eval) ===")
def smooth_k0(sigma):
    Wt = gaussW(sigma)
    out = k0.copy()
    for m in np.unique(val_ta[ok0]):
        selm = np.where((val_ta == m) & ok0)[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[val_cc[selm]] = k0[selm].astype(np.float32)
        sm = from_grid(kernel_pool(to_grid(cm), Wt))
        out[selm] = sm[val_cc[selm]]
    return out
for sig in [1.0, 1.5, 2.0, 2.5, 3.0]:
    ps = smooth_k0(sig)
    ok = ok0 & np.isfinite(ps)
    r = float(np.sqrt(np.mean((ps[ok]-tv[ok])**2)))
    print(f"  sigma={sig:.1f}: k=0 RMSE {r:.4f}  ({r-r_k0:+.4f})")

# ---- blended total (what public LB sees, roughly) ----
print("\n=== total-score impact estimate (0.335*k0 + 0.665*masked) ===")
# masked-row side: use known CV numbers (v2b 0.6945 -> v5b ~0.675 effective scale 0.44 realized on LB)
# here just k=0 side, honest:
for sig in [1.5, 2.0, 2.5]:
    ps = smooth_k0(sig)
    ok = ok0 & np.isfinite(ps)
    r = float(np.sqrt(np.mean((ps[ok]-tv[ok])**2)))
    tot_before = np.sqrt(0.335*r_k0**2 + 0.665*0.70**2)
    tot_after  = np.sqrt(0.335*r**2 + 0.665*0.70**2)
    print(f"  sigma={sig}: k=0 {r_k0:.4f}->{r:.4f}  |  est total {tot_before:.4f}->{tot_after:.4f} ({tot_after-tot_before:+.4f})")

print("\nDone.", flush=True)
