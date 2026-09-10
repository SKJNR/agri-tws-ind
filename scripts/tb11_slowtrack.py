"""2-b step 11: (G2) Do covariate monthly deviations track the SLOW wander (D-drift analog)?
In train: slow_wander(t) = 12-mo moving average of detrended anomaly. Check per-cov tracking,
band structure, and SPEI_1 vs SPEI_12 component mapping. Also D-drift tracking at anchors."""
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, yms, t_abs_tr, lat_c, lon_c = d['F'].astype(np.float64), d['yms'], d['t_abs_tr'], d['lat_c'], d['lon_c']
mu_c, beta_c, tbar_c = d['mu_c'], d['beta_c'], d['tbar_c']
SPEI1, SPEI3, SPEI6, SPEI12, SM = [d[k].astype(np.float64) for k in ['SPEI1','SPEI3','SPEI6','SPEI12','SM']]
T, n_cells = F.shape
td = t_abs_tr[:,None] - tbar_c[None,:]
A_dt = F - mu_c[None,:] - td*beta_c[None,:]          # detrended anomaly (slow wander + fast)
covs = {'SPEI_01':SPEI1,'SPEI_03':SPEI3,'SPEI_06':SPEI6,'SPEI_12':SPEI12,'SM':SM}

def movavg(M, w):
    """centered moving average along axis 0, NaN-aware"""
    out = np.full_like(M, np.nan)
    M0 = np.where(np.isfinite(M), M, 0.0); Wm = np.isfinite(M).astype(float)
    for t in range(M.shape[0]):
        a = max(0, t-w//2); b = min(M.shape[0], t+w//2+1)
        out[t] = np.where(Wm[a:b].sum(0)>0, M0[a:b].sum(0)/np.maximum(Wm[a:b].sum(0),1), np.nan)
    return out

W12 = 13
slow = movavg(A_dt, W12)                              # slow wander estimate
fast = A_dt - slow
print("(G2) TRAIN-ERA component tracking, pooled corr across cells+months:")
print(f"  var shares: slow(13-mo MA)={np.nanvar(slow)/np.nanvar(A_dt)*100:.0f}%, fast={np.nanvar(fast)/np.nanvar(A_dt)*100:.0f}%")
for nm, M in covs.items():
    Mdev = M - np.nanmean(M, axis=0, keepdims=True)
    Mdev_s = movavg(Mdev, W12); Mdev_f = Mdev - Mdev_s
    o = np.isfinite(Mdev)&np.isfinite(A_dt)
    r_tot = np.corrcoef(Mdev[o], A_dt[o])[0,1]
    o1 = np.isfinite(Mdev_s)&np.isfinite(slow)
    r_slow = np.corrcoef(Mdev_s[o1], slow[o1])[0,1]
    o2 = np.isfinite(Mdev_f)&np.isfinite(fast)
    r_fast = np.corrcoef(Mdev_f[o2], fast[o2])[0,1]
    o3 = np.isfinite(Mdev)&np.isfinite(fast)
    r_fast_raw = np.corrcoef(Mdev[o3], fast[o3])[0,1]
    print(f"  {nm:8s}: corr(dev, total)={r_tot:+.3f}  corr(dev_slow, SLOW)={r_slow:+.3f}  corr(dev_fast, FAST)={r_fast:+.3f}  corr(dev_raw, FAST)={r_fast_raw:+.3f}")

# band-wise: which TWS bands do the SMOOTHED covs track? (large-scale slow channel)
lats = np.sort(np.unique(lat_c)); lons = np.sort(np.unique(lon_c))
nl, no = len(lats), len(lons)
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
lat_idx_all = np.array([lat_i[v] for v in lat_c]); lon_idx_all = np.array([lon_i[v] for v in lon_c])
Wg = np.zeros((nl,no)); Wg[lat_idx_all, lon_idx_all] = 1.0
def gsmooth(field, sigma_cells):
    G = np.full((nl,no), np.nan); G[lat_idx_all, lon_idx_all] = field
    Gf = np.where(np.isnan(G), 0.0, G)
    num = gaussian_filter(Gf, sigma_cells, mode='constant')
    den = gaussian_filter(Wg, sigma_cells, mode='constant')
    vals = num[lat_idx_all, lon_idx_all]/np.maximum(den[lat_idx_all,lon_idx_all],1e-9)
    return np.where(np.isfinite(field), vals, np.nan)
print("\n  band-wise corr (pooled months) of cov-dev vs SLOW wander and vs FAST:")
print("  {:8s} {}".format("cov", "  ".join(f"b{k}slow  b{k}fast" for k in range(3))))
for nm, M in covs.items():
    Mdev = M - np.nanmean(M, axis=0, keepdims=True)
    Mdev_s = movavg(Mdev, W12); Mdev_f = Mdev - Mdev_s
    row = f"  {nm:8s}"
    for k, sig in enumerate([3.0, 6.0, 12.0]):
        rs, rf = [], []
        for t in range(3, T-3, 5):
            gs = gsmooth(slow[t], sig); gf = gsmooth(fast[t], sig)
            cs = gsmooth(Mdev_s[t], sig); cf = gsmooth(Mdev_f[t], sig)
            o1 = np.isfinite(gs)&np.isfinite(cs); o2 = np.isfinite(gf)&np.isfinite(cf)
            if o1.sum()>1000: rs.append(np.corrcoef(cs[o1], gs[o1])[0,1])
            if o2.sum()>1000: rf.append(np.corrcoef(cf[o2], gf[o2])[0,1])
        row += f"  {np.mean(rs):+.3f}  {np.mean(rf):+.3f}   "
    print(row)

# ---------- D-drift tracking at anchors: does cov deviation co-move with anchor-D deviations? ----------
print("\n(G3) ANCHOR D-drift: does the cov static+deviation structure at anchor months explain (anchor - D-hat)?")
DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = pd.to_datetime(train['time']).dt.year*100 + pd.to_datetime(train['time']).dt.month
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos_map = {(int(round(float(la)*10)), int(round(float(lo)*10))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['ym'] = pd.to_datetime(test['time']).dt.year*100 + pd.to_datetime(test['time']).dt.month
test['cc'] = [pos_map.get((int(round(float(a)*10)), int(round(float(b)*10))), -1) for a,b in zip(test['lat'], test['lon'])]
test['masked'] = test['TWS_t_masked'].astype(bool)
for c in ['TWS_t']+COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
cnt = test.groupby('ym')['masked'].agg(['size','sum'])
anchor_yms = [int(y) for y in cnt[cnt['sum']==0].index]
A_anchor = {}
for y in anchor_yms:
    sub = test[(test['ym']==y) & (~test['masked'])]
    f = np.full(n_cells, np.nan); f[sub['cc'].values] = sub['TWS_t'].values
    A_anchor[y] = f - mu_c
Dhat = np.nanmean(np.stack([A_anchor[y] for y in anchor_yms]), axis=0)
# cov train static patterns
S_stat = {nm: np.nanmean(M, axis=0) for nm, M in covs.items()}
# cov deviation fields at anchors
Cdev_anchor = {}
for nm, col in [('SPEI_01','SPEI_01_t'),('SPEI_03','SPEI_03_t'),('SPEI_06','SPEI_06_t'),('SPEI_12','SPEI_12_t'),('SM','SOIL_MOISTURE_t')]:
    Cdev_anchor[nm] = {}
    for y in anchor_yms:
        sub = test[(test['ym']==y) & (~test['masked'])]
        f = np.full(n_cells, np.nan); f[sub['cc'].values] = sub[col].values
        Cdev_anchor[nm][y] = f - S_stat[nm]
print("  corr(anchor anomaly - D-hat, cov deviation at anchor) per anchor month:")
for y in anchor_yms:
    fa = A_anchor[y] - Dhat
    row = f"  {y}: "
    for nm in Cdev_anchor:
        cd = Cdev_anchor[nm][y]
        o = np.isfinite(fa)&np.isfinite(cd)
        row += f"{nm}={np.corrcoef(cd[o], fa[o])[0,1]:+.3f} "
    print(row)
# large-scale only (>24 deg band)
print("  same but >24-deg band only (slow channel):")
for y in anchor_yms:
    fa = A_anchor[y] - Dhat
    fa_s = gsmooth(np.where(np.isfinite(fa), fa, 0.0), 12.0)
    row = f"  {y}: "
    for nm in Cdev_anchor:
        cd = Cdev_anchor[nm][y]
        cd_s = gsmooth(np.where(np.isfinite(cd), cd, 0.0), 12.0)
        o = np.isfinite(fa)&np.isfinite(cd)
        row += f"{nm}={np.corrcoef(cd_s[o], fa_s[o])[0,1]:+.3f} "
    print(row)
