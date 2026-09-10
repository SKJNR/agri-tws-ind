"""2-b step 10: (E) D decomposition; (G) covariate band/scale structure + component mapping;
anchor band persistence in test era."""
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, yms, t_abs_tr, lat_c, lon_c = d['F'].astype(np.float64), d['yms'], d['t_abs_tr'], d['lat_c'], d['lon_c']
mu_c, beta_c, tbar_c = d['mu_c'], d['beta_c'], d['tbar_c']
SPEI1, SPEI3, SPEI6, SPEI12, SM = [d[k].astype(np.float64) for k in ['SPEI1','SPEI3','SPEI6','SPEI12','SM']]
T, n_cells = F.shape
full = ~np.isnan(F).any(axis=0)
lats = np.sort(np.unique(lat_c)); lons = np.sort(np.unique(lon_c))
nl, no = len(lats), len(lons)
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
lat_idx_all = np.array([lat_i[v] for v in lat_c]); lon_idx_all = np.array([lon_i[v] for v in lon_c])
W = np.zeros((nl,no)); W[lat_idx_all, lon_idx_all] = 1.0
def gsmooth(field, sigma_cells):
    G = np.full((nl,no), np.nan); G[lat_idx_all, lon_idx_all] = field
    Gf = np.where(np.isnan(G), 0.0, G)
    num = gaussian_filter(Gf, sigma_cells, mode='constant')
    den = gaussian_filter(W, sigma_cells, mode='constant')
    vals = num[lat_idx_all, lon_idx_all]/np.maximum(den[lat_idx_all,lon_idx_all],1e-9)
    return np.where(np.isfinite(field), vals, np.nan)
SIGMAS = [1.5, 3.0, 6.0, 12.0]
def bands(field):
    sm = [gsmooth(field, s) for s in SIGMAS]
    b = [field - sm[0]]
    for k in range(3): b.append(sm[k] - sm[k+1])
    b.append(sm[-1])
    return b  # 5 bands: <2, 2-4, 4-9, 9-24, >24 deg

# ---------- load test ----------
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
print("anchors:", anchor_yms)

A_anchor = {}
for y in anchor_yms:
    sub = test[(test['ym']==y) & (~test['masked'])]
    f = np.full(n_cells, np.nan); f[sub['cc'].values] = sub['TWS_t'].values
    A_anchor[y] = f - mu_c
Dhat = np.nanmean(np.stack([A_anchor[y] for y in anchor_yms]), axis=0)

# ---------- (E) D decomposition ----------
print("\n(E) D-HAT DECOMPOSITION  [std(D-hat) = %.4f]" % np.nanstd(Dhat))
t_mid = 24210  # ~2016-10
trendex = (t_mid - tbar_c)*beta_c
S_cov = {}
for name, M in [('SPEI1',SPEI1),('SPEI3',SPEI3),('SPEI6',SPEI6),('SPEI12',SPEI12),('SM',SM)]:
    S_cov[name] = np.nanmean(M, axis=0)
ok = np.isfinite(Dhat) & np.isfinite(trendex) & np.isfinite(S_cov['SPEI12'])
X = np.column_stack([np.ones(ok.sum()), trendex[ok], mu_c[ok], S_cov['SPEI12'][ok], S_cov['SM'][ok], S_cov['SPEI3'][ok]])
y = Dhat[ok]
c,res_,rank,sv = np.linalg.lstsq(X, y, rcond=None)
pred = X@c
r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
print(f"  joint linear [trendex, mu, S12, SM, S3]: R2={r2:.4f}, resid std={np.std(y-pred):.4f}")
for nm, cf in zip(['const','trendex','mu_c','S_SPEI12','S_SM','S_SPEI3'], c):
    print(f"    {nm:10s}: {cf:+.4f}")
# individual R2s
for nm, v in [('trendex', trendex), ('mu_c', mu_c), ('S_SPEI12', S_cov['SPEI12']), ('S_SM', S_cov['SM']), ('S_SPEI3', S_cov['SPEI3'])]:
    o = ok & np.isfinite(v)
    r = np.corrcoef(v[o], Dhat[o])[0,1]
    print(f"  corr(D-hat, {nm}) = {r:+.4f}  (R2={r*r:.3f})")
# nonlinearity: D vs trendex with quadratic / interaction
X2 = np.column_stack([X, trendex[ok]**2, mu_c[ok]**2, np.abs(trendex[ok]), trendex[ok]*mu_c[ok]])
c2, *_ = np.linalg.lstsq(X2, y, rcond=None)
r2b = 1 - ((y-X2@c2)**2).sum()/((y-y.mean())**2).sum()
print(f"  + quadratic/interaction terms: R2={r2b:.4f} (delta {r2b-r2:+.4f})")
# residual rank: PCA of residual field
resid = np.full(n_cells, np.nan); resid[ok] = y - pred
R = np.where(np.isfinite(resid), resid, 0.0)
# spectrum via SVD of the field seen as rank-1... use spatial smoothness + variogram instead
rb = bands(resid)
tot = np.nansum([np.nanvar(b) for b in rb])
print("  D-residual band variance shares:", " ".join(f"{np.nanvar(b)/tot*100:.0f}%" for b in rb))
r1_res = np.nanmean([np.corrcoef(rb[k][ok], rb[k][ok])[0,1] for k in range(1)])
# neighbor corr of D residual
key = {(int(round(a*10)), int(round(b*10))): i for i,(a,b) in enumerate(zip(lat_c, lon_c))}
cc_vals = []
for (a,b), i in key.items():
    js = [key.get((a+da,b+db)) for da,db in [(10,0),(-10,0),(0,10),(0,-10)]]
    js = [j for j in js if j is not None]
    if len(js)>=3 and np.isfinite(resid[i]) and all(np.isfinite(resid[j]) for j in js):
        cc_vals.append(resid[i]-np.mean([resid[j] for j in js]))
print(f"  D-residual local roughness (x - mean nbrs): var={np.var(cc_vals):.5f} -> white-noise-ish std={np.sqrt(np.var(cc_vals)/1.25):.4f}; resid std={np.nanstd(resid):.4f}")

# ---------- (G) covariates: band-wise correlation with FAST ----------
print("\n(G) COVARIATES: which spatial bands do they observe?")
# train-era: FAST proxy = detrended anomaly; per band corr(cov_t, FAST_t)
td = t_abs_tr[:,None] - tbar_c[None,:]
A_dt = F - mu_c[None,:] - td*beta_c[None,:]
covs_train = {'SPEI_01':SPEI1,'SPEI_03':SPEI3,'SPEI_06':SPEI6,'SPEI_12':SPEI12,'SM':SM}
# remove cov static pattern: deviation
print("  corr(cov deviation, TWS detrended anomaly) per band (pooled over months):")
hdr = "  {:10s}".format("cov") + "".join(f"  band{k}({['<2','2-4','4-9','9-24','>24'][k]}deg)" for k in range(5))
print(hdr)
for nm, M in covs_train.items():
    Mdev = M - np.nanmean(M, axis=0, keepdims=True)
    row = f"  {nm:10s}"
    for k in range(5):
        # pooled corr across all months/cells
        a = np.nan_to_num(Mdev); b = np.nan_to_num(A_dt)
        # band of cov deviation vs band of TWS anomaly
        bk_tws = bands(A_dt[100])  # template for shapes
        # compute per-month band fields is expensive; do a sample of months
        rs = []
        for t in range(0, T, 6):
            bb = bands(A_dt[t]); cb = bands(Mdev[t])
            o = np.isfinite(bb[k]) & np.isfinite(cb[k])
            if o.sum()>1000: rs.append(np.corrcoef(cb[k][o], bb[k][o])[0,1])
        row += f"  {np.mean(rs):+.3f}          "
    print(row)

# ---------- anchor FAST band persistence (test era) ----------
print("\nANCHOR FAST band persistence (test era): corr(band_k(anchor_i), band_k(anchor_j)) by gap")
anchor_fast = {y: A_anchor[y]-Dhat for y in anchor_yms}
gaps = []
for i in range(len(anchor_yms)):
    for j in range(i+1, len(anchor_yms)):
        ta = (anchor_yms[i]//100)*12 + anchor_yms[i]%100
        tb = (anchor_yms[j]//100)*12 + anchor_yms[j]%100
        gaps.append((tb-ta, i, j))
print("  pairs (gap_months):", [(g, i, j) for g,i,j in gaps])
band_anchor = {y: bands(anchor_fast[y]) for y in anchor_yms}
for k in range(5):
    cors = {}
    for g,i,j in gaps:
        a = band_anchor[anchor_yms[i]][k]; b = band_anchor[anchor_yms[j]][k]
        o = np.isfinite(a)&np.isfinite(b)
        cors[g] = np.corrcoef(a[o],b[o])[0,1]
    vals = [f"g{g}:{cors[g]:+.3f}" for g in sorted(cors)]
    print(f"  band{k} ({['<2','2-4','4-9','9-24','>24'][k]:>4}deg): " + " ".join(vals))
# also total anchor-anomaly (D+FAST) band persistence
band_anchor_tot = {y: bands(A_anchor[y]) for y in anchor_yms}
print("  (total anomaly = D+FAST):")
for k in range(5):
    cors = {}
    for g,i,j in gaps:
        a = band_anchor_tot[anchor_yms[i]][k]; b = band_anchor_tot[anchor_yms[j]][k]
        o = np.isfinite(a)&np.isfinite(b)
        cors[g] = np.corrcoef(a[o],b[o])[0,1]
    vals = [f"g{g}:{cors[g]:+.3f}" for g in sorted(cors)]
    print(f"  band{k} ({['<2','2-4','4-9','9-24','>24'][k]:>4}deg): " + " ".join(vals))
