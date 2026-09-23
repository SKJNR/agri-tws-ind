"""THE V19 CEILING MEASUREMENT:
target(t) = GDO(t). Estimators available at month t (no future data):
  - exact state: comp TWS_t(t) = GDO(t-1)                     [past, exact]
  - other GRACE products at month t: GravIS, COST-G, CSR      [current-month satellite]
  - comp covariates at t and t+1 (in-file)                     [provided by challenge]
Fit honest model on train<=2012, validate 2013-15.
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

# ---------- load all TWS products ----------
def load_gravis(path):
    ds = xr.open_dataset(path)
    f = ds['tws'].values.astype(np.float32)
    tv = pd.to_datetime(ds['time'].values)
    latg = ds['lat'].values; long = ds['lon'].values
    ds.close()
    out = {int(t.year*100+t.month): f[i] for i, t in enumerate(tv)}
    return out, latg, long

GR, lat_g, lon_g = load_gravis(f'{S}/gravis_tws_grid.nc')
CG, _, _ = load_gravis(f'{S}/gravis_costg_tws.nc')

# CSR (0.25 -> 1deg, precompute lazily per month)
ds_c = xr.open_dataset(f'{S}/csr_mascons_all.nc')
tv_c = pd.to_datetime(ds_c['time'].values, unit='D', origin=pd.Timestamp('2002-01-01'))
lwe = ds_c['lwe_thickness']
CS = {}
for i, t in enumerate(tv_c):
    CS[int(t.year*100+t.month)] = i
lat_c = np.array([-89.875 + 0.25*(4*a+1.5) for a in range(180)])[::-1]  # desc
lon_c = np.arange(0.5, 360.0, 1.0)

def csr_field(ym):
    if ym not in CS: return None
    i = CS[ym]
    blk = lwe.isel(time=i).values.astype(np.float32)  # (720,1440)
    out = np.full((180, 360), np.nan, dtype=np.float32)
    for a in range(180):
        for b in range(360):
            # rows: lat ascending -89.875.. ; we need desc lat1[a]=89.5-...
            aa = 179 - a  # descending index -> ascending row block
            sub = blk[4*aa:4*aa+4, 4*b:4*b+4]
            n = np.isfinite(sub).sum()
            if n: out[a, b] = np.nanmean(sub)
    return out

# GDO
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
GD = {}
for f in files:
    ds = xr.open_dataset(f)
    v = ds['twsan'].values
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = v[i, 0].astype(np.float32)
    gd_lat = ds['lat'].values; gd_lon = ds['lon'].values
    ds.close()

def lookup(fld, lat_arr, lon_arr, la, lo):
    li = int(np.argmin(np.abs(lat_arr - la)))
    lo_i = int(np.argmin(np.abs(((lon_arr - lo + 180) % 360) - 180)))
    return fld[li, lo_i]

print("products loaded")

# ---------- competition train ----------
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
tr['time'] = pd.to_datetime(tr['time'])
for c in ['TWS_t','target']+COVS:
    tr[c] = pd.to_numeric(tr[c], errors='coerce').astype('float32')
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month

# sample for speed: every 3rd cell
cells = tr[['lat','lon']].drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True)
cells_s = cells.iloc[::3].reset_index(drop=True)
cset = set(zip(cells_s['lat'].values, cells_s['lon'].values))
tr_s = tr[[(la, lo) in cset for la, lo in zip(tr['lat'], tr['lon'])]].copy()
print(f"sample: {len(tr_s):,} rows, {len(cells_s):,} cells")

# next-month covs
def shift_ym(ym, d=1):
    y, m = divmod(ym, 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

# build next-month cov lookup from train itself
cov_next = {}
for (la, lo, ym), row in tr_s.groupby(['lat','lon','ym']):
    pass
# faster: pivot
idx = {(la, lo, ym): i for i, (la, lo, ym) in enumerate(zip(tr_s['lat'], tr_s['lon'], tr_s['ym']))}
nxt_cov = {c: np.full(len(tr_s), np.nan, dtype=np.float32) for c in COVS}
rows = list(zip(tr_s['lat'], tr_s['lon'], tr_s['ym'], range(len(tr_s))))
for la, lo, ym, i in rows:
    j = idx.get((la, lo, shift_ym(ym, 1)))
    if j is not None:
        for c in COVS:
            nxt_cov[c][i] = tr_s[c].values[j]
print("next-month covs attached")

# ---------- attach external fields ----------
n = len(tr_s)
ext = {k: np.full(n, np.nan, dtype=np.float32) for k in ['gdo_m1','gravis_t','costg_t','csr_t','gdo_t']}
las = tr_s['lat'].values; los = tr_s['lon'].values; yms = tr_s['ym'].values
csr_cache = {}
for ym in np.unique(yms):
    ym = int(ym)
    m = yms == ym
    g_m1 = GD.get(shift_ym(ym, -1)); g_t = GD.get(ym)
    gr_t = GR.get(ym); cg_t = CG.get(ym)
    if ym not in csr_cache:
        csr_cache[ym] = csr_field(ym)
    cs_t = csr_cache[ym]
    idxs = np.where(m)[0]
    for k, fld, lat_arr, lon_arr in [('gdo_m1', g_m1, gd_lat, gd_lon), ('gdo_t', g_t, gd_lat, gd_lon),
                                      ('gravis_t', gr_t, lat_g, lon_g), ('costg_t', cg_t, lat_g, lon_g),
                                      ('csr_t', cs_t, lat_c, lon_c)]:
        if fld is None: continue
        vals = np.array([lookup(fld, lat_arr, lon_arr, la, lo) for la, lo in zip(las[m], los[m])], dtype=np.float32)
        ext[k][idxs] = vals
print("externals attached")

df = pd.DataFrame(ext)
for c in COVS:
    df[c] = tr_s[c].values
    df[c+'_nxt'] = nxt_cov[c]
df['target'] = tr_s['target'].values
df['tws'] = tr_s['TWS_t'].values
df['year'] = tr_s['time'].dt.year.values

# sanity: verify target == gdo_t
ok = np.isfinite(df['gdo_t']) & np.isfinite(df['target'])
print(f"verify target==GDO(t): corr={np.corrcoef(df['gdo_t'][ok], df['target'][ok])[0,1]:.6f} "
      f"rmse={np.sqrt(np.mean((df['gdo_t'][ok]-df['target'][ok])**2)):.5f} n={ok.sum():,}")
ok2 = np.isfinite(df['gdo_m1']) & np.isfinite(df['tws'])
print(f"verify TWS_t==GDO(t-1): rmse={np.sqrt(np.mean((df['gdo_m1'][ok2]-df['tws'][ok2])**2)):.5f}")

# ---------- honest model: fit <=2012, eval 2013-15 ----------
fit = df[(df['year'] <= 2012)]
ev = df[(df['year'] >= 2013) & (df['year'] <= 2015)]

def fit_eval(feats, use_lgb=False):
    Xf = fit[feats].values.astype(np.float64)
    okf = np.isfinite(Xf).all(axis=1) & np.isfinite(fit['target'].values)
    Xe = ev[feats].values.astype(np.float64)
    oke = np.isfinite(Xe).all(axis=1) & np.isfinite(ev['target'].values)
    if use_lgb:
        import lightgbm as lgb
        dtr = lgb.Dataset(Xf[okf], label=fit['target'].values[okf])
        params = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
                      min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                      num_threads=2, seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
        bst = lgb.train(params, dtr, num_boost_round=300)
        p = bst.predict(Xe[oke])
    else:
        A = np.column_stack([Xf[okf], np.ones(okf.sum())])
        co = np.linalg.solve(A.T@A + 1e-3*np.eye(A.shape[1]), A.T@fit['target'].values[okf])
        p = np.column_stack([Xe[oke], np.ones(oke.sum())])@co
    return float(np.sqrt(np.mean((p - ev['target'].values[oke])**2))), oke.sum()

print("\n=== honest 2013-15 eval (pooled linear) ===")
r, n_ = fit_eval(['tws']); print(f"  persistence:               {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws'] + [c+'_nxt' for c in COVS]); print(f"  state + covs(t+1):         {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws', 'gravis_t']); print(f"  state + GravIS(t):         {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws', 'gravis_t', 'costg_t', 'csr_t']); print(f"  state + 3 GRACE(t):        {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws', 'gravis_t', 'costg_t', 'csr_t'] + [c+'_nxt' for c in COVS])
print(f"  state + 3 GRACE(t)+covs:   {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws', 'gravis_t', 'costg_t'] + [c+'_nxt' for c in COVS])
print(f"  state + GravIS+COSTG+covs: {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws', 'gravis_t', 'costg_t', 'csr_t'] + COVS + [c+'_nxt' for c in COVS])
print(f"  everything:                {r:.4f}  n={n_:,}")
r, n_ = fit_eval(['tws', 'gravis_t', 'costg_t', 'csr_t'] + [c+'_nxt' for c in COVS], use_lgb=True)
print(f"  3 GRACE + covs (LGBM):     {r:.4f}  n={n_:,}")
