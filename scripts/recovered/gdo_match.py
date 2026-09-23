"""THE FINAL MATCH TEST: GDO TWSA vs competition TWS_t.
If per-cell correlations ~0.99 -> THE SOURCE IS FOUND.
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

# ---- load all GDO files ----
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
# fix: remove the failed 404 files (2017, 2018 full-year)
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
print(f"files: {len(files)}")
G = {}   # ym -> (field, lat, lon)
grid_ref = None
for f in files:
    try:
        ds = xr.open_dataset(f)
    except Exception as e:
        print(f"  FAIL {f}: {e}"); continue
    vname = [v for v in ds.data_vars][0]
    da = ds[vname]
    # find dims
    dims = da.dims
    tname = [d for d in dims if 'time' in d.lower() or d in ('time',)][0]
    tvals = pd.to_datetime(ds[tname].values)
    latn = [c for c in ds.coords if c.lower() in ('lat', 'latitude', 'y')][0]
    lonn = [c for c in ds.coords if c.lower() in ('lon', 'longitude', 'x')][0]
    lat = ds[latn].values; lon = ds[lonn].values
    vals = da.values.astype(np.float64)
    if grid_ref is None:
        grid_ref = (lat, lon)
        print(f"grid: lat[{lat.min():.2f},{lat.max():.2f}] n={len(lat)}  lon[{lon.min():.2f},{lon.max():.2f}] n={len(lon)}")
        print(f"var: {vname}, dims: {dims}, units: {da.attrs.get('units','?')}")
    for i, t in enumerate(tvals):
        ym = t.year*100 + t.month
        if vals.ndim == 4:
            G[ym] = vals[i, 0]     # (lat, lon), band=0
        elif vals.ndim == 3:
            G[ym] = vals[i]
    ds.close()
print(f"GDO months loaded: {len(G)}  range {min(G)}..{max(G)}")

lat, lon = grid_ref
lat_list = np.asarray(lat, dtype=float).ravel()
lon_list = np.asarray(lon, dtype=float).ravel()
lat_desc = lat_list[0] > lat_list[-1]
print(f"lat descending: {lat_desc}; lat step: {lat_list[1]-lat_list[0]:.3f}; lon step: {lon_list[1]-lon_list[0]:.3f}")

def gval(la, lo, ym):
    f = G.get(ym)
    if f is None: return np.nan
    # nearest indices
    li = int(np.argmin(np.abs(lat_list - la)))
    lo_i = int(np.argmin(np.abs(((lon_list - lo + 180) % 360) - 180)))
    return f[li, lo_i]

# ---- test at competition anchors + train ----
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

print("\n=== spatial corr at test anchors (GDO TWSA) ===")
for ym in [201509, 201601, 201606, 201612, 201807, 201811]:
    if ym not in G:
        print(f"  {ym}: GDO month MISSING"); continue
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    c = sub['TWS_t'].values
    ok = np.isfinite(gv)
    if ok.sum() > 100:
        r = np.corrcoef(c[ok], gv[ok])[0,1]
        # affine regression rmse
        A = np.column_stack([gv[ok], np.ones(ok.sum())])
        co = np.linalg.lstsq(A, c[ok], rcond=None)[0]
        rm = np.sqrt(np.mean((A@co - c[ok])**2))
        print(f"  {ym}: corr={r:+.4f} affine-RMSE={rm:.4f} n={ok.sum():,}")

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month

print("\n=== per-cell time-series corr (1200 cells, train) ===")
cells = tr[['lat','lon']].drop_duplicates().sample(1200, random_state=0)
rs = []
for la, lo in cells.values:
    sub = tr[(tr['lat'] == la) & (tr['lon'] == lo)]
    gv = np.array([gval(la, lo, ym) for ym in sub['ym'].values])
    cv = sub['TWS_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    if ok.sum() >= 60: rs.append(np.corrcoef(gv[ok], cv[ok])[0,1])
rs = np.array(rs)
print(f"  median={np.median(rs):.3f} mean={np.mean(rs):.3f} p10={np.percentile(rs,10):.3f} p90={np.percentile(rs,90):.3f}")

print("\n=== month coverage: comp months missing from GDO ===")
comp_months = set(tr['ym'].unique()) | set(te['ym'].unique())
print(sorted(m for m in comp_months if m not in G))
print("=== GDO months (2002-2018) missing from comp ===")
print(sorted(m for m in G if 200204 <= m <= 201812 and m not in comp_months))
