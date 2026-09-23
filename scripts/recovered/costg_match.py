"""Test COST-G combined TWS vs competition TWS (same protocol as GravIS-GFZOP test)."""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
ds = xr.open_dataset(f'{S}/gravis_costg_tws.nc')
print("dims:", dict(ds.sizes), "vars:", list(ds.data_vars))
tws = ds['tws'].values.astype(np.float64)
tv = pd.to_datetime(ds['time'].values)
lat_g = ds['lat'].values; lon_g = ds['lon'].values
ym_idx = {t.year*100+t.month: i for i, t in enumerate(tv)}
lat_pos = {v: i for i, v in enumerate(lat_g)}
print(f"coverage: {tv[0].date()}..{tv[-1].date()}")

def gval(la, lo, ym):
    if ym not in ym_idx: return np.nan
    glo = lo if lo >= 0 else lo + 360
    li = lat_pos.get(round(float(la), 1))
    if li is None: li = int(np.argmin(np.abs(lat_g - la)))
    return tws[ym_idx[ym], li, int(round((glo - 0.5))) % 360]

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month

te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

print("\n=== spatial corr at test anchors (COST-G) ===")
for ym in [201509, 201601, 201606, 201612, 201807, 201811]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    c = sub['TWS_t'].values
    ok = np.isfinite(gv)
    print(f"  {ym}: corr={np.corrcoef(c[ok], gv[ok])[0,1]:+.4f} n={ok.sum():,}")

print("\n=== per-cell time-series corr (sample 1500 cells, train) ===")
cells = tr[['lat','lon']].drop_duplicates().sample(1500, random_state=0)
rs = []
for la, lo in cells.values:
    sub = tr[(tr['lat'] == la) & (tr['lon'] == lo)]
    gv = np.array([gval(la, lo, ym) for ym in sub['ym'].values])
    cv = sub['TWS_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    if ok.sum() >= 60: rs.append(np.corrcoef(gv[ok], cv[ok])[0,1])
rs = np.array(rs)
print(f"  median={np.median(rs):.3f} mean={np.mean(rs):.3f} p10={np.percentile(rs,10):.3f} p90={np.percentile(rs,90):.3f}")

print("\n=== per-year spatial corr drift ===")
for yr in [2004, 2008, 2012, 2015]:
    yms = [y for y in tr['ym'].unique() if y//100 == yr][:3]
    rr = []
    for ym in yms:
        sub = tr[tr['ym'] == ym]
        gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv)
        rr.append(np.corrcoef(sub['TWS_t'].values[ok], gv[ok])[0,1])
    print(f"  {yr}: {np.mean(rr):+.3f}")
