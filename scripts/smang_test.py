"""Test GDO SMI Anomaly (smang, 0.05deg ASCAT) vs comp SOIL_MOISTURE_t with shift."""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

ds = xr.open_dataset(f'{S}/gdo_soil/smang_2016.nc')
v = ds['smang'].values
tvals = pd.to_datetime(ds['time'].values)
lat = ds['lat'].values; lon = ds['lon'].values
print(f"smang: {v.shape}, lat[{lat.min()},{lat.max()}], lon[{lon.min()},{lon.max()}]")

# aggregate 0.05 -> 1 deg on the fly for needed months (block = 20x20)
la_step = lat[1]-lat[0]; lo_step = lon[1]-lon[0]
def agg_month(i):
    f = v[i]  # (3000, 7200)
    out = np.full((180, 360), np.nan)
    # build 1deg centers covering the file's range
    lat1 = np.arange(89.5, -90.5, -1.0)
    lon1 = np.arange(-179.5, 180.5, 1.0)
    for a in range(180):
        la_hi = lat1[a]+0.5; la_lo = lat1[a]-0.5
        mi = np.where((lat <= la_hi) & (lat > la_lo))[0]
        if len(mi) == 0: continue
        for b in range(360):
            lo_lo = lon1[b]-0.5; lo_hi = lon1[b]+0.5
            mj = np.where((lon >= lo_lo) & (lon < lo_hi))[0]
            if len(mj) == 0: continue
            blk = f[np.ix_(mi, mj)]
            n = np.isfinite(blk).sum()
            if n: out[a, b] = np.nanmean(blk)
    return out, lat1, lon1

# last dekad of each month: index 2,5,8,... (dekad 1,2,3 per month)
by_month = {}
for i, t in enumerate(tvals):
    by_month[int(t.year*100+t.month)] = i
# use the LAST dekad of each month
last_dekad = {}
ym_list = sorted(set(int(f"{t.year}{t.month:02d}") for t in tvals))
for ym in ym_list:
    idxs = [i for i, t in enumerate(tvals) if int(t.year*100+t.month) == ym]
    last_dekad[ym] = idxs[-1]

def shift_ym(ym, d=-1):
    y, m = divmod(ym, 100)
    tt = y*12 + (m-1) + d
    return (tt//12)*100 + (tt % 12) + 1

te = pd.read_csv(f'{DATA}/Test (2).csv')
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

def lookup(fld, lat1, lon1, la, lo):
    li = int(np.argmin(np.abs(lat1 - la)))
    lo_i = int(np.argmin(np.abs(lon1 - lo)))
    return fld[li, lo_i]

for ym in [201601, 201606]:
    ym2 = shift_ym(ym, -1)
    if ym2 not in last_dekad:
        print(f"{ym}: t-1 ({ym2}) not in smang"); continue
    fld, lat1, lon1 = agg_month(last_dekad[ym2])
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    gv = np.array([lookup(fld, lat1, lon1, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    cv = sub['SOIL_MOISTURE_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    r = np.corrcoef(gv[ok], cv[ok])[0,1]
    print(f"  {ym} vs smang(t-1): r={r:+.4f} n={ok.sum():,} gdo_std={np.nanstd(gv[ok]):.3f} rmse={np.sqrt(np.mean((gv[ok]-cv[ok])**2)):.3f}")
