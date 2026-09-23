"""Test soil moisture candidates vs comp SOIL_MOISTURE_t (with 1-month shift).
- GDO Ensemble Soil Moisture Anomaly (smant)  [dekadal 0.25deg?]
- GDO Soil Moisture Index Anomaly (smang)
"""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

def load_nc_series(path):
    ds = xr.open_dataset(path)
    vname = [v for v in ds.data_vars if v != 'spatial_ref'][0]
    v = ds[vname].values
    tvals = pd.to_datetime(ds['time'].values)
    lat = ds['lat'].values; lon = ds['lon'].values
    out = {}
    for i, t in enumerate(tvals):
        ym = int(t.year*100+t.month)
        f25 = v[i, 0] if v.ndim == 4 else v[i]
        if f25.shape[0] == 720:
            f1 = np.full((180, 360), np.nan)
            for a in range(180):
                for b in range(360):
                    blk = f25[4*a:4*a+4, 4*b:4*b+4]
                    n = np.isfinite(blk).sum()
                    if n: f1[a, b] = np.nanmean(blk)
            lat1 = np.array([np.nanmean(lat[4*a:4*a+4]) for a in range(180)])
            lon1 = np.array([np.nanmean(lon[4*b:4*b+4]) for b in range(360)])
        else:
            f1 = f25; lat1 = lat; lon1 = lon
        out[ym] = (f1, lat1, lon1)   # last dekad of month
    ds.close()
    return out

def lookup(fld, lat, lon, la, lo):
    li = int(np.argmin(np.abs(lat - la)))
    lo_i = int(np.argmin(np.abs(((lon - lo + 180) % 360) - 180)))
    return fld[li, lo_i]

def shift_ym(ym, d=-1):
    y, m = divmod(ym, 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

te = pd.read_csv(f'{DATA}/Test (2).csv')
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

print("=== SOIL_MOISTURE_t vs candidates at t-1 ===")
SM = {}
for y in [2015, 2016]:
    SM.update(load_nc_series(f'{S}/gdo_soil/smant_{y}.nc'))
SMI = {}
try:
    SMI = load_nc_series(f'{S}/gdo_soil/smang_2016.nc')
except Exception as e:
    print(f"  smang load failed: {e}")

for ym in [201509, 201601, 201606]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    cv = sub['SOIL_MOISTURE_t'].values
    ym2 = shift_ym(ym, -1)
    for name, C in [("smant(ens SMA)", SM), ("smang(SMI anom)", SMI)]:
        if ym2 not in C:
            print(f"  {ym} {name}: month missing"); continue
        fld, lat, lon = C[ym2]
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv) & np.isfinite(cv)
        if ok.sum() < 100:
            print(f"  {ym} {name}: too few ({ok.sum()})"); continue
        r = np.corrcoef(gv[ok], cv[ok])[0,1]
        rm = np.sqrt(np.mean((gv[ok]-cv[ok])**2))
        print(f"  {ym} {name}: r={r:+.4f} rmse={rm:.4f} n={ok.sum():,} gdo_std={np.nanstd(gv[ok]):.3f}")
    # also unshifted for reference
    if ym in SM:
        fld, lat, lon = SM[ym]
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv) & np.isfinite(cv)
        if ok.sum() > 100:
            print(f"  {ym} smant UNshifted: r={np.corrcoef(gv[ok], cv[ok])[0,1]:+.4f}")
