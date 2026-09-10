"""Verify SPEI shift-match + identify SOIL_MOISTURE product (with 1-month shift).
Candidates for soil: GDO Ensemble Soil Moisture Anomaly, GDO Soil Moisture Index Anomaly.
"""
import numpy as np, pandas as pd, xarray as xr, glob, os

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

def load_spei(h, years=(2015, 2016)):
    out = {}
    for y in years:
        f = f'{S}/gdo_spei/spe{h}_{y}.nc'
        if not os.path.exists(f): continue
        ds = xr.open_dataset(f)
        vname = [v for v in ds.data_vars if v != 'spatial_ref'][0]
        v = ds[vname].values
        tvals = pd.to_datetime(ds['time'].values)
        lat = ds['lat'].values; lon = ds['lon'].values
        for i, t in enumerate(tvals):
            ym = int(t.year*100+t.month)
            f25 = v[i, 0]
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
            out[ym] = (f1, lat1, lon1)
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

print("=== SPEI shift match: comp SPEI(t) vs GDO SPEI(t-1) ===")
cache = {}
for h in [1, 3, 6, 12]:
    cache[h] = load_spei(h)
for ym in [201509, 201601, 201606]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    line = f"  {ym}:"
    for h, col in [(1, 'SPEI_01_t'), (3, 'SPEI_03_t'), (6, 'SPEI_06_t'), (12, 'SPEI_12_t')]:
        ym2 = shift_ym(ym, -1)
        if ym2 not in cache[h]:
            line += f"  S{h}: n/a"; continue
        fld, lat, lon = cache[h][ym2]
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        cv = sub[col].values
        ok = np.isfinite(gv) & np.isfinite(cv)
        r = np.corrcoef(gv[ok], cv[ok])[0,1]
        rm = np.sqrt(np.mean((gv[ok]-cv[ok])**2))
        line += f"  S{h}: r={r:.4f} rmse={rm:.3f}"
    print(line)
