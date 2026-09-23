"""SPEI match test v2 (clean)."""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

def load_gdo_spei(h, year):
    f = f'{S}/gdo_spei/spe{h}_{year}.nc'
    ds = xr.open_dataset(f)
    vname = [v for v in ds.data_vars if v != 'spatial_ref'][0]
    v = ds[vname].values
    tvals = pd.to_datetime(ds['time'].values)
    lat = ds['lat'].values; lon = ds['lon'].values
    out = {}
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
        out[ym] = (f1, lat1, lon1)   # last dekad of month overwrites earlier ones
    ds.close()
    return out

def lookup(fld, lat, lon, la, lo):
    li = int(np.argmin(np.abs(lat - la)))
    lo_i = int(np.argmin(np.abs(((lon - lo + 180) % 360) - 180)))
    return fld[li, lo_i]

te = pd.read_csv(f'{DATA}/Test (2).csv')
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

print("=== SPEI match at anchor months ===")
for ym in [201509, 201601, 201606]:
    year = ym // 100
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    for h, col in [(1, 'SPEI_01_t'), (3, 'SPEI_03_t'), (6, 'SPEI_06_t'), (12, 'SPEI_12_t')]:
        try:
            G = load_gdo_spei(h, year)
        except Exception as e:
            print(f"  {ym} h={h}: LOAD ERROR {e}")
            continue
        if ym not in G:
            print(f"  {ym} SPEI{h}: month missing; have {sorted(G)[:4]}")
            continue
        fld, lat, lon = G[ym]
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        cv = sub[col].values
        ok = np.isfinite(gv) & np.isfinite(cv)
        if ok.sum() < 100:
            print(f"  {ym} {col}: too few finite ({ok.sum()})")
            continue
        r = np.corrcoef(gv[ok], cv[ok])[0,1]
        print(f"  {ym} {col}: corr={r:+.4f} n={ok.sum():,} gdo_std={np.nanstd(gv[ok]):.3f} comp_std={np.nanstd(cv[ok]):.3f}")
