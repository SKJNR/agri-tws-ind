"""Test GDO ERA5 SPEI products vs competition SPEI columns."""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

def load_gdo_spei(h, year):
    """returns dict ym -> 1deg field (lat desc 89.5..-89.5, lon -179.5..179.5)"""
    f = f'{S}/gdo_spei/spe{h}_{year}.nc'
    try:
        ds = xr.open_dataset(f)
    except Exception as e:
        print(f"  fail {f}: {e}"); return {}
    vname = [v for v in ds.data_vars if v != 'spatial_ref'][0]
    v = ds[vname].values  # (time, band, lat, lon)
    tvals = pd.to_datetime(ds['time'].values)
    lat = ds['lat'].values; lon = ds['lon'].values
    out = {}
    for i, t in enumerate(tvals):
        ym = t.year*100+t.month
        f25 = v[i, 0]
        # aggregate 0.25 -> 1deg
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
        # dedupe: keep last dekad per month (dekadal products)
        out[ym] = (f1, lat1, lon1)
    ds.close()
    # restructure: keep last entry per ym
    res = {}
    for ym, (f1, lat1, lon1) in out.items():
        res[ym] = (f1, lat1, lon1)
    return res

# load comp SPEI for a test month
te = pd.read_csv(f'{DATA}/Test (2).csv')
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

def comp_col(ym, col, masked_ok=False):
    sub = te[te['ym'] == ym]
    if not masked_ok:
        sub = sub[~sub['TWS_t_masked']]
    return sub

def lookup(fld, lat, lon, la, lo):
    li = int(np.argmin(np.abs(lat - la)))
    lo_i = int(np.argmin(np.abs(((lon - lo + 180) % 360) - 180)))
    return fld[li, lo_i]

# test months: 201601 (anchor) and 201509
print("=== SPEI match at anchor months (pooled corr over cells) ===")
for ym in [201509, 201601, 201606]:
    year = ym // 100
    sub = comp_col(ym, None)
    for h, col in [(1, 'SPEI_01_t'), (3, 'SPEI_03_t'), (6, 'SPEI_06_t'), (12, 'SPEI_12_t')]:
        try:
            G, lat, lon = load_gdo_spei(h, year)
        except Exception as e:
            print(f"  {ym} h={h}: load fail"); continue
        fld = G.get(ym)
        if fld is None:
            print(f"  {ym} SPEI{h}: month missing ({sorted(G)[:3]}...)")
            continue
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        cv = sub[col].values
        ok = np.isfinite(gv) & np.isfinite(cv)
        r = np.corrcoef(gv[ok], cv[ok])[0,1]
        print(f"  {ym} {col}: corr={r:+.4f} n={ok.sum():,} gdo_std={np.nanstd(gv[ok]):.3f} comp_std={np.nanstd(cv[ok]):.3f}")
