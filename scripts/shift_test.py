"""TIME ALIGNMENT TEST for SPEI_01: corr at shifts -2..+2.
Also test TWSA alignment at shifts."""
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
        out[ym] = (f1, lat1, lon1)
    ds.close()
    return out

# cache both years
G1 = {}
for y in [2015, 2016]:
    G1.update(load_gdo_spei(1, y))

def lookup(fld, lat, lon, la, lo):
    li = int(np.argmin(np.abs(lat - la)))
    lo_i = int(np.argmin(np.abs(((lon - lo + 180) % 360) - 180)))
    return fld[li, lo_i]

def shift_ym(ym, d):
    y, m = divmod(ym, 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

te = pd.read_csv(f'{DATA}/Test (2).csv')
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

print("=== SPEI_01 shift test at anchor months ===")
for ym in [201509, 201601, 201606]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    cv = sub['SPEI_01_t'].values
    line = f"  {ym}:"
    for d in [-2, -1, 0, 1, 2]:
        ym2 = shift_ym(ym, d)
        if ym2 not in G1:
            line += f"  t{d:+d}: n/a"
            continue
        fld, lat, lon = G1[ym2]
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv) & np.isfinite(cv)
        line += f"  t{d:+d}: {np.corrcoef(gv[ok], cv[ok])[0,1]:+.3f}"
    print(line)

print("\n=== SPEI_12 shift test ===")
G12 = {}
for y in [2015, 2016]:
    G12.update(load_gdo_spei(12, y))
for ym in [201509, 201601, 201606]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    cv = sub['SPEI_12_t'].values
    line = f"  {ym}:"
    for d in [-1, 0, 1]:
        ym2 = shift_ym(ym, d)
        if ym2 not in G12:
            line += f"  t{d:+d}: n/a"; continue
        fld, lat, lon = G12[ym2]
        gv = np.array([lookup(fld, lat, lon, la, lo) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv) & np.isfinite(cv)
        line += f"  t{d:+d}: {np.corrcoef(gv[ok], cv[ok])[0,1]:+.3f}"
    print(line)

# === TWSA shift test ===
print("\n=== TWSA shift test at anchors ===")
import glob
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
G = {}
for f in files:
    ds = xr.open_dataset(f)
    v = ds['twsan'].values
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        G[int(t.year*100+t.month)] = v[i, 0].astype(np.float64)
    ds.close()
latg = ds['lat'].values; long = ds['lon'].values
def gval(la, lo, ym):
    f = G.get(ym)
    if f is None: return np.nan
    li = int(np.argmin(np.abs(latg - la)))
    lo_i = int(np.argmin(np.abs(((long - lo + 180) % 360) - 180)))
    return f[li, lo_i]
for ym in [201509, 201601, 201606]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    cv = sub['TWS_t'].values
    line = f"  {ym}:"
    for d in [-1, 0, 1]:
        ym2 = shift_ym(ym, d)
        gv = np.array([gval(la, lo, ym2) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv) & np.isfinite(cv)
        line += f"  t{d:+d}: {np.corrcoef(gv[ok], cv[ok])[0,1]:+.3f}"
    print(line)
