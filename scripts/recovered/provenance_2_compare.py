"""PROVENANCE TEST 2: compare competition columns vs ERA5-Land swvl1-4 (European box).

If SOIL_MOISTURE_t == standardized(ERA5-Land soil moisture) -> pipeline confirmed.
If TWS_t == standardized(swvl-combo) -> FULL reconstruction possible.
"""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

# ---- load ERA5-Land, aggregate 0.1 -> 1 deg ----
ds1 = xr.open_dataset(f'{S}/15a_era5land_sm12_2002_2018.nc')
ds3 = xr.open_dataset(f'{S}/15a_era5land_sm34_2002_2018.nc')
lat = ds1.latitude.values; lon = ds1.longitude.values  # 0.1deg, lat descending
times = pd.to_datetime(ds1.valid_time.values)
ym_idx = {t.year*100+t.month: i for i, t in enumerate(times)}

def to1deg(da):
    """average 10x10 blocks -> 1deg centers"""
    la = da.latitude.values; lo = da.longitude.values; f = da.values  # (T, lat, lon)
    lat_c = np.arange(30.5, 72.0, 1.0)
    lon_c = np.arange(-14.5, 45.0, 1.0)
    out = np.full((f.shape[0], len(lat_c), len(lon_c)), np.nan, dtype=np.float32)
    for i, lc in enumerate(lat_c):
        mi = (la >= lc-0.5) & (la < lc+0.5)
        if not mi.any(): continue
        for j, oc in enumerate(lon_c):
            mj = (lo >= oc-0.5) & (lo < oc+0.5)
            if not mj.any(): continue
            out[:, i, j] = np.nanmean(f[:, mi, :][:, :, mj], axis=(1, 2))
    return lat_c, lon_c, out

lat_c, lon_c, SW1 = to1deg(ds1.swvl1)
_, _, SW2 = to1deg(ds1.swvl2)
_, _, SW3 = to1deg(ds3.swvl3)
_, _, SW4 = to1deg(ds3.swvl4)
print(f"ERA5-Land 1deg: {SW1.shape}, months {times[0].date()}..{times[-1].date()}")

# ERA5 storage proxy (sum of layer volumetric*depth? use raw sum first)
STOR = SW1 + SW2 + SW3 + SW4  # crude

# ---- competition data in the box ----
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','SOIL_MOISTURE_t','target'])
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','SOIL_MOISTURE_t','TWS_t_masked'])
for df in (tr, te):
    df['time'] = pd.to_datetime(df['time'])
    df['ym'] = df['time'].dt.year*100 + df['time'].dt.month

def era_lookup(df):
    """attach era5 fields at each (lat, lon, ym)"""
    i_lat = np.searchsorted(lat_c, df['lat'].values)  # lat_c ascending
    ok_lat = (i_lat > 0) & (i_lat < len(lat_c))
    i_lon = np.searchsorted(lon_c, df['lon'].values)
    ok_lon = (i_lon > 0) & (i_lon < len(lon_c))
    mi = df['ym'].map(ym_idx).values
    ok = ok_lat & ok_lon & (mi >= 0)
    ii = np.clip(i_lat-1, 0, len(lat_c)-1); jj = np.clip(i_lon-1, 0, len(lon_c)-1)
    out = {k: np.full(len(df), np.nan) for k in ['sw1','sw2','sw3','sw4','stor']}
    idx = (mi[ok], ii[ok], jj[ok])
    out['sw1'][ok] = SW1[idx]; out['sw2'][ok] = SW2[idx]
    out['sw3'][ok] = SW3[idx]; out['sw4'][ok] = SW4[idx]
    out['stor'][ok] = STOR[idx]
    return out, ok

era_tr, ok_tr = era_lookup(tr)
print(f"train rows in box+era: {ok_tr.sum():,}")

m = ok_tr & np.isfinite(tr['SOIL_MOISTURE_t'].values) & np.isfinite(era_tr['sw1'])
sm = tr['SOIL_MOISTURE_t'].values[m]
print("\n=== SOIL_MOISTURE_t vs ERA5-Land (pooled over box, train) ===")
for k in ['sw1','sw2','sw3','sw4','stor']:
    r = np.corrcoef(sm, era_tr[k][m])[0,1]
    print(f"  corr(SOIL_MOISTURE_t, {k}) = {r:+.4f}")

tw = tr['TWS_t'].values[m]
print("\n=== TWS_t vs ERA5-Land (pooled over box, train) ===")
for k in ['sw1','sw2','sw3','sw4','stor']:
    r = np.corrcoef(tw, era_tr[k][m])[0,1]
    print(f"  corr(TWS_t, {k}) = {r:+.4f}")

# ---- per-cell time-series correlation (the real test) ----
print("\n=== PER-CELL time-series correlations (cells with >= 60 matching train months) ===")
tr_m = tr[m].copy()
for k in ['sw1','sw2','sw3','sw4','stor']:
    tr_m[k] = era_tr[k][m]
g = tr_m.groupby(['lat','lon'])
res = {k: [] for k in ['sw1','sw2','sw3','sw4','stor']}
resT = {k: [] for k in ['sw1','sw2','sw3','sw4','stor']}
for key, sub in g:
    if len(sub) < 60: continue
    for k in ['sw1','sw2','sw3','sw4','stor']:
        if sub[k].notna().sum() < 60: continue
        r = np.corrcoef(sub['SOIL_MOISTURE_t'].values, sub[k].values)[0,1]
        res[k].append(r)
        r2 = np.corrcoef(sub['TWS_t'].values, sub[k].values)[0,1]
        resT[k].append(r2)
print(f"cells tested: {len(res['sw1'])}")
print("SOIL_MOISTURE_t per-cell corr distribution (median / mean / p90):")
for k in ['sw1','sw2','sw3','sw4','stor']:
    a = np.array(res[k]); print(f"  {k}: {np.median(a):+.3f} / {np.mean(a):+.3f} / {np.percentile(a,90):+.3f}")
print("TWS_t per-cell corr distribution:")
for k in ['sw1','sw2','sw3','sw4','stor']:
    a = np.array(resT[k]); print(f"  {k}: {np.median(a):+.3f} / {np.mean(a):+.3f} / {np.percentile(a,90):+.3f}")

# ---- standardization check on competition columns ----
print("\n=== standardization check (per-cell mean/std over TRAIN, box cells) ===")
for col in ['TWS_t','SOIL_MOISTURE_t']:
    g2 = tr_m.groupby(['lat','lon'])[col]
    mu = g2.mean(); sd = g2.std()
    print(f"  {col}: mean-of-means={mu.mean():+.4f} mean-of-stds={sd.mean():.4f} std-of-stds={sd.std():.4f}")

# ---- check exact relationship for best match: regression TWS on stor ----
print("\n=== TWS_t ~ stor per-cell regression R2 (in-sample diagnostic) ===")
r2s = []
for key, sub in g:
    if len(sub) < 60 or sub['stor'].notna().sum() < 60: continue
    x = sub['stor'].values; y = sub['TWS_t'].values
    A = np.column_stack([x, np.ones(len(x))])
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A@c
    r2s.append(1 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2))
r2s = np.array(r2s)
print(f"  per-cell R2: median={np.median(r2s):.3f} mean={np.mean(r2s):.3f} p90={np.percentile(r2s,90):.3f}")
