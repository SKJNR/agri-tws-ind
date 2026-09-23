"""Aggregate ERA5 NetCDFs -> our 15,715-cell grid (centers at x.5) -> parquet (ym, cc).

ERA5 1deg grid: lat integer -90..90, lon integer 0..359.
Our grid:      lat/lon at x.5 -> each cell sits exactly between 4 ERA5 pixels.
Aggregation: 4-pixel box average (exactly bilinear at 0.5/0.5 offset).

Output: data/era5_grid.parquet [cc, ym, swvl1..4, sd, tp, e, ro, t2m]
"""
import numpy as np
import pandas as pd
import xarray as xr

DATA = '/home/z/my-project/data'
OUT = f'{DATA}/era5_grid.parquet'

# ---- our cells ----
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon'])
cells = train[['lat','lon']].drop_duplicates().reset_index(drop=True)
cells['cc'] = (cells['lat'].round(2).astype(str) + '_' + cells['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
print(f"our cells: {len(cells)}")
our_lat = cells['lat'].values
our_lon = cells['lon'].values
cc_arr = cells['cc'].values

# ---- load both files, merge vars ----
ds_ad = xr.open_dataset(f'{DATA}/era5_extract/data_stream-moda_stepType-avgad.nc')   # tp, e, ro
ds_ua = xr.open_dataset(f'{DATA}/era5_extract/data_stream-moda_stepType-avgua.nc')   # swvl1-4, sd, t2m

lat_era = ds_ua['latitude'].values          # -90..90 integers, ascending
lon_era = ds_ua['longitude'].values         # 0..359 integers
lon_norm = np.where(lon_era > 180, lon_era - 360, lon_era)   # -> -180..179

t_ad = pd.to_datetime(ds_ad['valid_time'].values).normalize()   # strip 06:00 offset
t_ua = pd.to_datetime(ds_ua['valid_time'].values).normalize()
assert (t_ad == t_ua).all(), "time axes differ between files"
ym_arr = t_ua.year.values * 100 + t_ua.month.values
n_time = len(ym_arr)
print(f"ERA5 months: {n_time} ({ym_arr.min()}..{ym_arr.max()})")

# ---- 4-pixel box indices for each of our cells ----
# lat: our x.5 -> pixels floor(x.5)=x and x+1
la0 = np.floor(our_lat).astype(int)          # e.g. -55.5 -> -56
la1 = la0 + 1                                # -55
# lon: our y.5 -> pixels floor and floor+1 (in normalized -180..179 space)
lo0 = np.floor(our_lon).astype(int)          # e.g. -179.5 -> -180
lo1 = lo0 + 1                                # -179

# index into ERA5 arrays
lat_idx = {int(v): i for i, v in enumerate(lat_era)}
lon_idx = {int(v): i for i, v in enumerate(np.sort(lon_norm))}   # sorted lons with their positions
lon_order = np.argsort(lon_norm)
lon_sorted = lon_norm[lon_order]

def ilat(v): return lat_idx[int(v)]
def ilon(v):
    # find position of value v in sorted normalized lons
    p = np.searchsorted(lon_sorted, v)
    return lon_order[p]

# vectorized: for each cell, 4 pixel (ilat, ilon) pairs
Ia0 = np.array([lat_idx[v] for v in la0])
Ia1 = np.array([lat_idx[v] for v in la1])
Io0 = np.array([ilon(v) for v in lo0])
Io1 = np.array([ilon(v) for v in lo1])

# sanity: distances
print(f"pixel offsets: lat {la0[0]}+{la1[0]} vs center {our_lat[0]}; lon {lo0[0]}+{lo1[0]} vs center {our_lon[0]}")

# ---- extract 4-pixel average for all vars ----
def box_avg(da):
    """da: (time, lat, lon) -> (time, n_cells) 4-pixel average"""
    v = da.values.astype(np.float32)
    return 0.25 * (v[:, Ia0, Io0] + v[:, Ia0, Io1] + v[:, Ia1, Io0] + v[:, Ia1, Io1])

out_cols = {}
for v in ['swvl1','swvl2','swvl3','swvl4','sd','t2m']:
    out_cols[v] = box_avg(ds_ua[v])
    print(f"  {v}: nan%={np.isnan(out_cols[v]).mean()*100:.2f}")
for v in ['tp','e','ro']:
    out_cols[v] = box_avg(ds_ad[v])
    print(f"  {v}: nan%={np.isnan(out_cols[v]).mean()*100:.2f}")

# ---- build dataframe ----
n_cells = len(cells)
df = pd.DataFrame({
    'ym': np.repeat(ym_arr, n_cells),
    'cc': np.tile(cc_arr, n_time),
})
for v, arr in out_cols.items():
    df[v] = arr.reshape(-1)
df = df.drop_duplicates(['ym','cc']).sort_values(['cc','ym']).reset_index(drop=True)
df.to_parquet(OUT, index=False)

print(f"\nsaved {OUT}: {len(df):,} rows x {len(df.columns)} cols")
print(f"months per cell: {df.groupby('cc').size().describe().to_dict()}")
print(f"\nsample stats:")
print(df[['swvl1','swvl2','swvl3','swvl4','sd','tp','e','ro','t2m']].describe().T[['mean','std','min','max']])
