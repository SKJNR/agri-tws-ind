"""VERIFY DATA PROVENANCE: is the competition data derived from ERA5(-Land)?

Files on disk (downloaded this morning by the wiped session):
  15a_era5land_sm12_2002_2018.nc  (swvl1, swvl2)
  15a_era5land_sm34_2002_2018.nc  (swvl3, swvl4)
  15a_era5_tp_sd_2002_2018.nc     (tp, sd)

Tests:
  V1: SOIL_MOISTURE_t (train + test) vs ERA5-Land swvl combos (standardized).
  V2: TWS_t (train + test anchors) vs storage proxy = swvl1+2+3+4 (+sd).
Key: if columns match after per-cell standardization, the full pipeline is
     Copernicus/ERA5-based => we can DOWNLOAD the real values for masked months.
"""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

for f, tag in [('15a_era5land_sm12_2002_2018.nc','sm12'), ('15a_era5land_sm34_2002_2018.nc','sm34'), ('15a_era5_tp_sd_2002_2018.nc','tpsd')]:
    ds = xr.open_dataset(f'{S}/{f}')
    print(f"=== {tag}: {f}")
    print("  vars:", list(ds.data_vars))
    print("  dims:", dict(ds.sizes))
    t = ds[list(ds.data_vars)[0]].valid_time.values if 'valid_time' in ds[list(ds.data_vars)[0]].dims else ds[list(ds.data_vars)[0]].time.values if 'time' in ds[list(ds.data_vars)[0]].dims else None
    if t is not None:
        print("  time:", pd.to_datetime(t[0]).date(), "->", pd.to_datetime(t[-1]).date(), f"n={len(t)}")
    for v in ds.data_vars:
        da = ds[v]
        latn = 'latitude' if 'latitude' in ds.coords else 'lat'
        lonn = 'longitude' if 'longitude' in ds.coords else 'lon'
        print(f"  {v}: shape={da.shape} lat[{ds[latn].min().values:.1f},{ds[latn].max().values:.1f}] lon[{ds[lonn].min().values:.1f},{ds[lonn].max().values:.1f}]")
    ds.close()
