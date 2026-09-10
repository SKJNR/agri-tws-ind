"""Test CSR RL06.3 mascon grids vs competition TWS — THE match test."""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
ds = xr.open_dataset(f'{S}/csr_mascons_all.nc')
print("dims:", dict(ds.sizes))
print("vars:", list(ds.data_vars)[:10])
print("coords:", list(ds.coords))
for c in ds.coords:
    try:
        v = ds[c].values
        if v.ndim == 1 and len(v) < 300:
            print(f"  {c}: len={len(v)} [{v[0]} .. {v[-1]}]")
    except Exception:
        pass
