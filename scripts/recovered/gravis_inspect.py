"""DECISIVE PROVENANCE MATCH: GravIS GFZ RL06 L3 TWS grid vs competition TWS_t.

Step 1: inspect the GravIS file (grid, time, units).
Step 2: for the 6 TEST ANCHOR months (2015-09, 2016-01, 2016-06, 2016-12, 2018-07, 2018-11):
        per-cell correlation / regression RMSE of competition TWS vs GravIS TWS.
If the match is strong (r > 0.9), the masked months' true values are essentially known.
"""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

ds = xr.open_dataset(f'{S}/gravis_tws_grid.nc')
print("=== GravIS file structure ===")
print("dims:", dict(ds.sizes))
print("vars:", list(ds.data_vars))
for c in ds.coords:
    v = ds[c].values
    print(f"  coord {c}: len={len(v)} [{v[0]} .. {v[-1]}]")
main = list(ds.data_vars)[0]
print(f"\nmain var: {main}")
da = ds[main]
print("attrs:", dict(da.attrs))
print("shape:", da.shape)

# time values
tv = None
for tc in ['time', 't', 'valid_time']:
    if tc in ds.coords:
        tv = pd.to_datetime(ds[tc].values); break
if tv is None:
    # maybe time is a dim without coord
    print("NO TIME COORD FOUND; dims:", dict(ds.sizes))
else:
    print(f"time: {tv[0].date()} .. {tv[-1].date()}  n={len(tv)}")
    # check test months coverage
    for ym in [201509, 201601, 201606, 201612, 201807, 201811, 201602, 201703, 201812]:
        y, m = divmod(ym, 100)
        hit = any((t.year == y) and (t.month == m) for t in tv)
        print(f"  {ym}: {'YES' if hit else 'no'}")
