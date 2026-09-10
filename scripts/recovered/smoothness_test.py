"""TWO DECISIVE DIAGNOSTICS:
D1. Smoothness: LOO neighbor prediction error of comp TWS vs real GRACE products.
    If comp is much smoother than ALL real products -> synthetic generator, stop hunt.
D2. Error decomposition on M2 mirror: D-tilde error vs fast-state error.
"""
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

# ---------- D1: smoothness ----------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month

cells = train[['lat','lon']].drop_duplicates().sort_values(['lat','lon'])
cell_xy = cells[['lat','lon']].values
n_cells = len(cells)
cmap = {(la, lo): i for i, (la, lo) in enumerate(cell_xy)}
train['ci'] = [cmap[(la, lo)] for la, lo in zip(train['lat'], train['lon'])]

# neighbor structure (within 1.6 deg)
la_r = np.deg2rad(cell_xy[:,0]); lo_r = np.deg2rad(cell_xy[:,1])
pts = np.column_stack([np.cos(la_r)*np.cos(lo_r), np.cos(la_r)*np.sin(lo_r), np.sin(la_r)])
tree = cKDTree(pts)
pairs = tree.query_pairs(np.deg2rad(1.6))
print(f"neighbor pairs (<=1.6deg): {len(pairs):,}")

def loo_err(fields_by_month, label):
    """LOO: predict cell from mean of its neighbors; return pooled err / std."""
    errs, stds = [], []
    for ym, f in list(fields_by_month.items())[::12]:
        vals = f
        ok = np.isfinite(vals)
        ii, jj = np.array(list(pairs)).T
        m = ok[ii] & ok[jj]
        # LOO: value at i = mean of neighbors j
        s = np.zeros(n_cells); c = np.zeros(n_cells)
        np.add.at(s, ii[m], vals[jj[m]]); np.add.at(c, ii[m], 1)
        np.add.at(s, jj[m], vals[ii[m]]); np.add.at(c, jj[m], 1)
        has = (c > 0) & ok
        pred = s[has]/c[has]
        errs.append(np.sqrt(np.mean((pred - vals[has])**2)))
        stds.append(np.nanstd(vals))
    e = np.mean(errs); s_ = np.mean(stds)
    print(f"  {label}: LOO err = {e:.4f}, field std = {s_:.4f}, ratio = {e/s_:.4f}")

# comp fields
comp_fields = {}
for ym in [200305, 200501, 200706, 200906, 201206, 201406]:
    sub = train[train['ym'] == ym]
    f = np.full(n_cells, np.nan)
    f[sub['ci'].values] = sub['TWS_t'].values
    comp_fields[ym] = f
print("D1. SMOOTHNESS (LOO neighbor error ratio; lower = smoother):")
loo_err(comp_fields, "COMPETITION TWS")

# gravis fields
ds = xr.open_dataset(f'{S}/gravis_tws_grid.nc')
tws = ds['tws'].values.astype(np.float64)
tv = pd.to_datetime(ds['time'].values)
latg = ds['lat'].values; long = ds['lon'].values
latpos = {round(float(v),1): i for i, v in enumerate(latg)}
gravis_fields = {}
for ym in [200305, 200501, 200706, 200906, 201206, 201406]:
    y, m = divmod(ym, 100)
    idx = [i for i, t in enumerate(tv) if t.year == y and t.month == m]
    if not idx: continue
    f = np.full(n_cells, np.nan)
    for ci, (la, lo) in enumerate(cell_xy):
        glo = lo if lo >= 0 else lo + 360
        li = latpos.get(round(float(la),1))
        lo_i = int(round(glo - 0.5)) % 360
        f[ci] = tws[idx[0], li, lo_i]
    gravis_fields[ym] = f
loo_err(gravis_fields, "GravIS GFZ (real GRACE)")

# csr mascon fields (1deg aggregated)
ds2 = xr.open_dataset(f'{S}/csr_mascons_all.nc')
tv2 = pd.to_datetime(ds2['time'].values, unit='D', origin=pd.Timestamp('2002-01-01'))
lwe = ds2['lwe_thickness'].values.astype(np.float32)
csr_fields = {}
for ym in [200305, 200501, 200706, 200906, 201206, 201406]:
    y, m = divmod(ym, 100)
    idx = [i for i, t in enumerate(tv2) if t.year == y and t.month == m]
    if not idx: continue
    f = np.full(n_cells, np.nan)
    for ci, (la, lo) in enumerate(cell_xy):
        glo = lo if lo >= 0 else lo + 360
        # 0.25deg: 4 rows/deg, lat ascending from -89.875
        li = int((la + 89.875)/0.25); lo_i = int(glo/0.25) % 1440
        blk = lwe[idx[0], max(li-2,0):li+2, max(lo_i-2,0):lo_i+2]
        if np.isfinite(blk).any(): f[ci] = np.nanmean(blk)
    csr_fields[ym] = f
loo_err(csr_fields, "CSR mascons (real GRACE)")
