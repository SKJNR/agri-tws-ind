"""THE DECISIVE MATCH TEST: GravIS GFZ TWS vs competition TWS_t.

A) Spatial-pattern correlation at the 6 test anchor months.
B) Per-cell affine fit on TRAIN months (competition = a*gravis + b), then
   out-of-sample RMSE at the 6 test anchors.  <-- expected submission quality
C) Same for a few train-era months (sanity).
"""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

ds = xr.open_dataset(f'{S}/gravis_tws_grid.nc')
tws = ds['tws']          # (time, lat, lon), lat desc 89.5..-89.5, lon 0.5..359.5
std = ds['std_tws']; leak = ds['leakage']
tv = pd.to_datetime(ds['time'].values)
lat_g = ds['lat'].values  # descending
lon_g = ds['lon'].values  # 0.5..359.5
ym_idx = {t.year*100+t.month: i for i, t in enumerate(tv)}
# lat index lookup (descending)
lat_pos = {v: i for i, v in enumerate(lat_g)}

def gravis_field(ym, use_leak=False, use_std=False):
    if ym not in ym_idx: return None, None
    i = ym_idx[ym]
    f = tws.isel(time=i).values.astype(np.float64)          # (180, 360)
    if use_leak: f = f + leak.isel(time=i).values.astype(np.float64)
    return f, lat_g, lon_g

# competition cells
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

def comp_field(ym):
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    return sub

def lookup_gravis(f, lats, lons):
    """map competition (lat, lon) to gravis grid values"""
    out = np.full(len(lats), np.nan)
    for k, (la, lo) in enumerate(zip(lats, lons)):
        glo = lo if lo >= 0 else lo + 360
        # nearest 0.5-offset
        li = lat_pos.get(round(float(la), 1), None)
        if li is None:
            # nearest
            li = int(np.argmin(np.abs(lat_g - la)))
        lo_i = int(round((glo - 0.5))) % 360
        out[k] = f[li, lo_i]
    return out

print("=== A) spatial corr at test anchors: competition TWS vs GravIS ===")
anchors = [201509, 201601, 201606, 201612, 201807, 201811]
for ym in anchors:
    sub = comp_field(ym)
    f, _, _ = gravis_field(ym)
    f2, _, _ = gravis_field(ym, use_leak=True)
    g = lookup_gravis(f, sub['lat'].values, sub['lon'].values)
    g2 = lookup_gravis(f2, sub['lat'].values, sub['lon'].values)
    c = sub['TWS_t'].values
    ok = np.isfinite(g) & np.isfinite(c)
    r = np.corrcoef(c[ok], g[ok])[0,1]
    r2 = np.corrcoef(c[ok], g2[ok])[0,1]
    # regression rmse
    A = np.column_stack([g[ok], np.ones(ok.sum())])
    coef = np.linalg.lstsq(A, c[ok], rcond=None)[0]
    rm = np.sqrt(np.mean((A@coef - c[ok])**2))
    print(f"  {ym}: n={ok.sum():,} corr={r:+.4f} corr(+leak)={r2:+.4f} regRMSE={rm:.4f}  comp_std={c.std():.3f} gravis_std={g[ok].std():.3f}")

print("\n=== B) per-cell affine fit on TRAIN, out-of-sample at anchors ===")
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month
# precompute gravis fields for all needed yms
needed = set(tr['ym'].unique()) | set(anchors)
gf = {}
for ym in needed:
    f, _, _ = gravis_field(ym)
    if f is not None: gf[int(ym)] = f
print(f"gravis months available for train: {len(set(tr['ym'].unique()) & set(gf))}/{tr['ym'].nunique()}")

# per-cell regression using train months
cells = tr[['lat','lon']].drop_duplicates()
print(f"fitting per-cell affine on {len(cells):,} cells ...")
a_c = np.zeros(len(cells)); b_c = np.zeros(len(cells)); n_c = np.zeros(len(cells), dtype=int)
lats_c = cells['lat'].values; lons_c = cells['lon'].values
tr_g = tr.groupby(['lat','lon'])
i = 0
for (la, lo), sub in tr_g:
    gv = []
    for ym in sub['ym'].values:
        f = gf.get(int(ym))
        if f is None: gv.append(np.nan); continue
        glo = lo if lo >= 0 else lo + 360
        li = lat_pos.get(round(float(la), 1), None)
        if li is None: li = int(np.argmin(np.abs(lat_g - la)))
        lo_i = int(round((glo - 0.5))) % 360
        gv.append(f[li, lo_i])
    gv = np.array(gv); cv = sub['TWS_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    n_c[i] = ok.sum()
    if ok.sum() >= 40:
        A = np.column_stack([gv[ok], np.ones(ok.sum())])
        try:
            coef = np.linalg.lstsq(A, cv[ok], rcond=None)[0]
            a_c[i] = coef[0]; b_c[i] = coef[1]
        except Exception:
            a_c[i] = 0; b_c[i] = 0
    else:
        a_c[i] = 0; b_c[i] = 0
    i += 1
print(f"cells with >=40 months: {(n_c>=40).sum():,}")

# out-of-sample at anchors
cell_map = {(la, lo): i for i, (la, lo) in enumerate(zip(lats_c, lons_c))}
for ym in anchors:
    sub = comp_field(ym)
    f = gf.get(ym)
    if f is None: continue
    g = lookup_gravis(f, sub['lat'].values, sub['lon'].values)
    idx = [cell_map[(la, lo)] for la, lo in zip(sub['lat'].values, sub['lon'].values)]
    pred = a_c[idx]*g + b_c[idx]
    c = sub['TWS_t'].values
    ok = np.isfinite(pred) & np.isfinite(c) & (np.abs(a_c[idx]) > 1e-6)
    rm = np.sqrt(np.mean((pred[ok]-c[ok])**2))
    r = np.corrcoef(pred[ok], c[ok])[0,1]
    print(f"  anchor {ym}: OOS RMSE={rm:.4f} corr={r:+.4f} n={ok.sum():,}")

print("\n=== C) train-era sanity (2013-2015 monthly spatial corr) ===")
for ym in [201301, 201306, 201312, 201406, 201506]:
    sub = tr[tr['ym']==ym]
    f = gf.get(ym)
    if f is None: continue
    g = lookup_gravis(f, sub['lat'].values, sub['lon'].values)
    c = sub['TWS_t'].values
    ok = np.isfinite(g) & np.isfinite(c)
    r = np.corrcoef(c[ok], g[ok])[0,1]
    print(f"  {ym}: spatial corr={r:+.4f} n={ok.sum():,}")
