"""Quantify GDO->comp mapping quality + residual structure.

Q1: per-cell affine (fit on train <=2012 / full train) -> RMSE at 6 test anchors.
Q2: residual structure: spatial smoothness, temporal ACF (version noise vs white noise).
Q3: fit including anchors themselves (in-sample) -> the theoretical best GDO-only RMSE.
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
G = {}
for f in files:
    ds = xr.open_dataset(f)
    v = ds['twsan'].values
    tvals = pd.to_datetime(ds['time'].values)
    for i, t in enumerate(tvals):
        G[t.year*100+t.month] = v[i, 0].astype(np.float64)
    ds.close()
latg = None
ds = xr.open_dataset(files[0]); latg = ds['lat'].values; long = ds['lon'].values; ds.close()
print(f"GDO months: {len(G)}")

def gval(la, lo, ym):
    f = G.get(ym)
    if f is None: return np.nan
    li = int(np.argmin(np.abs(latg - la)))
    lo_i = int(np.argmin(np.abs(((long - lo + 180) % 360) - 180)))
    return f[li, lo_i]

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

# attach gdo to train
print("attaching GDO values to train (vectorized)...")
cells = tr[['lat','lon']].drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True)
la_arr = cells['lat'].values; lo_arr = cells['lon'].values
ci_map = {(la, lo): i for i, (la, lo) in enumerate(zip(la_arr, lo_arr))}
tr['ci'] = [ci_map[(la, lo)] for la, lo in zip(tr['lat'], tr['lon'])]
li_arr = np.array([int(np.argmin(np.abs(latg - la))) for la in la_arr])
lo_arr2 = np.array([int(np.argmin(np.abs(((long - lo + 180) % 360) - 180))) for lo in lo_arr])
tr_gdo = np.full(len(tr), np.nan)
for ym in tr['ym'].unique():
    f = G.get(ym)
    if f is None: continue
    m = tr['ym'].values == ym
    tr_gdo[m] = f[li_arr[tr['ci'].values[m]], lo_arr2[tr['ci'].values[m]]]
tr['gdo'] = tr_gdo

# per-cell affine fit on train (two windows)
def fit_affine(sub, min_n=40):
    a = np.zeros(len(cells)); b = np.zeros(len(cells)); n = np.zeros(len(cells), dtype=int)
    for ci, s in sub.groupby('ci'):
        ok = np.isfinite(s['gdo'].values) & np.isfinite(s['TWS_t'].values)
        n[ci] = ok.sum()
        if ok.sum() >= min_n:
            A = np.column_stack([s['gdo'].values[ok], np.ones(ok.sum())])
            try:
                co = np.linalg.lstsq(A, s['TWS_t'].values[ok], rcond=None)[0]
                a[ci], b[ci] = co
            except Exception: pass
    return a, b, n

print("fitting per-cell affine (train<=2012)...")
a1, b1, n1 = fit_affine(tr[tr['time'].dt.year <= 2012])
print("fitting per-cell affine (full train)...")
a2, b2, n2 = fit_affine(tr)

# Q1: predict the 6 test anchors
print("\n=== Q1: per-cell affine -> test anchors (OUT-OF-SAMPLE) ===")
anchors = [201509, 201601, 201606, 201612, 201807, 201811]
for tag, (a_, b_) in [("fit<=2012", (a1, b1)), ("fit-full-train", (a2, b2))]:
    errs = []
    for ym in anchors:
        sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
        ci = np.array([ci_map[(la, lo)] for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        pred = a_[ci]*gv + b_[ci]
        ok = np.isfinite(pred) & np.isfinite(sub['TWS_t'].values) & (np.abs(a_[ci]) > 1e-9)
        errs.append(((pred - sub['TWS_t'].values)**2 * ok).sum())
        n_ok = ok.sum()
        rm = np.sqrt(errs[-1]/n_ok)
        print(f"  [{tag}] {ym}: RMSE={rm:.4f} n={n_ok:,}")
    tot_n = sum(((te[(te['ym']==ym)&(~te['TWS_t_masked'])]).shape[0] for ym in anchors))
    print(f"  [{tag}] POOLED anchor RMSE: {np.sqrt(sum(errs)/tot_n):.4f}")

# Q3: in-sample fit including anchors (theoretical best single-month GDO estimate)
print("\n=== Q3: in-sample (fit on anchors themselves) ===")
for ym in anchors:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    c = sub['TWS_t'].values
    ok = np.isfinite(gv)
    A = np.column_stack([gv[ok], np.ones(ok.sum())])
    co = np.linalg.lstsq(A, c[ok], rcond=None)[0]
    rm = np.sqrt(np.mean((A@co - c[ok])**2))
    print(f"  {ym}: pooled-affine RMSE={rm:.4f}")

# Q2: residual structure per cell (fit full train)
print("\n=== Q2: residual structure (full-train affine) ===")
res_by_cell = {}
for ci, s in tr.groupby('ci'):
    ok = np.isfinite(s['gdo'].values) & np.isfinite(s['TWS_t'].values)
    if ok.sum() < 40: continue
    res = s['TWS_t'].values[ok] - (a2[ci]*s['gdo'].values[ok] + b2[ci])
    res_by_cell[ci] = (s['ym'].values[ok], res)
# temporal lag-1 ACF of residuals
acfs = []
for ci, (yms_, res) in list(res_by_cell.items())[::5]:
    o = np.argsort(yms_)
    r = res[o]
    if len(r) > 30:
        acfs.append(np.corrcoef(r[:-1], r[1:])[0,1])
print(f"residual lag-1 temporal ACF: median={np.median(acfs):.3f}")
# residual std
rstd = [np.std(r) for _, r in res_by_cell.values()]
print(f"residual std: median={np.median(rstd):.3f}")
# spatial structure of residual at a month
sub = tr[tr['ym'] == 201306]
f = np.full(len(cells), np.nan)
civ = sub['ci'].values
f[civ] = sub['TWS_t'].values - (a2[civ]*sub['gdo'].values + b2[civ])
ok = np.isfinite(f)
print(f"residual field (201306): std={np.nanstd(f):.3f}")
# neighbor corr of residual field
from scipy.spatial import cKDTree
la_r = np.deg2rad(la_arr); lo_r = np.deg2rad(lo_arr)
pts = np.column_stack([np.cos(la_r)*np.cos(lo_r), np.cos(la_r)*np.sin(lo_r), np.sin(la_r)])
tree = cKDTree(pts)
pairs = tree.query_pairs(np.deg2rad(1.6))
i_, j_ = np.array(list(pairs)).T
m = ok[i_] & ok[j_]
print(f"residual neighbor corr (1.6deg): {np.corrcoef(f[i_[m]], f[j_[m]])[0,1]:.3f}  (=0 white, ~1 smooth)")
