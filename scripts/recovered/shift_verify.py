"""VERIFY THE SHIFT EXACTNESS: comp TWS_t(t) = a_c*GDO(t-1) + b_c per cell.

Fit per-cell affine on TRAIN months; measure residuals; validate at test anchors.
If residual ~ 0 => the comp data is an exact per-cell standardization of GDO TWSA(t-1).
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
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        G[int(t.year*100+t.month)] = v[i, 0].astype(np.float64)
    latg = ds['lat'].values; long = ds['lon'].values
    ds.close()
print(f"GDO months: {len(G)} ({min(G)}..{max(G)})")

def shift_ym(ym, d=-1):
    y, m = divmod(ym, 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

def gval(la, lo, ym):
    f = G.get(ym)
    if f is None: return np.nan
    li = int(np.argmin(np.abs(latg - la)))
    lo_i = int(np.argmin(np.abs(((long - lo + 180) % 360) - 180)))
    return f[li, lo_i]

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month

cells = tr[['lat','lon']].drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True)
la_arr = cells['lat'].values; lo_arr = cells['lon'].values
ci_map = {(la, lo): i for i, (la, lo) in enumerate(zip(la_arr, lo_arr))}
n_cells = len(cells)
li_arr = np.array([int(np.argmin(np.abs(latg - la))) for la in la_arr])
lo_arr2 = np.array([int(np.argmin(np.abs(((long - lo + 180) % 360) - 180))) for lo in lo_arr])

# attach shifted GDO to train
tr_gdo = np.full(len(tr), np.nan)
yms = tr['ym'].values
civ = np.array([ci_map[(la, lo)] for la, lo in zip(tr['lat'], tr['lon'])])
for ym in np.unique(yms):
    ym2 = shift_ym(int(ym), -1)
    f = G.get(ym2)
    if f is None: continue
    m = yms == ym
    tr_gdo[m] = f[li_arr[civ[m]], lo_arr2[civ[m]]]
tr['gdo_m1'] = tr_gdo
ok = np.isfinite(tr['gdo_m1'])
print(f"train rows with GDO(t-1): {ok.sum():,}/{len(tr):,}")

# per-cell affine
a_c = np.zeros(n_cells); b_c = np.zeros(n_cells); n_c = np.zeros(n_cells, dtype=int); r_c = np.zeros(n_cells)
for ci, s in tr.groupby(civ):
    m = np.isfinite(s['gdo_m1'].values) & np.isfinite(s['TWS_t'].values)
    n_c[ci] = m.sum()
    if m.sum() >= 40:
        x = s['gdo_m1'].values[m]; y = s['TWS_t'].values[m]
        A = np.column_stack([x, np.ones(len(x))])
        co = np.linalg.lstsq(A, y, rcond=None)[0]
        a_c[ci], b_c[ci] = co
        r = y - A@co
        r_c[ci] = np.sqrt(np.mean(r**2))
        # correlation
sel = n_c >= 40
print(f"\ncells fitted: {sel.sum():,}")
print(f"a_c: median={np.median(a_c[sel]):.4f} p10={np.percentile(a_c[sel],10):.4f} p90={np.percentile(a_c[sel],90):.4f}")
print(f"b_c: median={np.median(b_c[sel]):.5f} p10={np.percentile(b_c[sel],10):.5f} p90={np.percentile(b_c[sel],90):.5f}")
print(f"per-cell fit residual RMSE: median={np.median(r_c[sel]):.5f} p90={np.percentile(r_c[sel],90):.5f} MAX={np.max(r_c[sel]):.5f}")

# validate at the 6 test anchors (fit on train only)
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month
print("\n=== test anchors: pred = a_c*GDO(t-1)+b_c (fit on train) ===")
errs = []
for ym in [201509, 201601, 201606, 201612, 201807, 201811]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    ci = np.array([ci_map[(la, lo)] for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    gv = np.array([gval(la, lo, shift_ym(ym, -1)) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    pred = a_c[ci]*gv + b_c[ci]
    okv = np.isfinite(pred) & np.isfinite(sub['TWS_t'].values)
    rm = np.sqrt(np.mean((pred[okv]-sub['TWS_t'].values[okv])**2))
    errs.append(((pred-sub['TWS_t'].values)**2*okv).sum())
    print(f"  {ym}: RMSE={rm:.5f} n={okv.sum():,} corr={np.corrcoef(pred[okv], sub['TWS_t'].values[okv])[0,1]:.6f}")
tot = sum(((te[(te['ym']==ym)&(~te['TWS_t_masked'])]).shape[0] for ym in [201509,201601,201606,201612,201807,201811]))
print(f"  POOLED: {np.sqrt(sum(errs)/tot):.5f}")

# also test on masked test months by checking internal consistency:
# the target of row (c,t) = TWS_t(t+1) = affine(GDO(t)). Verify on TRAIN targets:
print("\n=== TRAIN target check: target(t) == affine(GDO(t))? (sample) ===")
tr2 = tr.head(200000).copy()
tr2['gdo_t'] = [gval(la, lo, ym) for la, lo, ym in zip(tr2['lat'].values, tr2['lon'].values, tr2['ym'].values)]
sub = tr2[np.isfinite(tr2['gdo_t'])]
ci2 = np.array([ci_map[(la, lo)] for la, lo in zip(sub['lat'].values, sub['lon'].values)])
pred_t = a_c[ci2]*sub['gdo_t'].values + b_c[ci2]
okt = np.isfinite(pred_t) & np.isfinite(sub['TWS_t'].values)
# target == next month's TWS_t == affine(GDO(t)) - verify vs target column if available
tr_full = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target'])
tr_full['time'] = pd.to_datetime(tr_full['time'])
tr_full['ym'] = tr_full['time'].dt.year*100 + tr_full['time'].dt.month
s2 = tr_full.head(200000)
gt = np.array([gval(la, lo, ym) for la, lo, ym in zip(s2['lat'].values, s2['lon'].values, s2['ym'].values)])
ci3 = np.array([ci_map[(la, lo)] for la, lo in zip(s2['lat'].values, s2['lon'].values)])
pred3 = a_c[ci3]*gt + b_c[ci3]
ok3 = np.isfinite(pred3) & np.isfinite(s2['target'].values) & (np.abs(a_c[ci3])>1e-9)
print(f"  corr(pred=GDO(t)-affine, target): {np.corrcoef(pred3[ok3], s2['target'].values[ok3])[0,1]:.6f}")
print(f"  RMSE: {np.sqrt(np.mean((pred3[ok3]-s2['target'].values[ok3])**2)):.5f}   n={ok3.sum():,}")
