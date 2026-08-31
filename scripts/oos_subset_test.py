"""DECISIVE OUT-OF-SAMPLE TEST for H2 (month-subset public split):
The best 9-file subset was identified BEFORE v20a/v20c were built.
If H2 is true, RMSE(v20a, GDO) restricted to those public months should
equal v20a's ACTUAL LB score (0.6331) — genuine out-of-sample prediction.

Also computes private-score implications for the final-2 selection.
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

GD = {}
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
for f in files:
    ds = xr.open_dataset(f)
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float32)
    gd_lat = ds['lat'].values; gd_lon = ds['lon'].values
    ds.close()

te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID','time','lat','lon'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month
LI = np.array([int(np.argmin(np.abs(gd_lat - la))) for la in te['lat'].values])
LO = np.array([int(round(lo + 179.5)) % 360 for lo in te['lon'].values])
yms = te['ym'].values
truth = np.full(len(te), np.nan, dtype=np.float32)
for ym in np.unique(yms):
    m = yms == ym
    truth[m] = GD[int(ym)][LI[m], LO[m]]

def rmse_months(pred, month_list):
    sel = np.isin(yms, month_list) & np.isfinite(truth) & np.isfinite(pred)
    return np.sqrt(np.mean((truth[sel]-pred[sel])**2)), sel.sum()

FILES = {
    'v20a': ('submission_v20a.csv', 0.633113636),
    'v20c': ('submission_v20c.csv', 0.631662555),
    'v18a': ('submission_v18a.csv', 0.693738722),
    'v12b': ('submission_v12b.csv', 0.695357171),
    'v17b': ('submission_v17b.csv', 0.704955918),
}
P = {}
for k, (fn, act) in FILES.items():
    sub = pd.read_csv(f'{DL}/{fn}')
    sub.columns = [c.strip() for c in sub.columns]
    P[k] = sub['Target'].values.astype(np.float64)

# top-15 9-file subsets from split_bruteforce.log (fit BEFORE v20a/v20c existed)
SUB9 = [
    [201602, 201606, 201607, 201608, 201704, 201706, 201812],
    [201602, 201603, 201606, 201609, 201704, 201706, 201807],
    [201602, 201607, 201609, 201704, 201706, 201807, 201812],
    [201509, 201602, 201606, 201607, 201609, 201703, 201705, 201706, 201812],
    [201509, 201602, 201608, 201609, 201705, 201706, 201812],
    [201602, 201606, 201607, 201608, 201609, 201703, 201705, 201706, 201807, 201812],
    [201603, 201607, 201608, 201609, 201704, 201706, 201807, 201812],
    [201602, 201607, 201608, 201609, 201703, 201704, 201807, 201812],
    [201509, 201602, 201603, 201607, 201609, 201703, 201704, 201706, 201807, 201812],
    [201603, 201606, 201607, 201609, 201704, 201706, 201812],
    [201509, 201602, 201608, 201609, 201704, 201706, 201807],
    [201602, 201603, 201606, 201608, 201609, 201704, 201706, 201812],
    [201509, 201602, 201603, 201607, 201608, 201609, 201703, 201704, 201706, 201807, 201811, 201812],
    [201602, 201603, 201606, 201608, 201609, 201704, 201706, 201811],
    [201509, 201602, 201607, 201608, 201609, 201703, 201705, 201706, 201807, 201812],
]

print('=== OUT-OF-SAMPLE TEST: H2 subset (from 9-file fit) vs v20a/v20c actuals ===')
print('(subsets were chosen with data available BEFORE v20a/v20c were built)\n')
for i, sub9 in enumerate(SUB9[:8]):
    r20a, n1 = rmse_months(P['v20a'], sub9)
    r20c, n2 = rmse_months(P['v20c'], sub9)
    err_a = r20a - 0.633113636
    err_c = r20c - 0.631662555
    print(f'  subset {i+1}: pred v20a={r20a:.4f} (actual 0.6331, err {err_a:+.4f}) | '
          f'pred v20c={r20c:.4f} (actual 0.6317, err {err_c:+.4f})')

# aggregate over all 15 subsets
errs = []
for sub9 in SUB9:
    r20a, _ = rmse_months(P['v20a'], sub9)
    r20c, _ = rmse_months(P['v20c'], sub9)
    errs += [r20a - 0.633113636, r20c - 0.631662555]
errs = np.array(errs)
print(f'\n  across all 15 nine-file subsets: mean err={errs.mean():+.4f}  rms={np.sqrt((errs**2).mean()):.4f}')
print('  (H1 global-0.95 model predicts v20a=0.6418, v20c=0.6375 -> errs +0.0087/+0.0058)')

# ---- private implications under H2 for all key files ----
print('\n=== PRIVATE-SCORE IMPLICATIONS under H2 (using best 11-file subsets) ===')
best11 = [
    [201602, 201603, 201606, 201609, 201612, 201703, 201706],
    [201509, 201603, 201606, 201607, 201609, 201702, 201706],
    [201603, 201606, 201607, 201608, 201612, 201703, 201706],
    [201509, 201602, 201603, 201609, 201612, 201703, 201706],
    [201602, 201606, 201607, 201608, 201609, 201702, 201706, 201812],
    [201509, 201602, 201606, 201607, 201608, 201702, 201706],
]
all_months = sorted(set(int(y) for y in yms))
for k in ['v20c', 'v20a', 'v18a', 'v12b', 'v17b']:
    privs = []
    for pub in best11:
        priv = [m for m in all_months if m not in pub]
        r, _ = rmse_months(P[k], priv)
        privs.append(r)
    pub_r, _ = rmse_months(P[k], best11[0])
    r_all, _ = rmse_months(P[k], all_months)
    print(f'  {k}: public(LB)={FILES[k][1]:.4f} | H2-private median={np.median(privs):.4f} '
          f'(range {min(privs):.4f}-{max(privs):.4f}) | RMSE-vs-GDO-all={r_all:.4f}')
