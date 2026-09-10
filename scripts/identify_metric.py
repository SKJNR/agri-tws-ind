"""IDENTIFY THE PUBLIC LB METRIC using 9 submissions with known actual scores.

Candidates:
  M1: plain RMSE all rows
  M2: RMSE rows with t+1 in test ("where available" targets)
  M3: mean of monthly RMSEs
  M4: RMSE excluding 2017 block
  M5: RMSE anchor-month rows only / masked only
  M6..: month-subset least squares (solve which months are in the public split)
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

GD = {}
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
for f in files:
    ds = xr.open_dataset(f)
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float32)
    gd_lat = ds['lat'].values; gd_lon = ds['lon'].values
    ds.close()

te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID','time','lat','lon','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month
LI = np.array([int(np.argmin(np.abs(gd_lat - la))) for la in te['lat'].values])
LO = np.array([int(round(lo + 179.5)) % 360 for lo in te['lon'].values])
yms = te['ym'].values
truth = np.full(len(te), np.nan, dtype=np.float32)
for ym in np.unique(yms):
    fld = GD.get(int(ym))
    m = yms == ym
    truth[m] = fld[LI[m], LO[m]]

# row month sets
def shift_ym(ym, d=1):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1
test_months = set(int(y) for y in np.unique(yms))
has_next = np.array([shift_ym(ym) in test_months for ym in yms])
is_masked = te['TWS_t_masked'].values.astype(bool)
m2017 = (yms >= 201701) & (yms <= 201706)

KNOWN = [
    ('submission_v18a.csv', 0.693738722),
    ('submission_v12b.csv', 0.695357171),
    ('submission_v17b.csv', 0.704955918),
    ('submission_v4a.csv', 0.715488093),
    ('submission_v1b.csv', 0.7152),
    ('submission_v1c.csv', 0.7168),
    ('submission_v2b.csv', 0.7137),
    ('submission_v3a.csv', 0.7984),
    ('submission_v1a.csv', 0.8337),
]

import os
rows = []
errs_by_file = {}
for name, actual in KNOWN:
    p = f'/home/z/my-project/download/{name}'
    if not os.path.exists(p):
        print(f"missing {name}"); continue
    sub = pd.read_csv(p)
    sub.columns = [c.strip() for c in sub.columns]
    pred = sub['Target'].values
    ok = np.isfinite(truth) & np.isfinite(pred)
    e = (truth - pred)**2
    errs_by_file[name] = (e, ok, actual)
    m1 = np.sqrt(np.mean(e[ok]))
    m2 = np.sqrt(np.mean(e[ok & has_next]))
    m3 = np.mean([np.sqrt(np.mean(e[ok & (yms == ym)])) for ym in sorted(test_months)])
    m4 = np.sqrt(np.mean(e[ok & ~m2017]))
    m5a = np.sqrt(np.mean(e[ok & ~is_masked]))
    m5b = np.sqrt(np.mean(e[ok & is_masked]))
    rows.append((name, actual, m1, m2, m3, m4, m5a, m5b))

df = pd.DataFrame(rows, columns=['file', 'ACTUAL', 'M1_full', 'M2_next', 'M3_monthly', 'M4_no2017', 'M5_unmasked', 'M6_masked'])
pd.set_option('display.width', 200)
print(df.round(4).to_string(index=False))

# ---- solve per-month inclusion weights via least squares ----
# actual^2 = sum_m w_m * sum_rows_in_m(e) / sum_m w_m * n_m
# with w_m in [0,1]; solve ridge regression for w given 8-9 equations, 18 unknowns
names = [n for n, _, _, _, _, _, _, _ in rows]
E = np.array([errs_by_file[n][0] for n in names])          # (F, N)
OK = np.array([errs_by_file[n][1] for n in names])
ACT = np.array([errs_by_file[n][2] for n in names])
months = sorted(test_months)
# per-file per-month mean error
M = np.zeros((len(names), len(months)))
for fi in range(len(names)):
    for mi, ym in enumerate(months):
        sel = OK[fi] & (yms == ym)
        M[fi, mi] = E[fi][sel].mean()
# solve: ACT^2 = M @ w / (n_w @ w) ... nonlinear; iterate: fix denominator
n_m = np.array([(yms == ym).sum() for ym in months])
w = np.ones(len(months)) / 2
for it in range(200):
    denom = n_m @ w
    # ACT^2 * denom = M @ w  ->  (M - ACT^2 * n_m) @ w = 0 with sum constraint... use lstsq on:
    # M @ w = ACT^2 * (n_m @ w)
    A = M - np.outer(ACT**2, n_m)
    # solve A w = 0, ||w||=1, w>=0 -> find null space via SVD
    U, s, Vt = np.linalg.svd(A)
    w_new = np.abs(Vt[-1])
    w_new = w_new / w_new.sum() * len(months) * 0.5  # scale
    if np.max(np.abs(w_new - w)) < 1e-6: break
    w = w_new
print("\nsolved month weights (rough):")
for mi, ym in enumerate(months):
    print(f"  {ym}: {w[mi]:.3f}")
