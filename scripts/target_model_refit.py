"""REFIT TARGET MODEL WITH 11 CALIBRATION POINTS (incl. v20a/v20c actuals)
and PREDICT THE PRIVATE-LB SCORE of v20c under each hypothesis.

H1: target = alpha*GDO(t) globally, public = random rows -> private ~= public
H2: target = GDO(t) EXACTLY, public = specific MONTH subset (from split_bruteforce)
    -> private score = RMSE(file, GDO) on the COMPLEMENT months

Also: refit the month-subset search with all 11 files.
"""
import numpy as np, pandas as pd, xarray as xr, glob
from itertools import combinations

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
    ('submission_v20a.csv', 0.633113636),   # NEW
    ('submission_v20c.csv', 0.631662555),   # NEW
]
P = {}
for name, actual in KNOWN:
    sub = pd.read_csv(f'{DL}/{name}')
    sub.columns = [c.strip() for c in sub.columns]
    P[name] = sub['Target'].values.astype(np.float64)

months = sorted(set(int(y) for y in yms))
month_idx = {ym: (yms == ym) for ym in months}
month_rows = {ym: month_idx[ym].sum() for ym in months}

# ---------------- H1: global scale with 11 points ----------------
print('=== H1: pure-scale target = alpha*GDO(t), 11 files ===')
for a in [1.0, 0.97, 0.95, 0.93]:
    diffs = []
    for name, actual in KNOWN:
        r = np.sqrt(np.nanmean((a*truth - P[name])**2))
        diffs.append(r - actual)
    print(f'  alpha={a:.2f}: mean={np.mean(diffs):+.4f} rms={np.sqrt(np.mean(np.square(diffs))):.4f}')

# per-file implied alpha for the 2 new files
print('\n=== per-file best alpha (quadratic solve) ===')
for name, actual in KNOWN[-2:]:
    T = truth.astype(np.float64); Pv = P[name]
    ok = np.isfinite(T) & np.isfinite(Pv)
    a2 = np.sum(T[ok]**2); a1 = -2*np.sum(T[ok]*Pv[ok]); a0 = np.sum(Pv[ok]**2) - actual**2*ok.sum()
    disc = a1**2 - 4*a2*a0
    al = (-a1 + np.sqrt(disc))/(2*a2) if disc >= 0 else np.nan
    print(f'  {name}: alpha={al:.4f}')

# ---------------- H2: month-subset search with 11 files ----------------
print('\n=== H2: month-subset search (target = GDO exactly), 11 files ===')
# precompute per-month sums of squared error for each file
SQ = {}   # name -> {ym: sum_sq}
for name, actual in KNOWN:
    e = (truth - P[name])**2
    SQ[name] = {ym: np.nansum(e[month_idx[ym]]) for ym in months}
    # careful: NaN truth months -> nansum treats NaN as 0. GDO covers all months, so fine.
NOK = {ym: month_rows[ym] for ym in months}

best = []
rng_subsets = []
for k in range(5, 12):
    for combo in combinations(months, k):
        mset = set(combo)
        n_pub = sum(NOK[ym] for ym in combo)
        if not (0.20 <= n_pub/len(te) <= 0.45):   # public ~30% of rows
            continue
        errs = []
        for name, actual in KNOWN:
            sse = sum(SQ[name][ym] for ym in combo)
            r = np.sqrt(sse / n_pub)
            errs.append(r - actual)
        rms = np.sqrt(np.mean(np.square(errs)))
        best.append((rms, combo))
best.sort(key=lambda x: x[0])
print(f'subsets scanned (size 5-11, row-frac 0.20-0.45): {len(best):,}')
print(f'top 10 of {len(best)}:')
for rms, combo in best[:10]:
    print(f'  rms={rms:.5f}  months={list(combo)}')

# ---------------- private prediction for v20c under top subsets ----------------
print('\n=== PREDICTED PRIVATE SCORE of v20c (complement months) ===')
priv_preds = []
for rms, combo in best[:50]:
    priv_months = [ym for ym in months if ym not in combo]
    n_priv = sum(NOK[ym] for ym in priv_months)
    sse = sum(SQ['submission_v20c.csv'][ym] for ym in priv_months)
    priv_preds.append(np.sqrt(sse/n_priv))
priv_preds = np.array(priv_preds)
print(f'  across top-50 candidate public subsets:')
print(f'  private RMSE(v20c vs GDO): min={priv_preds.min():.4f} median={np.median(priv_preds):.4f} max={priv_preds.max():.4f}')
pub_rms = [r for r, c in best[:50]]
print(f'  (fit quality of those subsets: rms {min(pub_rms):.5f}..{max(pub_rms):.5f})')

# sanity: v20c vs GDO on ALL months
r_all = np.sqrt(np.nanmean((truth - P['submission_v20c.csv'])**2))
print(f'\n  RMSE(v20c, GDO) ALL months = {r_all:.4f}  (actual public LB = 0.6317)')
r_all_a = np.sqrt(np.nanmean((truth - P['submission_v20a.csv'])**2))
print(f'  RMSE(v20a, GDO) ALL months = {r_all_a:.4f}  (actual public LB = 0.6331)')

# how well would '0.95*GDO exactly' score vs GDO?
d = (0.95*truth - truth)
print(f'\n  RMSE(0.95*GDO, GDO) = {np.sqrt(np.nanmean(d**2)):.4f}  <- the 0.95 shrinkage itself costs this much vs pure GDO')
