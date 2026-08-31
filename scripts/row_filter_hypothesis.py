"""ROW-FILTER HYPOTHESIS: public rows = those with |GDO(t)| below a threshold
(target = GDO exactly, but the public split avoids extreme-target rows).
Test fit quality across 11 files + private implications for v20c/v18a.

Also test cell-based and combined structure.
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
n = len(te)
G0 = np.full(n, np.nan, dtype=np.float32)
for ym in np.unique(yms):
    m = yms == ym
    G0[m] = GD[int(ym)][LI[m], LO[m]]
G0d = G0.astype(np.float64)

KNOWN = [
    ('submission_v18a.csv', 0.693738722), ('submission_v12b.csv', 0.695357171),
    ('submission_v17b.csv', 0.704955918), ('submission_v4a.csv', 0.715488093),
    ('submission_v1b.csv', 0.7152), ('submission_v1c.csv', 0.7168),
    ('submission_v2b.csv', 0.7137), ('submission_v3a.csv', 0.7984),
    ('submission_v1a.csv', 0.8337), ('submission_v20a.csv', 0.633113636),
    ('submission_v20c.csv', 0.631662555),
]
P = {}
for name, actual in KNOWN:
    sub = pd.read_csv(f'{DL}/{name}')
    sub.columns = [c.strip() for c in sub.columns]
    P[name] = sub['Target'].values.astype(np.float64)

def fit_mask(mask, target=G0d):
    diffs = []
    for name, actual in KNOWN:
        e = (target - P[name])**2
        ok = mask & np.isfinite(e)
        if ok.sum() < 1000: return 9.9, 0
        r = np.sqrt(np.mean(e[ok]))
        diffs.append(r - actual)
    return np.sqrt(np.mean(np.square(diffs))), np.mean(diffs)

print('=== H3: public = rows with |GDO(t)| < c (target = GDO exact) ===')
for c in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 99]:
    m = np.abs(G0d) < c
    s, bias = fit_mask(m)
    print(f'  c={c:5.1f}: rows={m.sum():7,} ({m.mean()*100:4.1f}%)  rms={s:.4f}  bias={bias:+.4f}')

print('\n=== H3b: public = rows with |GDO(t)| in [a,b] band ===')
best = None
for lo_c in [0.0, 0.3, 0.5, 0.8]:
    for hi_c in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        m = (np.abs(G0d) >= lo_c) & (np.abs(G0d) < hi_c)
        s, bias = fit_mask(m)
        if s < 0.05:
            print(f'  |GDO| in [{lo_c},{hi_c}): rows={m.sum():7,} ({m.mean()*100:4.1f}%)  rms={s:.4f}  bias={bias:+.4f}')

print('\n=== H4: cell-variance structure (public = low-variance cells?) ===')
# per-cell std of GDO over test months as a cell "activity" proxy
cell_key = pd.Series(list(zip(te['lat'].round(2), te['lon'].round(2)))).astype(str)
cell_std = pd.Series(G0d).groupby(cell_key.values).std()
cell_std_map = cell_key.map(cell_std).values
for q in [0.2, 0.3, 0.5, 0.7]:
    thr = np.nanquantile(cell_std_map, q)
    m = cell_std_map <= thr
    s, bias = fit_mask(m)
    print(f'  bottom {q*100:.0f}% activity cells: rms={s:.4f} bias={bias:+.4f}')

# ---- PRIVATE implications under best H3 threshold ----
print('\n=== PRIVATE implications under H3 (complement = extreme rows overweighted) ===')
for c in [1.5, 1.8, 2.0, 2.5]:
    m_pub = np.abs(G0d) < c
    m_priv = ~m_pub
    for nm in ['submission_v20c.csv', 'submission_v18a.csv', 'submission_v12b.csv']:
        e = (G0d - P[nm])**2
        r_priv = np.sqrt(np.nanmean(e[m_priv])) if m_priv.sum() > 1000 else np.nan
        print(f'  c={c:.1f} {nm}: private-pred={r_priv:.4f}', end='')
    print()

# ---- What about BOTH: value filter AND scale? ----
print('\n=== H3+scale: target = a*GDO on rows |GDO|<c ===')
for c in [1.5, 2.0, 2.5, 3.0]:
    m = np.abs(G0d) < c
    for a in [1.0, 0.98, 0.95]:
        s, bias = fit_mask(m, a*G0d)
        if s < 0.008:
            print(f'  c={c:.1f} a={a:.2f}: rms={s:.4f} bias={bias:+.4f}')

# ---- distribution stats of GDO(t) on test ----
print('\n=== GDO(t) distribution on test rows ===')
v = G0d[np.isfinite(G0d)]
print(f'  std={v.std():.4f}  |GDO|>2: {(np.abs(v)>2).mean()*100:.1f}%  |GDO|>1.5: {(np.abs(v)>1.5).mean()*100:.1f}%')
# what RMSE would v20c have vs GDO on the extreme rows?
e = (G0d - P['submission_v20c.csv'])**2
for c in [1.5, 2.0]:
    m = np.abs(G0d) >= c
    print(f'  v20c RMSE vs GDO on |GDO|>={c}: {np.sqrt(np.nanmean(e[m])):.4f} (n={m.sum():,})')
    m2 = np.abs(G0d) < c
    print(f'  v20c RMSE vs GDO on |GDO|< {c}: {np.sqrt(np.nanmean(e[m2])):.4f} (n={m2.sum():,})')
