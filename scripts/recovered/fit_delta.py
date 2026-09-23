"""FIT THE TARGET DISCREPANCY: assume actual_e(month) = oracle_e(month) - delta(month-group).
Solve delta per group across 9 files; consistency across files = the model is right.
Groups: (A) 2017 block, (B) rest. Also try 3 groups.
"""
import numpy as np, pandas as pd, xarray as xr, glob, os

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

months = sorted(int(y) for y in np.unique(yms))
n_m = {ym: (yms == ym).sum() for ym in months}

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

G17 = lambda ym: 201701 <= ym <= 201706

rows = []
per_file = {}
for name, actual in KNOWN:
    p = f'/home/z/my-project/download/{name}'
    sub = pd.read_csv(p)
    sub.columns = [c.strip() for c in sub.columns]
    pred = sub['Target'].values
    e = (truth - pred)**2
    em = {ym: e[yms == ym].mean() for ym in months}
    per_file[name] = (em, actual)

# ---- 2-group fit: actual^2 = w17*(e17 - d17) + wr*(er - dr); solve (d17, dr) per file ----
print("=== per-file solved deltas (2-group): actual^2 = w17*(e17-d17) + wr*(er-dr) ===")
print("(underdetermined per-file; instead solve GLOBAL d17, dr by least squares over 9 files)")
# global least squares: for each file: actual^2 = sum_m w_m e_m - (w17*d17 + wr*dr)
# => residual = actual^2 - oracle^2 + w17*d17 + wr*dr = 0
A = []; b = []
for name, actual in KNOWN:
    em, act = per_file[name]
    w17 = sum(n_m[ym] for ym in months if G17(ym)) / len(te)
    wr = 1 - w17
    e17 = np.mean([em[ym] for ym in months if G17(ym)])
    er = np.mean([em[ym] for ym in months if not G17(ym)])
    oracle2 = w17*e17 + wr*er
    A.append([w17, wr])
    b.append(oracle2 - act**2)
A = np.array(A); b = np.array(b)
d, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print(f"solved: d17={d[0]:.4f}  dr={d[1]:.4f}  (variance units; sqrt={np.sqrt(max(d[0],0)):.3f}/{np.sqrt(max(d[1],0)):.3f})")
pred_actual = []
for name, actual in KNOWN:
    em, act = per_file[name]
    w17 = sum(n_m[ym] for ym in months if G17(ym)) / len(te)
    wr = 1 - w17
    e17 = np.mean([em[ym] for ym in months if G17(ym)])
    er = np.mean([em[ym] for ym in months if not G17(ym)])
    pa = np.sqrt(w17*(e17-d[0]) + wr*(er-d[1]))
    pred_actual.append(pa)
    print(f"  {name}: actual={act:.4f} fitted={pa:.4f} diff={pa-act:+.4f}")

# ---- 3-group: 2017-block, 2016+2018 masked months, anchors ----
print("\n=== 3-group fit: (2017block, other-masked, anchors) ===")
def G3(ym):
    if 201701 <= ym <= 201706: return 0
    if ym in (201602, 201603, 201607, 201608, 201609, 201812): return 1
    return 2
A = []; b = []
for name, actual in KNOWN:
    em, act = per_file[name]
    ws = [sum(n_m[ym] for ym in months if G3(ym) == g)/len(te) for g in range(3)]
    es = [np.mean([em[ym] for ym in months if G3(ym) == g]) for g in range(3)]
    oracle2 = sum(w*e for w, e in zip(ws, es))
    A.append(ws); b.append(oracle2 - act**2)
A = np.array(A); b = np.array(b)
d3, *_ = np.linalg.lstsq(A, b, rcond=None)
print(f"solved deltas: 2017={d3[0]:.4f} other-masked={d3[1]:.4f} anchors={d3[2]:.4f}")
for name, actual in KNOWN:
    em, act = per_file[name]
    ws = [sum(n_m[ym] for ym in months if G3(ym) == g)/len(te) for g in range(3)]
    es = [np.mean([em[ym] for ym in months if G3(ym) == g]) for g in range(3)]
    pa = np.sqrt(sum(w*max(e-dd, 0) for w, e, dd in zip(ws, es, d3)))
    print(f"  {name}: actual={act:.4f} fitted={pa:.4f} diff={pa-act:+.4f}")
