#!/usr/bin/env python3
"""
E-A: LEAD/LAG structure — are covariates UPSTREAM (predict eta with a lead)
     or DOWNSTREAM (respond to TWS with a lag)?
     Cross-correlate eta fields with cov fields at leads -3..+3 months.
E-D: Seasonality of eta (month-of-year structure).
If no lead structure: covs are downstream -> the linear Kalman is optimal ->
the ONLY remaining levers are D-tracking and anchor-state quality.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

DATA = '/home/z/my-project/data'
rng = np.random.default_rng(11)
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1
def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
train['cc'] = ckey(train['lat'], train['lon'])
grid = train[['cc', 'lat', 'lon']].drop_duplicates('cc').set_index('cc').sort_index()
mu_c = train.groupby('cc')['TWS_t'].mean()
n = len(grid)
pts = np.column_stack([grid['lat'].values, grid['lon'].values])

# 2-deg smoother
tree = cKDTree(pts)
pairs = tree.query_pairs(r=6.0, output_type='ndarray')
dist = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
w = np.exp(-dist**2 / 2 / 2.0**2)
ii = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
jj = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
ww = np.concatenate([w, w, np.ones(n)])
Sm = coo_matrix((ww / ww[ii].sum(), (ii, jj)), shape=(n, n)).tocsr() if False else coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
rs = np.asarray(Sm.sum(axis=1)).ravel()
Sm = coo_matrix((ww / rs[ii], (ii, jj)), shape=(n, n)).tocsr()

piv = train.pivot_table(index='cc', columns='t_abs', values='TWS_t').reindex(grid.index)
anom = piv.sub(mu_c.reindex(grid.index), axis=0)
t_cols = anom.columns.values
cov_pivs = {c: train.pivot_table(index='cc', columns='t_abs', values=c).reindex(grid.index) for c in COVS}

# detrend
t_norm = t_cols.astype(float)
X = np.vstack([np.ones_like(t_norm), t_norm - t_norm.mean()]).T
A = anom.values
trend = np.zeros((n, 2))
for c in range(n):
    y = A[c]; m = ~np.isnan(y)
    if m.sum() > 24:
        trend[c] = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
y_dt = A - (X @ trend.T).T

PHI = 0.82
# eta field per transition (smoothed — the predictable part)
etas = {}
for i in range(len(t_cols) - 1):
    if t_cols[i + 1] - t_cols[i] == 1:
        y0, y1 = y_dt[:, i], y_dt[:, i + 1]
        m = ~np.isnan(y0) & ~np.isnan(y1)
        e = np.full(n, np.nan)
        e[m] = y1[m] - PHI * y0[m]
        etas[t_cols[i]] = e

# smoothed cov fields per month
cov_sm = {c: {} for c in COVS}
for c in COVS:
    P = cov_pivs[c]
    for t in t_cols:
        cov_sm[c][t] = Sm @ np.nan_to_num(P[t].values) if t in P.columns else None

# ===== E-A: lead/lag cross-correlations =====
print("===== E-A: corr(eta(t), cov-innovation(t+lead)) by cov, lead in months =====")
print(f"{'lead':>5} | " + " | ".join(f"{c[:9]:>9}" for c in COVS) + " |   eta(t)~eta(t+lead) self")
trans = sorted(etas.keys())
for lead in [-3, -2, -1, 0, 1, 2, 3]:
    cors = []
    for c in COVS:
        num, den_a, den_b = 0.0, 0.0, 0.0
        pairs_xy = []
        for t in trans:
            t_cov = t + lead
            if t_cov not in cov_sm[c] or t_cov + 1 not in cov_sm[c]:
                continue
            if (t_cov - 1) not in cov_sm[c]:
                continue
            # cov innovation at month t_cov+1? define innovation index = the y1 month
            t1_cov = t_cov  # we want cov-innov whose y1 month = t + lead
            t0_cov = t1_cov - 1
            if t0_cov not in cov_sm[c] or t1_cov not in cov_sm[c]:
                continue
            e = etas[t]
            cinn = cov_sm[c][t1_cov] - PHI * cov_sm[c][t0_cov]
            mm = ~np.isnan(e)
            if mm.sum() < 8000:
                continue
            pairs_xy.append((e[mm], cinn[mm]))
        if not pairs_xy:
            cors.append(np.nan)
            continue
        # pooled z-corr per transition then average (avoids field-magnitude drift)
        rs_ = []
        for a_, b_ in pairs_xy:
            a_ = (a_ - a_.mean()) / (a_.std() + 1e-12)
            b_ = (b_ - b_.mean()) / (b_.std() + 1e-12)
            rs_.append((a_ * b_).mean())
        cors.append(np.mean(rs_))
    # eta self-lag for reference
    rs_self = []
    for t in trans:
        t2 = t + lead
        if t2 in etas:
            a_, b_ = etas[t], etas[t2]
            mm = ~np.isnan(a_) & ~np.isnan(b_)
            if mm.sum() > 8000:
                a_, b_ = a_[mm], b_[mm]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-12)
                b_ = (b_ - b_.mean()) / (b_.std() + 1e-12)
                rs_self.append((a_ * b_).mean())
    print(f"{lead:>5} | " + " | ".join(f"{r:>9.3f}" if np.isfinite(r) else "     n/a" for r in cors) +
          f" |   {np.mean(rs_self) if rs_self else float('nan'):>6.3f}")

# ===== E-D: seasonality of eta =====
print("\n===== E-D: eta seasonality (per month-of-year eta std & mean) =====")
by_month = {}
for t, e in etas.items():
    m = t % 12 + 1
    v = e[~np.isnan(e)]
    by_month.setdefault(m, []).append((v.std(), v.mean()))
for m in sorted(by_month):
    arr = np.array(by_month[m])
    print(f"  month {m:02d}: eta std {arr[:,0].mean():.3f}  mean {arr[:,1].mean():+.3f}")
