#!/usr/bin/env python3
"""
THE FLOOR INVESTIGATION — where is the true irreducible error?

Adjacent-diff says spatially-white noise is only ~0.10-0.15, but our error
budget assumes 0.456 "target noise". The 0.456 came from POOLED ACF factoring
which cannot distinguish:
  (a) per-cell white observation noise  -> truly irreducible
  (b) spatially-smooth temporally-white innovations -> the fast process itself
      (partially trackable via covariates + spatial assimilation!)

FOUR DECISIVE MEASUREMENTS on train (full targets available):
  F1. Spatial LOO interpolation of TWS fields -> direct white-noise floor
  F2. Per-cell AR(1)+noise moments (rho1,rho2 -> phi, lambda) -> clean noise est
  F3. Innovation field spatial structure (white or smooth?)
  F4. Cov-visible fraction of the innovation (how much of eta do covs see?)
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
rng = np.random.default_rng(7)

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1
def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
train['cc'] = ckey(train['lat'], train['lon'])

grid = train[['cc', 'lat', 'lon']].drop_duplicates('cc').set_index('cc').sort_index()
mu_c = train.groupby('cc')['TWS_t'].mean()

# adjacency (rook)
from scipy.spatial import cKDTree
pts = grid[['lat', 'lon']].values
tree = cKDTree(pts)
pairs = tree.query_pairs(r=1.5, output_type='ndarray')
d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
adj = pairs[d < 1.05]
# neighbor lists for LOO interpolation
nbr = [[] for _ in range(len(grid))]
for i, j in adj:
    nbr[i].append(int(j)); nbr[j].append(int(i))
nbr = [np.array(x, dtype=int) if x else np.array([], dtype=int) for x in nbr]

# pivot: cells x months
piv = train.pivot_table(index='cc', columns='t_abs', values='TWS_t')
piv = piv.reindex(grid.index)
anom = piv.sub(mu_c.reindex(grid.index), axis=0)
t_cols = anom.columns.values
print(f"pivot: {anom.shape[0]:,} cells x {anom.shape[1]} months; "
      f"missing frac {anom.isna().mean().mean():.3f}")

# ================= F1: spatial LOO interpolation floor =================
print("\n===== F1: predict TWS(c,t) from same-month spatial neighbors =====")
sample_months = rng.choice(len(t_cols), size=30, replace=False)
errs_loo, errs_1nbr = [], []
for mi in sample_months:
    v = anom.iloc[:, mi].values
    ok = ~np.isnan(v)
    pred = np.full(len(v), np.nan)
    for c in np.where(ok)[0]:
        nn = nbr[c]
        if len(nn) == 0:
            continue
        nn = nn[ok[nn]]
        if len(nn) >= 3:
            pred[c] = v[nn].mean()
    m = ok & ~np.isnan(pred)
    if m.sum() > 100:
        errs_loo.append(np.sqrt(np.mean((v[m] - pred[m]) ** 2)))
print(f"LOO neighbor-mean error (30 months): mean {np.mean(errs_loo):.3f}  "
      f"range [{min(errs_loo):.3f}, {max(errs_loo):.3f}]")
print("  ^ this ~= sigma_white * sqrt(1+1/n_nbr) + small-scale structure = the SPATIAL floor")
field_std = anom.iloc[:, sample_months].std().mean()
print(f"  (mean field anomaly std for reference: {field_std:.3f})")

# ================= F2: per-cell AR(1)+noise moments =================
print("\n===== F2: per-cell detrended AR(1)+noise (rho1,rho2 -> phi, lambda) =====")
# detrend per cell: linear fit on t
t_norm = (t_cols - t_cols.mean())
X = np.vstack([np.ones_like(t_norm), t_norm]).T
A = anom.values  # cells x months (NaN possibly)
ok = ~np.isnan(A)
# fit per-cell trend via normal equations with nan-masking (loop is fine: 15.7k cells)
phis, lambdas, var_res = [], [], []
t_arr = t_cols.astype(float)
for c in rng.choice(A.shape[0], size=3000, replace=False):
    y = A[c]
    m = ok[c]
    if m.sum() < 60:
        continue
    tt = t_arr[m]; yy = y[m]
    Xc = np.vstack([np.ones_like(tt), tt - tt.mean()]).T
    beta, *_ = np.linalg.lstsq(Xc, yy, rcond=None)
    r = yy - Xc @ beta
    # calendar-consecutive lag pairs
    order = np.argsort(tt)
    tt_o, r_o = tt[order], r[order]
    # calendar-correct lag pairs: positions i, i+k with calendar distance exactly k
    k1 = np.where(tt_o[1:] - tt_o[:-1] == 1)[0]
    k2 = np.where((np.arange(len(tt_o) - 2) >= 0) &
                  (tt_o[2:] - tt_o[:-2] == 2))[0]
    if len(k1) < 40 or len(k2) < 20:
        continue
    a, b = r_o[k1], r_o[k1 + 1]
    a2, b2 = r_o[k2], r_o[k2 + 2]
    c1 = np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-9 and np.std(b) > 1e-9 else np.nan
    c2 = np.corrcoef(a2, b2)[0, 1] if np.std(a2) > 1e-9 and np.std(b2) > 1e-9 else np.nan
    if np.isfinite(c1) and np.isfinite(c2) and c1 > 0.1 and c2 > 0.02:
        phi_c = c2 / c1
        lam_c = c1 ** 2 / c2
        phis.append(phi_c); lambdas.append(lam_c); var_res.append(np.var(r))
phis = np.array(phis); lambdas = np.array(lambdas); var_res = np.array(var_res)
print(f"n cells fitted: {len(phis):,}")
print(f"phi = rho2/rho1: median {np.median(phis):.3f} (IQR {np.percentile(phis,25):.3f}-{np.percentile(phis,75):.3f})")
print(f"lambda = rho1^2/rho2: median {np.median(lambdas):.3f} (IQR {np.percentile(lambdas,25):.3f}-{np.percentile(lambdas,75):.3f})")
sig_w_cell = np.sqrt(var_res * (1 - np.clip(lambdas, 0, 1)))
print(f"implied sigma_noise per cell: median {np.median(sig_w_cell):.3f}")
print(f"  vs the 0.456 assumed in our error budget!")

# ================= F3: innovation field spatial structure =================
print("\n===== F3: innovation field e(c,t) = r(t) - phi*r(t-1): white or smooth? =====")
phi_use = np.median(phis)
# build innovation fields for a sample of month transitions
innov_errs_adj = []
resid_fields = []
for mi in rng.choice(np.arange(2, len(t_cols) - 1), size=40, replace=False):
    tm1, t0 = t_cols[mi - 1], t_cols[mi]
    r_prev = anom[tm1].values if tm1 in anom.columns else None
    r_now = anom[t0].values
    if r_prev is None:
        continue
    m = ~np.isnan(r_prev) & ~np.isnan(r_now)
    if m.sum() < 5000:
        continue
    e = np.full(len(r_now), np.nan)
    e[m] = r_now[m] - phi_use * r_prev[m]  # anomaly space already
    # remove cross-cell mean (global level)
    e_mean = np.nanmean(e[m])
    e[m] = e[m] - e_mean
    resid_fields.append((t0, e.copy()))
    # adjacent diff variance of e
    ok_e = ~np.isnan(e)
    ap = adj[ok_e[adj[:, 0]] & ok_e[adj[:, 1]]]
    diff = e[ap[:, 0]] - e[ap[:, 1]]
    innov_errs_adj.append((np.nanstd(e[m]), np.sqrt(np.var(diff) / 2)))
ie = np.array(innov_errs_adj)
print(f"innovation field std: mean {ie[:,0].mean():.3f}")
print(f"adjacent-diff implied white part: mean {ie[:,1].mean():.3f}")
frac_white = (ie[:,1].mean() / ie[:,0].mean()) ** 2
print(f"  => white fraction of innovation variance: {frac_white*100:.0f}%  "
      f"({'INNOVATION IS SMOOTH FIELD (trackable!)' if frac_white < 0.5 else 'innovation mostly white'})")

# ================= F4: cov-visible fraction of the innovation =================
print("\n===== F4: how much of y(t+1) is explained by y(t) + covs(t+1)? =====")
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']
piv_covs = {c: train.pivot_table(index='cc', columns='t_abs', values=c).reindex(grid.index) for c in COVS}
r2_base_list, r2_full_list = [], []
for mi in rng.choice(np.arange(1, len(t_cols) - 1), size=40, replace=False):
    tm1, t0 = t_cols[mi], t_cols[mi + 1]
    if tm1 not in anom.columns or t0 not in anom.columns:
        continue
    y_prev = anom[tm1].values; y_now = anom[t0].values
    cv = np.column_stack([piv_covs[c][t0].values if t0 in piv_covs[c].columns else np.nan for c in COVS])
    m = ~np.isnan(y_prev) & ~np.isnan(y_now) & ~np.isnan(cv).any(1)
    if m.sum() < 5000:
        continue
    Xb = np.column_stack([y_prev[m], np.ones(m.sum())])
    Xf = np.column_stack([y_prev[m], cv[m], np.ones(m.sum())])
    yn = y_now[m]
    # demean y_now cross-cell (remove global level)
    yn = yn - yn.mean()
    Xb = Xb - Xb.mean(0); Xf = Xf - Xf.mean(0)
    bb, *_ = np.linalg.lstsq(Xb, yn, rcond=None)
    bf, *_ = np.linalg.lstsq(Xf, yn, rcond=None)
    ss = ((yn - yn.mean()) ** 2).sum()
    r2b = 1 - ((yn - Xb @ bb) ** 2).sum() / ss
    r2f = 1 - ((yn - Xf @ bf) ** 2).sum() / ss
    r2_base_list.append(r2b); r2_full_list.append(r2f)
print(f"persistence-only cross-cell R^2: {np.mean(r2_base_list):.3f}")
print(f"+ covs(t+1) cross-cell R^2:      {np.mean(r2_full_list):.3f}")
print(f"  => covariates add {np.mean(r2_full_list)-np.mean(r2_base_list):.3f} R^2")
resid_std = np.sqrt((1 - np.mean(r2_full_list))) * field_std
print(f"  => residual std after y(t)+covs: ~{resid_std:.3f} (vs field std {field_std:.3f})")

print("\n===== STRATEGIC VERDICT =====")
print(f"F1 spatial floor:  {np.mean(errs_loo):.3f}")
print(f"F2 per-cell noise: {np.median(sig_w_cell):.3f} (lambda median {np.median(lambdas):.3f})")
print(f"F3 innovation:     std {ie[:,0].mean():.3f}, white part {ie[:,1].mean():.3f} "
      f"-> smooth fraction {(1-frac_white)*100:.0f}%")
print(f"F4 cov increment:  +{np.mean(r2_full_list)-np.mean(r2_base_list):.3f} R^2")
