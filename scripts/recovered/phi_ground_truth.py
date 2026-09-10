#!/usr/bin/env python3
"""
PHI GROUND TRUTH: is the fast process phi=0.82 (our Kalman) or much lower?

Static-free estimator: pooled cross-sectional lag slopes on detrended y:
  b_h = cov(y(t+h), y(t)) / var(y(t))   (pooled over cells & months)
  For AR(1)+static: phi_hat = (b2-b3)/(b1-b2)   [static cancels exactly]
Also: ARMA(1,1) vs AR(1) walk-forward on PC scores, and the anchor r(h) profile.
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
rng = np.random.default_rng(5)

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1
def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
train['cc'] = ckey(train['lat'], train['lon'])
grid = train[['cc', 'lat', 'lon']].drop_duplicates('cc').set_index('cc').sort_index()
mu_c = train.groupby('cc')['TWS_t'].mean()
n = len(grid)

piv = train.pivot_table(index='cc', columns='t_abs', values='TWS_t').reindex(grid.index)
anom = piv.sub(mu_c.reindex(grid.index), axis=0)
t_cols = anom.columns.values

t_norm = t_cols.astype(float)
X = np.vstack([np.ones_like(t_norm), t_norm - t_norm.mean()]).T
A = anom.values
trend = np.zeros((n, 2))
for c in range(n):
    y = A[c]; m = ~np.isnan(y)
    if m.sum() > 24:
        trend[c] = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
y_dt = A - (X @ trend.T).T

# ===== static-free phi from pooled lag slopes =====
print("===== Pooled cross-sectional lag slopes (detrended y) =====")
def lag_slope(h):
    """pooled cov(y(t+h), y(t))/var(y(t)) over all valid (cell, t)"""
    num, den = [], []
    for i in range(len(t_cols) - h):
        if t_cols[i + h] - t_cols[i] != h:
            continue
        y0 = y_dt[:, i]; yh = y_dt[:, i + h]
        m = ~np.isnan(y0) & ~np.isnan(yh)
        if m.sum() < 8000:
            continue
        a, b = y0[m], yh[m]
        a = a - a.mean(); b = b - b.mean()
        num.append((a * b).sum()); den.append((a * a).sum())
    return np.sum(num) / np.sum(den)

bs = {}
for h in [1, 2, 3, 4, 6, 12, 24]:
    bs[h] = lag_slope(h)
    print(f"  b_{h:<2} = {bs[h]:.3f}")
phi_hat = (bs[2] - bs[3]) / (bs[1] - bs[2])
print(f"\n  STATIC-FREE PHI = (b2-b3)/(b1-b2) = {phi_hat:.3f}")
print(f"  (our Kalman uses phi=0.80-0.85; F2's rho2/rho1 said 0.823)")

# also with different lag triplets for robustness
for (l1, l2, l3) in [(1, 2, 3), (2, 4, 6), (3, 6, 9), (1, 3, 5)]:
    if l3 in bs and l2 in bs and l1 in bs:
        continue
# compute missing
for h in [5, 9]:
    if h not in bs:
        bs[h] = lag_slope(h)
for (l1, l2, l3) in [(2, 4, 6), (3, 6, 9), (1, 3, 5)]:
    print(f"  triplet ({l1},{l2},{l3}): phi = {(bs[l2]-bs[l3])/(bs[l1]-bs[l2]) if bs[l1]!=bs[l2] else float('nan'):.3f}")

# implied static + fast decomposition from b1,b2,b3
# b_h = (phi^h F + S)/(F+S): with phi=phi_hat, solve F,S from b1,b2
phi = phi_hat
V = 1.0
S = (bs[2] - phi**2 * bs[1]) / (1 - phi**2) * 0  # placeholder
# solve: b1 = (phi F + S)/(F+S), b2 = (phi^2 F + S)/(F+S)
# => S/F = (b1 - phi... ) let r = S/F:
# b1(F)(1) ... do numerically
from scipy.optimize import brentq
def eq(r):
    F = 1.0
    S = r
    return (phi**1 * F + S) / (F + S) - bs[1]
try:
    r_sol = brentq(eq, 0.01, 50)
    print(f"\n  implied S/F (static/fast variance ratio) = {r_sol:.2f}")
    print(f"  => fast fraction of detrended variance = {1/(1+r_sol)*100:.0f}%")
except Exception as e:
    print(f"  (S/F solve failed: {e})")

# ===== ARMA(1,1) vs AR(1) walk-forward on pooled detrended y =====
print("\n===== Walk-forward: AR(1) vs ARMA(1,1) one-step on pooled data =====")
# build monthly pooled sample: for each consecutive pair, per-cell y_t -> y_t+1
rows = []
for i in range(len(t_cols) - 1):
    if t_cols[i + 1] - t_cols[i] != 1:
        continue
    y0 = y_dt[:, i]; y1 = y_dt[:, i + 1]
    m = ~np.isnan(y0) & ~np.isnan(y1)
    if m.sum() < 8000:
        continue
    rows.append((i, y0[m], y1[m]))
# for each cell-month: maintain running AR(1) pred error e, ARMA pred: phi*y + theta*e
def wf_compare(phi, thetas):
    errs = {th: [] for th in thetas}
    errs_ar = []
    e_prev = None
    prev_i = None
    # state per cell across months (cells align within rows because same mask? no —
    # use position-independent: rebuild per transition with e from previous transition same cells)
    # simpler: evaluate on cells present in consecutive transitions
    for k in range(1, len(rows)):
        i_prev, y0p, y1p = rows[k - 1]
        i_cur, y0c, y1c = rows[k]
        if i_cur - i_prev != 1:
            e_prev = None
            continue
        # e_prev = y1p - phi*y0p (innovation at month i_prev+1 = current y0 month)
        e = y1p - phi * y0p
        # ARMA pred for y1c given y0c and e (same cells order? need same mask)
        # masks may differ; use only where both valid via index alignment — approximate:
        if len(e) == len(y0c):
            errs_ar.append(((y1c - phi * y0c) ** 2).mean())
            for th in thetas:
                pred = phi * y0c + th * e
                errs[th].append(((y1c - pred) ** 2).mean())
    rmse_ar = np.sqrt(np.mean(errs_ar))
    print(f"  AR(1) phi={phi}: one-step RMSE = {rmse_ar:.4f}")
    for th in thetas:
        if errs[th]:
            rmse_arma = np.sqrt(np.mean(errs[th]))
            print(f"  ARMA(1,1) theta={th:+.2f}: one-step RMSE = {rmse_arma:.4f} "
                  f"({(rmse_arma/rmse_ar-1)*100:+.1f}%)")

# NOTE: len(e)==len(y0c) only if same cells valid in both transitions — check
for phi_try in [0.82, 0.60, 0.50]:
    print(f"\n  --- phi = {phi_try} ---")
    wf_compare(phi_try, [-0.5, -0.3, -0.2, -0.1, 0.1])
