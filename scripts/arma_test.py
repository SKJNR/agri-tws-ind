#!/usr/bin/env python3
"""ARMA(1,1) innovation-form test with proper cell alignment (full-grid vectors)."""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
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
y_dt = A - (X @ trend.T).T   # (cells, months)

THETAS = [-0.4, -0.3, -0.2, -0.1, 0.1]

def run(phi):
    # single pass collecting AR and ARMA errors simultaneously
    e_prev = np.full(n, np.nan)
    sq_ar, sq_arma = [], {th: [] for th in THETAS}
    for i in range(len(t_cols) - 1):
        if t_cols[i + 1] - t_cols[i] != 1:
            e_prev = np.full(n, np.nan)
            continue
        y0, y1 = y_dt[:, i], y_dt[:, i + 1]
        m_ar = ~np.isnan(y0) & ~np.isnan(y1)
        if m_ar.sum() > 5000:
            sq_ar.append(((y1[m_ar] - phi * y0[m_ar]) ** 2).mean())
        m = m_ar & ~np.isnan(e_prev)
        if m.sum() > 5000:
            for th in THETAS:
                pred = phi * y0[m] + th * e_prev[m]
                sq_arma[th].append(((y1[m] - pred) ** 2).mean())
        # update carried innovation
        e_prev = np.where(m_ar, y1 - phi * y0, np.nan)
    rmse_ar = np.sqrt(np.mean(sq_ar))
    line = f"phi={phi:.2f}: AR(1) RMSE = {rmse_ar:.4f} | "
    parts = []
    for th in THETAS:
        r = np.sqrt(np.mean(sq_arma[th]))
        parts.append(f"th={th:+.1f}: {r:.4f} ({(r/rmse_ar-1)*100:+.1f}%)")
    print(line + "  ".join(parts))

print("Walk-forward one-step RMSE: AR(1) vs ARMA(1,1) innovation form\n")
for phi in [0.82, 0.70, 0.60, 0.50]:
    run(phi)
print("\n(if negative theta materially reduces one-step error, the innovation-form")
print(" correction is real and worth adding to the Kalman output stage)")
