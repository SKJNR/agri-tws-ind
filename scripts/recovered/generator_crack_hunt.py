#!/usr/bin/env python3
"""
GENERATOR CRACK HUNT: temporal structure of the innovation field's PC scores.

eta fields are 97% spatially smooth (rank-limited). If the generator draws
eta(t) = PCs @ scores(t) with scores following predictable dynamics (seasonal,
oscillatory, AR with long memory, or even deterministic), we can predict the
leading scores -> recover a chunk of eta variance -> masked RMSE breakthrough.

Tests on train (~127 monthly eta fields):
  P1: PCA of eta fields; spectrum (how many PCs?)
  P2: per-PC-score ACF at lags 1..12 (is there memory?)
  P3: seasonality of scores (corr with month-of-year dummies)
  P4: spectral peaks (periodogram of score series)
  P5: one-step predictability of top-10 scores (AR(2) + seasonal)
  P6: same analysis on the w-field (y - AR-prediction residual structure)
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

PHI = 0.82
etas, eta_months = [], []
for i in range(len(t_cols) - 1):
    if t_cols[i + 1] - t_cols[i] == 1:
        y0, y1 = y_dt[:, i], y_dt[:, i + 1]
        m = ~np.isnan(y0) & ~np.isnan(y1)
        if m.sum() < 10000:
            continue
        e = np.full(n, np.nan)
        e[m] = y1[m] - PHI * y0[m]
        # mask-consistent: keep cells present in both, fill others with 0 after demean
        e_mean = np.nanmean(e)
        e = np.nan_to_num(e - e_mean, nan=0.0)
        etas.append(e)
        eta_months.append(t_cols[i + 1])
E = np.array(etas)  # (T, n)
print(f"eta fields: {E.shape[0]} months x {E.shape[1]:,} cells; "
      f"mean field std {E.std(axis=1).mean():.3f}")

# ===== P1: PCA spectrum =====
Ec = E - E.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
var_expl = (S**2) / (S**2).sum()
print("\n===== P1: PCA spectrum of eta fields =====")
for k in [1, 2, 5, 10, 20, 50, 100]:
    print(f"  PC1-{k:<4d}: {var_expl[:k].sum()*100:5.1f}% of eta variance")

# ===== P2: PC-score ACF =====
print("\n===== P2: ACF of PC scores (lags 1,2,3,6,12) =====")
scores = U * S  # (T, n) time series of scores
print(f"{'PC':>4} {'var%':>6} | " + " | ".join(f"lag{l:>2}" for l in [1, 2, 3, 6, 12]))
for k in range(10):
    s = scores[:, k]
    acs = []
    for lag in [1, 2, 3, 6, 12]:
        if len(s) > lag + 5:
            a, b = s[:-lag], s[lag:]
            acs.append(np.corrcoef(a, b)[0, 1])
        else:
            acs.append(np.nan)
    print(f"PC{k+1:>3} {var_expl[k]*100:>5.1f}% | " +
          " | ".join(f"{a:>5.2f}" if np.isfinite(a) else "  n/a" for a in acs))

# ===== P3: seasonality =====
print("\n===== P3: seasonality of PC scores (R^2 vs month-of-year dummies) =====")
months = np.array([t % 12 + 1 for t in eta_months])
for k in range(5):
    s = scores[:, k]
    D = np.zeros((len(s), 12))
    D[np.arange(len(s)), months - 1] = 1
    beta, *_ = np.linalg.lstsq(D, s, rcond=None)
    r2 = 1 - ((s - D @ beta)**2).sum() / ((s - s.mean())**2).sum()
    # strongest month contrast
    print(f"  PC{k+1}: seasonal R^2 = {r2:.3f}")

# ===== P4: dominant period =====
print("\n===== P4: periodogram peaks of top-5 PC scores =====")
for k in range(5):
    s = scores[:, k]
    s = s - s.mean()
    f = np.fft.rfft(s)
    p = np.abs(f)**2
    freqs = np.fft.rfftfreq(len(s), d=1.0)
    k_peak = np.argmax(p[1:]) + 1
    period = 1/freqs[k_peak] if freqs[k_peak] > 0 else np.inf
    print(f"  PC{k+1}: peak period ~{period:.1f} months (power {p[k_peak]/p[1:].sum()*100:.0f}% of spectrum)")

# ===== P5: one-step predictability of scores =====
print("\n===== P5: one-step-ahead prediction of top-10 PC scores =====")
from numpy.linalg import lstsq
tot_var = sum(var_expl[k] * scores[:, k].var() for k in range(10))
resid_var = 0
print(f"{'PC':>4} {'R2_ar':>6} {'R2_sar':>7}")
r2_list = []
for k in range(10):
    s = scores[:, k]
    # AR(2) with month dummies
    Dm = np.zeros((len(s), 12)); Dm[np.arange(len(s)), months - 1] = 1
    Xa = np.column_stack([np.roll(s, 1), np.roll(s, 2), Dm])[2:]
    ya = s[2:]
    # walk-forward eval
    preds, actuals = [], []
    for t_split in range(max(24, len(s)//3), len(s)-1):
        Xtr, ytr = Xa[:t_split], ya[:t_split]
        b, *_ = lstsq(Xtr, ytr, rcond=None)
        preds.append(Xa[t_split] @ b); actuals.append(ya[t_split])
    preds, actuals = np.array(preds), np.array(actuals)
    r2 = 1 - ((actuals-preds)**2).sum() / ((actuals-actuals.mean())**2).sum()
    r2_list.append((k, r2))
    print(f"PC{k+1:>3} {r2:>6.3f}")
pred_var_total = sum(max(r2,0) * var_expl[k] * scores[:, k].var() for k, r2 in r2_list)
eta_var_total = (E.std(axis=1)**2).mean()
print(f"\n  => predictable fraction of top-10 PC variance: {pred_var_total/sum(var_expl[k]*scores[:,k].var() for k,r2 in r2_list)*100:.1f}%")
print(f"  => top-10 PCs carry {var_expl[:10].sum()*100:.1f}% of eta variance")
print(f"  => eta variance predictable overall: ~{pred_var_total/eta_var_total*100:.1f}%")

# ===== P6: same for w (smooth temporally-white field) =====
print("\n===== P6: sanity — eta field temporal white? corr(eta_t, eta_{t+1}) field-space =====")
fl = []
for i in range(len(E) - 1):
    fl.append(np.corrcoef(E[i], E[i+1])[0, 1])
print(f"  field-to-field lag-1 corr: mean {np.mean(fl):.3f} (0 = temporally white)")
