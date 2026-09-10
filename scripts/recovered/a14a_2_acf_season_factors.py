"""Exp 2: ACF shape + seasonality probe + factor dynamics.

A) Calendar-correct pooled ACF of detrended anomaly, lags 1..36 (consecutive-pairs based).
B) Per-cell seasonal harmonics (a*sin+b*cos on month-of-year): amplitude distribution,
   variance share, phase-vs-latitude. ACF after seasonal removal.
C) PCA of seasonally-adjusted detrended field: spectrum again; top-30 PC scores:
   AR(1) fits (phi, innovation var), residual whiteness, cross-lag correlations (VAR?),
   seasonality of scores.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)

df = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float)
tc = t - t.mean()

mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)   # detrended anomaly

# ---------------- A) calendar-correct pooled ACF ----------------
def pooled_acf(Xf, months, lags):
    """ACF averaged over cells using pairs of calendar months exactly `lag` apart.
    Per-cell normalization by per-cell variance; then average over cells."""
    out = {}
    dt = np.diff(months)
    for lag in lags:
        # index pairs (i, j) with months[j]-months[i] == lag
        pairs = [(i, j) for i in range(len(months)) for j in range(i + 1, len(months))
                 if months[j] - months[i] == lag]
        num = np.zeros(Xf.shape[1]); den = np.zeros(Xf.shape[1])
        for i, j in pairs:
            num += Xf[i] * Xf[j]
            den += 1
        # normalize per cell by its own variance, weight pairs equally
        varc = Xf.var(axis=0, ddof=1)
        r = (num / den) / varc
        out[lag] = (np.nanmean(r), len(pairs))
    return out

lags = list(range(1, 37))
acf = pooled_acf(Xd, months, lags)
print("=== pooled ACF of detrended anomaly (calendar-correct) ===")
print(" ".join(f"{k}:{v[0]:+.3f}" for k, v in acf.items()))

# ---------------- B) per-cell seasonal harmonics ----------------
moy = (months % 12) + 1  # 1..12
ang = 2 * np.pi * (moy - 1) / 12
S = np.sin(ang); Cs = np.cos(ang)
# OLS per cell of Xd on [S, Cs] (S,Cs orthogonal over full 12-month cycles only roughly; do joint)
Dmat = np.column_stack([S, Cs])
coef, *_ = np.linalg.lstsq(Dmat, Xd, rcond=None)
seas = Dmat @ coef
Xds = Xd - seas
amp = np.sqrt((coef ** 2).sum(axis=0))
seas_share = (seas.var(axis=0) / Xd.var(axis=0, ddof=1))
print("\n=== per-cell seasonal harmonic (single annual freq) ===")
print(f"amplitude: mean={amp.mean():.4f} p5={np.percentile(amp,5):.4f} med={np.median(amp):.4f} "
      f"p95={np.percentile(amp,95):.4f} max={amp.max():.4f}")
print(f"seasonal variance share: mean={seas_share.mean():.4f} p50={np.median(seas_share):.4f} "
      f"p95={np.percentile(seas_share,95):.4f}")
phase = np.arctan2(coef[0], coef[1])  # season peak month
lat = cells["lat"].values[full]
# correlation of phase with latitude (circular: use sin/cos of phase vs lat)
rsl = np.corrcoef(np.sin(phase), lat)[0, 1]
rcl = np.corrcoef(np.cos(phase), lat)[0, 1]
print(f"corr(sin(phase), lat)={rsl:.3f}  corr(cos(phase), lat)={rcl:.3f}")
# amplitude vs |lat|
print(f"corr(amp, |lat|)={np.corrcoef(amp, np.abs(lat))[0,1]:.3f}")

acf2 = pooled_acf(Xds, months, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 24, 30, 36])
print("\n=== pooled ACF after seasonal removal ===")
print(" ".join(f"{k}:{v[0]:+.3f}" for k, v in acf2.items()))

# ---------------- C) spectrum of seasonally-adjusted field ----------------
U, s, Vt = np.linalg.svd(Xds, full_matrices=False)
v = s ** 2 / (C * (T - 1))
cum = np.cumsum(v) / v.sum()
print(f"\n=== spectrum of seasonally-adjusted detrended field (sum={v.sum():.4f}) ===")
for k in [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 110, 120, 130, 134, 136]:
    print(f"  k={k:>3}: v={v[k-1]:.6f} cum={100*cum[k-1]:.3f}%")

# ---------------- D) factor dynamics on top-30 PCs of Xds ----------------
K = 30
scores = U[:, :K] * s[:K]          # (T, K) PC scores (time series)
print("\n=== per-PC AR(1) fits (seasonally-adjusted field) ===")
print(f"{'PC':>3} {'var':>9} {'phi':>7} {'innov_sd':>9} {'res_lag1':>8} {'res_lag2':>8} "
      f"{'res_lag12':>9} {'seasF':>7} {'seasP':>7}")
band = 2 / np.sqrt(T)
phi_all = []; q_all = []
for k in range(K):
    y = scores[:, k]
    ylag = y[:-1]; ycur = y[1:]
    phi = (ylag * ycur).sum() / (ylag ** 2).sum()
    resid = ycur - phi * ylag
    q = resid.std()
    # whiteness of AR residuals
    r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    r2 = np.corrcoef(resid[:-2], resid[2:])[0, 1]
    r12 = np.corrcoef(resid[:-12], resid[12:])[0, 1]
    # seasonality of score
    a = np.corrcoef(y, S)[0, 1]; b = np.corrcoef(y, Cs)[0, 1]
    phi_all.append(phi); q_all.append(q)
    print(f"{k+1:>3} {y.var():>9.3f} {phi:>7.3f} {q:>9.3f} {r1:>+8.3f} {r2:>+8.3f} "
          f"{r12:>+9.3f} {a:>+7.3f} {b:>+7.3f}")
print(f"2/sqrt(T) whiteness band = ±{band:.3f}")
print(f"phi: mean={np.mean(phi_all):.3f} std={np.std(phi_all):.3f} min={np.min(phi_all):.3f} "
      f"max={np.max(phi_all):.3f}")

# cross-factor lagged correlations: innovation correlation matrix
R = scores[1:] - np.array(phi_all) * scores[:-1]  # (T-1, K) innovations
Cinn = np.corrcoef(R.T)
offdiag = Cinn[np.triu_indices(K, 1)]
print(f"\ninnovation cross-correlations: mean={offdiag.mean():+.3f} std={offdiag.std():.3f} "
      f"max|.|={np.abs(offdiag).max():.3f}")
# lagged score cross-corr: corr(score_i(t), score_j(t+1)) vs corr(score_i(t), score_j(t))
Csl = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        Csl[i, j] = np.corrcoef(scores[:-1, i], scores[1:, j])[0, 1]
Csym = np.corrcoef(scores.T)
asym = Csl - Csym.T * 0  # compare Csl to phi_i*Csym
pred = np.diag(np.array(phi_all)) @ Csym
print(f"VAR(1) test: max|Csl - diag(phi)@Csym| = {np.abs(Csl - pred).max():.3f} "
      f"(if ~0, independent AR(1)s suffice; else VAR)")

np.save("/home/z/my-project/scripts/a14a_cache_dyn.npy",
        {"v_s": v, "cum_s": cum, "phi_all": phi_all, "q_all": q_all}, allow_pickle=True)
print("\nsaved a14a_cache_dyn.npy")
