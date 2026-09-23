"""Exp 3: Noise structure + full mode-dynamics map + artifacts.

A) Per-PC AR(1) for ALL 136 modes: phi_k, score var, innov var, stationary check,
   phi->0 transition point (effective factor rank vs white noise).
B) Per-cell AR(1): phi_c distribution, innovation field structure:
   temporal autocorr (lags 1-3), spatial neighbor corr at same t, kurtosis,
   innovation var vs total var (per-cell SNR).
C) Monte Carlo: is the negative long-lag pooled ACF an artifact of per-cell
   mean+trend removal on a persistent process?
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(14)

df = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)

U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
scores = U * s  # (T, 136) — score time series (already time-mean-zero-ish)

# ---------------- A) per-mode AR(1) map ----------------
print("=== A) per-mode AR(1) map (all modes) ===")
print(f"{'k':>4} {'svar':>9} {'phi':>7} {'innov_v':>9} {'stat_var':>9} {'stat/svar':>9} {'R2':>6}")
phi_k = np.zeros(T); q_k = np.zeros(T); svar_k = np.zeros(T)
for k in range(T):
    y = scores[:, k]
    phi = (y[:-1] * y[1:]).sum() / (y[:-1] ** 2).sum()
    q2 = ((y[1:] - phi * y[:-1]) ** 2).mean()
    svar = y.var(ddof=1)
    stat = q2 / max(1 - phi ** 2, 1e-9)
    phi_k[k] = phi; q_k[k] = q2; svar_k[k] = svar
    if k < 40 or k % 5 == 0:
        print(f"{k+1:>4} {svar:>9.2f} {phi:>7.3f} {q2:>9.2f} {stat:>9.2f} "
              f"{stat/svar:>9.3f} {1-q2/svar:>6.3f}")

# where does phi hit the noise band (2/sqrt(T))?
band = 2 / np.sqrt(T)
persist = np.where(phi_k > band)[0]
print(f"\nphi > {band:.3f} (2/sqrt(T)) for modes 1..{persist.max()+1} "
      f"({(phi_k>band).sum()} modes persistent)")
# white-noise-variance estimate: sum of per-cell variance of modes with phi<=band
white_modes = phi_k <= band
v = s ** 2 / (C * (T - 1))
sigma2_white = v[white_modes].sum()
print(f"per-cell variance in non-persistent modes: {sigma2_white:.5f} "
      f"(sigma={np.sqrt(sigma2_white):.4f})")
# gradual view: cumulative per-cell variance of modes 50+, 75+, 100+
for K in [30, 50, 75, 100, 110, 120]:
    print(f"  per-cell var beyond mode {K}: {v[K:].sum():.5f} (sigma={np.sqrt(v[K:].sum()):.4f})")

# ---------------- B) per-cell AR(1) + innovation field ----------------
print("\n=== B) per-cell AR(1) on detrended anomaly ===")
# per-cell OLS AR(1) using consecutive calendar months only
dt_ok = np.diff(months) == 1
idx_pairs = np.where(dt_ok)[0]
Xa, Xb = Xd[idx_pairs], Xd[idx_pairs + 1]
num = (Xa * Xb).sum(axis=0); den = (Xa ** 2).sum(axis=0)
phi_c = num / den
resid_c = Xb - phi_c * Xa         # innovation field at times idx_pairs+1
varc = Xd.var(axis=0, ddof=1)
print(f"phi_c: mean={phi_c.mean():.3f} std={phi_c.std():.3f} p5={np.percentile(phi_c,5):.3f} "
      f"p50={np.median(phi_c):.3f} p95={np.percentile(phi_c,95):.3f}")
print(f"corr(phi_c, var_c)={np.corrcoef(phi_c, varc)[0,1]:.3f}")
innov_var_c = (resid_c ** 2).mean(axis=0)
print(f"per-cell innovation std: mean={np.sqrt(innov_var_c).mean():.4f} "
      f"p5={np.percentile(np.sqrt(innov_var_c),5):.4f} p95={np.percentile(np.sqrt(innov_var_c),95):.4f}")
print(f"corr(innov_var_c, var_c)={np.corrcoef(innov_var_c, varc)[0,1]:.3f}  "
      f"(1.0 => noise scales with signal)")

# temporal autocorr of innovation field (mixture residual): lag 1,2,3 within resid_c
def field_acf(Fm, months, lag):
    dt = np.diff(months)
    ok = dt == lag
    i = np.where(ok)[0]
    num = (Fm[i] * Fm[i + lag]).sum(axis=0)
    den = (Fm ** 2).sum(axis=0) * len(i) / len(Fm)  # approx normalization
    return np.nanmean(num / den / (Fm.var(axis=0, ddof=1)))
for lag in [1, 2, 3]:
    print(f"innovation field temporal autocorr lag {lag}: {field_acf(resid_c, months[idx_pairs], lag):+.4f}")

# spatial neighbor correlation of innovation field at same time (average over times)
lat = cells["lat"].values[full]; lon = cells["lon"].values[full]
key = {(a, b): i for i, (a, b) in enumerate(zip(lat, lon))}
def neighbor_corr(Fm, dlat, dlon):
    cors = []
    for i, (a, b) in enumerate(zip(lat, lon)):
        j = key.get((a + dlat, b + dlon))
        if j is not None:
            cors.append(np.corrcoef(Fm[:, i], Fm[:, j])[0, 1])
    return np.mean(cors), len(cors)
print("\ninnovation field spatial neighbor corr:")
for (dl, do) in [(1, 0), (0, 1), (2, 0), (5, 0), (10, 0)]:
    m, n = neighbor_corr(resid_c, dl, do)
    print(f"  dlat={dl:+d} dlon={do:+d}: mean corr={m:.4f} (n={n})")
print("detrended field spatial neighbor corr (for comparison):")
for (dl, do) in [(1, 0), (0, 1), (5, 0)]:
    m, n = neighbor_corr(Xd, dl, do)
    print(f"  dlat={dl:+d} dlon={do:+d}: mean corr={m:.4f} (n={n})")

# kurtosis / normality of innovation field (pooled)
w = resid_c.ravel()
w = w - w.mean()
kurt = (w ** 4).mean() / (w ** 2).mean() ** 2
print(f"\ninnovation pooled kurtosis={kurt:.4f} (3=Gaussian), skew={((w**3).mean()/w.var()**1.5):+.4f}")
# per-cell kurtosis distribution
kc = ((resid_c - resid_c.mean(0)) ** 4).mean(0) / resid_c.var(0) ** 2
print(f"per-cell kurtosis: mean={kc.mean():.3f} p5={np.percentile(kc,5):.3f} p95={np.percentile(kc,95):.3f}")

# ---------------- C) Monte Carlo mean-removal ACF bias ----------------
print("\n=== C) MC: pooled ACF after per-cell mean+trend removal ===")
def mc_acf(phi, T=138, ncell=2000, reps=3):
    out = {}
    for rep in range(reps):
        tt = np.arange(T, dtype=float)
        e = rng.normal(size=(T, ncell))
        x = np.empty_like(e)
        x[0] = rng.normal(scale=1/np.sqrt(1-phi**2), size=ncell) if phi < 1 else 0
        for i in range(1, T):
            x[i] = phi * x[i - 1] + np.sqrt(1 - phi ** 2) * e[i] if phi < 1 else x[i-1] + e[i]
        # per-cell mean + trend removal
        xm = x - x.mean(0)
        b = (xm * (tt - tt.mean())[:, None]).sum(0) / ((tt - tt.mean()) ** 2).sum()
        xr = xm - np.outer(tt - tt.mean(), b)
        varc = xr.var(0, ddof=1)
        for lag in [1, 6, 12, 24, 36]:
            num = np.zeros(ncell)
            for i in range(T - lag):
                num += xr[i] * xr[i + lag]
            r = (num / (T - lag)) / varc
            out.setdefault(lag, []).append(np.nanmean(r))
    return {k: np.mean(v) for k, v in out.items()}
for phi in [0.65, 0.85, 0.98, 1.0]:
    print(f"  phi={phi}: {mc_acf(phi)}")
print("observed pooled ACF: lag1=+0.653 lag6=+0.315 lag12=+0.075 lag24=-0.093 lag36=-0.122")

np.save("/home/z/my-project/scripts/a14a_cache_modes.npy",
        {"phi_k": phi_k, "q_k": q_k, "svar_k": svar_k, "v": v,
         "phi_c": phi_c, "innov_var_c": innov_var_c, "varc": varc}, allow_pickle=True)
print("\nsaved a14a_cache_modes.npy")
