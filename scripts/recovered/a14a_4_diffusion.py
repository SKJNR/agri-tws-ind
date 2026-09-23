"""Exp 4: DIFFUSION OPERATOR TEST (the decisive experiment).

Hypothesis: generator evolves the anomaly field as
    X(t+1) - mu = A (X(t) - mu) + W(t+1),   A = rho*I - D*L  (graph Laplacian L on land grid)
Mode k of A has eigenvalue a_k = rho - D*lambda_k; negative at fine scales
(observed phi_k down to -0.6!). If phi_k vs lambda_k(L) is a TIGHT LINE,
the generator is a discrete diffusion (heat equation) on the land grid.

A) lambda_k per PC (plain graph Laplacian + normalized variant); regress phi_k.
B) Basis-free check: high-pass field (X - 3deg box mean) lag-1 autocorr < 0?
C) Diffusion residual W: temporal whiteness, spatial smoothness, spectrum.
D) HONEST HOLDOUT (fit 2002-2012, predict 2013-2015, horizons 1..7):
   persistence | trend-only | per-cell AR(1) | scalar-phi spectral (phi=0.74)
   | per-mode spectral phi_k | diffusion A^h. If diffusion ~ per-mode and both
   beat the rest -> generator identified.
"""
import numpy as np
import pandas as pd
import sys
from scipy import sparse
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)

df = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
lat = cells["lat"].values[full]; lon = cells["lon"].values[full]

# ---- graph Laplacian on the land grid (5-point, lon wraps) ----
key = {(a, b): i for i, (a, b) in enumerate(zip(lat, lon))}
rows, cols_, vals = [], [], []
for i, (a, b) in enumerate(zip(lat, lon)):
    nb = [(a + 1, b), (a - 1, b), (a, b + 1 if b + 1 <= 179.5 else -179.5 + 0.0),
          (a, b - 1 if b - 1 >= -179.5 else 179.5 - 0.0)]
    deg = 0
    for (na, nbn) in nb:
        j = key.get((na, nbn))
        if j is not None:
            rows.append(i); cols_.append(j); vals.append(-1.0); deg += 1
    rows.append(i); cols_.append(i); vals.append(float(deg))
L = sparse.csr_matrix((vals, (rows, cols_)), shape=(C, C))
deg = np.asarray(L.diagonal()); Aadj = -L + sparse.diags(deg)
B_rw = sparse.diags(1.0 / np.maximum(deg, 1)) @ Aadj   # random-walk (neighbor-average) operator

mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)

U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
V = Vt.T  # (C, T) loading vectors, unit norm
scores = U * s

# ---------------- A) phi_k vs Laplacian eigenvalue ----------------
lam = np.array([V[:, k] @ (L @ V[:, k]) for k in range(T)])
b_rw = np.array([V[:, k] @ (B_rw @ V[:, k]) for k in range(T)])
phi_k = np.zeros(T); q_k = np.zeros(T)
for k in range(T):
    y = scores[:, k]
    phi = (y[:-1] * y[1:]).sum() / (y[:-1] ** 2).sum()
    phi_k[k] = phi
    q_k[k] = ((y[1:] - phi * y[:-1]) ** 2).mean()

# regression phi = rho - D*lam (weights = mode variance share)
w = scores.var(axis=0)
def wlstsq(x, y, w):
    Xd_ = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    coef = np.linalg.lstsq(Xd_ * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    pred = Xd_ @ coef
    ss_res = (w * (y - pred) ** 2).sum(); ss_tot = (w * (y - y.mean()) ** 2).sum()
    return coef, 1 - ss_res / ss_tot

coef_l, r2_l = wlstsq(lam, phi_k, w)
coef_n, r2_n = wlstsq(b_rw, phi_k, w)
print("=== A) phi_k ~ a + b*lambda_k(L) (variance-weighted) ===")
print(f"plain Laplacian : rho={coef_l[0]:+.4f} D={-coef_l[1]:.5f}  R2={r2_l:.4f}")
print(f"random-walk B   : rho={coef_n[0]:+.4f} D={-coef_n[1]:.5f}  R2={r2_n:.4f}")
# unweighted too
cu, r2u = wlstsq(lam, phi_k, np.ones(T))
print(f"plain Lapl (unw): rho={cu[0]:+.4f} D={-cu[1]:.5f}  R2={r2u:.4f}")
# top modes only (best estimated)
sel = w > np.percentile(w, 50)
cs, r2s = wlstsq(lam[sel], phi_k[sel], w[sel])
print(f"plain Lapl (top-68 modes by var): rho={cs[0]:+.4f} D={-cs[1]:.5f}  R2={r2s:.4f}")

print("\n(mode, lambda_k, phi_k) sorted by lambda — check linearity:")
o = np.argsort(lam)
for k in o[::9]:
    print(f"  mode {k+1:>3}: lam={lam[k]:.3f} b_rw={b_rw[k]:+.3f} phi={phi_k[k]:+.3f} svar={scores[:,k].var():.1f}")

# ---------------- B) basis-free high-pass anti-persistence ----------------
# smooth: average over 5x5 box (via sparse)
rows, cols_, vals = [], [], []
for i, (a, b) in enumerate(zip(lat, lon)):
    cnt = 0
    for da in range(-2, 3):
        for db in range(-2, 3):
            j = key.get((a + da, (b + db + 180) % 360 - 180 + 0.0 if abs(b + db) > 179.5 else b + db))
            if j is not None:
                rows.append(i); cols_.append(j); vals.append(1.0); cnt += 1
    vals_last = cnt
    # normalize row
Sm = sparse.csr_matrix((vals, (rows, cols_)), shape=(C, C))
Sm = sparse.diags(1.0 / np.maximum(np.asarray(Sm.sum(axis=1)).ravel(), 1)) @ Sm
Xsm = (Sm @ Xd.T).T
Xhp = Xd - Xsm
dt_ok = np.diff(months) == 1
ip = np.where(dt_ok)[0]
r_hp = ((Xhp[ip] * Xhp[ip + 1]).sum(0) / (Xhp[ip] ** 2).sum(0))
r_full = ((Xd[ip] * Xd[ip + 1]).sum(0) / (Xd[ip] ** 2).sum(0))
print(f"\n=== B) basis-free lag-1 autocorr: high-pass(5deg)={np.nanmean(r_hp):+.4f}  "
      f"full field={np.nanmean(r_full):+.4f} ===")

# ---------------- C) diffusion residual W ----------------
rho, D = coef_l[0], -coef_l[1]
Amat = rho * sparse.identity(C) - D * L
R = Xd[ip + 1] - (Amat @ Xd[ip].T).T
print(f"\n=== C) diffusion residual W (rho={rho:.4f}, D={D:.5f}) ===")
Wf = R  # (n_pairs, C) innovations at consecutive months
# temporal whiteness of W (lag1 within consecutive residual pairs — approximate)
tt = months[ip + 1]
r1 = []
for i in range(len(tt) - 1):
    if tt[i + 1] - tt[i] == 1:
        r1.append((Wf[i] * Wf[i + 1]).sum() / np.sqrt((Wf[i] ** 2).sum() * (Wf[i + 1] ** 2).sum()))
print(f"W temporal corr (consecutive W's): mean={np.mean(r1):+.4f} (n={len(r1)})")
# spatial smoothness of W
def spat_corr(Fm, dlat, dlon):
    cs = []
    for i, (a, b) in enumerate(zip(lat, lon)):
        j = key.get((a + dlat, b + dlon if abs(b + dlon) <= 179.5 else (b + dlon + 180) % 360 - 180))
        if j is not None:
            cs.append(np.corrcoef(Fm[:, i], Fm[:, j])[0, 1])
    return np.mean(cs)
print("W spatial neighbor corr: ", end="")
for (dl, do) in [(1, 0), (0, 1), (2, 0), (5, 0), (10, 0)]:
    print(f"d({dl},{do})={spat_corr(Wf, dl, do):.3f} ", end="")
print()
# spectrum of W (is the innovation field smooth / low-rank?)
Uw, sw, _ = np.linalg.svd(Wf - Wf.mean(0), full_matrices=False)
vw = sw ** 2 / (C * (len(Wf) - 1))
print(f"W spectrum: PC1={100*vw[0]/vw.sum():.1f}% PC1-10={100*np.cumsum(vw)[:9][-1]/vw.sum():.1f}% "
      f"PC1-50={100*np.cumsum(vw)[49]/vw.sum():.1f}% (smooth kernel innovations if top-heavy)")
print(f"W per-cell std: mean={np.sqrt((Wf**2).mean(0)).mean():.4f}")
kw = (Wf - Wf.mean(0))
kurtw = ((kw**4).mean(0) / kw.var(0)**2)
print(f"W per-cell kurtosis: mean={kurtw.mean():.3f}")

# ---------------- D) honest holdout 2013-2015 ----------------
print("\n=== D) holdout: fit on 2002-2012, predict 2013-2015 (horizons 1..7) ===")
tr_mask = months < 2013 * 12
te_mask = ~tr_mask
Xtr = X[tr_mask]; mtr = months[tr_mask]; ttr = t[tr_mask]
tctr = ttr - ttr.mean()
mu_h = Xtr.mean(axis=0)
Atr = Xtr - mu_h
beta_h = (Atr * tctr[:, None]).sum(axis=0) / (tctr ** 2).sum()
Xdtr = Atr - np.outer(tctr, beta_h)
Uh, sh, Vth = np.linalg.svd(Xdtr, full_matrices=False)
Vh = Vth.T
scores_h = Uh * sh
phi_h = np.zeros(min(Xdtr.shape))
for k in range(scores_h.shape[1]):
    y = scores_h[:, k]
    p = (y[:-1] * y[1:]).sum() / (y[:-1] ** 2).sum()
    phi_h[k] = p
# per-cell AR(1) on train window
ip_tr = np.where(np.diff(mtr) == 1)[0]
phi_c = (Xdtr[ip_tr] * Xdtr[ip_tr + 1]).sum(0) / (Xdtr[ip_tr] ** 2).sum(0)
# diffusion params from train window only
lam_h = np.array([Vh[:, k] @ (L @ Vh[:, k]) for k in range(Vh.shape[1])])
wh = scores_h.var(axis=0)
coef_h, _ = wlstsq(lam_h, phi_h, wh)
rho_h, Dh = coef_h[0], -coef_h[1]
Amat_h = rho_h * sparse.identity(C) - Dh * L
print(f"train-only fit: rho={rho_h:.4f} D={Dh:.5f}")

# evaluation pairs: (tau, tau+h) both in test window, h<=7
te_idx = np.where(te_mask)[0]
res = {h: {"pers": [], "percell": [], "scalar": [], "permode": [], "diff": [], "diff2": [],
           "trend": [], "n": 0} for h in range(1, 8)}
Kspec = 100
for tau in te_idx:
    for h in range(1, 8):
        j = tau + h
        # find month months[tau]+h in months
        pos = np.searchsorted(months, months[tau] + h)
        if pos >= len(months) or months[pos] != months[tau] + h:
            continue
        x_tau = Xd[tau]  # NOTE: Xd uses full-window detrend; for honesty recompute below
        # honest anomaly at tau: use train-window mu/beta
        x_tau_h = X[tau] - mu_h - beta_h * (t[tau] - ttr.mean())
        x_true = X[j] - mu_h - beta_h * (t[j] - ttr.mean())
        # predictors (anomaly space)
        pers = x_tau_h
        percell = (phi_c ** h) * x_tau_h
        # spectral scalar phi=0.74 with K=100 denoise
        sc = Vh.T @ x_tau_h
        sc[Kspec:] = 0
        xdn = Vh @ sc
        scalar = xdn.copy()
        sc2 = sc.copy()
        sc2[:Kspec] = (0.74 ** h) * sc2[:Kspec]
        scalar = Vh @ sc2
        # per-mode
        scm = (Vh.T @ x_tau_h).copy()
        Km = len(phi_h)
        scm[:Km] = (phi_h[:Km] ** h) * scm[:Km]
        permode = Vh @ scm
        # diffusion powers (two compositions)
        xd_ = x_tau_h.copy()
        for _ in range(h):
            xd_ = rho_h * xd_ - Dh * (L @ xd_)
        diff = xd_
        xd2_ = x_tau_h.copy()
        Ah_ = sparse.identity(C) - (Dh / rho_h) * L
        for _ in range(h):
            xd2_ = rho_h * (Ah_ @ xd2_)
        diff2 = xd2_
        r = res[h]
        r["pers"].append(((pers - x_true) ** 2).mean())
        r["percell"].append(((percell - x_true) ** 2).mean())
        r["scalar"].append(((scalar - x_true) ** 2).mean())
        r["permode"].append(((permode - x_true) ** 2).mean())
        r["diff"].append(((diff - x_true) ** 2).mean())
        r["diff2"].append(((diff2 - x_true) ** 2).mean())
        r["trend"].append((x_true ** 2).mean())
        r["n"] += 1

print(f"{'h':>2} {'n_pairs':>7} {'trend0':>8} {'pers':>8} {'percell':>8} {'scalar.74':>9} "
      f"{'permode':>8} {'diffus':>8} {'diffus2':>8}")
for h in range(1, 8):
    r = res[h]
    if r["n"] == 0:
        continue
    g = lambda k: np.sqrt(np.mean(r[k]))
    print(f"{h:>2} {r['n']:>7} {g('trend'):>8.4f} {g('pers'):>8.4f} {g('percell'):>8.4f} "
          f"{g('scalar'):>9.4f} {g('permode'):>8.4f} {g('diff'):>8.4f} {g('diff2'):>8.4f}")

np.save("/home/z/my-project/scripts/a14a_cache_diff.npy",
        {"rho": rho, "D": D, "rho_h": rho_h, "Dh": Dh, "r2_l": r2_l, "lam": lam,
         "phi_k": phi_k, "r2_n": r2_n}, allow_pickle=True)
print("\nsaved a14a_cache_diff.npy")
