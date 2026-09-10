"""Exp 7: operator identification v2 + innovation structure + D-in-subspace.

1) Gap-corrected per-mode phi: only calendar-consecutive pairs (rule out the
   22 gap months contaminating score regressions).
2) Fast-only operator: remove slow modes (PC1-5 of detrended field), then
   (a) per-mode phi vs Laplacian eigenvalue (fit, R2, stability);
   (b) STAR ring regression on the fast field.
3) Innovation field structure: spectrum (rank), spatial corr length, kurtosis,
   share of innovation variance inside top-K PC subspace.
4) D-in-subspace: how much of D-hat (6-anchor mean anomaly) lies inside the
   train PC subspace (K=10/30/50/136)?
5) Global mode: innovation field global-mean std (is there a spatially uniform
   random component?).
"""
import numpy as np
import pandas as pd
import sys
from scipy import sparse
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, load_test, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)

df = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
lat = cells["lat"].values[full]; lon = cells["lon"].values[full]
key = {(a, b): i for i, (a, b) in enumerate(zip(lat, lon))}
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)
U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
V = Vt.T
S = U * s
Tm = V.shape[1]
ip = np.where(np.diff(months) == 1)[0]     # calendar-consecutive pairs

def wrap(b):
    if b > 179.5: return b - 360.0
    if b < -179.5: return b + 360.0
    return b

# graph Laplacian
rows, cols_, vals = [], [], []
for i, (a, b) in enumerate(zip(lat, lon)):
    nb = [(a + 1, b), (a - 1, b), (a, wrap(b + 1)), (a, wrap(b - 1))]
    deg = 0
    for (na, nbn) in nb:
        j = key.get((na, nbn))
        if j is not None:
            rows.append(i); cols_.append(j); vals.append(-1.0); deg += 1
    rows.append(i); cols_.append(i); vals.append(float(deg))
L = sparse.csr_matrix((vals, (rows, cols_)), shape=(C, C))

# ---------------- 1) gap-corrected per-mode phi ----------------
print("=== 1) per-mode phi: consecutive-only vs all-row pairs ===")
phi_c = np.zeros(Tm); phi_a = np.zeros(Tm)
for k in range(Tm):
    y = S[:, k]
    phi_c[k] = (S[ip, k] * S[ip + 1, k]).sum() / (S[ip, k] ** 2).sum()
    phi_a[k] = (y[:-1] * y[1:]).sum() / (y[:-1] ** 2).sum()
for k in [0, 1, 2, 3, 4, 5, 7, 9, 11, 15, 19, 25, 30, 39, 49, 69, 99, 120, 135]:
    print(f"  mode {k+1:>3}: phi_consec={phi_c[k]:+.3f}  phi_allrows={phi_a[k]:+.3f}")
d = phi_c - phi_a
print(f"  mean diff (consec - allrows) over modes 6-136: {d[5:].mean():+.4f}")
print(f"  negative-phi modes (consec): {int((phi_c < -0.17).sum())} of {Tm}")

# ---------------- 2) fast-only operator ----------------
print("\n=== 2) fast-only (PC1-5 removed) operator ===")
NSLOW = 5
Sf = S.copy(); Sf[:, :NSLOW] = 0.0
Xf = Sf @ V.T
phi_f = np.zeros(Tm)
for k in range(NSLOW, Tm):
    phi_f[k] = (Sf[ip, k] * Sf[ip + 1, k]).sum() / (Sf[ip, k] ** 2).sum()
lam = np.array([V[:, k] @ (L @ V[:, k]) for k in range(Tm)])
sel = np.arange(NSLOW, Tm)
w = Sf.var(axis=0)
def wlstsq(x, y, w):
    Xm = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(Xm * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    pred = Xm @ coef
    return coef, 1 - (w * (y - pred) ** 2).sum() / (w * (y - y.mean()) ** 2).sum()
coef_l, r2_l = wlstsq(lam[sel], phi_f[sel], w[sel])
print(f"  phi_k ~ rho - D*lam_k  (modes {NSLOW+1}-136, var-weighted): "
      f"rho={coef_l[0]:+.4f} D={-coef_l[1]:.5f} R2={r2_l:.4f}")
# restrict to well-estimated modes (var share)
thr = np.percentile(w[sel], 75)
m2 = sel[w[sel] > thr]
coef_l2, r2_l2 = wlstsq(lam[m2], phi_f[m2], w[m2])
print(f"  top-quartile-variance modes only: rho={coef_l2[0]:+.4f} D={-coef_l2[1]:.5f} R2={r2_l2:.4f}")
# quadratic fit
Xq = np.column_stack([np.ones(len(sel)), lam[sel], lam[sel] ** 2])
Wq = np.diag(w[sel])
cq = np.linalg.lstsq(Xq * np.sqrt(w[sel])[:, None], phi_f[sel] * np.sqrt(w[sel]), rcond=None)[0]
print(f"  quadratic: phi = {cq[0]:+.4f} {cq[1]:+.4f}*lam {cq[2]:+.4f}*lam^2")
print("  (mode, lam, phi_fast): ", [(int(k+1), round(float(lam[k]), 3), round(float(phi_f[k]), 3))
      for k in [5, 7, 9, 11, 14, 19, 25, 30, 40, 50, 70, 90, 110, 130, 135]])

# STAR on fast field
def ring_matrix(rlo, rhi):
    rows, cols_, cnt = [], [], np.zeros(C)
    for i, (a, b) in enumerate(zip(lat, lon)):
        for da in range(-rhi, rhi + 1):
            for db in range(-rhi, rhi + 1):
                d = max(abs(da), abs(db))
                if d <= rlo or d > rhi: continue
                j = key.get((a + da, wrap(b + db)))
                if j is not None:
                    rows.append(i); cols_.append(j); cnt[i] += 1
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols_)), shape=(C, C))
    return sparse.diags(1.0 / np.maximum(cnt, 1)) @ M
R1 = ring_matrix(0, 1); R2 = ring_matrix(1, 2); R3 = ring_matrix(2, 4)
y = Xf[ip + 1].ravel()
Xm = np.column_stack([Xf[ip].ravel(), (R1 @ Xf[ip].T).T.ravel(),
                      (R2 @ Xf[ip].T).T.ravel(), (R3 @ Xf[ip].T).T.ravel()])
coef = np.linalg.lstsq(Xm, y, rcond=None)[0]
resid = y - Xm @ coef
print(f"\n  STAR(fast): a0={coef[0]:+.4f} a1={coef[1]:+.4f} a2={coef[2]:+.4f} a3={coef[3]:+.4f}"
      f"  sum={coef.sum():+.4f}")
print(f"  resid std={resid.std():.4f}; var explained vs self-AR: "
      f"{1 - resid.var() / (y - coef[0]*Xm[:,0]).var():.4f}")

# ---------------- 3) innovation field structure ----------------
print("\n=== 3) innovation field W (per-mode AR resid on consecutive pairs) ===")
# use fast field; innovations live in cell space
Wmat = Xf[ip + 1] - (phi_f[None, :] * Sf[ip]) @ V.T    # (npairs, C)
Wc = Wmat - Wmat.mean(axis=0, keepdims=True)
Uw, sw, _ = np.linalg.svd(Wc, full_matrices=False)
vw = sw ** 2 / (C * (len(Wc) - 1))
print(f"  per-cell innovation variance: {vw.sum():.4f} (std {np.sqrt(vw.sum()):.4f})")
cum = np.cumsum(vw) / vw.sum()
for k in [1, 5, 10, 20, 50, 100]:
    print(f"    W spectrum: PC1-{k} = {100*cum[k-1]:.1f}%")
def spat(Fm, dl, do):
    cs = []
    for i, (a, b) in enumerate(zip(lat, lon)):
        j = key.get((a + dl, wrap(b + do)))
        if j is not None:
            cs.append(np.corrcoef(Fm[:, i], Fm[:, j])[0, 1])
    return np.mean(cs)
print("  W spatial corr: ", " ".join(f"d({dl},{do})={spat(Wc, dl, do):.3f}"
      for dl, do in [(1, 0), (0, 1), (2, 0), (5, 0), (10, 0)]))
# share of innovation variance inside TWS top-K subspace
proj = (Wc @ V[:, :100])
print(f"  share of W var inside TWS top-100 PC subspace: {(proj**2).sum()/(Wc**2).sum():.4f}")
proj50 = (Wc @ V[:, :50])
print(f"  share inside top-50: {(proj50**2).sum()/(Wc**2).sum():.4f}")
kw = (Wc ** 4).mean() / (Wc ** 2).mean() ** 2
print(f"  W pooled kurtosis={kw:.3f}")
# global mode
gm = Wc.mean(axis=1)
print(f"  W global-mean (uniform mode) std={gm.std():.4f} "
      f"(vs per-cell std {np.sqrt(vw.sum()):.4f}); corr(gm_t, gm_t+1)="
      f"{np.corrcoef(gm[:-1], gm[1:])[0,1]:+.3f}")

# ---------------- 4) D-in-subspace ----------------
print("\n=== 4) D-hat inside train PC subspace? ===")
dte = load_test(cols=("time", "lat", "lon", "TWS_t", "TWS_t_masked"))
dte["unmask"] = ~dte["TWS_t_masked"].astype(bool)
tms = dte.groupby("t_abs")["unmask"].mean()
anchors = sorted(tms[tms > 0.9].index.tolist())
allB = np.full((len(anchors), C), np.nan)
for j, ta in enumerate(anchors):
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    lut = dict(zip(zip(sub["lat"].values, sub["lon"].values), sub["TWS_t"].values))
    for idx, (la, lo) in enumerate(zip(lat, lon)):
        if (la, lo) in lut:
            allB[j, idx] = lut[(la, lo)] - mu[idx]
Dhat = np.nanmean(allB, axis=0)
ok = ~np.isnan(Dhat)
Dv_ = np.where(ok, Dhat, 0.0)
for K in [10, 30, 50, 136]:
    proj = V[:, :K] @ (V[:, :K].T @ Dv_)
    print(f"  K={K:>3}: var(Dhat) in-subspace = {(proj[ok]**2).sum()/(Dv_[ok]**2).sum():.4f}")
# trend part of D
trendex = beta * (24207 - t.mean())     # mid test era
residD = Dv_ - 0.53 * trendex           # a=0.53 from a14c
print(f"  corr(Dhat, mu)={np.corrcoef(Dhat[ok], mu[ok])[0,1]:+.3f}; "
      f"corr(Dhat-0.53*trendex, mu)={np.corrcoef(residD[ok], mu[ok])[0,1]:+.3f}")
print(f"  std(Dhat)={Dhat[ok].std():.4f}; std(resid after 0.53*trendex)={residD[ok].std():.4f}")
# is the D residual correlated with the cov STATIC field? (via -mu channel)
print("done")
