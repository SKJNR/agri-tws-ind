"""Exp 4b: STAR (spatio-temporal AR) regression — basis-free identification of the
generator's update operator.

x_c(t+1) = a0*x_c(t) + a1*nb1(t) + a2*ring5(t) + a3*ring9(t) + w_c(t+1)

- If a1..a3 ~ 0  -> per-cell AR(1) + spatially correlated innovations.
- If a0 + neighbor stencil ~ diffusion -> local PDE update
  (e.g. x(t+1) = rho*[ (1-4D)x + D*sum_nb ] + w  =>  a0=rho(1-4D), a1=4*rho*D).
Also: honest holdout with the STAR operator iterated h=1..7; per-cell variance map.
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
key = {(a, b): i for i, (a, b) in enumerate(zip(lat, lon))}

def wrap(b):
    if b > 179.5: return b - 360.0
    if b < -179.5: return b + 360.0
    return b

def ring_matrix(radius_lo, radius_hi):
    """mean over cells with Chebyshev distance in (lo, hi] box ring around each cell"""
    rows, cols_, cnt = [], [], np.zeros(C)
    for i, (a, b) in enumerate(zip(lat, lon)):
        for da in range(-radius_hi, radius_hi + 1):
            for db in range(-radius_hi, radius_hi + 1):
                d = max(abs(da), abs(db))
                if d <= radius_lo or d > radius_hi:
                    continue
                j = key.get((a + da, wrap(b + db)))
                if j is not None:
                    rows.append(i); cols_.append(j); cnt[i] += 1
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols_)), shape=(C, C))
    M = sparse.diags(1.0 / np.maximum(cnt, 1)) @ M
    return M

mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)

R1 = ring_matrix(0, 1)    # 4 adjacent neighbors
R2 = ring_matrix(1, 2)    # 5x5 ring
R3 = ring_matrix(2, 4)    # 9x9 ring (up to 4deg)

ip = np.where(np.diff(months) == 1)[0]   # consecutive calendar months

# ---- pooled STAR regression ----
y = Xd[ip + 1].ravel()
feats = [Xd[ip], (R1 @ Xd[ip].T).T, (R2 @ Xd[ip].T).T, (R3 @ Xd[ip].T).T]
Xm = np.column_stack([f.ravel() for f in feats])
coef, res_, rank_, sv_ = np.linalg.lstsq(Xm, y, rcond=None)
resid = y - Xm @ coef
names = ["a0 (self)", "a1 (nb 1deg)", "a2 (ring 2deg)", "a3 (ring 3-4deg)"]
# standard errors
n, p = Xm.shape
sigma2 = (resid ** 2).sum() / (n - p)
XtX_inv = np.linalg.inv(Xm.T @ Xm)
se = np.sqrt(np.diag(XtX_inv) * sigma2)
print("=== pooled STAR regression (global coefficients) ===")
for nm, c, s in zip(names, coef, se):
    print(f"  {nm:>16}: {c:+.5f} ± {s:.5f}  (t={c/s:+.1f})")
print(f"  resid var={resid.var():.5f} (std {resid.std():.4f}); "
      f"var explained vs persistence-only: {1 - resid.var()/ (y - Xm[:,0]*coef[0]).var():.4f}")
r2_vs_ar = 1 - resid.var() / (y - coef[0] * Xm[:, 0]).var()
print(f"  STAR vs pure-AR improvement in resid var: {r2_vs_ar:.4f}")

# spatial residual structure
W = resid.reshape(len(ip), C)
def spat_corr(Fm, dlat, dlon):
    cs = []
    for i, (a, b) in enumerate(zip(lat, lon)):
        j = key.get((a + dlat, wrap(b + dlon)))
        if j is not None:
            cs.append(np.corrcoef(Fm[:, i], Fm[:, j])[0, 1])
    return np.mean(cs)
print("\nSTAR residual W spatial corr:", end="")
for (dl, do) in [(1, 0), (0, 1), (2, 0), (5, 0), (10, 0)]:
    print(f" d({dl},{do})={spat_corr(W, dl, do):.3f}", end="")
print()
tt = months[ip + 1]
r1w = [ (W[i]*W[i+1]).sum()/np.sqrt((W[i]**2).sum()*(W[i+1]**2).sum())
        for i in range(len(tt)-1) if tt[i+1]-tt[i]==1 ]
print(f"STAR residual temporal corr (consecutive): mean={np.mean(r1w):+.4f}")
kw = (W - W.mean(0))
print(f"STAR residual per-cell kurtosis mean={( (kw**4).mean(0)/kw.var(0)**2 ).mean():.3f}")

# ---- implied operator interpretation ----
a0, a1, a2, a3 = coef
print("\n=== implied operator ===")
print(f"sum of coefficients (response to spatially uniform field) = {coef.sum():+.5f}"
      f"  (phi of the zero/largest mode)")
print(f"self coefficient a0 = {a0:+.5f} (negative => fine-scale anti-persistence)")
print(f"diffusion interpretation: rho*(1-4D)=a0, 4*rho*D=a1  => rho={a0+a1:.4f}, "
      f"D={a1/(4*max(a0+a1,1e-9)):.4f}")

# ---- honest holdout with STAR operator ----
print("\n=== holdout: fit STAR on 2002-2012, predict 2013-2015 ===")
tr = months < 2013 * 12
Xtr = X[tr]; ttr = t[tr]; tctr = ttr - ttr.mean()
mu_h = Xtr.mean(0); Atr = Xtr - mu_h
beta_h = (Atr * tctr[:, None]).sum(0) / (tctr ** 2).sum()
Xdtr = Atr - np.outer(tctr, beta_h)
ip_tr = np.where(np.diff(months[tr]) == 1)[0]
y_tr = Xdtr[ip_tr + 1].ravel()
feats = [Xdtr[ip_tr], (R1 @ Xdtr[ip_tr].T).T, (R2 @ Xdtr[ip_tr].T).T, (R3 @ Xdtr[ip_tr].T).T]
Xm_tr = np.column_stack([f.ravel() for f in feats])
coef_tr = np.linalg.lstsq(Xm_tr, y_tr, rcond=None)[0]
print("train-only STAR coefs:", coef_tr)
# per-cell AR baseline
phi_c = (Xdtr[ip_tr] * Xdtr[ip_tr + 1]).sum(0) / (Xdtr[ip_tr] ** 2).sum(0)

res = {h: {} for h in range(1, 8)}
for tau in np.where(~tr)[0]:
    for h in range(1, 8):
        pos = np.searchsorted(months, months[tau] + h)
        if pos >= len(months) or months[pos] != months[tau] + h:
            continue
        x_tau = X[tau] - mu_h - beta_h * (t[tau] - ttr.mean())
        x_true = X[pos] - mu_h - beta_h * (t[pos] - ttr.mean())
        # STAR iterated
        x_ = x_tau.copy()
        for _ in range(h):
            x_ = coef_tr[0] * x_ + coef_tr[1] * (R1 @ x_) + coef_tr[2] * (R2 @ x_) + coef_tr[3] * (R3 @ x_)
        # per-cell AR^h
        pc = (phi_c ** h) * x_tau
        res[h].setdefault("star", []).append(((x_ - x_true) ** 2).mean())
        res[h].setdefault("percell", []).append(((pc - x_true) ** 2).mean())
        res[h].setdefault("pers", []).append(((x_tau - x_true) ** 2).mean())
        res[h].setdefault("trend", []).append((x_true ** 2).mean())
        # slow-only STAR (zero mode kept): predict with uniform-response operator
print(f"{'h':>2} {'n':>3} {'trend0':>8} {'pers':>8} {'percellAR':>10} {'STAR':>8}")
for h in range(1, 8):
    if not res[h]:
        continue
    g = lambda k: np.sqrt(np.mean(res[h][k]))
    n = len(res[h]["pers"])
    print(f"{h:>2} {n:>3} {g('trend'):>8.4f} {g('pers'):>8.4f} {g('percell'):>10.4f} {g('star'):>8.4f}")

# ---- per-cell variance map (heterogeneity) ----
pv = Xd.var(axis=0, ddof=1)
print("\n=== per-cell variance structure ===")
print(f"corr(var, |lat|)={np.corrcoef(pv, np.abs(lat))[0,1]:+.3f}  "
      f"corr(var, lat)={np.corrcoef(pv, lat)[0,1]:+.3f}")
q20, q80 = np.percentile(pv, [20, 80])
lo = pv < q20; hi = pv > q80
print(f"low-var cells mean |lat|={np.abs(lat[lo]).mean():.1f}; high-var cells mean |lat|={np.abs(lat[hi]).mean():.1f}")
# spatial smoothness of the variance field
vfield = np.sqrt(pv)
cs = []
for i, (a, b) in enumerate(zip(lat, lon)):
    j = key.get((a + 1, wrap(b)))
    if j is not None: cs.append(abs(vfield[i] - vfield[j]))
print(f"neighbor |diff| of per-cell std: mean={np.mean(cs):.4f} vs overall std={vfield.std():.4f} "
      f"(smooth if <<)")
np.save("/home/z/my-project/scripts/a14a_cache_star.npy",
        {"coef": coef, "coef_tr": coef_tr, "pv": pv, "lat": lat, "lon": lon}, allow_pickle=True)
print("saved a14a_cache_star.npy")
