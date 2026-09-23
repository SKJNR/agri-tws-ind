"""Exp 6+7: COVARIATE GENERATOR FORENSICS.

Questions:
A) Do covariates have seasonality? (per-cell monthly climatology variance share,
   harmonic amplitudes) — and why do month_sin/month_cos columns exist?
B) Are covariates generated from the SAME latent factors as TWS?
   - per-mode correlation profile r_k = corr(cov score_k, TWS detrended score_k)
   - loading matrix diagonal or dense? (cross-mode correlation matrix)
   - field-level reconstruction: R^2 of TWS anomaly from cov fields (per-mode
     regression, honest 2002-2012 fit -> 2013-15 eval)
C) Internal consistency: SPEI_03 ~ 3-mo MA of SPEI_01? SPEI_06/12 similar?
D) SOIL_MOISTURE <-> TWS: per-cell R^2; lead structure (does SOIL_t see TWS_{t+1}?)
E) Innovation observability: does the cov field at month t observe the TWS
   innovation AT month t?  (per-mode corr of cov score with AR residual)
F) Cov field spectra: white-noise floor? rank profile vs TWS.
G) Test-set structure: which rows have covariates at t+1 available?
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, load_test, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)
COVS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]

# ---------------- load ----------------
df = load_train(cols=("time", "lat", "lon", "TWS_t") + tuple(COVS))
cells, months = build_grid(df)
F = field_matrix(df, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]                          # (T, Cf)
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)             # detrended anomaly
print(f"T={T} C={C}")

# TWS PC basis (cell space), full 136 modes
U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
V = Vt.T                                # (C, Tm)
S = U * s                               # (T, Tm) TWS scores
Tm = V.shape[1]

# cov fields on full-history cells
Dv = {}
for c in COVS:
    Fc = field_matrix(df, c, cells, months)[:, full]
    Dv[c] = Fc - np.nanmean(Fc, axis=0, keepdims=True)   # deviation from per-cell mean

# ---------------- A) seasonality of covariates ----------------
print("\n=== A) seasonality: per-cell monthly climatology variance share ===")
moy = (months % 12) + 1
for c in COVS + ["TWS_t"]:
    Fld = Xd if c == "TWS_t" else Dv[c]
    num = np.zeros(C); den = np.zeros(C)
    for m in range(1, 13):
        sel = moy == m
        mm = Fld[sel].mean(axis=0)
        num += (sel.sum() * mm ** 2)
        den += (Fld[sel] ** 2).sum(axis=0)
    share = num / np.maximum(den, 1e-12)
    print(f"  {c:>16}: monthly-clim variance share mean={share.mean():.4f} "
          f"p50={np.median(share):.4f} p95={np.percentile(share, 95):.4f}")

# ---------------- F) cov field spectra ----------------
print("\n=== F) cov deviation-field spectra (variance units) vs TWS ===")
print(f"{'field':>16} {'v1':>8} {'v1-10%':>8} {'v1-50%':>8} {'v1-100%':>9} {'sum':>8} "
      f"{'v_136':>10} {'floor_sig':>9}")
spec = {}
for c in COVS:
    Uc, sc, _ = np.linalg.svd(Dv[c], full_matrices=False)
    vc = sc ** 2 / (C * (T - 1))
    spec[c] = vc
    cum = np.cumsum(vc) / vc.sum()
    # white noise floor: bottom eigenvalues (after detrend rank loss, use 130..135)
    floor = vc[125:135].mean() * (T - 1)   # sigma^2 if flat
    print(f"{c:>16} {vc[0]:>8.4f} {100*cum[9]:>8.2f} {100*cum[49]:>8.2f} "
          f"{100*cum[99]:>9.2f} {vc.sum():>8.4f} {vc[135]:>10.6f} {np.sqrt(max(floor,0)):>9.4f}")
vc = s ** 2 / (C * (T - 1))
print(f"{'TWS detrended':>16} {vc[0]:>8.4f} {100*np.cumsum(vc)[9]/vc.sum():>8.2f} "
      f"{100*np.cumsum(vc)[49]/vc.sum():>8.2f} {100*np.cumsum(vc)[99]/vc.sum():>9.2f} "
      f"{vc.sum():>8.4f} {vc[135]:>10.6f} {np.sqrt(vc[125:135].mean()*(T-1)):>9.4f}")

# ---------------- B) shared latent structure ----------------
print("\n=== B) per-mode corr r_k = corr(cov score_k(t), TWS detrended score_k(t)) ===")
Zs = {}
for c in COVS:
    Zs[c] = Dv[c] @ V                     # (T, Tm) cov scores in TWS basis
print(f"{'mode':>4} {'varTWS':>8} " + " ".join(f"{c[:8]:>9}" for c in COVS) + f" {'all5_R2':>8}")
rmat = {c: np.zeros(Tm) for c in COVS}
for k in range(Tm):
    for c in COVS:
        rmat[c][k] = np.corrcoef(Zs[c][:, k], S[:, k])[0, 1]
for k in list(range(0, 30)) + [39, 49, 59, 79, 99, 135]:
    # combined: multi-cov regression of S[:,k] on Zs[all][:,k]
    Zm = np.column_stack([Zs[c][:, k] for c in COVS])
    coef = np.linalg.lstsq(Zm, S[:, k], rcond=None)[0]
    r2 = 1 - ((S[:, k] - Zm @ coef) ** 2).mean() / S[:, k].var()
    print(f"{k+1:>4} {S[:,k].var():>8.2f} " + " ".join(f"{rmat[c][k]:>9.3f}" for c in COVS)
          + f" {r2:>8.3f}")
print("r_k profile summary (modes 2-50): " + " | ".join(
    f"{c[:8]}: mean={rmat[c][1:50].mean():.3f} min={rmat[c][1:50].min():.3f}" for c in COVS))

# loading matrix: diagonal or dense? cross-mode corr for top 20 modes (SOIL)
k20 = 20
Cm = np.zeros((k20, k20))
for i in range(k20):
    for j in range(k20):
        Cm[i, j] = np.corrcoef(Zs["SOIL_MOISTURE_t"][:, i], S[:, j])[0, 1]
off = Cm - np.diag(np.diag(Cm))
print(f"\nSOIL loading matrix (top-20 modes): diag mean={np.diag(Cm).mean():.3f}, "
      f"offdiag |.| mean={np.abs(off).mean():.3f} max={np.abs(off).max():.3f}")

# field-level reconstruction (honest: fit on 2002-2012, eval 2013-15)
tr = months < 2013 * 12
te = ~tr
print("\n=== B2) field-level reconstruction of TWS detrended anomaly from cov fields ===")
print("(per-mode least squares, fit 2002-2012, eval 2013-2015; K = mode cutoff)")
fastpart = S.copy()
fastpart[:, :5] = 0.0            # remove slow modes PC1-5 for 'fast' metric
Xfast = fastpart @ V.T           # fast component of detrended anomaly
var_full_tr = (Xd[tr] ** 2).mean()
var_fast_tr = (Xfast[tr] ** 2).mean()
var_full_te = (Xd[te] ** 2).mean()
var_fast_te = (Xfast[te] ** 2).mean()
print(f"train var: full={var_full_tr:.4f} fast={var_fast_tr:.4f}; "
      f"test win var: full={var_full_te:.4f} fast={var_fast_te:.4f}")
for K in [10, 30, 50, 100, Tm]:
    Ztr = np.column_stack([Zs[c][tr, :K] for c in COVS])       # (ntr, 5K)
    Str = S[tr, :K]
    W = np.linalg.lstsq(Ztr, Str, rcond=None)[0]               # (5K, K)
    # eval
    Zte = np.column_stack([Zs[c][te, :K] for c in COVS])
    pred_scores = Zte @ W                                        # (nte, K)
    Xhat = pred_scores @ V[:, :K].T                              # (nte, C)
    r2_full = 1 - ((Xd[te] - Xhat) ** 2).mean() / var_full_te
    fh = np.zeros_like(Xhat)
    # fast-only reconstruction: zero out first 5 modes in pred and in target
    pred_scores_f = pred_scores.copy(); pred_scores_f[:, :5] = 0
    Xhat_f = pred_scores_f @ V[:, :K].T
    r2_fast = 1 - ((Xfast[te] - Xhat_f) ** 2).mean() / var_fast_te
    rmse = np.sqrt(((Xd[te] - Xhat) ** 2).mean())
    print(f"  K={K:>3}: R2(full)={r2_full:.4f}  R2(fast,PC6+)={r2_fast:.4f}  "
          f"RMSE(full)={rmse:.4f}")

# ---------------- C) SPEI internal consistency ----------------
print("\n=== C) SPEI MA structure: corr(SPEI_{m}, mean of SPEI_01 over window) ===")
F1 = Dv["SPEI_01_t"]
def ma_field(field, w):
    """calendar-correct trailing MA of width w (NaN when any month missing)"""
    out = np.full_like(field, np.nan)
    for i in range(len(months)):
        need = months[i] - np.arange(w - 1, -1, -1)
        idx = [np.searchsorted(months, m) if np.searchsorted(months, m) < len(months)
               and months[np.searchsorted(months, m)] == m else -1 for m in need]
        if all(ix >= 0 for ix in idx):
            out[i] = field[idx].mean(axis=0)
    return out
for w, cname in [(3, "SPEI_03_t"), (6, "SPEI_06_t"), (12, "SPEI_12_t")]:
    ma = ma_field(F1, w)
    ok = ~np.isnan(ma[:, 0])
    Fc = Dv[cname]
    # per-cell correlation
    cc = np.array([np.corrcoef(ma[ok, j], Fc[ok, j])[0, 1] for j in range(0, C, 7)])
    # pooled regression slope
    slope = (ma[ok] * Fc[ok]).sum() / (ma[ok] ** 2).sum()
    resid = Fc[ok] - slope * ma[ok]
    r2 = 1 - resid.var() / Fc[ok].var()
    print(f"  {cname} ~ MA{w}(SPEI_01): per-cell corr mean={np.nanmean(cc):.4f} "
          f"p5={np.nanpercentile(cc,5):.3f} p95={np.nanpercentile(cc,95):.3f}; "
          f"pooled slope={slope:.3f} R2={r2:.4f} resid_std={resid.std():.4f} "
          f"(cov std {Fc[ok].std():.4f})")

# ---------------- D) SOIL <-> TWS coupling, lead structure ----------------
print("\n=== D) SOIL vs TWS: contemporaneous + lead structure ===")
Fs = Dv["SOIL_MOISTURE_t"]
# per-cell contemporaneous R^2
r2c = np.array([1 - ((Xd[:, j] - np.polyval(np.polyfit(Fs[:, j], Xd[:, j], 1), Fs[:, j]))**2).mean()
                / Xd[:, j].var() for j in range(0, C, 7)])
print(f"per-cell R2(TWS_anom ~ SOIL_anom): mean={r2c.mean():.4f} p50={np.median(r2c):.4f} "
      f"p95={np.percentile(r2c,95):.4f}")
ip = np.where(np.diff(months) == 1)[0]      # consecutive calendar pairs (t -> t+1)
# does SOIL(t) predict TWS(t+1) beyond TWS(t)? pooled per-cell regression
y = Xd[ip + 1].ravel()
Xm = np.column_stack([Xd[ip].ravel(), Fs[ip].ravel(), np.ones(T * 0 + y.size)])
coef = np.linalg.lstsq(Xm[:, :2], y, rcond=None)[0]
resid = y - Xm[:, :2] @ coef
resid0 = y - coef[0] * Xm[:, 0]
print(f"TWS(t+1) ~ TWS(t) + SOIL(t): coef TWS={coef[0]:+.4f} SOIL={coef[1]:+.4f}; "
      f"var reduction from SOIL = {1 - resid.var()/resid0.var():.4f}")
# reverse: TWS(t) predicting SOIL(t+1) beyond SOIL(t)
y2 = Fs[ip + 1].ravel()
Xm2 = np.column_stack([Fs[ip].ravel(), Xd[ip].ravel()])
coef2 = np.linalg.lstsq(Xm2, y2, rcond=None)[0]
r2b = 1 - (y2 - Xm2 @ coef2).var() / (y2 - coef2[0] * Xm2[:, 0]).var()
print(f"SOIL(t+1) ~ SOIL(t) + TWS(t): TWS adds var reduction = {r2b:.4f}")

# ---------------- E) innovation observability ----------------
print("\n=== E) does the cov field at month t observe the TWS innovation at t? ===")
# per-mode AR(1) on consecutive pairs only
phi_k = np.zeros(Tm)
for k in range(Tm):
    yv = S[ip + 1, k]; xl = S[ip, k]
    phi_k[k] = (xl * yv).sum() / (xl ** 2).sum()
innov = np.full((T, Tm), np.nan)
innov[ip + 1] = S[ip + 1] - phi_k[None, :] * S[ip]   # innovation at month ip+1
okr = ~np.isnan(innov[:, 0])
print(f"{'mode':>4} {'phi':>7} {'innovvar':>9} " + " ".join(f"{c[:8]:>9}" for c in COVS))
gam = {c: np.zeros(Tm) for c in COVS}
for k in list(range(0, 12)) + [14, 19, 24, 29, 39, 49]:
    row = [f"{k+1:>4}", f"{phi_k[k]:>7.3f}", f"{innov[okr,k].var():>9.3f}"]
    for c in COVS:
        gam[c][k] = np.corrcoef(Zs[c][okr, k], innov[okr, k])[0, 1]
        row.append(f"{gam[c][k]:>9.3f}")
    print(" ".join(row))
print("gamma profile (modes 2-50): " + " | ".join(
    f"{c[:8]}: mean={gam[c][1:50].mean():+.3f}" for c in COVS))

# ---------------- G) test-set structure: covs(t+1) availability ----------------
print("\n=== G) test rows with next-month row (covs at t+1) available ===")
dte = load_test(cols=("ID", "time", "lat", "lon", "TWS_t_masked"))
dte["masked"] = dte["TWS_t_masked"].astype(bool)
tm = dte.groupby("t_abs")["masked"].agg(["mean", "count"])
test_months = np.sort(dte["t_abs"].unique())
nxt = {m: (m + 1 in test_months) for m in test_months}
dte["has_next"] = dte["t_abs"].map(nxt)
for mk in [False, True]:
    sub = dte[dte["masked"] == mk]
    hn = sub["has_next"].mean()
    print(f"  masked={mk}: n={len(sub):>6}, frac with covs(t+1) available = {hn:.4f}")
print(f"  total rows: {len(dte)}, overall frac with covs(t+1) = {dte['has_next'].mean():.4f}")

# ---------------- month_sin/month_cos sanity ----------------
print("\nmonth_sin/cos check: first rows")
d2 = pd.read_csv("/home/z/my-project/data/Train (1).csv", nrows=3,
                 usecols=["time", "month_sin", "month_cos"])
for _, r in d2.iterrows():
    m = pd.to_datetime(r["time"]).month
    print(f"  time={r['time']} month={m}: month_sin={r['month_sin']:.4f} "
          f"sin(2pi(m-1)/12)={np.sin(2*np.pi*(m-1)/12):.4f} "
          f"month_cos={r['month_cos']:.4f} cos={np.cos(2*np.pi*(m-1)/12):.4f}")
print("\ndone")
