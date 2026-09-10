"""Exp 5: THE REGIME-CHANGE TEST — does the test era continue the train process?

Decisive question: is the test-era field the CONTINUATION of the train-era
latent process (no regime change), i.e.
    TWS(c, t) = mu_c + beta_c*(t-tbar) + sum_k L_ck f_k(t),  f_k = AR(1)(phi_k)
extrapolated from the train end (2015-08)?  If yes, the mysterious test-era
offset "D" is just the slow modes' continuation + window-mean offset, and the
optimal predictor fuses train-end extrapolation with anchors/covariates.

Tests:
 A) Predict each of the 6 anchors (and 12 partial months) from train-end:
    [trend-only, persistence (phi^0), per-mode phi^D, slow-only (top-K) phi^D,
     PC1-persistence variants]  -> RMSE, R^2, corr.
 B) Compare with D-hat (LOO anchor mean) — the current V4 slow-state estimator.
 C) Residual structure: corr with mu_c, spatial smoothness, cross-anchor corr.
 D) Anchor->anchor chains with per-mode extrapolation.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from a14a_common import load_train, load_test, build_grid, field_matrix

np.set_printoptions(precision=4, suppress=True)

dtr = load_train(cols=("time", "lat", "lon", "TWS_t"))
cells, months = build_grid(dtr)
cmap = {(a, b): i for i, (a, b) in enumerate(zip(cells["lat"], cells["lon"]))}
F = field_matrix(dtr, "TWS_t", cells, months)
full = ~np.isnan(F).any(axis=0)
X = F[:, full]                       # (138, 14558)
T, C = X.shape
t = months.astype(float); tc = t - t.mean()
mu = X.mean(axis=0)
A = X - mu
beta = (A * tc[:, None]).sum(axis=0) / (tc ** 2).sum()
Xd = A - np.outer(tc, beta)
U, s, Vt = np.linalg.svd(Xd, full_matrices=False)
V = Vt.T
scores = U * s
phi_k = np.zeros(T)
for k in range(T):
    y = scores[:, k]
    phi_k[k] = (y[:-1] * y[1:]).sum() / (y[:-1] ** 2).sum()

# ---------------- test anchors + partial months ----------------
dte = load_test(cols=("time", "lat", "lon", "TWS_t", "TWS_t_masked"))
dte["unmask"] = ~dte["TWS_t_masked"].astype(bool)
tmask_stats = dte.groupby("t_abs")["unmask"].agg(["mean", "sum"])
anchors = tmask_stats[tmask_stats["mean"] > 0.9].index.values
partials = tmask_stats[(tmask_stats["mean"] > 0) & (tmask_stats["mean"] <= 0.9)].index.values
print(f"anchor months (t_abs): {sorted(anchors.tolist())}")
print(f"partial months: {sorted(partials.tolist())} (unmasked counts: "
      f"{[int(tmask_stats.loc[m,'sum']) for m in sorted(partials)]})")

t_end = months[-1]                   # 2015-08
x_end = Xd[-1]                       # detrended anomaly at train end (full-window mu/beta)
lat_f = cells["lat"].values[full]; lon_f = cells["lon"].values[full]
full_cells = set(zip(lat_f, lon_f))
cidx_full = {c: i for i, c in enumerate(full_cells)}

def project_modes(x, K):
    sc = V[:, :K].T @ x
    return sc

def predict_from(x_tau, delta, mode="permodes", K=None, phi_override=None):
    """predict anomaly delta months ahead from anomaly x_tau"""
    if mode == "permodes":
        K = T if K is None else K
        sc = V[:, :K].T @ x_tau
        sc = (phi_k[:K] ** delta) * sc
        return V[:, :K] @ sc
    if mode == "slowpermodes":   # slow modes per-mode, fast tail -> 0
        sc = V[:, :K].T @ x_tau
        sc = (phi_k[:K] ** delta) * sc
        return V[:, :K] @ sc
    if mode == "persistence":
        return x_tau.copy()
    if mode == "pc1persist":     # PC1 persistence, others per-mode
        sc = V.T @ x_tau
        d = phi_k.copy() ** delta
        d[0] = 1.0
        return V @ (d * sc)
    if mode == "phi_global":     # all modes decay at single phi
        phi = phi_override
        return V @ ((phi ** delta) * (V.T @ x_tau))
    raise ValueError(mode)

print("\n=== A) predict anchors from train-end (2015-08), full-history cells ===")
print(f"{'anchor':>8} {'D':>3} {'n':>6} {'trend0':>8} {'persist':>8} {'permode':>8} "
      f"{'slowK10':>8} {'pc1pers':>8} {'phi.98':>8} {'anom_sd':>8} {'corr_pm':>7}")
for ta in sorted(anchors):
    delta = ta - t_end
    preds = {
        "trend0": predict_from(x_end, delta, "phi_global", phi_override=0.0),
        "persist": predict_from(x_end, delta, "persistence"),
        "permode": predict_from(x_end, delta, "permodes"),
        "slowK10": predict_from(x_end, delta, "slowpermodes", K=10),
        "pc1pers": predict_from(x_end, delta, "pc1persist"),
        "phi.98": predict_from(x_end, delta, "phi_global", phi_override=0.98),
    }
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    ii = [cidx_full[(la, lo)] for la, lo in zip(sub["lat"].values, sub["lon"].values)
          if (la, lo) in cidx_full]
    yv = [tws for (la, lo), tws in zip(zip(sub["lat"].values, sub["lon"].values),
                                       sub["TWS_t"].values) if (la, lo) in cidx_full]
    ii = np.array(ii); yv = np.array(yv)
    a_true = yv - mu[ii] - beta[ii] * (ta - t.mean())
    row = [f"{ta:>8}", f"{delta:>3}", f"{len(ii):>6}"]
    for nm in ["trend0", "persist", "permode", "slowK10", "pc1pers", "phi.98"]:
        p = preds[nm][ii]
        row.append(f"{np.sqrt(np.mean((p - a_true) ** 2)):>8.4f}")
    row.append(f"{a_true.std():>8.4f}")
    p = preds["permode"][ii]
    row.append(f"{np.corrcoef(p, a_true)[0,1]:>7.3f}")
    print(" ".join(row))

# ---------------- B) D-hat LOO baseline on anchors ----------------
print("\n=== B) anchor LOO: predict anchor j from mean of other anchors (D-hat) vs train-end ===")
anch_list = sorted(anchors)
anch_fields = {}
for ta in anch_list:
    sub = dte[(dte["t_abs"] == ta) & dte["unmask"]]
    ii = [cidx_full[(la, lo)] for la, lo in zip(sub["lat"].values, sub["lon"].values)
          if (la, lo) in cidx_full]
    yv = [tws for (la, lo), tws in zip(zip(sub["lat"].values, sub["lon"].values),
                                       sub["TWS_t"].values) if (la, lo) in cidx_full]
    ii = np.array(ii); yv = np.array(yv)
    a_true = yv - mu[ii] - beta[ii] * (ta - t.mean())
    anch_fields[ta] = (ii, a_true)

print(f"{'anchor':>8} {'n':>6} {'DhatLOO':>8} {'permode':>8} {'slowK10':>8} {'pc1pers':>8}")
for ta in anch_list:
    ii, a_true = anch_fields[ta]
    others = [a for a in anch_list if a != ta]
    # D-hat: mean anomaly of other anchors on the same cells
    dh = np.zeros(len(ii)); cnt = 0
    for ta2 in others:
        ii2, a2 = anch_fields[ta2]
        m = np.isin(ii, ii2)
        # align
        lookup = dict(zip(ii2, a2))
        vals = np.array([lookup.get(c, np.nan) for c in ii])
        dh += np.nan_to_num(vals)
        cnt += np.sum(~np.isnan(vals))
    dh /= np.maximum(cnt, 1)
    delta = ta - t_end
    pm = predict_from(x_end, delta, "permodes")[ii]
    sl = predict_from(x_end, delta, "slowpermodes", K=10)[ii]
    pc = predict_from(x_end, delta, "pc1persist")[ii]
    print(f"{ta:>8} {len(ii):>6} {np.sqrt(np.mean((dh - a_true)**2)):>8.4f} "
          f"{np.sqrt(np.mean((pm - a_true)**2)):>8.4f} {np.sqrt(np.mean((sl - a_true)**2)):>8.4f} "
          f"{np.sqrt(np.mean((pc - a_true)**2)):>8.4f}")

# ---------------- C) residual structure of the best predictor ----------------
print("\n=== C) residual structure (permode predictor from train-end) ===")
for ta in anch_list:
    ii, a_true = anch_fields[ta]
    delta = ta - t_end
    pm = predict_from(x_end, delta, "permodes")[ii]
    res = a_true - pm
    lat_a = lat_f[ii]
    print(f"anchor {ta} (D={delta}): resid std={res.std():.4f} corr(res, mu)={np.corrcoef(res, mu[ii])[0,1]:+.3f} "
          f"corr(res, a_true)={np.corrcoef(res, a_true)[0,1]:+.3f} corr(pm, a_true)={np.corrcoef(pm, a_true)[0,1]:+.3f}")

# cross-anchor residual correlation
resids = {}
for ta in anch_list:
    ii, a_true = anch_fields[ta]
    pm = predict_from(x_end, ta - t_end, "permodes")[ii]
    resids[ta] = pd.Series(res_pm := (a_true - pm), index=ii)
common = None
for ta in anch_list:
    rs = resids[ta]
    common = rs.index if common is None else common.intersection(rs.index)
M = np.column_stack([resids[ta].loc[common].values for ta in anch_list])
Cc = np.corrcoef(M.T)
print("\ncross-anchor residual corr (permode-from-train-end):")
print(Cc)

# ---------------- D) anchor -> anchor chains ----------------
print("\n=== D) anchor->anchor prediction (per-mode from previous anchor) ===")
print(f"{'from':>8} {'to':>8} {'gap':>4} {'persist':>8} {'permode':>8}")
for i, ta in enumerate(anch_list):
    ii, a_true = anch_fields[ta]
    # previous anchor field -> full-space interpolation
    prev = anch_list[i - 1] if i > 0 else None
    if prev is None:
        continue
    ii2, a2 = anch_fields[prev]
    x_prev = np.zeros(C); x_prev[:] = np.nan
    x_prev[ii2] = a2
    # fill nan with 0 for projection (approximation)
    x_prev = np.nan_to_num(x_prev)
    gap = ta - prev
    pm = predict_from(x_prev, gap, "permodes")[ii]
    pers = x_prev[ii]
    print(f"{prev:>8} {ta:>8} {gap:>4} {np.sqrt(np.mean((pers - a_true)**2)):>8.4f} "
          f"{np.sqrt(np.mean((pm - a_true)**2)):>8.4f}")
