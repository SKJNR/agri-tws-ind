"""a14d_01_baselines.py — honest CV baselines for the k=0 subproblem (Agent 14-d).

Baselines (all fit on <=2012 data only):
  1. persistence            : pred = TWS_t
  2. slope persistence      : pred = mu_c + s*(TWS_t - mu_c), s fit on fit rows
  3. v4 linear (full/red)   : the current production k=0 model, re-fit on <=2012
  4. persistence + covs(t+1): linear [x_p, covs(t+1) anomalies, 1]
  5. persistence + covs(t)  : linear [x_p, covs(t) anomalies, 1]  (shows covs(t) negativity)

Evaluated on:
  A. CONTINUITY: k0_diagnostic protocol (all val months, has_next rows) -> should reproduce
     persistence 0.6635 / linear 0.6407 and validates the machinery.
  B. STRICT PROTOCOL: val anchor months only, test-structure covs(t+1) availability,
     subgroup split + test-weighted overall + per-anchor table.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '/home/z/my-project/scripts')
from a14d_common import (load_train, cell_coords, Infra, Fields, val_anchor_info,
                         COVS, rmse, ANCHOR_CAL)

t0 = time.time()
tr = load_train()
n_cells = int(tr['cc'].max()) + 1
print(f"train {len(tr):,} rows, {n_cells} cells  [{time.time()-t0:.0f}s]")

fit = tr[tr['time'].dt.year <= 2012].copy()
val = tr[(tr['time'].dt.year >= 2013) & (tr['time'].dt.year <= 2015)].copy()
print(f"fit rows {len(fit):,} | val rows {len(val):,}")

infra = Infra(fit, n_cells, None)
fields = Fields(tr, n_cells)          # all train months (fit+val); used only for covs(t+1) lookup
print(f"infra done [{time.time()-t0:.0f}s]")

va = val_anchor_info(val)
print("\nval anchor months:")
print(va.to_string(index=False))

FIT_END = 2012 * 12 + 11              # t_abs of 2012-12
STRICT_FIT_MAX_T = FIT_END - 1        # rows with t_abs <= 2012-11 keep target within fit period


def next_month_idx(ta_arr):
    """vectorized: month-index of t_abs+1 in `fields`, or -1."""
    mp = {int(m): i for i, m in enumerate(fields.months)}
    return np.array([mp.get(int(m) + 1, -1) for m in ta_arr], np.int64)


def row_arrays(rows, strict_next_from_fields=True):
    """Features in anomaly space for arbitrary row sets (fit or val)."""
    cc = rows['cc'].values.astype(np.int64)
    ta = rows['t_abs'].values.astype(np.int64)
    x_p = (rows['TWS_t'].values - infra.mu_c[cc]).astype(np.float32)
    y = (rows['target'].values - infra.mu_c[cc]).astype(np.float32)
    covs_t = np.column_stack([rows[c].values for c in COVS]).astype(np.float32) - infra.clim[cc]
    nidx = next_month_idx(ta)
    covs_t1 = np.full((len(rows), 5), np.nan, np.float32)
    ok = nidx >= 0
    covs_t1[ok] = fields.COV[nidx[ok], cc[ok]]
    has_next = ok.copy()
    return dict(cc=cc, ta=ta, x_p=x_p, y=y, covs_t=covs_t, covs_t1=covs_t1, has_next=has_next)


# ==================== A. CONTINUITY (k0_diagnostic replication) ====================
print("\n" + "=" * 70)
print("A. CONTINUITY CHECK — k0_diagnostic protocol (all val months, has_next rows)")
print("=" * 70)

# k0_diagnostic: fit rows = has_nxt & yr<=2012 (their convention, includes 2012-12 rows
# whose target is 2013-01 TWS); eval = has_nxt & 2013<=yr<2016.
cd_fit = fit.copy()
cd_fit['t_next'] = cd_fit['t_abs'] + 1
Rk = tr[['cc', 't_abs'] + COVS].rename(columns={'t_abs': 't_next', **{c: c + '_nxt' for c in COVS}})
mg = cd_fit.merge(Rk, on=['cc', 't_next'], how='left')
mg = mg[mg['target'].notna() & mg['TWS_t'].notna()]
has_nxt_cd = mg[[c + '_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mg['cc'].values
yr = mg['t_abs'].values // 12
Xf = np.column_stack([
    mg['TWS_t'].values - infra.mu_c[ccm],
    *[mg[c].values - infra.clim[ccm, j] for j, c in enumerate(COVS)],
    *[mg[c + '_nxt'].values - infra.clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mg))]).astype(np.float32)
Xf = np.nan_to_num(Xf, nan=0.0)
yA = (mg['target'].values - infra.mu_c[ccm]).astype(np.float32)
w_rec = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
sw = np.sqrt(w_rec)
fitm_cd = has_nxt_cd & (yr <= 2012)
A_ = Xf[fitm_cd] * sw[fitm_cd, None]
cf_cd = np.linalg.solve(A_.T @ A_ + 1e-3 * np.eye(12), A_.T @ (yA[fitm_cd] * sw[fitm_cd]))

val_cd = val.copy()
val_cd['t_next'] = val_cd['t_abs'] + 1
mgv = val_cd.merge(Rk, on=['cc', 't_next'], how='left')
has_nxt_v = mgv[[c + '_nxt' for c in COVS]].notna().all(axis=1).values
ccv = mgv['cc'].values
Xv = np.column_stack([
    mgv['TWS_t'].values - infra.mu_c[ccv],
    *[mgv[c].values - infra.clim[ccv, j] for j, c in enumerate(COVS)],
    *[mgv[c + '_nxt'].values - infra.clim[ccv, j] for j, c in enumerate(COVS)],
    np.ones(len(mgv))]).astype(np.float32)
Xv = np.nan_to_num(Xv, nan=0.0)
yAv = (mgv['target'].values - infra.mu_c[ccv]).astype(np.float32)
ev_cd = has_nxt_v  # all 2013-15 months with next-month covs
slope_cd = float(np.sum(Xv[ev_cd][:, 0] * 0))  # placeholder
xfit = Xf[fitm_cd][:, 0]; yfit = yA[fitm_cd]
slope_cd = float(np.sum(xfit * yfit) / np.sum(xfit * xfit))
rmse_pers_cd = rmse(slope_cd * Xv[ev_cd][:, 0], yAv[ev_cd])
rmse_lin_cd = rmse(Xv[ev_cd] @ cf_cd, yAv[ev_cd])
print(f"persistence (slope {slope_cd:.3f}): RMSE={rmse_pers_cd[0]:.4f}  n={rmse_pers_cd[1]:,}   [ref 0.6635]")
print(f"v4 linear full:                  RMSE={rmse_lin_cd[0]:.4f}  n={rmse_lin_cd[1]:,}   [ref 0.6407]")
print(f"   coefs: TWS={cf_cd[0]:.3f} covs_t={np.round(cf_cd[1:6],3)} covs_t1={np.round(cf_cd[6:11],3)}")

# ==================== B. STRICT PROTOCOL ====================
print("\n" + "=" * 70)
print("B. STRICT PROTOCOL — val anchor months only, test-structure covs(t+1)")
print("=" * 70)

# fit rows (strict): t_abs <= 2012-11 so target month <= 2012-12
fit_strict = fit[fit['t_abs'] <= STRICT_FIT_MAX_T]
F = row_arrays(fit_strict)
print(f"strict fit rows: {len(fit_strict):,}  (has_next: {F['has_next'].sum():,})")

# val k=0 rows: rows at val anchor months; covs(t+1) only per protocol rule
val_anchor_set = set(va['t_abs'].tolist())
vk = val[val['t_abs'].isin(val_anchor_set)].copy()
V = row_arrays(vk)
anchor_cal_map = dict(zip(va['t_abs'], va['has_next']))
proto_has_next = vk['t_abs'].map(anchor_cal_map).values.astype(bool)
# enforce: protocol says no covs(t+1) for cal {7,9}; for cal {1,6,11,12} only if t+1 in file
V['covs_t1'][~proto_has_next] = np.nan
V['has_next'] = proto_has_next
print(f"val k=0 rows: {len(vk):,}  (with covs(t+1): {proto_has_next.sum():,} = "
      f"{proto_has_next.mean()*100:.1f}%  | test mix: 66.8%)")

with_grp = proto_has_next
without_grp = ~proto_has_next
yv = vk['target'].values.astype(np.float64)     # raw target for RMSE reporting
mu_v = infra.mu_c[V['cc']].astype(np.float64)
yA_v = V['y'].astype(np.float64)
x_p_v = V['x_p'].astype(np.float64)

# recency weights on strict fit rows
yr_f = F['ta'] // 12
w_f = np.where(yr_f <= 2009, 1.0, 2.0).astype(np.float32)
swf = np.sqrt(w_f)

results = {}

def eval_model(name, pred_anom):
    """pred_anom: anomaly-space predictions for ALL val k=0 rows; report subgroups + weighted."""
    pred = mu_v + pred_anom
    r_with, n_with = rmse(pred[with_grp], yv[with_grp])
    r_wo, n_wo = rmse(pred[without_grp], yv[without_grp])
    mse_with = np.mean((pred[with_grp] - yv[with_grp]) ** 2)
    mse_wo = np.mean((pred[without_grp] - yv[without_grp]) ** 2)
    overall_rowwise = np.sqrt(np.mean((pred - yv) ** 2))
    overall_testwt = np.sqrt(0.668 * mse_with + 0.332 * mse_wo)
    results[name] = dict(with_=r_with, without=r_wo, overall=overall_rowwise, testwt=overall_testwt)
    print(f"{name:34s} | with {r_with:.4f} (n={n_with:,}) | without {r_wo:.4f} (n={n_wo:,}) "
          f"| rowwise {overall_rowwise:.4f} | TEST-WTD {overall_testwt:.4f}")
    return pred

# ---- 1. persistence ----
print("\n--- baselines (strict protocol) ---")
eval_model("persistence (raw)", x_p_v)

# ---- 2. slope persistence ----
xf, yf = F['x_p'].astype(np.float64), F['y'].astype(np.float64)
slope_uw = float(np.sum(xf * yf) / np.sum(xf * xf))
slope_w = float(np.sum(w_f * xf * yf) / np.sum(w_f * xf * xf))
eval_model(f"slope persistence (uw s={slope_uw:.3f})", slope_uw * x_p_v)
eval_model(f"slope persistence (w  s={slope_w:.3f})", slope_w * x_p_v)

# ---- 3. v4 linear (full / reduced), recency-weighted, ridge 1e-3 ----
X_full_f = np.column_stack([F['x_p'], F['covs_t'], F['covs_t1'], np.ones(len(F['x_p']))])
X_full_f = np.nan_to_num(X_full_f, nan=0.0).astype(np.float32)
selF = F['has_next']
Af = X_full_f[selF] * swf[selF, None]
coefF = np.linalg.solve(Af.T @ Af + 1e-3 * np.eye(12), Af.T @ (F['y'][selF] * swf[selF]))
colsR = [0, 1, 2, 3, 4, 5, 11]
Ar = X_full_f[:, colsR] * swf[:, None]
coefR = np.linalg.solve(Ar.T @ Ar + 1e-3 * np.eye(7), Ar.T @ (F['y'] * swf))

Xv_full = np.column_stack([V['x_p'], V['covs_t'], V['covs_t1'], np.ones(len(V['x_p']))])
Xv_full = np.nan_to_num(Xv_full, nan=0.0).astype(np.float32)
predF = Xv_full @ coefF
predR = Xv_full[:, colsR] @ coefR
print(f"   linear-F coefs: TWS={coefF[0]:.3f} covs_t={np.round(coefF[1:6],3)} covs_t1={np.round(coefF[6:11],3)}")
print(f"   linear-R coefs: TWS={coefR[0]:.3f} covs_t={np.round(coefR[1:6],3)}")
pred_v4 = np.where(with_grp, predF, predR).astype(np.float64)
eval_model("v4 linear (F/R dual)", pred_v4)
eval_model("v4 linear-F everywhere (cheat chk)", predF.astype(np.float64))

# ---- 4. persistence + covs(t+1) ----
Xp1_f = np.column_stack([F['x_p'], F['covs_t1'], np.ones(len(F['x_p']))])
Xp1_f = np.nan_to_num(Xp1_f, nan=0.0).astype(np.float32)
Ap1 = Xp1_f[selF] * swf[selF, None]
coefP1 = np.linalg.solve(Ap1.T @ Ap1 + 1e-3 * np.eye(7), Ap1.T @ (F['y'][selF] * swf[selF]))
print(f"   pers+covs(t+1) coefs: TWS={coefP1[0]:.3f} covs_t1={np.round(coefP1[1:6],3)}")
Xv_p1 = np.column_stack([V['x_p'], V['covs_t1'], np.ones(len(V['x_p']))])
Xv_p1 = np.nan_to_num(Xv_p1, nan=0.0).astype(np.float32)
predP1 = Xv_p1 @ coefP1
eval_model("persistence + covs(t+1) (F only)", predP1.astype(np.float64))

# ---- 5. persistence + covs(t) ----
Xp0_f = np.column_stack([F['x_p'], F['covs_t'], np.ones(len(F['x_p']))]).astype(np.float32)
Xp0_f = np.nan_to_num(Xp0_f, nan=0.0)
Ap0 = Xp0_f * swf[:, None]
coefP0 = np.linalg.solve(Ap0.T @ Ap0 + 1e-3 * np.eye(7), Ap0.T @ (F['y'] * swf))
print(f"   pers+covs(t) coefs: TWS={coefP0[0]:.3f} covs_t={np.round(coefP0[1:6],3)}")
Xv_p0 = np.column_stack([V['x_p'], V['covs_t'], np.ones(len(V['x_p']))]).astype(np.float32)
Xv_p0 = np.nan_to_num(Xv_p0, nan=0.0)
eval_model("persistence + covs(t)", (Xv_p0 @ coefP0).astype(np.float64))

# ---- per-anchor table for the best linear ----
print("\nper-anchor RMSE (v4 linear F/R dual):")
vk2 = vk.copy()
vk2['pred'] = mu_v + pred_v4
vk2['err2'] = (vk2['pred'] - vk2['target']) ** 2
tab = vk2.groupby(['t_abs', 'has_next' if False else 't_abs']).agg(
    n=('err2', 'size'), rmse=('err2', lambda s: float(np.sqrt(np.mean(s)))))
tab2 = va.set_index('t_abs').join(tab[['n', 'rmse']])
print(tab2.to_string())

print(f"\nDONE [{time.time()-t0:.0f}s]")

# summary table
print("\n=== SUMMARY (strict protocol) ===")
print(f"{'model':36s} {'with':>8s} {'without':>8s} {'rowwise':>8s} {'TESTWTD':>8s}")
for k, v in results.items():
    print(f"{k:36s} {v['with_']:8.4f} {v['without']:8.4f} {v['overall']:8.4f} {v['testwt']:8.4f}")
