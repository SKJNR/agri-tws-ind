"""a15_2_k0.py — V8 factor k=0 predictor, STRICT 2013-15 protocol (a14d_01 conventions).

Protocol:
  - val anchor months = cal in {1,6,7,9,11,12}; whole TWS_t field observable there.
  - covs(t+1) visible iff cal in {1,6,11,12} AND t+1 month present in train.
  - metrics: with/without/rowwise/TESTWTD (0.668/0.332), per-anchor table.

Baselines (from a15_0_baselines_run.txt, reproduced this session):
  persistence 0.6880 | slope-pers 0.6572 | v4 linear F/R dual 0.6451
  (with 0.6688 / without 0.5945)
Target: TESTWTD < 0.60.

Model: pred = mu + Dtil(t+1) + r(t) + V_ext @ KF(f(t) -> t+1, cov update if avail)
  Dtil = W1*Dhat_val + W2*S_val + W3*trendex(t)  (LOO-fitted weights, cv_lb protocol)
"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/scripts')
from a15_common import load_train, FactorCore, ModeKF, estimate_VarD, COVS

t0 = time.time()
OUT = '/home/z/my-project/download/a15_2_k0.txt'
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

ANCHOR_CAL = {1, 6, 7, 9, 11, 12}
NEXT_CAL = {1, 6, 11, 12}

train = load_train()
n_cells = int(train['cc'].max()) + 1
fit = train[train['time'].dt.year <= 2012]
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]
tr_months = set(int(m) for m in train['t_abs'].unique())

results = {}

# ------------------------------------------------------------------ model builder
def build(K):
    core = FactorCore(fit, n_cells, K=K, rec_edges=(2010, 2013), verbose=False)
    return core

def run_variant(tag, K=100, VarD_scale=1.0, use_cov=True, use_r=True, use_linblend=0.0,
                blend_core=None):
    core = build(K)
    V, Vx, phi, q = core.V, core.Vx, core.phi, core.q
    vmonths, vfields = core.era_cov(val)
    vz = core.zscores(vfields, vmonths)
    S_val = np.nanmean(np.array([vfields[int(m)][0] for m in vmonths]), axis=0)
    val_months = np.sort(val['t_abs'].unique())
    vm2i = {int(m): i for i, m in enumerate(val_months)}

    # val anchor fields + Dhat + LOO Dtil weights
    val_anchor_months = [int(m) for m in val_months if (m % 12 + 1) in ANCHOR_CAL]
    AF = {}
    vta = val['t_abs'].values
    vcc = val['cc'].values
    vtw = val['TWS_t'].values
    for a in val_anchor_months:
        f = np.full(n_cells, np.nan, dtype=np.float32)
        sel = vta == a
        f[vcc[sel]] = vtw[sel]
        AF[a] = f - core.mu_c
    Dhat = np.nanmean(np.array([AF[a] for a in val_anchor_months]), axis=0)
    Xs, ys = [], []
    for a in val_anchor_months:
        dloo = np.nanmean(np.array([AF[b] for b in val_anchor_months if b != a]), axis=0)
        tx = core.trendex(a)
        ok = np.isfinite(dloo) & np.isfinite(S_val) & np.isfinite(AF[a]) & np.isfinite(tx)
        Xs.append(np.column_stack([dloo[ok], S_val[ok], tx[ok]]))
        ys.append(AF[a][ok])
    w_ = np.linalg.solve(np.vstack(Xs).T @ np.vstack(Xs) + 1e-3 * np.eye(3),
                         np.vstack(Xs).T @ np.concatenate(ys))
    W1, W2, W3 = float(w_[0]), float(w_[1]), float(w_[2])
    def Dtil(tt):
        return W1 * Dhat + W2 * S_val + W3 * core.trendex(tt)

    # VarD from val anchors
    YA = np.stack([core.anchor_y(AF[a], Dtil(a)) for a in val_anchor_months])
    gaps = [(i, i + 1, val_anchor_months[i + 1] - val_anchor_months[i])
            for i in range(len(val_anchor_months) - 1)]
    VarD = estimate_VarD(YA, gaps, phi, Vx) * VarD_scale
    kf = ModeKF(phi, Vx, q, core.H, core.Rcov, VarD)

    # ---- predict every val anchor month ----
    rows_pred = {}
    for a in val_anchor_months:
        has_next = ((a % 12 + 1) in NEXT_CAL) and ((a + 1) in tr_months)
        g = AF[a] - Dtil(a)
        gf = g[core.full]
        gf = np.where(np.isfinite(gf), gf, 0.0)
        f = V.T @ gf
        r = np.where(np.isfinite(g), g, 0.0) - f @ core.V_ext.T
        kf.reset()
        kf.anchor_update(f)
        kf.step(1)
        if use_cov and has_next and (a + 1) in vz:
            kf.cov_update(vz[a + 1])
        m = kf.mean_var()[0]
        pred = core.mu_c + Dtil(a + 1) + (f @ core.V_ext.T * 0)  # placeholder
        pred = core.mu_c + Dtil(a + 1) + (m @ core.V_ext.T)
        if use_r:
            pred = pred + r
        rows_pred[a] = (pred, has_next)
    return core, rows_pred, val_anchor_months, Dtil

# ------------------------------------------------------------------ evaluation
val_q = val.copy()
vta = val['t_abs'].values
vy = val['target'].values.astype(np.float64)
vcc = val['cc'].values

# ------------------------------------------------------------------ v4 linear (reproduction for blend)
def v4_linear_preds(core):
    """v4 linear F/R dual k=0 model, refit on fit rows (a14d_01 conventions)."""
    tr_fields_months = np.sort(train['t_abs'].unique())
    fm2i = {int(m): i for i, m in enumerate(tr_fields_months)}
    COVmat = np.full((len(tr_fields_months), n_cells, len(COVS)), np.nan, np.float32)
    mi_all = train['t_abs'].map(fm2i).values
    COVmat[mi_all, train['cc'].values] = train[COVS].values

    def feats(rows):
        cc_ = rows['cc'].values
        ta_ = rows['t_abs'].values
        x_p = rows['TWS_t'].values - core.mu_c[cc_]
        covs_t = np.column_stack([rows[c].values for c in COVS]) - core.clim[cc_]
        nidx = np.array([fm2i.get(int(m) + 1, -1) for m in ta_])
        covs_t1 = np.full((len(rows), 5), np.nan, np.float32)
        okn = nidx >= 0
        covs_t1[okn] = COVmat[nidx[okn], cc_[okn]]
        return x_p, covs_t, covs_t1, okn

    # strict fit rows: t_abs <= 2012-11 (targets stay within fit period)
    fitS = fit[fit['t_abs'] <= 2012 * 12 + 10]
    x_p, covs_t, covs_t1, okn = feats(fitS)
    yA = fitS['target'].values - core.mu_c[fitS['cc'].values]
    yr = fitS['t_abs'].values // 12
    wrec = np.where(yr <= 2009, 1.0, 2.0)
    sw = np.sqrt(wrec)
    Xf_ = np.column_stack([x_p, covs_t, covs_t1, np.ones(len(x_p))])
    Xf_ = np.nan_to_num(Xf_, nan=0.0).astype(np.float32)
    selF_ = okn & np.isfinite(yA) & np.isfinite(x_p)
    A_ = Xf_[selF_] * sw[selF_, None]
    coefF = np.linalg.solve(A_.T @ A_ + 1e-3 * np.eye(12), A_.T @ (yA[selF_] * sw[selF_]))
    colsR = [0, 1, 2, 3, 4, 5, 11]
    selR_ = np.isfinite(yA) & np.isfinite(x_p)
    A_ = Xf_[selR_][:, colsR] * sw[selR_, None]
    coefR = np.linalg.solve(A_.T @ A_ + 1e-3 * np.eye(7), A_.T @ (yA[selR_] * sw[selR_]))

    # val anchor rows
    vanchor_mask = np.isin(vta, [a for a in val_anchor_months_global])
    vr = val[vanchor_mask]
    x_p, covs_t, covs_t1, okn = feats(vr)
    Xv_ = np.column_stack([x_p, covs_t, covs_t1, np.ones(len(x_p))])
    Xv_ = np.nan_to_num(Xv_, nan=0.0).astype(np.float32)
    predF = Xv_ @ coefF
    predR = Xv_[:, colsR] @ coefR
    has_next_cal = np.array([((int(a) % 12 + 1) in NEXT_CAL) and ((int(a) + 1) in tr_months)
                             for a in vr['t_abs'].values])
    pred = np.where(has_next_cal, predF, predR)
    lin_pred = np.full(len(val), np.nan)
    lin_pred[vanchor_mask] = core.mu_c[vr['cc'].values] + pred
    return lin_pred

def rows_from_fields(rows_pred, val_anchor_months):
    """Convert per-anchor field predictions to row-level arrays."""
    pred_by_row = np.full(len(val), np.nan)
    has_next_by_row = np.zeros(len(val), bool)
    for a in val_anchor_months:
        sel = vta == a
        pred_by_row[sel] = rows_pred[a][0][vcc[sel]]
        has_next_by_row[sel] = rows_pred[a][1]
    return pred_by_row, has_next_by_row

def evaluate(tag, pred_by_row, has_next_by_row):
    ok = np.isfinite(pred_by_row) & np.isfinite(vy)
    selA = ok & has_next_by_row
    selB = ok & ~has_next_by_row
    rA = float(np.sqrt(np.mean((pred_by_row[selA] - vy[selA]) ** 2)))
    rB = float(np.sqrt(np.mean((pred_by_row[selB] - vy[selB]) ** 2)))
    rw = float(np.sqrt(np.mean((pred_by_row[ok] - vy[ok]) ** 2)))
    tw = float(np.sqrt(0.668 * rA ** 2 + 0.332 * rB ** 2))
    results[tag] = (rA, rB, rw, tw, int(selA.sum()), int(selB.sum()))
    P(f"{tag:38s} | with {rA:.4f} (n={selA.sum():,}) | without {rB:.4f} (n={selB.sum():,})"
      f" | rowwise {rw:.4f} | TESTWTD {tw:.4f}")
    return pred_by_row

P("=" * 100)
P("STRICT k=0 PROTOCOL — baselines (a15_0 run): persistence 0.6880 | slope-pers 0.6572 | v4 linear 0.6451")
P("  (with 0.6688 / without 0.5945). Target TESTWTD < 0.60")
P("=" * 100)

# persistence reference on this row set
pers = np.where(np.isfinite(val['TWS_t'].values), val['TWS_t'].values, np.nan)
okp = np.isfinite(pers) & np.isfinite(vy)
anchor_rows = np.isin(vta, list(range(24156, 24187)))
sel = okp & anchor_rows
P(f"persistence (anchor rows only):       rowwise {np.sqrt(np.mean((pers[sel]-vy[sel])**2)):.4f}")

# global list of val anchor months for the linear model
val_months_g = np.sort(val['t_abs'].unique())
val_anchor_months_global = [int(m) for m in val_months_g if (m % 12 + 1) in ANCHOR_CAL]

for K in [50, 100, 114]:
    core, rp, vam, _ = run_variant(f"K{K}", K=K)
    p, h = rows_from_fields(rp, vam)
    evaluate(f"factor K={K} (cov+r+Dtil)", p, h)
P("")
core, rp, vam, _ = run_variant("K100 nor", K=100, use_r=False)
p, h = rows_from_fields(rp, vam)
evaluate("factor K=100 no r", p, h)
core, rp, vam, _ = run_variant("K100 nocov", K=100, use_cov=False)
p, h = rows_from_fields(rp, vam)
evaluate("factor K=100 no cov(t+1)", p, h)
core, rp, vam, _ = run_variant("K100 VarD2", K=100, VarD_scale=2.0)
p, h = rows_from_fields(rp, vam)
evaluate("factor K=100 VarDx2", p, h)
core, rp, vam, _ = run_variant("K100 VarD.5", K=100, VarD_scale=0.5)
p, h = rows_from_fields(rp, vam)
evaluate("factor K=100 VarDx0.5", p, h)

# ---- blend factor + v4 linear ----
P("")
lin_pred = v4_linear_preds(core)
core, rp, vam, _ = run_variant("final", K=100)
factor_pred_rows, hn = rows_from_fields(rp, vam)
# lin_pred is already a row array over val rows:
lin_rows = np.where(np.isfinite(lin_pred), lin_pred, factor_pred_rows)
evaluate("v4 linear F/R dual (repro)", lin_rows, hn)
evaluate("factor K=100 (final)", factor_pred_rows, hn)
for w_ in [0.25, 0.5, 0.75]:
    blended = w_ * np.nan_to_num(factor_pred_rows) + (1 - w_) * np.nan_to_num(lin_rows)
    blended = np.where(np.isfinite(lin_rows), blended, factor_pred_rows)
    evaluate(f"blend {w_:.2f}*factor + {1-w_:.2f}*linear", blended, hn)
P("\nper-anchor RMSE (factor K=100):")
for a in vam:
    sel = (vta == a) & np.isfinite(factor_pred_rows) & np.isfinite(vy)
    if sel.sum() == 0:
        continue
    r = float(np.sqrt(np.mean((factor_pred_rows[sel] - vy[sel]) ** 2)))
    P(f"  {a} (cal {a%12+1:2d}, next={rp[a][1]}): rmse={r:.4f}  n={int(sel.sum()):,}")

P("\n=== SUMMARY ===")
P(f"{'model':38s} {'with':>8s} {'without':>8s} {'rowwise':>8s} {'TESTWTD':>8s}")
for k, v in results.items():
    P(f"{k:38s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} {v[3]:8.4f}")

with open(OUT, 'w') as fh:
    fh.write('\n'.join(log) + '\n')
P(f"\nsaved {OUT} [{time.time()-t0:.0f}s]")
