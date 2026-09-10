"""
build_v21_phaseA.py — v21 Phase A: reproduce the a15 factor k=0 CV in THIS environment,
head-to-head vs the current v17b k0 (linear+LGB 50/50), + 3-way blend probe.

Protocol (identical to recovered a15_2_k0.py):
  - fit <= 2012, val 2013-01..2015-08 (train ends 2015-08).
  - val anchor months = cal in {1,6,7,9,11,12}; whole TWS_t field observable there.
  - covs(t+1) visible iff cal in {1,6,11,12} AND t+1 in train  (mirrors the test
    structure: 4 of 6 test k0 anchors have t+1 in the test month set, 2 do not).
  - metrics: with / without / rowwise / TESTWTD (0.668/0.332).

Reference numbers (recovered a15_2_k0.txt, lost session):
  factor K=100            TESTWTD 0.6310
  v4 linear F/R dual      TESTWTD 0.6451
  blend 0.50 factor+linear TESTWTD 0.6166   <- v21 k0 design
  (a14d, v17b k0): linear 0.6287 / LGBM 0.6273 / blend 0.6254 (rowwise)
"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/scripts')
from a15_common import load_train, FactorCore, ModeKF, estimate_VarD, COVS
import lightgbm as lgb

t0 = time.time()
OUT = '/home/z/my-project/download/build_v21_phaseA.txt'
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

vta = val['t_abs'].values
vcc = val['cc'].values
vy = val['target'].values.astype(np.float64)

val_months = np.sort(val['t_abs'].unique())
val_anchor_months = [int(m) for m in val_months if (m % 12 + 1) in ANCHOR_CAL]
P(f"val anchor months ({len(val_anchor_months)}): {val_anchor_months}")

# ---------------- factor model (verbatim a15_2_k0.run_variant, K=100) ----------------
def build_factor_preds():
    core = FactorCore(fit, n_cells, K=100, rec_edges=(2010, 2013), verbose=False)
    V, Vx, phi, q = core.V, core.Vx, core.phi, core.q
    vmonths, vfields = core.era_cov(val)
    vz = core.zscores(vfields, vmonths)
    S_val = np.nanmean(np.array([vfields[int(m)][0] for m in vmonths]), axis=0)

    AF = {}
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
    P(f"factor Dtil LOO weights: {W1:.3f}/{W2:.3f}/{W3:.3f}")
    def Dtil(tt):
        return W1 * Dhat + W2 * S_val + W3 * core.trendex(tt)

    YA = np.stack([core.anchor_y(AF[a], Dtil(a)) for a in val_anchor_months])
    gaps = [(i, i + 1, val_anchor_months[i + 1] - val_anchor_months[i])
            for i in range(len(val_anchor_months) - 1)]
    VarD = estimate_VarD(YA, gaps, phi, Vx)
    kf = ModeKF(phi, Vx, q, core.H, core.Rcov, VarD)

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
        if has_next and (a + 1) in vz:
            kf.cov_update(vz[a + 1])
        m = kf.mean_var()[0]
        pred = core.mu_c + Dtil(a + 1) + (m @ core.V_ext.T) + r
        rows_pred[a] = (pred, has_next)
    return core, rows_pred

def rows_from_fields(rows_pred):
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
    P(f"{tag:40s} | with {rA:.4f} (n={selA.sum():,}) | without {rB:.4f} (n={selB.sum():,})"
      f" | rowwise {rw:.4f} | TESTWTD {tw:.4f}")
    return pred_by_row, (rA, rB, rw, tw)

P("=" * 108)
P("PHASE A — v21 k=0 lane: a15 factor reproduction + v17b k0 head-to-head (this env)")
P("=" * 108)

# 1) factor model
core, rp = build_factor_preds()
factor_rows, hn = rows_from_fields(rp)
evaluate("factor K=100 (repro)", factor_rows, hn)

# 2) v4 linear F/R dual (verbatim a15_2_k0.v4_linear_preds)
def v4_linear_preds(core):
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

    vanchor_mask = np.isin(vta, val_anchor_months)
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
    return lin_pred, coefF, coefR

lin_rows, coefF, coefR = v4_linear_preds(core)
evaluate("v4 linear F/R dual (repro)", lin_rows, hn)

# 3) v17b k0 head-to-head: linear + LGBM blend with the SAME with/without structure.
#    Features per a14d: TWS_t anomaly + covs(t) + covs(t+) [NaN'd when cal not in NEXT_CAL].
def build_k0_X(rows, core, COVmat, fm2i, respect_next_cal):
    cc_ = rows['cc'].values
    ta_ = rows['t_abs'].values
    x_p = rows['TWS_t'].values - core.mu_c[cc_]
    covs_t = np.column_stack([rows[c].values for c in COVS]) - core.clim[cc_]
    nidx = np.array([fm2i.get(int(m) + 1, -1) for m in ta_])
    covs_t1 = np.full((len(rows), 5), np.nan, np.float32)
    okn = nidx >= 0
    covs_t1[okn] = COVmat[nidx[okn], cc_[okn]]
    if respect_next_cal:
        allow = np.array([((int(a) % 12 + 1) in NEXT_CAL) for a in ta_])
        covs_t1 = np.where(allow[:, None], covs_t1, np.nan)
        okn = okn & allow
    X = np.column_stack([x_p, covs_t, covs_t1]).astype(np.float32)
    return X, okn

tr_fields_months = np.sort(train['t_abs'].unique())
fm2i = {int(m): i for i, m in enumerate(tr_fields_months)}
COVmat = np.full((len(tr_fields_months), n_cells, len(COVS)), np.nan, np.float32)
mi_all = train['t_abs'].map(fm2i).values
COVmat[mi_all, train['cc'].values] = train[COVS].values

fitS = fit[fit['t_abs'] <= 2012 * 12 + 10]
Xtr, ok_tr = build_k0_X(fitS, core, COVmat, fm2i, respect_next_cal=False)
ytr = (fitS['target'].values - core.mu_c[fitS['cc'].values]).astype(np.float32)
okm = np.isfinite(ytr) & np.isfinite(Xtr[:, 0])
Xtr, ytr, ok_tr = Xtr[okm], ytr[okm], ok_tr[okm]
yr = fitS['t_abs'].values[okm] // 12
wtr = np.where(yr <= 2009, 1.0, 2.0).astype(np.float32)

vanchor_mask = np.isin(vta, val_anchor_months)
vr = val[vanchor_mask]
Xev, ok_ev = build_k0_X(vr, core, COVmat, fm2i, respect_next_cal=True)
yvr = vr['target'].values.astype(np.float64)

# linear F/R dual (same as v4 but through this feature path)
def k0_linear(Xtr, ytr, wtr, Xev):
    has_nxt_tr = np.isfinite(Xtr[:, 6:11]).all(axis=1)
    sw = np.sqrt(wtr)
    Xtr0 = np.nan_to_num(Xtr, nan=0.0)
    A = Xtr0[has_nxt_tr] * sw[has_nxt_tr, None]
    cF = np.linalg.solve(A.T @ A + 1e-3 * np.eye(11), A.T @ (ytr[has_nxt_tr] * sw[has_nxt_tr]))
    colsR = np.array([0, 1, 2, 3, 4, 5])
    A_r = Xtr0[:, colsR] * sw[:, None]
    cR = np.linalg.solve(A_r.T @ A_r + 1e-3 * np.eye(6), A_r.T @ (ytr * sw))
    has_nxt_ev = np.isfinite(Xev[:, 6:11]).all(axis=1)
    Xe = np.nan_to_num(Xev, nan=0.0)
    pred = np.empty(len(Xev), dtype=np.float32)
    pred[has_nxt_ev] = Xe[has_nxt_ev] @ cF
    pred[~has_nxt_ev] = Xe[~has_nxt_ev][:, colsR] @ cR
    return pred

p_lin_ev = k0_linear(Xtr, ytr, wtr, Xev)

params = dict(objective='regression', metric='rmse', num_leaves=63,
              learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, num_threads=2,
              seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
bst = lgb.train(params, dtr, num_boost_round=400)
p_lgb_ev = bst.predict(Xev)

# assemble row-level arrays for the head-to-head (val anchor rows)
lin2 = np.full(len(val), np.nan)
lgb2 = np.full(len(val), np.nan)
idx = np.where(vanchor_mask)[0]
lin2[idx] = core.mu_c[vr['cc'].values] + p_lin_ev
lgb2[idx] = core.mu_c[vr['cc'].values] + p_lgb_ev
hn_v = hn[idx]

def ev_rows(tag, pred_rows):
    ok = np.isfinite(pred_rows) & np.isfinite(vy)
    selA = ok & hn; selB = ok & ~hn
    rA = float(np.sqrt(np.mean((pred_rows[selA] - vy[selA]) ** 2)))
    rB = float(np.sqrt(np.mean((pred_rows[selB] - vy[selB]) ** 2)))
    rw = float(np.sqrt(np.mean((pred_rows[ok] - vy[ok]) ** 2)))
    tw = float(np.sqrt(0.668 * rA ** 2 + 0.332 * rB ** 2))
    P(f"{tag:40s} | with {rA:.4f} | without {rB:.4f} | rowwise {rw:.4f} | TESTWTD {tw:.4f}")
    return pred_rows

P("")
P("--- v17b k0 head-to-head (same rows, same with/without split) ---")
ev_rows("v17b linear (a14d repro)", lin2)
ev_rows("v17b LGBM(400) (a14d repro)", lgb2)
v17b_rows = 0.5 * lin2 + 0.5 * lgb2
ev_rows("v17b k0 = 0.5 lin + 0.5 LGB  [CURRENT]", v17b_rows)

P("")
P("--- v21 k0 candidates ---")
evaluate("v21 k0 = 0.5 factor + 0.5 linear", 0.5 * factor_rows + 0.5 * lin_rows, hn)
for wf, wl, wg in [(0.5, 0.25, 0.25), (0.4, 0.4, 0.2), (0.5, 0.3, 0.2)]:
    cand = wf * factor_rows + wl * lin_rows + wg * lgb2
    cand = np.where(np.isfinite(lin_rows), cand, factor_rows)
    evaluate(f"3-way {wf}F+{wl}L+{wg}G", cand, hn)

P("")
P(f"elapsed {time.time()-t0:.0f}s")
with open(OUT, 'w') as fh:
    fh.write('\n'.join(log) + '\n')
P(f"saved {OUT}")
