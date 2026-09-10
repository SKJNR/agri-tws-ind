"""a14d_02_lgbm.py — LightGBM for the k=0 subproblem, honest strict-protocol CV (Agent 14-d).

Models:
  F (full)   : for k=0 rows WITH covs(t+1)  — features incl. covs(t+1) anomalies + has_next flag
  R (reduced): for k=0 rows WITHOUT covs(t+1) — no covs(t+1) features

Feature blocks (all constructible in TEST from TWS_t at the anchor month + covs at the
18 in-file months + static cell attributes — NO t-1 history, no target-period TWS):
  core    : x_p = TWS_t - mu_c, mu_c
  trend   : beta_c, trendex_t, trendex_t1, fast_hat = x_p - trendex_t
  covt    : 5 covariate anomalies at t
  covt1   : 5 covariate anomalies at t+1 (NaN when absent) + has_next flag   [F only]
  spatial : 4-neighbor mean/std/min/max of the anomaly field at t, neighbor count,
            5x5-box mean, laplacian x_p - nb_mean
  geo     : lat, lon, |lat|
  mon     : sin/cos of the TARGET month (t+1)

Training: strict fit rows (t_abs <= 2012-11 so targets stay in fit period);
early stopping on 2011-2012 anchor-month rows; refit on all fit rows at best_iter.
Ablations: anchor-cal months vs all months; recency weights 1/2 vs unweighted; blocks.
"""
import sys, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/z/my-project/scripts')
from a14d_common import (load_train, cell_coords, build_nb_graph, Infra, Fields,
                         val_anchor_info, nb_stats_matrix, COVS, ANCHOR_CAL, rmse)

t0 = time.time()
OUT = '/home/z/my-project/scripts/a14d_val_preds.npz'

tr = load_train()
n_cells = int(tr['cc'].max()) + 1
lat_c, lon_c = cell_coords(tr)
fit = tr[tr['time'].dt.year <= 2012]
val = tr[(tr['time'].dt.year >= 2013) & (tr['time'].dt.year <= 2015)]
infra = Infra(fit, n_cells, None)
fields = Fields(tr, n_cells)
va = val_anchor_info(val)
print(f"data loaded [{time.time()-t0:.0f}s]")

nb1, nb2 = build_nb_graph(lat_c, lon_c)
print(f"neighbor graph: ring1 avail/cell mean={(nb1 >= 0).sum(1).mean():.2f}/4, "
      f"box2 avail/cell mean={(nb2 >= 0).sum(1).mean():.2f}/24  [{time.time()-t0:.0f}s]")

A = fields.TWS - infra.mu_c[None, :]          # anomaly field, all 138 train months
nb = nb_stats_matrix(A, nb1, nb2)
print(f"spatial stats done [{time.time()-t0:.0f}s]")

FIT_END = 2012 * 12 + 11                      # 2012-12
STRICT_MAX_T = FIT_END - 1
ES_START = 2011 * 12                          # 2011-01
month_mp = {int(m): i for i, m in enumerate(fields.months)}

BLOCK_NAMES = dict(
    core=['x_p', 'mu_c'],
    trend=['beta_c', 'trendex_t', 'trendex_t1', 'fast_hat'],
    covt=[f'covt_{c}' for c in ['S1', 'S3', 'S6', 'S12', 'SM']],
    covt1=[f'covt1_{c}' for c in ['S1', 'S3', 'S6', 'S12', 'SM']] + ['has_next'],
    spatial=['nb_mean', 'nb_std', 'nb_min', 'nb_max', 'nb_cnt', 'box2_mean', 'lap'],
    geo=['lat', 'lon', 'abs_lat'],
    mon=['mon_sin', 'mon_cos'],
)


def build_rows(rows_df):
    """Assemble feature blocks for a set of k=0 rows (order preserved)."""
    cc = rows_df['cc'].values.astype(np.int64)
    ta = rows_df['t_abs'].values.astype(np.int64)
    mi = np.array([month_mp[int(m)] for m in ta], np.int64)
    x_p = A[mi, cc].astype(np.float32)
    y = (rows_df['target'].values - infra.mu_c[cc]).astype(np.float32)
    trex_t = ((ta - infra.tbar_c[cc]) * infra.beta_c[cc]).astype(np.float32)
    beta = infra.beta_c[cc].astype(np.float32)
    covs_t = np.column_stack([rows_df[c].values for c in COVS]).astype(np.float32) - infra.clim[cc]
    nidx = np.array([month_mp.get(int(m) + 1, -1) for m in ta], np.int64)
    covs_t1 = np.full((len(rows_df), 5), np.nan, np.float32)
    ok = nidx >= 0
    covs_t1[ok] = fields.COV[nidx[ok], cc[ok]]
    cal1 = ((ta + 1) % 12) + 1
    feats = dict(
        core=[x_p, infra.mu_c[cc].astype(np.float32)],
        trend=[beta, trex_t.astype(np.float32), (trex_t + beta).astype(np.float32),
               (x_p - trex_t).astype(np.float32)],
        covt=[covs_t[:, j] for j in range(5)],
        covt1=[covs_t1[:, j] for j in range(5)] + [ok.astype(np.float32)],
        spatial=[nb['nb_mean'][mi, cc], nb['nb_std'][mi, cc], nb['nb_min'][mi, cc],
                 nb['nb_max'][mi, cc], nb['nb_cnt'][mi, cc],
                 nb['box2_mean'][mi, cc], (x_p - nb['nb_mean'][mi, cc]).astype(np.float32)],
        geo=[lat_c[cc].astype(np.float32), lon_c[cc].astype(np.float32),
             np.abs(lat_c[cc]).astype(np.float32)],
        mon=[np.sin(2 * np.pi * cal1 / 12).astype(np.float32),
             np.cos(2 * np.pi * cal1 / 12).astype(np.float32)],
    )
    return dict(feats=feats, y=y.astype(np.float64), ok=ok, ta=ta, cc=cc)


def X_of(pack, blocks):
    cols, nms = [], []
    for b in blocks:
        cols += pack['feats'][b]
        nms += BLOCK_NAMES[b]
    return np.column_stack(cols).astype(np.float32), nms


def wrec(ta):
    return np.where(ta // 12 <= 2009, 1.0, 2.0)


# ---------- row sets ----------
fit_strict = fit[fit['t_abs'] <= STRICT_MAX_T]
fitA_all = fit_strict[fit_strict['cal_mon'].isin(ANCHOR_CAL)]      # anchor-cal months
fitA_tr = fitA_all[fitA_all['t_abs'] < ES_START]                   # train part
fitB_tr = fit_strict[fit_strict['t_abs'] < ES_START]               # all-months train part
es_df = fit_strict[(fit_strict['t_abs'] >= ES_START)
                   & (fit_strict['cal_mon'].isin(ANCHOR_CAL))]     # ES: 2011-12 anchor months

P_A_all = build_rows(fitA_all)
P_A_tr = build_rows(fitA_tr)
P_B_tr = build_rows(fitB_tr)
P_es = build_rows(es_df)

val_anchor_set = set(va['t_abs'].tolist())
vk = val[val['t_abs'].isin(val_anchor_set)].reset_index(drop=True)
proto_has_next = vk['t_abs'].map(dict(zip(va['t_abs'], va['has_next']))).values.astype(bool)
P_v = build_rows(vk)
# enforce protocol: no covs(t+1) for without-group val rows
for j in range(5):
    P_v['feats']['covt1'][j][~proto_has_next] = np.nan
P_v['feats']['covt1'][5][:] = proto_has_next.astype(np.float32)

print(f"rows: fitA_train={len(fitA_tr):,} fitA_all={len(fitA_all):,} "
      f"fitB_train={len(fitB_tr):,} es={len(es_df):,} val={len(vk):,} "
      f"(with={proto_has_next.sum():,})  [{time.time()-t0:.0f}s]")

yv_raw = vk['target'].values.astype(np.float64)
mu_v = infra.mu_c[P_v['cc']].astype(np.float64)
with_grp = proto_has_next
without_grp = ~proto_has_next

PARAMS = dict(objective='regression', metric='rmse', learning_rate=0.07, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, num_threads=2, verbosity=-1)
MAX_ROUNDS, ES_PATIENCE = 2000, 100
results, importances, preds_store = {}, {}, {}


def run_lgbm(tag, tr_pack, es_pack, refit_pack, blocks, weights_mode='recency'):
    Xtr, nms = X_of(tr_pack, blocks)
    Xes, _ = X_of(es_pack, blocks)
    use_covt1 = 'covt1' in blocks
    sel_tr = tr_pack['ok'] if use_covt1 else np.ones(len(tr_pack['y']), bool)
    sel_es = es_pack['ok'] if use_covt1 else np.ones(len(es_pack['y']), bool)
    ytr = tr_pack['y'][sel_tr].astype(np.float32)
    yes = es_pack['y'][sel_es].astype(np.float32)
    w = wrec(tr_pack['ta'][sel_tr]) if weights_mode == 'recency' else np.ones(len(ytr))
    dtr = lgb.Dataset(Xtr[sel_tr], label=ytr, weight=w, feature_name=nms)
    des = lgb.Dataset(Xes[sel_es], label=yes, feature_name=nms)
    t1 = time.time()
    booster = lgb.train(PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[des],
                        callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    best_iter = booster.best_iteration
    imp = pd.Series(booster.feature_importance('gain'), index=nms).sort_values(ascending=False)
    importances[tag] = imp
    # refit on the full fit-period pack at best_iter
    Xrf, _ = X_of(refit_pack, blocks)
    sel_rf = refit_pack['ok'] if use_covt1 else np.ones(len(refit_pack['y']), bool)
    yrf = refit_pack['y'][sel_rf].astype(np.float32)
    wrf = wrec(refit_pack['ta'][sel_rf]) if weights_mode == 'recency' else np.ones(len(yrf))
    drf = lgb.Dataset(Xrf[sel_rf], label=yrf, weight=wrf, feature_name=nms)
    booster = lgb.train(PARAMS, drf, num_boost_round=best_iter)
    Xv, _ = X_of(P_v, blocks)
    pred = booster.predict(Xv).astype(np.float64)
    print(f"[{tag}] best_iter={best_iter} fit={time.time()-t1:.0f}s  top5: "
          + ", ".join(f"{k} {v/imp.sum():.0%}" for k, v in imp.head(5).items()))
    return pred


def eval_pred(name, pred_anom):
    pred = mu_v + pred_anom
    r_with, _ = rmse(pred[with_grp], yv_raw[with_grp])
    r_wo, _ = rmse(pred[without_grp], yv_raw[without_grp])
    mse_w = np.mean((pred[with_grp] - yv_raw[with_grp]) ** 2)
    mse_o = np.mean((pred[without_grp] - yv_raw[without_grp]) ** 2)
    testwt = np.sqrt(0.668 * mse_w + 0.332 * mse_o)
    results[name] = dict(with_=r_with, without=r_wo,
                         rowwise=float(np.sqrt(np.mean((pred - yv_raw) ** 2))), testwt=testwt)
    print(f"{name:46s} | with {r_with:.4f} | without {r_wo:.4f} | TESTWTD {testwt:.4f}")
    return pred


FULL_F = ['core', 'trend', 'covt', 'covt1', 'spatial', 'geo', 'mon']
FULL_R = ['core', 'trend', 'covt', 'spatial', 'geo', 'mon']

print("\n=== main runs (anchor-month training, recency-weighted, refit) ===")
predF = run_lgbm('F_anchor', P_A_tr, P_es, P_A_all, FULL_F)
eval_pred("LGBM-F (anchor-months train)", predF)
predR = run_lgbm('R_anchor', P_A_tr, P_es, P_A_all, FULL_R)
eval_pred("LGBM-R (anchor-months train)", predR)
eval_pred("LGBM F/R DUAL (anchor-months train)", np.where(with_grp, predF, predR))

print("\n=== ablation: all-months training ===")
predF_all = run_lgbm('F_all', P_B_tr, P_es, P_A_all, FULL_F)
eval_pred("LGBM-F (all-months train)", predF_all)
predR_all = run_lgbm('R_all', P_B_tr, P_es, P_A_all, FULL_R)
eval_pred("LGBM-R (all-months train)", predR_all)
eval_pred("LGBM F/R DUAL (all-months train)", np.where(with_grp, predF_all, predR_all))

print("\n=== ablation: unweighted (anchor-months) ===")
predF_nw = run_lgbm('F_anchor_nw', P_A_tr, P_es, P_A_all, FULL_F, 'none')
eval_pred("LGBM-F (anchor, unweighted)", predF_nw)

print("\n=== feature-block ablations (model F, anchor-months) ===")
for nm, blocks in {
    'core only': ['core'],
    'core+trend': ['core', 'trend'],
    'core+trend+covt': ['core', 'trend', 'covt'],
    'core+trend+covt1': ['core', 'trend', 'covt1'],
    'core+trend+covt+covt1': ['core', 'trend', 'covt', 'covt1'],
    '+spatial (full)': FULL_F,
    'no spatial': ['core', 'trend', 'covt', 'covt1', 'geo', 'mon'],
    'no trend': ['core', 'covt', 'covt1', 'spatial', 'geo', 'mon'],
    'no covt': ['core', 'trend', 'covt1', 'spatial', 'geo', 'mon'],
    'no geo/mon': ['core', 'trend', 'covt', 'covt1', 'spatial'],
    'minimal [core+covt1+spatial]': ['core', 'covt1', 'spatial'],
}.items():
    p = run_lgbm(f'F_blk[{nm}]', P_A_tr, P_es, P_A_all, blocks)
    eval_pred(f"LGBM-F {nm}", p)

print("\n=== feature-block ablations (model R, anchor-months) ===")
for nm, blocks in {
    'core only': ['core'],
    'core+trend': ['core', 'trend'],
    'core+trend+covt': ['core', 'trend', 'covt'],
    'full (no covt1)': FULL_R,
    'no spatial': ['core', 'trend', 'covt', 'geo', 'mon'],
}.items():
    p = run_lgbm(f'R_blk[{nm}]', P_A_tr, P_es, P_A_all, blocks)
    eval_pred(f"LGBM-R {nm}", p)

print("\n" + "=" * 78)
print("SUMMARY (strict honest protocol; ref: v4-linear dual with/without/testwtd = "
      "0.6688 / 0.5945 / 0.6451)")
print("=" * 78)
print(f"{'model':48s} {'with':>8s} {'without':>8s} {'TESTWTD':>8s}")
for k, v in results.items():
    print(f"{k:48s} {v['with_']:8.4f} {v['without']:8.4f} {v['testwt']:8.4f}")

print("\nTop-15 gain importances (F_anchor):")
print((importances['F_anchor'] / importances['F_anchor'].sum()).head(15).round(4).to_string())
print("\nTop-15 gain importances (R_anchor):")
print((importances['R_anchor'] / importances['R_anchor'].sum()).head(15).round(4).to_string())

np.savez_compressed(OUT,
                    pred_dual_anchor=np.where(with_grp, predF, predR).astype(np.float32),
                    pred_dual_all=np.where(with_grp, predF_all, predR_all).astype(np.float32),
                    predF=predF.astype(np.float32), predR=predR.astype(np.float32),
                    cc=P_v['cc'], ta=P_v['ta'], y=yv_raw.astype(np.float32),
                    mu=mu_v.astype(np.float32), with_grp=with_grp)
print(f"\nsaved val preds -> {OUT}  [{time.time()-t0:.0f}s total]")
