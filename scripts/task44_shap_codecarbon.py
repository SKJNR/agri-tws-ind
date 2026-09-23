"""task44_shap_codecarbon.py — report evidence run (Task 44, Sep 10).

Purpose (report §2 + §4, the two [V20] placeholders):
  1. Rebuild the FINAL k0-B compliant LightGBM EXACTLY per build_v24_compliant_k0.py
     (the k0 engine inside the selected slot-1 lineage v24->v28; 17 features,
     no raw lat/lon — the 19-Aug organizer-ruling compliance fix).
  2. SHAP: shap.TreeExplainer(bst).shap_values(X) on a 20k sample -> mean |SHAP|
     per feature, ranked (insert into report §2).
  3. CodeCarbon: EmissionsTracker wraps the WHOLE cycle (data load -> per-cell
     stats -> linear fits -> 500-round LGBM train -> 400-round reduced-LGBM train
     -> SHAP) -> kg CO2e for report §4.

All numbers land in download/report_shap_carbon.txt (+ .json for the report).
"""
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
OUT_TXT = '/home/z/my-project/download/report_shap_carbon.txt'
OUT_JSON = '/home/z/my-project/download/report_shap_carbon.json'
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

log_lines = []
def P(s=''):
    print(s, flush=True)
    log_lines.append(str(s))

# ------------------------------------------------------------------ CodeCarbon on
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(log_level='error')
tracker.start()
t0 = time.time()

# ================================================================== 1. data load (v24 verbatim)
P("=== [1] load train (v24 k0-B code path, verbatim) ===")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target'] + COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str) + '_' + train['lon'].round(2).astype(str)) \
    .astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
train['t_abs'] = (train['ym'] // 100) * 12 + (train['ym'] % 100) - 1
n_cells = int(train['cc'].max()) + 1
yms = np.sort(train['ym'].unique())
ym_to_i = {int(v): i for i, v in enumerate(yms)}
t_abs_train = np.array([(int(v) // 100) * 12 + (int(v) % 100) - 1 for v in yms], dtype=np.float64)

F = np.full((len(yms), n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_train[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxy = np.nansum(td * F64, axis=0)
sxx = np.nansum(td * td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), sxy / np.where(sxx > 0, sxx, 1), 0.0).astype('float32')
P(f"train rows={len(train)} cells={n_cells}  ({time.time()-t0:.1f}s)")

# ================================================================== 2. k0-B merge (v24 verbatim)
P("=== [2] k0-B Lk/Rk merge ===")
Lk = train[['cc', 't_abs', 'TWS_t', 'target'] + COVS].copy(); Lk['t_next'] = Lk['t_abs'] + 1
Rk = train[['cc', 't_abs'] + COVS].rename(columns={'t_abs': 't_next', **{c: c + '_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc', 't_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c + '_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values; yr = mgk['t_abs'].values // 12
colsR = [0, 1, 2, 3, 4, 5, 11]

def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr, dtype=np.float64) - tbar_c[cc_arr]) * beta_c[cc_arr]).astype(np.float32)

slow0 = trendex_vec(mgk['t_abs'].values, ccm)
slow1 = trendex_vec(mgk['t_next'].values, ccm)
Xlin = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm] - slow0,
    *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c + '_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
yB = (mgk['target'].values - mu_c[ccm] - slow1).astype('float32')
Xlin = np.nan_to_num(Xlin, nan=0.0)
w_recB = np.where(yr <= 2006, 1.0, np.where(yr <= 2009, 1.5, 2.0)).astype('float32')
swB = np.sqrt(w_recB); selF = has_nxt_tr

# linear fits (same as v24: two-component + reduced)
A_fB = Xlin[selF] * swB[selF, None]
coefF_B = np.linalg.solve(A_fB.T @ A_fB + 1e-3 * np.eye(12), A_fB.T @ (yB[selF] * swB[selF]))
A_rB = Xlin[:, colsR] * swB[:, None]
coefR_B = np.linalg.solve(A_rB.T @ A_rB + 1e-3 * np.eye(7), A_rB.T @ (yB * swB))
P(f"linear coef[FAST_est]={coefF_B[0]:.4f}  k0 rows={len(mgk)} (F={int(selF.sum())})  ({time.time()-t0:.1f}s)")

# ================================================================== 3. XLB (17 cols, compliant)
P("=== [3] XLB build (COMPLIANT: no raw lat/lon) + LGBM train ===")
mon_tr = (mgk['t_abs'].values % 12) + 1
XLB = np.column_stack([Xlin[:, 0], slow1, Xlin[:, 1:11],
                       has_nxt_tr.astype(np.float32),
                       np.sin(2 * np.pi * mon_tr / 12), np.cos(2 * np.pi * mon_tr / 12),
                       beta_c[ccm], mu_c[ccm]]).astype(np.float32)
okY = np.isfinite(yB)
FEATS = ['fast_resid(t)', 'trend(t+1)', 'SPEI_01(t)', 'SPEI_03(t)', 'SPEI_06(t)', 'SPEI_12(t)',
         'SOIL_MOIST(t)', 'SPEI_01(t+1)', 'SPEI_03(t+1)', 'SPEI_06(t+1)', 'SPEI_12(t+1)',
         'SOIL_MOIST(t+1)', 'has_covs(t+1)', 'sin(month)', 'cos(month)', 'beta_c', 'mu_c']
assert XLB.shape[1] == len(FEATS) == 17

params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=8)
ds = lgb.Dataset(XLB[selF], label=yB[selF], weight=w_recB[selF])
dsr = lgb.Dataset(XLB[~selF & okY], label=yB[~selF & okY], weight=w_recB[~selF & okY])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, dsr, num_boost_round=400)
P(f"bst(500 trees) + bst_r(400 trees) trained  ({time.time()-t0:.1f}s)")

# ================================================================== 4. SHAP
P("=== [4] SHAP TreeExplainer on bst (20k stratified sample of the full-k0 set) ===")
import shap
rng = np.random.default_rng(0)
idxF = np.where(selF)[0]
samp = rng.choice(idxF, size=min(20000, len(idxF)), replace=False)
Xs = XLb_S = XLB[samp]
expl = shap.TreeExplainer(bst)
sv = expl.shap_values(Xs)                     # (n, 17)
mabs = np.abs(sv).mean(axis=0)                # mean |SHAP| per feature
order = np.argsort(mabs)[::-1]
P("\nrank | feature | mean|SHAP| | LGB gain-importance (normalized)")
gain = bst.feature_importance('gain'); gainn = gain / gain.sum()
for r, i in enumerate(order, 1):
    P(f"{r:>2}  {FEATS[i]:<18} {mabs[i]:8.4f}   {gainn[i]:.3f}")

# ================================================================== 5. CodeCarbon off
tracker.stop()
emissions = float(tracker.final_emissions)
dur = time.time() - t0
try:
    cpu_kwh = float(tracker._total_cpu_energy.kWh)
except Exception:
    cpu_kwh = None
P(f"\n=== [5] CodeCarbon: {emissions:.5f} kg CO2e | wall {dur:.1f}s ({dur/60:.1f} min)"
  f" | cpu_kWh {cpu_kwh if cpu_kwh is None else round(cpu_kwh, 5)}")

res = dict(
    emissions_kg=float(f"{emissions:.6f}"),
    cpu_kwh=cpu_kwh,
    wall_seconds=round(dur, 1),
    trees=500, trees_reduced=400, features=FEATS,
    shap_mean_abs={FEATS[i]: float(f"{mabs[i]:.5f}") for i in order},
    lgb_gain_norm={FEATS[i]: float(f"{gainn[i]:.4f}") for i in order},
    top5_shap=[FEATS[i] for i in order[:5]],
    cycle_desc="full k0 cycle: CSV load -> per-cell stats -> linear fits -> 500-tree LGBM + 400-tree LGBM -> SHAP",
)
with open(OUT_JSON, 'w') as f:
    json.dump(res, f, indent=1)
with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(log_lines) + '\n')
print("saved:", OUT_TXT, "and", OUT_JSON)
