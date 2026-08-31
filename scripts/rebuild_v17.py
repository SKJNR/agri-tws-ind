"""
REBUILD v17 — post-reset reconstruction + decisive k=0 LGB experiment.

Context: env reset wiped the v5-v16 era (best was v12b LB 0.695357171).
Rebuilding from the V4 recipe (documented in TECHNICAL_HANDOFF.md) and
re-deriving improvements via the validated CV protocol (Task 12: Spearman=1.0,
overall CV-LB gap ~0.060).

New in v17 (the handoff's "decisive experiment", §6a):
  k=0 rows: LightGBM with covs(t+1), NaN-native handling, recency weights
             vs the incumbent linear model (honest val 0.6407).
  masked rows: phi-ensemble {0.70, 0.74, 0.80} over the v4a pipeline
             (PC-denoise init+obs, fwd+bwd Kalman, Dtil 0.70/0.45/0.073).

Protocol:
  CV: fit infra on train<=2012, evaluate on 2013-2015 with the test's
      calendar-month masking pattern (Task 12 protocol).
  FINAL: refit everything on full train, predict test.

Outputs: CV table, submission_v17a.csv (best), submission_v17b.csv (isolation).
"""
import numpy as np, pandas as pd, gc, time
import lightgbm as lgb

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
W1, W2, W3 = 0.70, 0.45, 0.073   # test-era Dtil weights (v4 LOO)
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ================= load =================
log("loading train...")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
log(f"train {len(train):,} rows, {n_cells} cells, months {int(train['ym'].min())}..{int(train['ym'].max())}")

log("loading test (masking pattern + final prediction)...")
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
assert (test['cc'] >= 0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)

cal_mask_frac = test.groupby(test['time'].dt.month)['masked'].mean()
log("calendar-month mask fractions: " + ", ".join(f"{m}:{f:.2f}" for m,f in cal_mask_frac.items()))

# ================= infra builder =================
def build_infra(fit_df, label):
    """All infrastructure from fit-era data only."""
    d = {}
    yms = np.sort(fit_df['ym'].unique()); T = len(yms)
    ym_to_i = {int(v):i for i,v in enumerate(yms)}
    t_abs_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)
    F = np.full((T, n_cells), np.nan, dtype=np.float32)
    F[fit_df['ym'].map(ym_to_i).values, fit_df['cc'].values] = fit_df['TWS_t'].values
    d['mu_c'] = np.nanmean(F, axis=0)
    d['clim'] = fit_df.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
    # global cov regression
    Z = fit_df[COVS].values.astype('float32'); yv = fit_df['TWS_t'].values.astype('float32')
    Z1 = np.column_stack([Z, np.ones(len(Z))])
    okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
    d['coef'] = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
    # per-cell trend
    F64 = F.astype(np.float64)
    ok_t = np.isfinite(F64)
    t_mat = np.where(ok_t, t_abs_arr[:, None], np.nan)
    tbar_c = np.nanmean(t_mat, axis=0)
    td = t_mat - tbar_c[None, :]
    sxx = np.nansum(td*td, axis=0)
    beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
    d['tbar_c'], d['beta_c'] = tbar_c, beta_c
    # PC-200 denoiser from detrended full-history cells
    full = ~np.isnan(F).any(axis=0)
    A_dt = F64 - d['mu_c'][None,:] - (t_abs_arr[:,None]-tbar_c[None,:])*beta_c[None,:]
    A = A_dt[:, full]; A = A - A.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    d['V'] = Vt[:200].T.astype(np.float32)
    d['full'] = full
    log(f"infra[{label}]: {full.sum()} full-history cells, V{d['V'].shape}")
    return d

def dn(infra, field, use_denoise=True):
    if not use_denoise:
        return field
    out = field.copy()
    x = field[infra['full']]
    okx = np.isfinite(x)
    V = infra['V']
    out[infra['full']] = np.where(okx, V @ (V.T @ np.where(okx, x, 0.0)), x)
    return out

# ================= masked-row pipeline (v4 recipe) =================
def run_masked(infra, ev_df, ev_tws_visible, phi, use_bwd=True, use_denoise=True, dtil_w=(W1,W2,W3)):
    """Full v4 pipeline on an evaluation window. Returns pred (len(ev_df),) NaN on unmasked."""
    mu_c = infra['mu_c']; n = len(ev_df)
    ta = ev_df['t_abs'].values; cc = ev_df['cc'].values; msk = ev_df['masked'].values
    covs_ok = np.isfinite(ev_df[COVS].values).all(axis=1)
    Zv = ev_df[COVS].values.astype('float32')
    cov_est = np.full(n, np.nan, dtype=np.float32)
    ok = covs_ok
    cov_est[ok] = np.column_stack([Zv[ok], np.ones(ok.sum())]) @ infra['coef']
    cov_field = {}
    for m in np.sort(ev_df['t_abs'].unique()):
        selm = (ta == m) & ok
        fm = np.full(n_cells, np.nan, dtype=np.float32)
        fm[cc[selm]] = cov_est[selm]
        cov_field[int(m)] = fm - mu_c
    all_m = np.array(sorted(cov_field.keys()))
    S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
    W = {int(m): cov_field[m] - S for m in all_m}
    mfrac = ev_df.groupby('t_abs')['masked'].mean()
    anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
    AF = {}
    tws_vis = ev_tws_visible.values
    for a in anchors:
        sel = (ta == a) & (~msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    w1,w2,w3 = dtil_w
    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)
    # calibration at anchors
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a); w_ = dn(infra, W[a], use_denoise)
        okc = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[okc], (AF[a]-dj)[okc])[0,1]); zs.append(np.var(w_[okc]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(LAM_F*var_f); R = max(varz - c_*c_/(LAM_F*var_f), 1e-4)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    def pass_kalman(i, tm, direction):
        f0 = dn(infra, AF[i] - Dtil(i), use_denoise)
        x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        rng = range(i+1, tm+1) if direction>0 else range(i-1, tm-1, -1)
        for m in rng:
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = dn(infra, W[m], use_denoise); okw = np.isfinite(w_)
                Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
                P = np.where(okw, (1-Kg*H)*P, P)
        return x, P
    pred = np.full(n, np.nan, dtype=np.float64)
    anchors_arr = np.array(anchors)
    masked_months = sorted(set(ta[msk].tolist()))
    for m in masked_months:
        tm = m + 1
        sel = np.where((ta == m) & msk)[0]
        if len(sel) == 0: continue
        i = int(anchors_arr[np.searchsorted(anchors_arr, m, side='right')-1])
        xf, Pf = pass_kalman(i, tm, +1)
        later = anchors_arr[anchors_arr > m]
        if use_bwd and len(later):
            k = int(later[0])
            xb, Pb = pass_kalman(k, tm, -1)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        dT = Dtil(tm)
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
    return pred

# ================= k=0 models =================
def build_k0_features(df, infra):
    """Feature matrix in anomaly space, with covs(t+1) via next-month merge."""
    L = df[['cc','t_abs','TWS_t','target']+COVS].copy(); L['t_next'] = L['t_abs']+1
    R = df[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
    mg = L.merge(R, on=['cc','t_next'], how='left')
    ccm = mg['cc'].values
    Xt = np.column_stack([
        mg['TWS_t'].values - infra['mu_c'][ccm],
        *[mg[c].values - infra['clim'][ccm, j] for j, c in enumerate(COVS)],
        *[mg[c+'_nxt'].values - infra['clim'][ccm, j] for j, c in enumerate(COVS)],
    ]).astype(np.float32)
    y = (mg['target'].values - infra['mu_c'][ccm]).astype(np.float32)
    yr = mg['t_abs'].values // 12
    return Xt, y, yr, mg['TWS_t'].notna().values

def k0_linear(Xtr, ytr, wtr, Xev):
    """Incumbent: recency-weighted ridge in anomaly space (v4 recipe, no intercept)."""
    has_nxt_tr = np.isfinite(Xtr[:, 6:11]).all(axis=1)
    sw = np.sqrt(wtr)
    Xtr0 = np.nan_to_num(Xtr, nan=0.0)
    A = Xtr0[has_nxt_tr]*sw[has_nxt_tr,None]
    coefF = np.linalg.solve(A.T@A + 1e-3*np.eye(11), A.T@(ytr[has_nxt_tr]*sw[has_nxt_tr]))
    colsR = np.array([0,1,2,3,4,5])
    A_r = Xtr0[:, colsR]*sw[:,None]
    coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(6), A_r.T@(ytr*sw))
    # on eval: rows with full features -> full model; else reduced (NaN->0)
    has_nxt_ev = np.isfinite(Xev[:, 6:11]).all(axis=1)
    Xe = np.nan_to_num(Xev, nan=0.0)
    pred = np.empty(len(Xev), dtype=np.float32)
    pred[has_nxt_ev] = Xe[has_nxt_ev] @ coefF
    pred[~has_nxt_ev] = Xe[~has_nxt_ev][:, colsR] @ coefR
    return pred

def k0_lgb(Xtr, ytr, wtr, Xev, rounds=400):
    params = dict(objective='regression', metric='rmse', num_leaves=63,
                  learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=2,
                  seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    bst = lgb.train(params, dtr, num_boost_round=rounds)
    return bst.predict(Xev, num_iteration=rounds), bst

# ================= CV PHASE =================
log("=== CV PHASE: fit<=2012, val=2013-2015 ===")
fit = train[train['time'].dt.year <= 2012].copy()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
log(f"fit {len(fit):,} / val {len(val):,}")
val['masked'] = val['time'].dt.month.map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_msk = val['masked'].values

infra_cv = build_infra(fit, 'cv')

# masked-row CV: A/B decomposition of v4 upgrades vs v2b recipe, then phi variants
log("masked-row CV: A/B decomposition (phi=0.74, Dtil fixed test weights)...")
for tag, ub, ud in [('v2b_equiv (no-dn, no-bwd)', False, False),
                    ('denoise only', False, True),
                    ('bwd only', True, False),
                    ('v4 full (dn+bwd)', True, True)]:
    p = run_masked(infra_cv, val, val_tws_visible, 0.74, use_bwd=ub, use_denoise=ud)
    mm = val_msk & np.isfinite(p) & np.isfinite(val_target)
    r = float(np.sqrt(np.mean((p[mm]-val_target[mm])**2)))
    log(f"  {tag:26s}: masked CV RMSE = {r:.4f}  (n={mm.sum():,})")
preds_phi = {}
for phi in [0.70, 0.74, 0.80]:
    preds_phi[phi] = run_masked(infra_cv, val, val_tws_visible, phi, use_bwd=True)
    mm = val_msk & np.isfinite(preds_phi[phi])
    r = float(np.sqrt(np.mean((preds_phi[phi][mm]-val_target[mm])**2)))
    log(f"  phi={phi}: masked CV RMSE = {r:.4f}  (n={mm.sum():,})")
ens = np.mean([preds_phi[p] for p in preds_phi], axis=0)
mm = val_msk & np.isfinite(ens)
r_ens = float(np.sqrt(np.mean((ens[mm]-val_target[mm])**2)))
log(f"  phi-ensemble: masked CV RMSE = {r_ens:.4f}")

# k=0 CV: linear vs LGB
log("k=0 CV (linear vs LGB, honest: labels strictly inside fit era)...")
# train rows for k=0 CV: t <= 2012-11 so that target (t+1) <= 2012-12 (no val leakage)
tmax_cv = 2012*12 + 10
tfit_k0 = train[train['t_abs'] <= tmax_cv]
fit_k0 = tfit_k0[tfit_k0['time'].dt.year <= 2012]
Xtr, ytr, yr, tws_ok_tr = build_k0_features(fit_k0, infra_cv)
ok_tr = np.isfinite(ytr) & tws_ok_tr
Xtr, ytr, yr = Xtr[ok_tr], ytr[ok_tr], yr[ok_tr]
wtr = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
log(f"  k=0 train rows: {len(Xtr):,}  (target months <= 2012-12)")
# eval rows: val unmasked with visible TWS_t and finite target
val_k0 = val.copy()  # TWS_t already NaN on masked rows
Xev, yev, _, tws_ok_ev = build_k0_features(val_k0, infra_cv)
ok_ev = (~val_msk) & tws_ok_ev & np.isfinite(yev)
Xev_o, yev_o = Xev[ok_ev], yev[ok_ev]
mu_ev = infra_cv['mu_c'][val['cc'].values[ok_ev]]
log(f"  k=0 eval rows: {len(Xev_o):,}")

p_lin = k0_linear(Xtr, ytr, wtr, Xev_o)
r_lin = float(np.sqrt(np.mean((p_lin-yev_o)**2)))
log(f"  LINEAR k=0 val RMSE = {r_lin:.4f}")

p_lgb, _ = k0_lgb(Xtr, ytr, wtr, Xev_o, rounds=400)
r_lgb = float(np.sqrt(np.mean((p_lgb-yev_o)**2)))
log(f"  LGBM   k=0 val RMSE = {r_lgb:.4f}  (400 rounds)")

# round-count sensitivity: subfit t<=2010-12, holdout t in 2011-01..2012-11
log("  round-count check (early stop on 2011-2012 holdout)...")
sub = yr <= 2010
dho_idx = (~sub)  # 2011..2012 rows (t_abs <= tmax_cv already ensures <= 2012-11)
Xes, yes, wes = Xtr[sub], ytr[sub], wtr[sub]
Xho, yho = Xtr[dho_idx], ytr[dho_idx]
dtr = lgb.Dataset(Xes, label=yes, weight=wes)
dho = lgb.Dataset(Xho, label=yho)
params = dict(objective='regression', metric='rmse', num_leaves=63,
              learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, num_threads=2,
              seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
bst_es = lgb.train(params, dtr, num_boost_round=1200, valid_sets=[dho],
                   callbacks=[lgb.early_stopping(75, verbose=False)])
best_it = bst_es.best_iteration or 400
if not (50 <= best_it <= 1200): best_it = 400
p_es = bst_es.predict(Xev_o, num_iteration=best_it)
r_es = float(np.sqrt(np.mean((p_es-yev_o)**2)))
log(f"  LGBM early-stopped: best_iter={best_it}, val RMSE = {r_es:.4f}")

# blend linear+LGB?
for bw in [0.0, 0.25, 0.5, 0.75, 1.0]:
    pb = (1-bw)*p_lin + bw*p_lgb
    log(f"  blend lin/lgb {1-bw:.2f}/{bw:.2f}: {float(np.sqrt(np.mean((pb-yev_o)**2))):.4f}")

# overall CV-equivalents (0.665 masked / 0.335 k0 as on test)
best_k0 = min(r_lin, r_lgb, r_es)
log(f"CV SUMMARY: masked best={min(r_ens, *[float(np.sqrt(np.mean((preds_phi[p][val_msk & np.isfinite(preds_phi[p])]-val_target[val_msk & np.isfinite(preds_phi[p])])**2))) for p in preds_phi]):.4f}, "
    f"k=0 linear={r_lin:.4f} lgb={r_lgb:.4f} es={r_es:.4f}")
log(f"  overall CV-equivalent (masked ens + best k0) = {np.sqrt(0.665*r_ens**2 + 0.335*best_k0**2):.4f}  -> projected LB ≈ +0.060")

# ================= FINAL PHASE =================
log("=== FINAL PHASE: refit on full train, predict test ===")
del fit, val, Xtr, ytr, Xev, preds_phi, ens; gc.collect()
infra_full = build_infra(train, 'full')

# masked: phi-ensemble with test Dtil weights (0.70/0.45/0.073)
tmf = test.groupby('t_abs')['masked'].mean()
log(f"test anchors: {sorted(int(v) for v in tmf[tmf<0.01].index)}")

test_for_pred = test.copy()
test_for_pred.loc[test_for_pred['masked'], 'TWS_t'] = np.nan
final_masked = {}
for phi in [0.70, 0.74, 0.80]:
    final_masked[phi] = run_masked(infra_full, test_for_pred, test_for_pred['TWS_t'], phi, use_bwd=True)
    log(f"  phi={phi}: test masked pred mean={np.nanmean(final_masked[phi]):.4f} std={np.nanstd(final_masked[phi]):.4f}")
masked_ens = np.mean([final_masked[p] for p in final_masked], axis=0)

# k=0 final: LGB on full train (rounds from CV choice)
RoundsFinal = int(best_it) if (100 <= best_it <= 1200) else 400
log(f"final k=0 LGB rounds: {RoundsFinal}")
XtrF, ytrF, yrF, okF = build_k0_features(train, infra_full)
okF = okF & np.isfinite(ytrF)
XtrF, ytrF, yrF = XtrF[okF], ytrF[okF], yrF[okF]
wtrF = np.where(yrF <= 2009, 1.0, np.where(yrF <= 2012, 2.0, 3.0)).astype(np.float32)
XevF, _, _, ok_evF = build_k0_features(test_for_pred, infra_full)
sel_k0 = (~test['masked'].values) & ok_evF
log(f"final k=0 rows: {sel_k0.sum():,}")
p_lgbF, bstF = k0_lgb(XtrF, ytrF, wtrF, XevF, rounds=RoundsFinal)
# linear fallback for rows LGB can't serve (should be none — NaN-native)
k0_pred = np.full(len(test), np.nan, dtype=np.float64)
k0_pred[sel_k0] = p_lgbF[sel_k0]
# any unmasked rows missed (no TWS_t)? fallback mu_c
unm = ~test['masked'].values
miss = unm & np.isnan(k0_pred)
if miss.any():
    log(f"  fallback mu_c for {miss.sum()} unmasked rows without TWS_t")
    k0_pred[miss] = infra_full['mu_c'][test['cc'].values[miss]]

# assemble
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
mu_all = infra_full['mu_c'][test['cc'].values]

def assemble(masked_pred, k0_p, tag):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_p[unm]
    pred[~unm] = masked_pred[~unm]
    pred = np.where(np.isnan(pred), mu_all, pred)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    log(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} "
        f"(k0 std={pred[unm].std():.3f}, masked std={pred[~unm].std():.3f})")

# v17a: phi-ensemble + LGB k=0
assemble(masked_ens, k0_pred, 'v17a')
# v17b: phi=0.74 only + LGB k=0 (isolates ensemble effect)
assemble(final_masked[0.74], k0_pred, 'v17b')

# linear k0 final (for comparison / possible v17c)
p_linF = k0_linear(XtrF, ytrF, wtrF, XevF)
k0_lin = np.full(len(test), np.nan, dtype=np.float64)
k0_lin[sel_k0] = p_linF[sel_k0]
k0_lin[miss] = mu_all[miss]
assemble(masked_ens, k0_lin, 'v17c')

log("DONE")
