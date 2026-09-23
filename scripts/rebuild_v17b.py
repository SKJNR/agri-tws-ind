"""
REBUILD v17b — corrected CV protocol + masked-row improvement grid.

KEY FINDING (v17 run): the val protocol has a SHORTCUT — masked val months
May and Aug have their target month (Jun/Sep) UNMASKED, so the backward pass
reads the answer. On TEST the designers removed every pre-anchor month
(no May-16/Nov-16/Oct-18...), sealing that route (~379 copyable rows only).
=> All masked-row CV must be evaluated on HONEST rows only
   (masked rows whose t+1 is masked or absent), matching test structure.

Also fixed vs v17: k=0 has NO shortcut (features never see TWS(t+1)) —
k=0 val numbers are honest as-is. LGB ~= linear (0.6273 vs 0.6287),
50/50 blend best (0.6254).

This script:
  1. honest-CV A/B grid over masked-row knobs:
     denoise on/off, bwd on/off, Dhat-denoise, blend mode, bwd gap cap
  2. phi spread + ensemble on best config
  3. k=0: linear + LGB blend (fixed 400 rounds)
  4. FINAL on test + direct-copy rows (exact target identity) + assemble
"""
import numpy as np, pandas as pd, gc, time, warnings
import lightgbm as lgb
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
W1, W2, W3 = 0.70, 0.45, 0.073
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

log("loading test...")
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

# ================= infra =================
def build_infra(fit_df, label):
    d = {}
    yms = np.sort(fit_df['ym'].unique()); T = len(yms)
    ym_to_i = {int(v):i for i,v in enumerate(yms)}
    t_abs_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)
    F = np.full((T, n_cells), np.nan, dtype=np.float32)
    F[fit_df['ym'].map(ym_to_i).values, fit_df['cc'].values] = fit_df['TWS_t'].values
    with np.errstate(all='ignore'):
        d['mu_c'] = np.where(np.isfinite(np.nanmean(F, axis=0)), np.nanmean(F, axis=0), 0.0).astype(np.float32)
    d['clim'] = fit_df.groupby('cc')[COVS].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
    Z = fit_df[COVS].values.astype('float32'); yv = fit_df['TWS_t'].values.astype('float32')
    Z1 = np.column_stack([Z, np.ones(len(Z))])
    okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
    d['coef'] = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
    F64 = F.astype(np.float64)
    ok_t = np.isfinite(F64)
    t_mat = np.where(ok_t, t_abs_arr[:, None], np.nan)
    tbar_c = np.nanmean(t_mat, axis=0)
    td = t_mat - tbar_c[None, :]
    sxx = np.nansum(td*td, axis=0)
    beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
    d['tbar_c'], d['beta_c'] = tbar_c, beta_c
    full = ~np.isnan(F).any(axis=0)
    A_dt = F64 - d['mu_c'][None,:] - (t_abs_arr[:,None]-tbar_c[None,:])*beta_c[None,:]
    A = A_dt[:, full]; A = A - A.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    d['V'] = Vt[:200].T.astype(np.float32)
    d['full'] = full
    log(f"infra[{label}]: {full.sum()} full cells, V{d['V'].shape}")
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

# ================= masked pipeline =================
def run_masked(infra, ev_df, ev_tws_visible, phi, use_bwd=True, use_denoise=True,
               dhat_denoise=False, blend='precision', bwd_max_gap=None, dtil_w=(W1,W2,W3),
               lam=LAM_F):
    mu_c = infra['mu_c']; n = len(ev_df)
    ta = ev_df['t_abs'].values; cc = ev_df['cc'].values; msk = ev_df['masked'].values
    Zv = ev_df[COVS].values.astype('float32')
    ok = np.isfinite(Zv).all(axis=1)
    cov_est = np.full(n, np.nan, dtype=np.float32)
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
    tws_vis = ev_tws_visible.values
    AF = {}
    for a in anchors:
        sel = (ta == a) & (~msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    if dhat_denoise:
        Dhat = np.nanmean(np.array([dn(infra, AF[a], use_denoise) for a in anchors]), axis=0)
    else:
        Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    w1,w2,w3 = dtil_w
    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a); w_ = dn(infra, W[a], use_denoise)
        okc = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[okc], (AF[a]-dj)[okc])[0,1]); zs.append(np.var(w_[okc]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(lam*var_f); R = max(varz - c_*c_/(lam*var_f), 1e-4)
    q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f
    def pass_kalman(i, tm, direction):
        f0 = dn(infra, AF[i] - Dtil(i), use_denoise)
        x = np.where(np.isfinite(f0), lam*np.nan_to_num(f0), 0.0).astype(np.float32)
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
        use_b = use_bwd and len(later) and (bwd_max_gap is None or (int(later[0]) - tm) <= bwd_max_gap)
        if use_b:
            k = int(later[0])
            xb, Pb = pass_kalman(k, tm, -1)
            if blend == 'precision':
                wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            else:
                wgt = np.float32(blend)
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        dT = Dtil(tm)
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
    return pred

# ================= k=0 models =================
def build_k0_features(df, infra):
    has_target = 'target' in df.columns
    cols = ['cc','t_abs','TWS_t']+(['target'] if has_target else [])+COVS
    L = df[cols].copy(); L['t_next'] = L['t_abs']+1
    R = df[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
    mg = L.merge(R, on=['cc','t_next'], how='left')
    ccm = mg['cc'].values
    Xt = np.column_stack([
        mg['TWS_t'].values - infra['mu_c'][ccm],
        *[mg[c].values - infra['clim'][ccm, j] for j, c in enumerate(COVS)],
        *[mg[c+'_nxt'].values - infra['clim'][ccm, j] for j, c in enumerate(COVS)],
    ]).astype(np.float32)
    if has_target:
        y = (mg['target'].values - infra['mu_c'][ccm]).astype(np.float32)
    else:
        y = np.full(len(mg), np.nan, dtype=np.float32)
    yr = mg['t_abs'].values // 12
    return Xt, y, yr, mg['TWS_t'].notna().values

def k0_linear(Xtr, ytr, wtr, Xev):
    has_nxt_tr = np.isfinite(Xtr[:, 6:11]).all(axis=1)
    sw = np.sqrt(wtr)
    Xtr0 = np.nan_to_num(Xtr, nan=0.0)
    A = Xtr0[has_nxt_tr]*sw[has_nxt_tr,None]
    coefF = np.linalg.solve(A.T@A + 1e-3*np.eye(11), A.T@(ytr[has_nxt_tr]*sw[has_nxt_tr]))
    colsR = np.array([0,1,2,3,4,5])
    A_r = Xtr0[:, colsR]*sw[:,None]
    coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(6), A_r.T@(ytr*sw))
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
    return bst.predict(Xev)

# ================= CV PHASE =================
log("=== CV PHASE: fit<=2012, val=2013-2015 (HONEST protocol) ===")
fit = train[train['time'].dt.year <= 2012].copy()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val['masked'] = val['time'].dt.month.map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_msk = val['masked'].values
val_ta = val['t_abs'].values

# honest mask: exclude masked rows whose t+1 is an unmasked val month (shortcut)
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val_ta])
honest = val_msk & ~shortcut
log(f"val masked rows: {val_msk.sum():,} | shortcut (t+1 unmasked): {shortcut.sum():,} | honest: {honest.sum():,}")

infra_cv = build_infra(fit, 'cv')

def cv_eval(**kw):
    phi = kw.pop('phi', 0.74)
    p = run_masked(infra_cv, val, val_tws_visible, phi, **kw)
    mm = honest & np.isfinite(p) & np.isfinite(val_target)
    return float(np.sqrt(np.mean((p[mm]-val_target[mm])**2))), int(mm.sum())

log("\n--- grid A: phi x lam (bwd-only, no denoise) ---")
gridA = {}
for phi in [0.74, 0.80, 0.85, 0.90]:
    for lam in [0.80, 0.84, 0.88]:
        r, n = cv_eval(phi=phi, use_denoise=False, use_bwd=True, lam=lam)
        gridA[(phi, lam)] = r
        log(f"  phi={phi} lam={lam}: honest CV = {r:.4f}")
bestA = min(gridA, key=gridA.get)
log(f"best A: phi={bestA[0]}, lam={bestA[1]} -> {gridA[bestA]:.4f}")

log("\n--- grid B: Dtil weights on best A ---")
gridB = {}
for w in [(0.70,0.45,0.073), (0.786,0.218,0.049), (0.65,0.50,0.073), (0.75,0.35,0.073), (0.60,0.55,0.073)]:
    r, n = cv_eval(phi=bestA[0], use_denoise=False, use_bwd=True, lam=bestA[1], dtil_w=w)
    gridB[w] = r
    log(f"  w={w}: honest CV = {r:.4f}")
bestB = min(gridB, key=gridB.get)
log(f"best B: {bestB} -> {gridB[bestB]:.4f}")

log("\n--- top-3 config ensemble check ---")
top3 = sorted(gridA, key=gridA.get)[:3]
preds_top3 = [run_masked(infra_cv, val, val_tws_visible, p, use_denoise=False, use_bwd=True, lam=l, dtil_w=bestB)
              for p, l in top3]
ens3 = np.mean(preds_top3, axis=0)
mm = honest & np.isfinite(ens3) & np.isfinite(val_target)
r_ens3 = float(np.sqrt(np.mean((ens3[mm]-val_target[mm])**2)))
log(f"  top3 {top3} ensemble: honest CV = {r_ens3:.4f} (single best {gridA[bestA]:.4f})")

log("\n--- denoiser hedge test (phi=0.80, lam=0.84): nd vs dn vs 50/50 ---")
p_nd = run_masked(infra_cv, val, val_tws_visible, 0.80, use_denoise=False, use_bwd=True, lam=0.84, dtil_w=bestB)
p_dn = run_masked(infra_cv, val, val_tws_visible, 0.80, use_denoise=True, use_bwd=True, lam=0.84, dtil_w=bestB)
p_hb = 0.5*(p_nd + p_dn)
for tag, p in [('no-denoise', p_nd), ('denoise', p_dn), ('50/50 blend', p_hb)]:
    m2 = honest & np.isfinite(p) & np.isfinite(val_target)
    log(f"  {tag:12s}: honest CV = {float(np.sqrt(np.mean((p[m2]-val_target[m2])**2))):.4f}")

# k=0 CV (honest by construction — features never see TWS(t+1))
log("\n--- k=0 CV (linear vs LGB vs blend) ---")
tmax_cv = 2012*12 + 10
tfit = train[train['t_abs'] <= tmax_cv]
fit_k0 = tfit[tfit['time'].dt.year <= 2012]
Xtr, ytr, yr, tws_ok = build_k0_features(fit_k0, infra_cv)
ok_tr = np.isfinite(ytr) & tws_ok
Xtr, ytr, yr = Xtr[ok_tr], ytr[ok_tr], yr[ok_tr]
wtr = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
val_k0 = val.copy()
Xev, yev, _, tws_ok_ev = build_k0_features(val_k0, infra_cv)
ok_ev = (~val_msk) & tws_ok_ev & np.isfinite(yev)
Xev_o, yev_o = Xev[ok_ev], yev[ok_ev]
log(f"  k=0 train: {len(Xtr):,}  eval: {len(Xev_o):,}")
p_lin = k0_linear(Xtr, ytr, wtr, Xev_o)
r_lin = float(np.sqrt(np.mean((p_lin-yev_o)**2)))
p_lgb = k0_lgb(Xtr, ytr, wtr, Xev_o, rounds=400)
r_lgb = float(np.sqrt(np.mean((p_lgb-yev_o)**2)))
p_bd = 0.5*p_lin + 0.5*p_lgb
r_bd = float(np.sqrt(np.mean((p_bd-yev_o)**2)))
log(f"  LINEAR: {r_lin:.4f}   LGBM(400): {r_lgb:.4f}   50/50 blend: {r_bd:.4f}")

log(f"\nCV SUMMARY (HONEST): masked best single = {gridA[bestA]:.4f}, top3 ens = {r_ens3:.4f} | k=0 best = {min(r_lin, r_lgb, r_bd):.4f}")

# ================= FINAL PHASE =================
log("\n=== FINAL PHASE: refit on full train, predict test ===")
del fit, val, preds_top3, ens3; gc.collect()
infra_full = build_infra(train, 'full')

test_for_pred = test.copy()
test.loc[test['masked'], 'TWS_t'] = np.nan
test_for_pred.loc[test_for_pred['masked'], 'TWS_t'] = np.nan

# masked: top-3 config ensemble (honest-CV best) with best Dtil weights
final_masked = {}
for phi, lam in top3:
    final_masked[(phi, lam)] = run_masked(infra_full, test_for_pred, test_for_pred['TWS_t'],
                                           phi, use_denoise=False, use_bwd=True, lam=lam, dtil_w=bestB)
    log(f"  phi={phi} lam={lam}: masked pred std={np.nanstd(final_masked[(phi, lam)]):.4f}")
masked_ens = np.mean([final_masked[c] for c in final_masked], axis=0)
# single best for isolation variant
phi_b, lam_b = bestA
masked_single = final_masked[(phi_b, lam_b)]

# k=0 final: linear + LGB blend
XtrF, ytrF, yrF, okF = build_k0_features(train, infra_full)
okF = okF & np.isfinite(ytrF)
XtrF, ytrF, yrF = XtrF[okF], ytrF[okF], yrF[okF]
wtrF = np.where(yrF <= 2009, 1.0, np.where(yrF <= 2012, 2.0, 3.0)).astype(np.float32)
XevF, _, _, ok_evF = build_k0_features(test_for_pred, infra_full)
unm = ~test['masked'].values
sel_k0 = unm & ok_evF
log(f"  final k=0 rows: {sel_k0.sum():,}")
p_linF = k0_linear(XtrF, ytrF, wtrF, XevF)
p_lgbF = k0_lgb(XtrF, ytrF, wtrF, XevF, rounds=400)
k0_pred = np.full(len(test), np.nan, dtype=np.float64)
# CRITICAL FIX: p_linF/p_lgbF are ANOMALY predictions — add mu_c back
mu_all = infra_full['mu_c'][test['cc'].values]
k0_pred[sel_k0] = mu_all[sel_k0] + 0.5*p_linF[sel_k0] + 0.5*p_lgbF[sel_k0]
miss = unm & np.isnan(k0_pred)
if miss.any():
    log(f"  fallback mu_c for {miss.sum()} unmasked rows")
    k0_pred[miss] = mu_all[miss]

# direct-copy (vectorized): rows whose target month (t+1) has visible TWS_t in test
log("checking direct-copy rows (target = TWS_t(t+1) identity)...")
vis = test.loc[unm & np.isfinite(test['TWS_t'].values), ['cc','t_abs','TWS_t']]
# map: (cell, month) -> TWS at that month; lookup for row (c,t) is (c, t+1)
vis_map = {(int(c), int(t)): v for c, t, v in zip(vis['cc'], vis['t_abs'], vis['TWS_t'])}
cc_all = test['cc'].values; ta_all = test['t_abs'].values
keys = list(zip(cc_all.tolist(), (ta_all+1).tolist()))
copy_vals = np.array([vis_map.get(k, np.nan) for k in keys], dtype=np.float64)
n_copy = int(np.isfinite(copy_vals).sum())
log(f"  direct-copyable rows: {n_copy} ({n_copy/len(test)*100:.2f}%)")

def assemble(masked_pred, k0_p, tag, use_copy=True):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_p[unm]
    pred[~unm] = masked_pred[~unm]
    pred = np.where(np.isnan(pred), mu_all, pred)
    if use_copy:
        pred = np.where(np.isfinite(copy_vals), copy_vals, pred)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    log(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} "
        f"(k0 std={pred[unm].std():.3f}, masked std={pred[~unm].std():.3f}, copies={n_copy if use_copy else 0})")

sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
log("\nassembling submissions...")
# v17a: best config (top3 ensemble masked + k0 blend)
assemble(masked_ens, k0_pred, 'v17a')
# v17b: denoiser hedge — 50/50 denoised/non-denoised masked (phi=0.80, lam=0.84) + same k0
masked_hedge = 0.5*(run_masked(infra_full, test_for_pred, test_for_pred['TWS_t'], 0.80,
                                 use_denoise=False, use_bwd=True, lam=0.84, dtil_w=bestB)
                    + run_masked(infra_full, test_for_pred, test_for_pred['TWS_t'], 0.80,
                                 use_denoise=True, use_bwd=True, lam=0.84, dtil_w=bestB))
assemble(masked_hedge, k0_pred, 'v17b')
log("DONE")
