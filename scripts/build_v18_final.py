"""
BUILD v18 FINAL — assemble submission_v18a (top-3 ensemble) + v18b (single best).

Winner config family (round-2 grid, wins on BOTH M1 and M2):
  phi=0.80/0.85, lam=0.80/0.84, bwd_max_gap=8, dhat_tau=24 (mode=full), smooth_sigma=2.0
vs v17b baseline: M1 0.6319->0.6077 (-0.024), M2 0.6905->0.6688 (-0.022).

Also: k0 spatial-smoothing A/B on standard val (include only if >0.002 win).
Pre-registered LB projection recorded at the end with decision rules.
"""
import numpy as np, pandas as pd, gc, time, warnings
import lightgbm as lgb
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
LOG = []
def P(s=''):
    print(s, flush=True); LOG.append(str(s))
def log(msg): P(f"[{time.time()-t0:7.1f}s] {msg}")

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)

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
    log(f"infra[{label}]: {full.sum()} full cells")
    return d

_SMOOTH = {}
def get_smoother(sigma_deg):
    key = float(sigma_deg)
    if key in _SMOOTH: return _SMOOTH[key]
    la = np.deg2rad(cell_xy[:,0]); lo = np.deg2rad(cell_xy[:,1])
    pts = np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
    tree = cKDTree(pts)
    rad = np.deg2rad(3.0*sigma_deg)
    rows, cols, vals = [], [], []
    for i in range(len(pts)):
        nb = tree.query_ball_point(pts[i], rad)
        if len(nb) == 0: nb = [i]
        d2 = ((pts[nb]-pts[i])**2).sum(axis=1)
        w = np.exp(-d2/(2*np.deg2rad(sigma_deg)**2))
        w = w/np.maximum(w.sum(), 1e-12)
        rows.extend([i]*len(nb)); cols.extend(nb); vals.extend(w.tolist())
    K = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    _SMOOTH[key] = K
    return K

def run_masked_v18(infra, ev_df, ev_tws_visible, phi, use_bwd=True,
                   dtil_w=(0.70,0.45,0.073), lam=0.84, bwd_max_gap=None,
                   dhat_tau=None, dhat_mode='full', smooth_sigma=None):
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
    Dhat_s = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    w1,w2,w3 = dtil_w
    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil_s(t): return (w1*Dhat_s + w2*S + w3*trendex(t)).astype(np.float32)
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil_s(a); w_ = W[a]
        okc = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[okc], (AF[a]-dj)[okc])[0,1]); zs.append(np.var(w_[okc]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(lam*var_f); R = max(varz - c_*c_/(lam*var_f), 1e-4)
    q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f
    A_arr = np.array(sorted(AF.keys()), dtype=np.float64)
    A_stack = np.array([AF[a] for a in A_arr])
    A_fin = np.isfinite(A_stack)
    def Dhat_era(t):
        wts = np.exp(-np.abs(A_arr - float(t))/dhat_tau)[:, None] * A_fin
        num = (np.where(A_fin, A_stack, 0.0) * wts).sum(axis=0)
        den = wts.sum(axis=0)
        return np.where(den > 1e-6, num/np.maximum(den,1e-6), Dhat_s).astype(np.float32)
    def Dtil_b(t, era):
        if not era: return Dtil_s(t)
        return (w1*Dhat_era(t) + w2*S + w3*trendex(t)).astype(np.float32)
    Sm = get_smoother(smooth_sigma) if smooth_sigma else None
    def pass_kalman(i, tm, direction):
        f0 = AF[i] - Dtil_b(i, era=(dhat_tau is not None and dhat_mode=='full'))
        x = np.where(np.isfinite(f0), lam*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        rng = range(i+1, tm+1) if direction>0 else range(i-1, tm-1, -1)
        for m in rng:
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = W[m]; okw = np.isfinite(w_)
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
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        if Sm is not None:
            x = Sm @ x
        dT = Dtil_b(tm, era=(dhat_tau is not None))
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
    return pred

# ---------- k0 machinery (verbatim v17b) ----------
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
    y = (mg['target'].values - infra['mu_c'][ccm]).astype(np.float32) if has_target else np.full(len(mg), np.nan, dtype=np.float32)
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

# ============ PHASE A: k0 smoothing A/B on standard val ============
log("=== k0 spatial-smoothing A/B (standard val, fit<=2012) ===")
fit = train[train['time'].dt.year <= 2012].copy()
infra_cv = build_infra(fit, 'cv')
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
cal_mask_frac = test.groupby(test['time'].dt.month)['masked'].mean()
val['masked'] = val['time'].dt.month.map(cal_mask_frac).values > 0.5
val_msk = val['masked'].values
tfit = train[train['t_abs'] <= 2012*12+10]; fit_k0 = tfit[tfit['time'].dt.year <= 2012]
Xtr, ytr, yr, tws_ok = build_k0_features(fit_k0, infra_cv)
ok_tr = np.isfinite(ytr) & tws_ok
Xtr, ytr, yr = Xtr[ok_tr], ytr[ok_tr], yr[ok_tr]
wtr = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
Xev, yev, _, tws_ok_ev = build_k0_features(val, infra_cv)
ok_ev = (~val_msk) & tws_ok_ev & np.isfinite(yev)
Xev_o, yev_o = Xev[ok_ev], yev[ok_ev]
ev_rows = val[ok_ev]
log(f"  k0 train: {len(Xtr):,}  eval: {len(Xev_o):,}")
p_lin = k0_linear(Xtr, ytr, wtr, Xev_o)
p_lgb = k0_lgb(Xtr, ytr, wtr, Xev_o, rounds=400)
p_blend = 0.5*p_lin + 0.5*p_lgb
r0 = float(np.sqrt(np.mean((p_blend-yev_o)**2)))
P(f"  k0 blend (v17): {r0:.4f}  [v17b log: 0.6254]")
# smoothing A/B: smooth anomaly predictions per month across cells
best_k0 = (None, r0)
for sig in [1.0, 1.5, 2.0]:
    K = get_smoother(sig)
    ta_ev = ev_rows['t_abs'].values; cc_ev = ev_rows['cc'].values
    fld = np.full(n_cells, np.nan, dtype=np.float64)
    smoothed = np.empty(len(p_blend))
    for m in np.unique(ta_ev):
        selm = ta_ev == m
        fld[:] = np.nan
        fld[cc_ev[selm]] = p_blend[selm]
        sm = K @ np.nan_to_num(fld, nan=0.0)
        smoothed[selm] = sm[cc_ev[selm]]
    rr = float(np.sqrt(np.mean((smoothed-yev_o)**2)))
    P(f"  k0 blend + smooth sigma={sig}: {rr:.4f}  ({rr-r0:+.4f})")
    if rr < best_k0[1]: best_k0 = (sig, rr)
K0_SIGMA = best_k0[0]
# DISCIPLINE OVERRIDE: -0.0013 is below the pre-registered 0.002 inclusion bar and
# inside CV noise (2SE~0.003). Revert k0 to the EXACT v17b block. Side benefit:
# v18a vs v17b then isolates the masked block exactly (same k0 rows both files).
K0_SIGMA = None
P(f"  => k0 smoothing verdict: sigma={best_k0[0]} ({best_k0[1]-r0:+.4f}) BUT below inclusion bar -> OFF (v17b k0 kept)")

# ============ PHASE B: confirm winner + ensemble on M1/M2 ============
log("=== M1/M2 confirmation of final configs ===")
val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()
M1_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]
M1_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2, 24173:3, 24176:3, 24177:3,
           24178:3, 24179:3, 24180:3, 24187:4}
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
M2_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2, 24173:3, 24176:3, 24177:3,
           24178:3, 24179:3, 24180:3, 24181:3, 24182:3, 24183:3, 24187:4}
def make_mirror(anchors, runs):
    months = set(anchors) | set(runs)
    mir = val_all[val_all['t_abs'].isin(months)].copy()
    mir['masked'] = mir['t_abs'].isin(runs)
    mir['TWS_t'] = val_tws_visible.loc[mir.index].values
    mir.loc[mir['masked'], 'TWS_t'] = np.nan
    return mir
M1 = make_mirror(M1_ANCHORS, M1_RUNS); M2 = make_mirror(M2_ANCHORS, M2_RUNS)
def sc(mir, pred):
    mm = mir['masked'].values & np.isfinite(pred) & np.isfinite(mir['target'].values)
    return float(np.sqrt(np.mean((pred[mm]-mir['target'].values[mm])**2)))
FINAL_CFGS = [
    dict(phi=0.80, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.85, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.80, lam=0.84, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
]
preds = {c: {} for c in range(len(FINAL_CFGS))}
for i, c in enumerate(FINAL_CFGS):
    kw = dict(c); phi = kw.pop('phi')
    p1 = run_masked_v18(infra_cv, M1, M1['TWS_t'], phi, **kw)
    p2 = run_masked_v18(infra_cv, M2, M2['TWS_t'], phi, **kw)
    preds[i]['M1'], preds[i]['M2'] = p1, p2
    P(f"  cfg{i} phi={c['phi']} lam={c['lam']}: M1={sc(M1,p1):.4f} | M2={sc(M2,p2):.4f}")
pe1 = np.mean([preds[i]['M1'] for i in range(len(FINAL_CFGS))], axis=0)
pe2 = np.mean([preds[i]['M2'] for i in range(len(FINAL_CFGS))], axis=0)
P(f"  ENSEMBLE top3:                 M1={sc(M1,pe1):.4f} | M2={sc(M2,pe2):.4f}")
P(f"  (v17b baseline: M1=0.6319 M2=0.6905; implied LB masked 0.7355; gap M2 +0.045)")

# ============ PHASE C: final test predictions ============
log("=== FINAL PHASE: refit on full train, predict test ===")
del fit, val, val_all, M1, M2, preds, pe1, pe2; gc.collect()
infra_full = build_infra(train, 'full')
test_for_pred = test.copy()
test.loc[test['masked'], 'TWS_t'] = np.nan
test_for_pred.loc[test_for_pred['masked'], 'TWS_t'] = np.nan
unm = ~test['masked'].values

final_masked = []
for c in FINAL_CFGS:
    kw = dict(c); phi = kw.pop('phi')
    p = run_masked_v18(infra_full, test_for_pred, test_for_pred['TWS_t'], phi, **kw)
    final_masked.append(p)
    log(f"  phi={phi} lam={kw['lam']}: masked pred std={np.nanstd(p):.4f}")
masked_ens = np.mean(final_masked, axis=0)
masked_single = final_masked[0]

# k0 final
XtrF, ytrF, yrF, okF = build_k0_features(train, infra_full)
okF = okF & np.isfinite(ytrF)
XtrF, ytrF, yrF = XtrF[okF], ytrF[okF], yrF[okF]
wtrF = np.where(yrF <= 2009, 1.0, np.where(yrF <= 2012, 2.0, 3.0)).astype(np.float32)
XevF, _, _, ok_evF = build_k0_features(test_for_pred, infra_full)
sel_k0 = unm & ok_evF
log(f"  final k0 rows: {sel_k0.sum():,}")
p_linF = k0_linear(XtrF, ytrF, wtrF, XevF)
p_lgbF = k0_lgb(XtrF, ytrF, wtrF, XevF, rounds=400)
mu_all = infra_full['mu_c'][test['cc'].values]
k0_anom = 0.5*p_linF + 0.5*p_lgbF
if K0_SIGMA:
    K = get_smoother(K0_SIGMA)
    ta_t = test['t_abs'].values; cc_t = test['cc'].values
    k0_sm = np.full(len(test), np.nan)
    fld = np.full(n_cells, np.nan)
    for m in np.unique(ta_t[sel_k0]):
        selm = sel_k0 & (ta_t == m)
        fld[:] = np.nan
        fld[cc_t[selm]] = k0_anom[selm]
        sm = K @ np.nan_to_num(fld, nan=0.0)
        k0_sm[selm] = sm[cc_t[selm]]
    k0_anom = np.where(np.isfinite(k0_sm), k0_sm, k0_anom)
    log(f"  k0 smoothing applied (sigma={K0_SIGMA})")
k0_pred = np.full(len(test), np.nan)
k0_pred[sel_k0] = mu_all[sel_k0] + k0_anom[sel_k0]
miss = unm & np.isnan(k0_pred)
if miss.any():
    log(f"  fallback mu_c for {miss.sum()} unmasked rows")
    k0_pred[miss] = mu_all[miss]

# direct-copy check (target identity)
vis = test.loc[unm & np.isfinite(test['TWS_t'].values), ['cc','t_abs','TWS_t']]
vis_map = {(int(c), int(t)): v for c, t, v in zip(vis['cc'], vis['t_abs'], vis['TWS_t'])}
cc_all = test['cc'].values; ta_all = test['t_abs'].values
keys = list(zip(cc_all.tolist(), (ta_all+1).tolist()))
copy_vals = np.array([vis_map.get(k, np.nan) for k in keys], dtype=np.float64)
n_copy = int(np.isfinite(copy_vals).sum())
log(f"  direct-copyable rows: {n_copy} ({n_copy/len(test)*100:.2f}%)")

def assemble(masked_pred, tag):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_pred[unm]
    pred[~unm] = masked_pred[~unm]
    pred = np.where(np.isnan(pred), mu_all, pred)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    log(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} "
        f"(k0 std={pred[unm].std():.3f}, masked std={pred[~unm].std():.3f})")
    return pred

sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
log("assembling...")
assemble(masked_ens, 'v18a')
assemble(masked_single, 'v18b')

# ============ PHASE D: pre-registered projections ============
P("\n================ PRE-REGISTERED LB PROJECTION ================")
# calibration: v17b config M2=0.6905 -> actual masked LB 0.7355 (gap +0.0450)
# gap band for v18 (era-weighted Dhat may close part of the D-drift share): [+0.030, +0.050]
for name, m2 in [('v18a (ensemble)', sc(None, None) if False else None), ]:
    pass
m2_ens = 0.0
# recompute M2 ensemble score quickly for the record (from stored preds is complex post-gc;
# use config scores from confirmation phase, tracked manually below)
P(f"config scores (M1/M2): cfg0 0.6077/0.6688  cfg1 0.6080/0.6703  cfg2 0.6081/0.6695  ens ~0.607/0.668")
P("gap model: v17b M2 0.6905 -> LB masked 0.7355 (+0.0450). v18 band [+0.030, +0.050]")
P("projected masked RMSE: 0.698-0.719 ; k0 assumed 0.640 (band 0.63-0.65)")
P("projected LB v18a: sqrt(0.6652*masked^2 + 0.3348*k0^2)")
for mk in [0.698, 0.708, 0.719]:
    lb = np.sqrt(0.6652*mk**2 + 0.3348*0.640**2)
    P(f"  masked={mk:.3f} -> LB={lb:.4f}  (v12b=0.6954, v17b=0.7050)")
P("DECISION RULES (pre-registered):")
P("  LB <= 0.692 : protocol fix + spatial smoothing confirmed -> double down (E1 D-evolution next)")
P("  0.692 < LB <= 0.700 : modest confirm -> keep iterating, gap model recalibrated")
P("  LB > 0.700 : projection missed -> suspect era-weight/smoothing transfer; fall back to v17a-config family")
P("  LB > 0.7050 : worse than v17b -> major protocol insight needed before next submission")

with open('/home/z/my-project/scripts/build_v18_final.log', 'w') as f:
    f.write('\n'.join(LOG))
P("\nDONE")
