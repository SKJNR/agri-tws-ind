"""
build_v21_phaseC.py — FINAL BUILD: submission_v21a.csv + submission_v21b.csv

v21a = v18a masked block EXACT (fixed dtil weights 0.70/0.45/0.073, top3 ensemble)
       + a15 k0 block (0.5 factor(K=100) + 0.5 v4-linear, NO LightGBM)
       -> isolates the k0 upgrade vs v18a (attribution: v21a - v18a = k0 effect)

v21b = masked block with LOO-refit D-tilde weights on the 6 test anchors
       + same a15 k0 block
       -> both upgrades (Phase B: LOO weights beat fixed on BOTH mirrors,
          M1 0.6078->0.6016, M2 0.6694->0.6622; ERA5 channels FAILED the bar -> excluded)

k0 structure on test (verified): 94,048 rows = 6 fully-unmasked anchor months
(Sep15/Jan16/Jun16/Dec16/Jul18/Nov18; 4 of them have t+1 in the test month set =
covs(t+1) available, 2 do not) + ~417 stray rows at ~fully-masked months
(strays -> linear-only prediction).

Data legality: competition CSVs ONLY. No GDO archive, no GRACE products, no ERA5
(Phase B excluded it). Fully deterministic (no LightGBM in v21).
"""
import numpy as np, pandas as pd, gc, time, warnings
import sys
sys.path.insert(0, '/home/z/my-project/scripts')
from a15_common import FactorCore, ModeKF, estimate_VarD, COVS
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
t0 = time.time()
LOG = []
def P(s=''):
    print(s, flush=True); LOG.append(str(s))
def log(msg): P(f"[{time.time()-t0:7.1f}s] {msg}")

# ====================================================================== load
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)
tr_months = set(int(m) for m in train['t_abs'].unique())

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
test_for_pred = test.copy()
test_for_pred.loc[test_for_pred['masked'], 'TWS_t'] = np.nan
unm = ~test['masked'].values
test_months_set = set(int(m) for m in test['t_abs'].unique())
log(f"data loaded: train {len(train):,} rows, test {len(test):,} rows, k0 rows {unm.sum():,}")

# ====================================================================== masked block (v18 architecture)
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
                   dhat_tau=None, dhat_mode='full', smooth_sigma=None,
                   dtil_fit='fixed'):
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
    if dtil_fit == 'loo':
        Xs, ys = [], []
        for a in anchors:
            dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
            tx = ((np.float64(a) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
            Xa = np.column_stack([dloo, S, tx])
            okc = np.isfinite(Xa).all(axis=1) & np.isfinite(AF[a])
            Xs.append(Xa[okc]); ys.append(AF[a][okc])
        Xw = np.vstack(Xs); yw = np.concatenate(ys)
        w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(3), Xw.T@yw)
        w1, w2, w3 = float(w_[0]), float(w_[1]), float(w_[2])
    else:
        w1, w2, w3 = dtil_w
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

log("=== masked block: final test predictions (v18a top3 ensemble) ===")
infra_full = build_infra(train, 'full')
FINAL_CFGS = [
    dict(phi=0.80, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.85, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.80, lam=0.84, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
]
masked_fixed = []
masked_loo = []
for c in FINAL_CFGS:
    kw = dict(c); phi = kw.pop('phi')
    pF = run_masked_v18(infra_full, test_for_pred, test_for_pred['TWS_t'], phi, dtil_fit='fixed', **kw)
    pL = run_masked_v18(infra_full, test_for_pred, test_for_pred['TWS_t'], phi, dtil_fit='loo', **kw)
    masked_fixed.append(pF); masked_loo.append(pL)
    log(f"  phi={phi} lam={kw['lam']}: fixed std={np.nanstd(pF):.4f} loo std={np.nanstd(pL):.4f}")
masked_v21a = np.mean(masked_fixed, axis=0)
masked_v21b = np.mean(masked_loo, axis=0)

# validate v21a masked block vs the SUBMITTED v18a.csv (bit-level toolchain check)
# (original v18a assemble replaced any NaN with mu_all — apply the same fill here)
n_nan_masked = int(np.isnan(masked_v21a[~unm]).sum())
mu_all = infra_full['mu_c'][test['cc'].values]
masked_v21a = np.where(np.isnan(masked_v21a), mu_all, masked_v21a)
masked_v21b = np.where(np.isnan(masked_v21b), mu_all, masked_v21b)
log(f"  masked rows with NaN D-tilde (mu_all filled): {n_nan_masked:,}")
v18a_csv = pd.read_csv(f'{DL}/submission_v18a.csv')
v18a_csv = v18a_csv.set_index('ID').loc[test['ID'].values, 'Target'].values
dv = np.abs(masked_v21a[~unm] - v18a_csv[~unm])
log(f"  v21a-masked vs v18a.csv masked rows: max|diff|={dv.max():.3e}  mean|diff|={dv.mean():.3e}  (n={dv.size:,})")
assert dv.max() < 1e-4, "masked block does not reproduce v18a!"

# ====================================================================== k0 block (a15 factor + v4 linear)
log("=== k0 block: a15 factor (K=100, full-train fit) + v4 linear ===")
core = FactorCore(train, n_cells, K=100, rec_edges=(2010, 2013), verbose=False)
vmonths, vfields = core.era_cov(test_for_pred)
vz = core.zscores(vfields, vmonths)
S_te = np.nanmean(np.array([vfields[int(m)][0] for m in vmonths]), axis=0)

# k0 anchor months = test months with fully-visible TWS_t
mfrac_te = test_for_pred.groupby('t_abs')['TWS_t'].apply(lambda s: s.isna().mean())
k0_anchors = sorted(int(m) for m, f in mfrac_te.items() if f < 0.01)
log(f"  k0 anchor months: {k0_anchors}")
tta = test['t_abs'].values; tcc = test['cc'].values; ttws = test_for_pred['TWS_t'].values
AF = {}
for a in k0_anchors:
    f = np.full(n_cells, np.nan, dtype=np.float32)
    sel = tta == a
    f[tcc[sel]] = ttws[sel]
    AF[a] = f - core.mu_c
Dhat = np.nanmean(np.array([AF[a] for a in k0_anchors]), axis=0)
Xs, ys = [], []
for a in k0_anchors:
    dloo = np.nanmean(np.array([AF[b] for b in k0_anchors if b != a]), axis=0)
    tx = core.trendex(a)
    okc = np.isfinite(dloo) & np.isfinite(S_te) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[okc], S_te[okc], tx[okc]]))
    ys.append(AF[a][okc])
w_ = np.linalg.solve(np.vstack(Xs).T @ np.vstack(Xs) + 1e-3*np.eye(3),
                     np.vstack(Xs).T @ np.concatenate(ys))
W1, W2, W3 = float(w_[0]), float(w_[1]), float(w_[2])
log(f"  k0 Dtil LOO weights (test anchors): {W1:.3f}/{W2:.3f}/{W3:.3f}")
def Dtil(tt): return W1*Dhat + W2*S_te + W3*core.trendex(tt)

YA = np.stack([core.anchor_y(AF[a], Dtil(a)) for a in k0_anchors])
gaps = [(i, i+1, k0_anchors[i+1]-k0_anchors[i]) for i in range(len(k0_anchors)-1)]
VarD = estimate_VarD(YA, gaps, core.phi, core.Vx)
kf = ModeKF(core.phi, core.Vx, core.q, core.H, core.Rcov, VarD)

factor_pred = np.full(len(test), np.nan)
for a in k0_anchors:
    has_next = (a + 1) in test_months_set
    g = AF[a] - Dtil(a)
    gf = g[core.full]
    gf = np.where(np.isfinite(gf), gf, 0.0)
    f = core.V.T @ gf
    r = np.where(np.isfinite(g), g, 0.0) - f @ core.V_ext.T
    kf.reset(); kf.anchor_update(f); kf.step(1)
    if has_next and (a + 1) in vz:
        kf.cov_update(vz[a + 1])
    m = kf.mean_var()[0]
    pred = core.mu_c + Dtil(a + 1) + (m @ core.V_ext.T) + r
    sel = (tta == a) & unm
    factor_pred[sel] = pred[tcc[sel]]
    log(f"  anchor {a}: rows={sel.sum():,} covs(t+1)={'Y' if has_next else 'N'} pred std={np.nanstd(pred):.3f}")

# ---- v4 linear final (full-train fit, F/R dual, covs(t+1) merged WITHIN test) ----
def build_linear_final():
    # covs(t+1) lookup within TEST
    te_idx = {(int(c), int(m)): j for j, (c, m) in enumerate(zip(tcc, tta))}
    tr_fields_months = np.sort(train['t_abs'].unique())
    fm2i = {int(m): i for i, m in enumerate(tr_fields_months)}
    COVmat = np.full((len(tr_fields_months), n_cells, len(COVS)), np.nan, np.float32)
    mi_all = train['t_abs'].map(fm2i).values
    COVmat[mi_all, train['cc'].values] = train[COVS].values

    def feats_train(rows):
        cc_ = rows['cc'].values; ta_ = rows['t_abs'].values
        x_p = rows['TWS_t'].values - core.mu_c[cc_]
        covs_t = np.column_stack([rows[c].values for c in COVS]) - core.clim[cc_]
        nidx = np.array([fm2i.get(int(m) + 1, -1) for m in ta_])
        covs_t1 = np.full((len(rows), 5), np.nan, np.float32)
        okn = nidx >= 0
        covs_t1[okn] = COVmat[nidx[okn], cc_[okn]]
        return x_p, covs_t, covs_t1

    Xtr, ytr = [], []
    yr_all = train['t_abs'].values // 12
    wtr = np.where(yr_all <= 2009, 1.0, np.where(yr_all <= 2012, 2.0, 3.0)).astype(np.float32)
    x_p, covs_t, covs_t1 = feats_train(train)
    yA = (train['target'].values - core.mu_c[train['cc'].values]).astype(np.float32)
    Xf_ = np.column_stack([x_p, covs_t, covs_t1, np.ones(len(x_p))])
    Xf_ = np.nan_to_num(Xf_, nan=0.0).astype(np.float32)
    selF_ = np.isfinite(covs_t1).all(axis=1) & np.isfinite(yA) & np.isfinite(x_p)
    selR_ = np.isfinite(yA) & np.isfinite(x_p)
    sw = np.sqrt(wtr)
    A_ = Xf_[selF_] * sw[selF_, None]
    coefF = np.linalg.solve(A_.T@A_ + 1e-3*np.eye(12), A_.T@(yA[selF_]*sw[selF_]))
    colsR = [0, 1, 2, 3, 4, 5, 11]
    A_ = Xf_[selR_][:, colsR] * sw[selR_, None]
    coefR = np.linalg.solve(A_.T@A_ + 1e-3*np.eye(7), A_.T@(yA[selR_]*sw[selR_]))

    # test rows (all k0 candidates incl. strays)
    cc_t = tcc; ta_t = tta
    x_p_t = ttws - core.mu_c[cc_t]
    covs_t_t = np.column_stack([test_for_pred[c].values for c in COVS]) - core.clim[cc_t]
    covs_t1_t = np.full((len(test), 5), np.nan, np.float32)
    for j in range(len(test)):
        key = (int(cc_t[j]), int(ta_t[j]) + 1)
        jj = te_idx.get(key)
        if jj is not None:
            covs_t1_t[j] = np.column_stack([test_for_pred[c].values[jj] for c in COVS]) - core.clim[cc_t[j]]
    Xv = np.column_stack([x_p_t, covs_t_t, covs_t1_t, np.ones(len(test))]).astype(np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0)
    predF = Xv @ coefF
    predR = Xv[:, colsR] @ coefR
    has_next = np.isfinite(covs_t1_t).all(axis=1)
    pred = np.where(has_next, predF, predR) + core.mu_c[cc_t]
    return pred, has_next

lin_pred, has_next_te = build_linear_final()
log(f"  linear final: rows with covs(t+1): {int((has_next_te & unm).sum()):,} / {int(unm.sum()):,}")

# ---- assemble k0: 0.5 factor + 0.5 linear at the 6 anchors; linear-only for strays ----
k0_pred = np.full(len(test), np.nan)
anchor_rows = np.isin(tta, k0_anchors) & unm
stray_rows = unm & ~np.isin(tta, k0_anchors)
k0_pred[anchor_rows] = 0.5*factor_pred[anchor_rows] + 0.5*lin_pred[anchor_rows]
k0_pred[stray_rows] = lin_pred[stray_rows]
nb_fb = int(np.isnan(k0_pred[anchor_rows]).sum())
k0_pred = np.where(np.isnan(k0_pred) & unm, lin_pred, k0_pred)
log(f"  k0 assembled: anchor rows {int(anchor_rows.sum()):,} (factor NaN fallback: {nb_fb}), strays {int(stray_rows.sum()):,}")

mu_all = infra_full['mu_c'][tcc]

# direct-copy check (record; expected 0 — train ends 2015-08, test starts 2015-09)
vis = test.loc[unm & np.isfinite(test['TWS_t'].values), ['cc','t_abs','TWS_t']]
vis_map = {(int(c), int(t)): v for c, t, v in zip(vis['cc'], vis['t_abs'], vis['TWS_t'])}
keys = list(zip(tcc.tolist(), (tta+1).tolist()))
copy_vals = np.array([vis_map.get(k, np.nan) for k in keys], dtype=np.float64)
log(f"  direct-copyable rows (within test): {int(np.isfinite(copy_vals).sum())} (expected 0)")

# ====================================================================== assemble
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
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
    return pred

log("=== assembling ===")
p_v21a = assemble(masked_v21a, k0_pred, 'v21a')
p_v21b = assemble(masked_v21b, k0_pred, 'v21b')

# also: v21a vs v21b masked-block diff stats (attribution log)
dmask = p_v21b[~unm] - p_v21a[~unm]
log(f"v21b-v21a masked diff: mean={dmask.mean():+.4f} std={dmask.std():.4f} max|d|={np.abs(dmask).max():.3f}")

# ====================================================================== projections
P("")
P("================ PRE-REGISTERED LB PROJECTION ================")
P("decomposition: LB^2 = 0.6652*masked^2 + 0.3348*k0^2")
P("calibration: v18a LB 0.6937 (masked ~0.719 via M2-gap model; k0 ~0.640)")
P("k0: strict-CV 0.6429 -> LB ~0.640 (transfer ~1:1); v21 k0 strict-CV 0.6166 -> LB 0.613-0.620")
P("masked (v21b): M2 0.6694 -> 0.6622 (-0.0072); LB masked 0.719 -> 0.712-0.716")
for mk in [0.719, 0.716, 0.712]:
    for kk in [0.613, 0.617, 0.621]:
        lb = np.sqrt(0.6652*mk**2 + 0.3348*kk**2)
        P(f"  masked={mk:.3f} k0={kk:.3f} -> LB={lb:.4f}")
P("v21a projected: 0.685-0.688   v21b projected: 0.679-0.685")
P("DECISION RULES (pre-registered):")
P("  v21a <= 0.692 : k0 upgrade confirmed -> keep a15 k0 in all future builds")
P("  v21b <  v21a  : LOO weights confirmed -> adopt for final selection")
P("  v21b >= v21a + 0.002 : LOO weights do not transfer -> final = v18a + v21a")
P("  both > 0.694  : projection missed -> investigate before any further submission")

with open('/home/z/my-project/scripts/build_v21_phaseC.log', 'w') as f:
    f.write('\n'.join(LOG) + '\n')
P("\nDONE")
