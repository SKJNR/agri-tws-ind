"""
build_v21_phaseB.py — v21 Phase B: ERA5 static channels in D-tilde, M1/M2 mirror A/B.

Question (Task 16 / era5_value_test C3): does adding ERA5 fit-period static per-cell
means as extra D-tilde regressors improve the MASKED-row block?  C3 measured -0.0107
on anchor-field LOO (0.7913 -> 0.7806).  Here we test it on the actual masked-row task
with the exact v18a winner architecture, under the v18 selection discipline:
include ONLY if it wins on BOTH M1 and M2 mirrors by > 0.002.

Variants (all with the v18a top3 ensemble: phi{.80,.85,.80}/lam{.80,.80,.84},
bwd_max_gap=8, dhat_tau=24/full, smooth_sigma=2.0):
  A  fixed dtil weights (0.70/0.45/0.073)          == v18a baseline (validate 0.6078/0.6694)
  B  LOO-refit base weights   [Dhat_s, S, trendex]  (fair LOO baseline)
  C  LOO base + 5 ERA5 static channels (z-scored)   [swvl4, sd, tp, e, t2m]
  D  LOO base + 2 ERA5 channels                      [swvl4, t2m]  (strongest singles)

ERA5 static = per-cell mean over the fit window (mirrors: <=2012-12; final build uses
full train).  ERA5 is a Copernicus reanalysis (not GRACE/TWS-derived) => legal,
prediction-time available, documented.
"""
import numpy as np, pandas as pd, gc, time, warnings
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
ERA5_VARS = ['swvl1','swvl2','swvl3','swvl4','sd','tp','e','ro','t2m']
ERA5_CH5 = ['swvl4','sd','tp','e','t2m']
ERA5_CH2 = ['swvl4','t2m']

t0 = time.time()
LOG = []
def P(s=''):
    print(s, flush=True); LOG.append(str(s))

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)

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
    P(f"  infra[{label}]: {full.sum()} full cells [{time.time()-t0:.0f}s]")
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
                   dtil_extras=None, dtil_fit='fixed'):
    """Generalized v18 masked predictor.

    dtil_extras : list of (n_cells,) static fields appended as D-tilde regressors.
    dtil_fit    : 'fixed' -> use dtil_w as-is (v18a exact when extras=None);
                  'loo'   -> LOO-refit [Dhat_s, S, trendex] + extras on the eval
                             window's anchors (ridge 1e-3), apply w/ era-weighted Dhat.
    """
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

    # ---- D-tilde weights ----
    extras = list(dtil_extras) if dtil_extras is not None else []
    if dtil_fit == 'loo':
        Xs, ys = [], []
        for a in anchors:
            dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
            tx = ((np.float64(a) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
            cols = [dloo, S, tx] + extras
            Xa = np.column_stack(cols)
            okc = np.isfinite(Xa).all(axis=1) & np.isfinite(AF[a])
            Xs.append(Xa[okc]); ys.append(AF[a][okc])
        Xw = np.vstack(Xs); yw = np.concatenate(ys)
        w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(Xw.shape[1]), Xw.T@yw)
        w1, w2, w3 = float(w_[0]), float(w_[1]), float(w_[2])
        wE = [float(v) for v in w_[3:]]
    else:
        w1, w2, w3 = dtil_w
        wE = [0.0]*len(extras)

    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil_s(t):
        out = w1*Dhat_s + w2*S + w3*trendex(t)
        for wj, Ej in zip(wE, extras):
            out = out + wj*Ej
        return out.astype(np.float32)
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
        out = w1*Dhat_era(t) + w2*S + w3*trendex(t)
        for wj, Ej in zip(wE, extras):
            out = out + wj*Ej
        return out.astype(np.float32)
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
    return pred, (w1, w2, w3, wE)

# ---------------- mirrors (verbatim build_v18_final) ----------------
fit = train[train['time'].dt.year <= 2012].copy()
infra_cv = build_infra(fit, 'cv')
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
M1 = make_mirror(M1_ANCHORS, M1_RUNS)
M2 = make_mirror(M2_ANCHORS, M2_RUNS)
def sc(mir, pred):
    mm = mir['masked'].values & np.isfinite(pred) & np.isfinite(mir['target'].values)
    return float(np.sqrt(np.mean((pred[mm]-mir['target'].values[mm])**2)))

# ---------------- ERA5 static channels (fit-window means, z-scored) ----------------
P("loading ERA5 static channels ...")
era5 = pd.read_parquet(f'{DATA}/era5_grid.parquet', columns=['cc','ym']+ERA5_CH5)
era5_fit = era5[era5['ym'] <= 201212]
E_mu = era5_fit.groupby('cc')[ERA5_CH5].mean().reindex(range(n_cells))
E_mu = E_mu.fillna(E_mu.mean()).values.astype(np.float64)      # (n_cells, 5)
Ez = (E_mu - E_mu.mean(axis=0)) / np.maximum(E_mu.std(axis=0), 1e-9)
E5 = [np.nan_to_num(Ez[:, j]).astype(np.float32) for j in range(5)]
E2 = [E5[0], E5[4]]                                            # swvl4, t2m
del era5, era5_fit; gc.collect()
P(f"  ERA5 static ready [{time.time()-t0:.0f}s]")

FINAL_CFGS = [
    dict(phi=0.80, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.85, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.80, lam=0.84, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
]

def run_ensemble(variant_name, extras, dtil_fit, dtil_w=(0.70,0.45,0.073)):
    preds = {'M1': [], 'M2': []}
    wlast = None
    for c in FINAL_CFGS:
        kw = dict(c); phi = kw.pop('phi')
        kw.update(dtil_extras=extras, dtil_fit=dtil_fit, dtil_w=dtil_w)
        p1, w1 = run_masked_v18(infra_cv, M1, M1['TWS_t'], phi, **kw)
        p2, w2 = run_masked_v18(infra_cv, M2, M2['TWS_t'], phi, **kw)
        preds['M1'].append(p1); preds['M2'].append(p2)
        wlast = w1
    pe1 = np.mean(preds['M1'], axis=0); pe2 = np.mean(preds['M2'], axis=0)
    s1, s2 = sc(M1, pe1), sc(M2, pe2)
    wtxt = f" w=({wlast[0]:.3f},{wlast[1]:.3f},{wlast[2]:.3f}" + \
           ("," + ",".join(f"{v:+.4f}" for v in wlast[3]) if wlast[3] else "") + ")"
    P(f"  {variant_name:34s} M1={s1:.4f} | M2={s2:.4f}{wtxt}")
    return s1, s2

P("")
P("=" * 100)
P("PHASE B — ERA5 static channels in D-tilde: M1/M2 A/B (v18a top3 ensemble architecture)")
P("=" * 100)
P("[validate] A fixed weights (v18a exact):  expect M1=0.6078 M2=0.6694")
rA = run_ensemble("A  fixed (v18a baseline)", None, 'fixed')
rB = run_ensemble("B  LOO base [Dhat,S,trendex]", None, 'loo')
rC = run_ensemble("C  LOO base + 5 ERA5 static", E5, 'loo')
rD = run_ensemble("D  LOO base + 2 ERA5 (swvl4,t2m)", E2, 'loo')

P("")
P("--- verdict (pre-registered: include ERA5 only if >0.002 win on BOTH mirrors vs A and B) ---")
for nm, r in [('C (5ch)', rC), ('D (2ch)', rD)]:
    dA1, dA2 = rA[0]-r[0], rA[1]-r[1]
    dB1, dB2 = rB[0]-r[0], rB[1]-r[1]
    okA = (dA1 > 0.002) and (dA2 > 0.002)
    okB = (dB1 > 0.002) and (dB2 > 0.002)
    P(f"  {nm}: vs A {dA1:+.4f}/{dA2:+.4f} | vs B {dB1:+.4f}/{dB2:+.4f} | "
      f"PASS={okA and okB}")

with open('/home/z/my-project/download/build_v21_phaseB.txt', 'w') as f:
    f.write('\n'.join(LOG) + '\n')
P(f"\nsaved download/build_v21_phaseB.txt  [{time.time()-t0:.0f}s]")
