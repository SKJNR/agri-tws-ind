"""
BUILD v18 ROUND 2 — extend grids after round-1 wins:
  - spherical-metric smoother (chordal 3D coords; round-1 used median-lat scaling)
  - sigma extension 2.0/2.5/3.0 (round-1 monotone at 1.5)
  - phi x lam re-sweep WITH smoothing (optimum may shift once noise drops)
  - af_smooth probe: smooth anchor fields pre-filter (feeds Dhat, f0, bwd init)
  - W-smooth probe (small sigma, pre-filter observation smoothing)
  - k0 spatial smoothing test on standard val protocol
All selection on M1 AND M2 (Auditor A rule). Baselines re-printed for reference.
"""
import numpy as np, pandas as pd, time, warnings
import lightgbm as lgb
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
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

# ---- spherical smoother: chordal distance on unit sphere ----
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
                   af_smooth_sigma=None, w_smooth_sigma=None):
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
    if w_smooth_sigma:
        Kw = get_smoother(w_smooth_sigma)
        W = {m: Kw @ w for m, w in W.items()}
    mfrac = ev_df.groupby('t_abs')['masked'].mean()
    anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
    tws_vis = ev_tws_visible.values
    AF = {}
    for a in anchors:
        sel = (ta == a) & (~msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    if af_smooth_sigma:
        Ka = get_smoother(af_smooth_sigma)
        for a in AF: AF[a] = Ka @ AF[a]
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

# ---- protocols ----
fit = train[train['time'].dt.year <= 2012].copy()
infra_cv = build_infra(fit, 'cv')
val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()

M1_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]
M1_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2,
           24173:3, 24176:3, 24177:3, 24178:3, 24179:3, 24180:3, 24187:4}
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
M2_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2,
           24173:3, 24176:3, 24177:3, 24178:3, 24179:3, 24180:3,
           24181:3, 24182:3, 24183:3, 24187:4}

def make_mirror(anchors, runs):
    months = set(anchors) | set(runs)
    mir = val_all[val_all['t_abs'].isin(months)].copy()
    mir['masked'] = mir['t_abs'].isin(runs)
    mir['TWS_t'] = val_tws_visible.loc[mir.index].values
    mir.loc[mir['masked'], 'TWS_t'] = np.nan
    return mir

M1 = make_mirror(M1_ANCHORS, M1_RUNS)
M2 = make_mirror(M2_ANCHORS, M2_RUNS)
TESTMIX = {1: 0.1673, 2: 0.2487, 3: 0.4995, 4: 0.0836}

def evaluate(cfg, mir, runs):
    kw = dict(cfg); phi = kw.pop('phi')
    pred = run_masked_v18(infra_cv, mir, mir['TWS_t'], phi, **kw)
    msk = mir['masked'].values; y = mir['target'].values
    mm = msk & np.isfinite(pred) & np.isfinite(y)
    rmse = float(np.sqrt(np.mean((pred[mm]-y[mm])**2)))
    num, den = 0.0, 0.0
    ta = mir['t_abs'].values
    for r, wt in TESTMIX.items():
        months = [m for m, rr in runs.items() if rr == r]
        sel = msk & np.isin(ta, months) & np.isfinite(pred) & np.isfinite(y)
        if sel.sum():
            num += wt*np.sum((pred[sel]-y[sel])**2); den += wt*sel.sum()
    return rmse, float(np.sqrt(num/den))

def show(tag, cfg):
    r1, m1 = evaluate(cfg, M1, M1_RUNS); r2, m2 = evaluate(cfg, M2, M2_RUNS)
    P(f"  {tag:44s} M1={r1:.4f} (mix {m1:.4f}) | M2={r2:.4f} (mix {m2:.4f})")
    return r1, r2

BASE = dict(phi=0.80, use_bwd=True, dtil_w=(0.70,0.45,0.073), lam=0.84)

P("\n============ R0: baselines (spherical kernel) ============")
show("v17b base", BASE)
C85 = dict(BASE); C85['phi']=0.85; C85['bwd_max_gap']=8
show("phi .85 cap8", C85)

P("\n============ R1: sigma extension (phi .85 cap8) ============")
for sig in [1.0, 1.5, 2.0, 2.5, 3.0]:
    cfg = dict(C85); cfg['smooth_sigma']=sig
    show(f"sigma={sig}", cfg)

P("\n============ R2: tau x mode x sigma (round-1 winners) ============")
for tau, mode in [(24,'full'), (24,'pred')]:
    for sig in [1.5, 2.0]:
        cfg = dict(C85); cfg['dhat_tau']=tau; cfg['dhat_mode']=mode; cfg['smooth_sigma']=sig
        show(f"tau={tau}/{mode} sigma={sig}", cfg)

P("\n============ R3: phi x lam re-sweep WITH smoothing ============")
for phi in [0.80, 0.85, 0.90]:
    for lam in [0.80, 0.84]:
        cfg = dict(C85); cfg['phi']=phi; cfg['lam']=lam; cfg['dhat_tau']=24; cfg['dhat_mode']='full'; cfg['smooth_sigma']=2.0
        show(f"phi={phi} lam={lam} tau24/full sig2.0", cfg)

P("\n============ R4: pre-smoothing probes ============")
cfg = dict(C85); cfg['smooth_sigma']=2.0; cfg['af_smooth_sigma']=1.0
show("af_smooth 1.0 + out 2.0", cfg)
cfg = dict(C85); cfg['smooth_sigma']=2.0; cfg['af_smooth_sigma']=2.0
show("af_smooth 2.0 + out 2.0", cfg)
cfg = dict(C85); cfg['smooth_sigma']=2.0; cfg['w_smooth_sigma']=0.5
show("w_smooth 0.5 + out 2.0", cfg)
cfg = dict(C85); cfg['smooth_sigma']=2.0; cfg['w_smooth_sigma']=1.0
show("w_smooth 1.0 + out 2.0", cfg)
cfg = dict(C85); cfg['af_smooth_sigma']=2.0
show("af_smooth 2.0 only", cfg)

P("\n============ R5: best-so-far refinement ============")
# filled after reading R1-R4 output; placeholder combos likely useful:
for sig in [2.5, 3.0]:
    cfg = dict(C85); cfg['dhat_tau']=24; cfg['dhat_mode']='full'; cfg['smooth_sigma']=sig
    show(f"tau24/full sigma={sig}", cfg)

with open('/home/z/my-project/scripts/build_v18_grid2.log', 'w') as f:
    f.write('\n'.join(LOG))
P("\nround-2 grid done")
