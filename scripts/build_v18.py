"""
BUILD v18 — protocol-corrected masked-row pipeline.

AUDIT-DRIVEN CHANGES vs v17b (audits A/B/C, 2026-08-30):
  A1. SELECTION PROTOCOL FIXED: old honest-CV (P0) overstates the backward pass by
      ~0.040 because val geometry (anchors every ~2.3 months, 33% rows at bwd-dist 1)
      does not exist on test (anchors Sep15/Jan16/Jun16/Dec16/Jul18/Nov18, gaps
      [4,5,6,19,4]; 50% of masked rows at bwd-dist 12-17; 8% fwd-only).
      -> select on MIRRORED protocols M1 + M2 (Auditor A's spec, hard-coded months).
  A2. phi re-tuned under mirror (M1 prefers 0.85-0.90 for bwd, NOT 0.80).
  A3. bwd_max_gap cap test: measured bwd gain at distance>=12 is NEGATIVE (-0.017);
      cap the backward pass so 2017-block rows run fwd-only.
  C1. E2 era-weighted D-hat (D drifts on ~2yr timescales; static mean of 6 anchors
      spanning 38 months blurs; weight anchors by temporal proximity to target t).
  C2. E3 spatial smoothing of the fast-state field (neighbor corr 0.99/0.97/0.84/0.61
      at 1/2/5/10 deg; per-cell filter errors are partly independent -> smoothable).
  C3. k0 spatial-smoothing probe (cheap test on standard val k0 rows).
  Denoiser stays OFF (verdict survived protocol fix: M1 0.6462 vs 0.6319).

PRE-REGISTERED SELECTION RULE (Auditor A): a config change is accepted only if it
wins on BOTH M1 and M2 vs the v17b baseline. Final = best avg rank among accepted.
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
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)

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

# ================= infra (verbatim v17b) =================
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

# ================= spatial smoother (E3) =================
_SMOOTH = {}
def get_smoother(sigma):
    """Row-normalized gaussian kernel over cells (equirectangular km-ish coords)."""
    key = float(sigma)
    if key in _SMOOTH: return _SMOOTH[key]
    la, lo = cell_xy[:,0], cell_xy[:,1]
    la_m = np.median(la)
    xy = np.column_stack([lo*np.cos(np.deg2rad(la_m))*111.0, la*111.0])  # km
    tree = cKDTree(xy)
    rad = 3.0*sigma*111.0
    rows, cols, vals = [], [], []
    for i in range(len(xy)):
        nb = tree.query_ball_point(xy[i], rad)
        d2 = ((xy[nb]-xy[i])**2).sum(axis=1)
        w = np.exp(-d2/(2*(sigma*111.0)**2))
        w = w/np.maximum(w.sum(), 1e-12)
        rows.extend([i]*len(nb)); cols.extend(nb); vals.extend(w.tolist())
    K = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    _SMOOTH[key] = K
    return K

# ================= masked pipeline v18 =================
def run_masked_v18(infra, ev_df, ev_tws_visible, phi, use_bwd=True,
                   dtil_w=(0.70,0.45,0.073), lam=0.84, bwd_max_gap=None,
                   dhat_tau=None, dhat_mode='full', smooth_sigma=None):
    """v17b run_masked + audit extensions. Denoiser permanently OFF.
    dhat_tau: None = static D-hat (v17b). float = era-weighted D-hat(t),
              weights exp(-|t_anchor - t|/tau) over the 6 anchors.
    dhat_mode: 'full'  = era-weighted baseline in state-init AND prediction
               'pred'  = static baseline in state-init, era-weighted in prediction only
    smooth_sigma: gaussian smoothing (deg) of the final fast-state field per month.
    bwd_max_gap: skip backward pass when next_anchor - target_month > gap.
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
    # static D-hat + calibration (EXACT v17b semantics)
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
    # era-weighted D-hat(t)
    A_arr = np.array(sorted(AF.keys()), dtype=np.float64)
    A_stack = np.array([AF[a] for a in A_arr])                    # [n_anchor, n_cells]
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

# ================= mirror protocols (Auditor A spec, HARD-CODED) =================
log("building M1/M2 mirror protocols (fit<=2012 infra)...")
fit = train[train['time'].dt.year <= 2012].copy()
infra_cv = build_infra(fit, 'cv')
val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()

M1_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]   # Jan13 May13 Nov13 May14 Apr15 Jul15
M1_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2,
           24173:3, 24176:3, 24177:3, 24178:3, 24179:3, 24180:3, 24187:4}
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]          # Jan13 May13 Nov13 May14 Jul15 (gap 14)
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

def score(mir, pred):
    mm = mir['masked'].values & np.isfinite(pred) & np.isfinite(mir['target'].values)
    return float(np.sqrt(np.mean((pred[mm]-mir['target'].values[mm])**2))), int(mm.sum())

def ev(cfg, mir):
    kw = dict(cfg)
    phi = kw.pop('phi')
    return run_masked_v18(infra_cv, mir, mir['TWS_t'], phi, **kw)

TESTMIX = {1: 0.1673, 2: 0.2487, 3: 0.4995, 4: 0.0836}   # test run mix (rows)
def runmix(mir, pred, runs):
    """test-mix-weighted RMSE (diagnostic only)."""
    num, den = 0.0, 0.0
    msk = mir['masked'].values
    ta = mir['t_abs'].values; y = mir['target'].values
    for r, wt in TESTMIX.items():
        months = [m for m, rr in runs.items() if rr == r]
        sel = msk & np.isin(ta, months) & np.isfinite(pred) & np.isfinite(y)
        if sel.sum() == 0: continue
        num += wt*np.sum((pred[sel]-y[sel])**2); den += wt*sel.sum()
    return float(np.sqrt(num/den)) if den > 0 else np.nan

BASE = dict(phi=0.80, use_bwd=True, dtil_w=(0.70,0.45,0.073), lam=0.84)

# ================= EXPERIMENTS (M1 + M2) =================
P("\n================ E0: baseline (v17b config) on M1/M2 ================")
p1 = ev(BASE, M1); p2 = ev(BASE, M2)
b1, n1 = score(M1, p1); b2, n2 = score(M2, p2)
bm1, bm2 = runmix(M1, p1, M1_RUNS), runmix(M2, p2, M2_RUNS)
P(f"BASE v17b: M1={b1:.4f} (mix {bm1:.4f}) | M2={b2:.4f} (mix {bm2:.4f})   [auditor A: 0.6319 / 0.6905]")
assert abs(b1-0.6319) < 0.002 and abs(b2-0.6905) < 0.002, "baseline drift vs auditor A!"

P("\n================ E1: phi x bwd-cap grid (bwd, no-dn) ================")
res = {}
for phi in [0.74, 0.80, 0.85, 0.90]:
    for cap in [None, 8]:
        cfg = dict(BASE); cfg['phi']=phi; cfg['bwd_max_gap']=cap
        r1,_ = score(M1, ev(cfg, M1)); r2,_ = score(M2, ev(cfg, M2))
        m1, m2 = runmix(M1, ev(cfg, M1), M1_RUNS), runmix(M2, ev(cfg, M2), M2_RUNS)
        res[(phi,cap)] = (r1, r2)
        P(f"  phi={phi} cap={cap}: M1={r1:.4f} (mix {m1:.4f}) | M2={r2:.4f} (mix {m2:.4f})")

P("\n================ E1b: lam check at best-ish phi ================")
best_e1 = min(res, key=lambda k: res[k][0]+res[k][1])
P(f"  best E1 so far: {best_e1} -> M1={res[best_e1][0]:.4f} M2={res[best_e1][1]:.4f}")
phi_b, cap_b = best_e1
for lam in [0.80, 0.88]:
    cfg = dict(BASE); cfg['phi']=phi_b; cfg['bwd_max_gap']=cap_b; cfg['lam']=lam
    r1,_ = score(M1, ev(cfg, M1)); r2,_ = score(M2, ev(cfg, M2))
    P(f"  phi={phi_b} cap={cap_b} lam={lam}: M1={r1:.4f} | M2={r2:.4f}")

P("\n================ E2: era-weighted D-hat (tau sweep) ================")
for tau in [6, 12, 24]:
    for mode in ['full', 'pred']:
        cfg = dict(BASE); cfg['phi']=phi_b; cfg['bwd_max_gap']=cap_b; cfg['dhat_tau']=tau; cfg['dhat_mode']=mode
        r1,_ = score(M1, ev(cfg, M1)); r2,_ = score(M2, ev(cfg, M2))
        P(f"  tau={tau} mode={mode}: M1={r1:.4f} | M2={r2:.4f}")

P("\n================ E3: spatial smoothing of fast state ================")
for sig in [0.5, 1.0, 1.5]:
    cfg = dict(BASE); cfg['phi']=phi_b; cfg['bwd_max_gap']=cap_b; cfg['smooth_sigma']=sig
    r1,_ = score(M1, ev(cfg, M1)); r2,_ = score(M2, ev(cfg, M2))
    P(f"  sigma={sig}deg: M1={r1:.4f} | M2={r2:.4f}")

P("\n================ E4: combos (tau x sigma) ================")
for tau, mode in [(12,'full'), (12,'pred'), (24,'pred')]:
    for sig in [0.5, 1.0]:
        cfg = dict(BASE); cfg['phi']=phi_b; cfg['bwd_max_gap']=cap_b
        cfg['dhat_tau']=tau; cfg['dhat_mode']=mode; cfg['smooth_sigma']=sig
        r1,_ = score(M1, ev(cfg, M1)); r2,_ = score(M2, ev(cfg, M2))
        P(f"  tau={tau}/{mode} sigma={sig}: M1={r1:.4f} | M2={r2:.4f}")

with open('/home/z/my-project/scripts/build_v18_grid.log', 'w') as f:
    f.write('\n'.join(LOG))
P("\ngrid phase done -> inspect, then finalize config in build_v18_final.py")
