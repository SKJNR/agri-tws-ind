"""Task 46b (2026-09-11): horizon-adaptive smoothing on the M1/M2 MIRROR
harness — the protocol with validated LB transfer (v21a projection 0.685-0.688
-> actual 0.687374). This is the honest test of the external AI's idea #2.

WHY NOT the tb-cache harness (task46 part 2): that harness initialized the
fast state from the TRUE Xres field — no D-tilde estimation error, no cov-
regression noise, no trend noise. It measured smoothing of a near-noiseless
init (verdict: hurts, sigma=0 best) — NOT the production problem. The mirrors
run run_masked_v18 verbatim: x = lam*(AF - Dtil) carries the real per-cell
estimation noise that smoothing exists to remove (build_v18 C2 note).

PRE-REGISTERED GATES (before any run):
  Gate A' (mirror, Auditor-A rule): a sigma(h) schedule is ACCEPTED only if
    it beats flat-2.0deg on BOTH M1 and M2 with avg gain >= 0.002.
  Gate B (LB, unchanged): v30 public <= 0.685874 (v21a - 0.0015) to swap
    final slot-2. Otherwise tombstone with numbers.

SWEEP (both directions — pre-registered):
  flat:   1.5, 2.0 (= production), 2.5, 3.0
  back-loaded (other-AI's direction, more sigma at long h):
          (2,2,3), (2,2,4), (2,2,5), (2,2,6), (2,3,4), (2,3,6), (2,4,6)
  front-loaded (opposite direction — init/cov noise highest at short h):
          (3,2,2), (4,2,2), (4,3,2), (3,2.5,2)
  piecewise by h: sigma(h<=3), sigma(h=4-5), sigma(h>=6)
"""
import numpy as np, pandas as pd, time, warnings
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/tmp/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def P(s=''): print(s, flush=True)

print('='*78)
print('TASK 46b: horizon-adaptive sigma(h) on M1/M2 mirrors (pre-registered)')
print('='*78)

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)
print(f'[{time.time()-t0:6.1f}s] train {len(train):,} rows, {n_cells:,} cells')

# ---------------- infra (fit <= 2012) — verbatim build_v18/build_v21 --------
fit = train[train['time'].dt.year <= 2012].copy()
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
    return d
infra_cv = build_infra(fit, 'cv')
print(f'[{time.time()-t0:6.1f}s] infra built ({int(infra_cv["full"].sum()):,} full cells)')

# ---------------- smoother (verbatim production) ----------------------------
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

# ---------------- run_masked_v18 with sigma_rule(h) --------------------------
def run_masked_v18(infra, ev_df, ev_tws_visible, phi, use_bwd=True,
                   dtil_w=(0.70,0.45,0.073), lam=0.84, bwd_max_gap=None,
                   dhat_tau=None, dhat_mode='full', sigma_rule=None,
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
        h = tm - i                       # horizon: months from anchor to TARGET
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
        if sigma_rule is not None:
            sig = sigma_rule(h)
            if sig and sig > 0:
                x = get_smoother(sig) @ x
        dT = Dtil_b(tm, era=(dhat_tau is not None))
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
    return pred

# ---------------- M1/M2 mirrors (verbatim) -----------------------------------
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
print(f'[{time.time()-t0:6.1f}s] mirrors: M1 {len(M1):,} rows ({int(M1["masked"].sum()):,} masked), '
      f'M2 {len(M2):,} rows ({int(M2["masked"].sum()):,} masked)')

def score(mir, pred):
    mm = mir['masked'].values & np.isfinite(pred) & np.isfinite(mir['target'].values)
    return float(np.sqrt(np.mean((pred[mm]-mir['target'].values[mm])**2))), int(mm.sum())

PROD = dict(phi=0.80, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full',
            dtil_fit='fixed')
def run_rule(rule, label):
    kw = dict(PROD); phi = kw.pop('phi')
    p1 = run_masked_v18(infra_cv, M1, M1['TWS_t'], phi, sigma_rule=rule, **kw)
    p2 = run_masked_v18(infra_cv, M2, M2['TWS_t'], phi, sigma_rule=rule, **kw)
    s1, n1 = score(M1, p1); s2, n2 = score(M2, p2)
    print(f'  {label:40s}: M1={s1:.4f} | M2={s2:.4f} | avg={0.5*(s1+s2):.4f}')
    return s1, s2

print(f'\n[{time.time()-t0:6.1f}s] --- flat sigma (production re-check) ---')
flat = {}
for s in [1.5, 2.0, 2.5, 3.0]:
    flat[s] = run_rule((lambda ss: (lambda h: ss))(s), f'flat sigma={s:.1f}deg' + (' (PRODUCTION)' if s==2.0 else ''))
b1, b2 = flat[2.0]
assert abs(b1-0.6090) < 0.003 and abs(b2-0.6716) < 0.003, \
    f'baseline drift! M1 {b1:.4f} vs 0.6090, M2 {b2:.4f} vs 0.6716'
print('  [baseline reproduces grid2 R2: M1 0.6090 / M2 0.6716 — OK]')

print(f'\n[{time.time()-t0:6.1f}s] --- horizon-adaptive sigma(h): h<=3 | h4-5 | h>=6 ---')
results = {}
for s1_, s2_, s3_ in [(2,2,3), (2,2,4), (2,2,5), (2,2,6), (2,3,4), (2,3,6), (2,4,6),
                      (3,2,2), (4,2,2), (4,3,2), (3,2.5,2), (2.5,2,2)]:
    rule = (lambda a, b, c: (lambda h: a if h <= 3 else (b if h <= 5 else c)))(s1_, s2_, s3_)
    results[(s1_, s2_, s3_)] = run_rule(rule, f'sigma(h): {s1_}/{s2_}/{s3_}')

print('\n' + '='*78)
base_avg = 0.5*(b1+b2)
cands = []
for k, (m1, m2) in results.items():
    gain = 0.5*((b1-m1)+(b2-m2))
    both = (m1 < b1) and (m2 < b2)
    print(f'  {k}: M1 {m1:.4f} ({b1-m1:+.4f}) | M2 {m2:.4f} ({b2-m2:+.4f}) '
          f'| avg gain {gain:+.4f} | beats-both: {"YES" if both else "no"}')
    if both and gain >= 0.002:
        cands.append((k, gain))
best_flat = min(flat, key=lambda s: 0.5*(flat[s][0]+flat[s][1]))
print(f'\nbest flat: {best_flat} deg (avg {0.5*(flat[best_flat][0]+flat[best_flat][1]):.4f} '
      f'vs prod-2.0 {base_avg:.4f})')
print(f"GATE A' (beats BOTH mirrors, avg gain >= 0.002): "
      f"{'PASS -> ' + str(cands) if cands else 'FAIL -> tombstone idea #2 (numbers above)'}")
print('='*78)
