"""
AUDIT A2b/A5 — MIRRORED-PROTOCOL EXPERIMENT
Tests the parent's hypothesis: the honest-CV val protocol (fit<=2012, val 2013-15,
calendar mask) is structurally EASIER than the real test (more anchors, shorter
backward distances), inflating the backward-pass config's CV.

Method: run the EXACT v17b masked-row pipeline (build_infra/run_masked copied
verbatim, plus optional anchor overrides) under:
  (P0) current honest protocol          [baseline, should reproduce 0.6535]
  (M1) test-mirrored protocol: anchors {Jan13,May13,Nov13,May14,Apr15,Jul15}
       (t_abs 24156,24160,24166,24172,24183,24186; gaps 4,6,6,11,3),
       scored masked runs: {Feb13},{Jun13,Jul13,Dec13,Jan14},
       {Jun14,Sep14,Oct14,Nov14,Dec14,Jan15},{Aug15}; all other val months
       treated as absent. 12 scored months, 6 anchors, ~5.95 obs/cell (test: 5.98).
  (M1d) M1 but D-hat also from 7 extra unmasked months {Dec13,Jan14,Apr14,Feb15,
       Mar15,May15,Jun15} -> 13-anchor D-hat (isolates D-hat noise vs geometry).
Configs: bwd (phi=.80,lam=.84) vs fwd-only, + denoiser check on M1.
Outputs: /home/z/my-project/download/auditA_mirror.txt
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/auditA_mirror.txt'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
t0 = time.time()
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))
def log(msg): P(f"[{time.time()-t0:6.1f}s] {msg}")

# ---------- load ----------
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

# ---------- verbatim pipeline from rebuild_v17b.py ----------
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

def run_masked(infra, ev_df, ev_tws_visible, phi, use_bwd=True, use_denoise=True,
               dhat_denoise=False, blend='precision', bwd_max_gap=None, dtil_w=(0.70,0.45,0.073),
               lam=LAM_F, anchors_override=None, extra_dhat_months=None, s_months=None):
    """Verbatim copy of rebuild_v17b.run_masked + two audit extensions:
       anchors_override: explicit anchor t_abs list (else auto mfrac<0.01)
       extra_dhat_months: extra months (unmasked rows in ev_df) added to the D-hat mean
       s_months: restrict S (static cov field) to these t_abs (else all ev_df months)"""
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
    if s_months is None:
        all_m = np.array(sorted(cov_field.keys()))
    else:
        all_m = np.array(sorted(s_months))
    S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
    W = {int(m): cov_field[m] - S for m in cov_field}
    if anchors_override is None:
        mfrac = ev_df.groupby('t_abs')['masked'].mean()
        anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
    else:
        anchors = sorted(int(v) for v in anchors_override)
    tws_vis = ev_tws_visible.values
    AF = {}
    dhat_months = list(anchors) + ([int(m) for m in extra_dhat_months] if extra_dhat_months else [])
    for a in sorted(set(list(anchors) + dhat_months)):
        sel = (ta == a) & (~msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    if dhat_denoise:
        Dhat = np.nanmean(np.array([dn(infra, AF[a], use_denoise) for a in dhat_months]), axis=0)
    else:
        Dhat = np.nanmean(np.array([AF[a] for a in dhat_months]), axis=0)
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

# ---------- infra (fit <= 2012, identical to v17b) ----------
log("building infra on fit<=2012...")
fit = train[train['time'].dt.year <= 2012].copy()
infra = build_infra(fit, 'cv')

# ---------- P0: current honest protocol (replicates v17b) ----------
log("\n=== P0: current honest protocol ===")
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val['masked'] = val['time'].dt.month.map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_ta = val['t_abs'].values
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val_ta])
honest = val['masked'].values & ~shortcut
P(f"P0 honest rows: {int(honest.sum()):,}  (v17b log: 93,829)")

def ev_p0(**kw):
    p = run_masked(infra, val, val_tws_visible, kw.pop('phi',0.80), **kw)
    mm = honest & np.isfinite(p) & np.isfinite(val_target)
    return float(np.sqrt(np.mean((p[mm]-val_target[mm])**2))), p, mm

r_p0_bwd, p_p0_bwd, mm0 = ev_p0(use_denoise=False, use_bwd=True, lam=0.84)
r_p0_fwd, _, _ = ev_p0(use_denoise=False, use_bwd=False, lam=0.84)
r_p0_fwd74, _, _ = ev_p0(phi=0.74, use_denoise=False, use_bwd=False, lam=0.84)
P(f"P0 bwd (phi=.80,lam=.84,no-dn): {r_p0_bwd:.4f}   [v17b: 0.6535]")
P(f"P0 fwd-only (phi=.80):          {r_p0_fwd:.4f}")
P(f"P0 fwd-only (phi=.74):          {r_p0_fwd74:.4f}   [Task13 v2b-equiv honest: 0.6919]")
P(f"P0 bwd advantage: {r_p0_fwd - r_p0_bwd:+.4f}")

# per-geometry-class breakdown of P0
geo_p0 = {}
for m in sorted(set(val_ta[honest].tolist())):
    sel = honest & (val_ta == m)
    rm = float(np.sqrt(np.mean((p_p0_bwd[sel & mm0]-val_target[sel & mm0])**2)))
    geo_p0[int(m)] = (int(sel.sum()), rm)
P("P0 per-month RMSE (bwd config): " + ", ".join(f"{m//12}-{m%12+1:02d}:{n}r:{r:.4f}" for m,(n,r) in geo_p0.items()))

# ---------- M1: mirrored protocol ----------
log("\n=== M1: test-mirrored protocol ===")
M_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]           # Jan13 May13 Nov13 May14 Apr15 Jul15
M_RUNS = {24157: 'run1', 24161: 'run2', 24162: 'run2', 24167: 'run2', 24168: 'run2',
          24173: 'run3', 24176: 'run3', 24177: 'run3', 24178: 'run3', 24179: 'run3', 24180: 'run3',
          24187: 'run4'}
m_months = set(M_ANCHORS) | set(M_RUNS)
mir = val[val['t_abs'].isin(m_months)].copy()
mir['masked'] = mir['t_abs'].isin(M_RUNS)
# CRITICAL: restore pristine TWS (val was TWS-nulled for the P0 protocol; May13/May14/Apr15
# are val-masked months but MIRROR ANCHORS -> must be readable)
mir['TWS_t'] = val_tws_visible.loc[mir.index].values
mir.loc[mir['masked'], 'TWS_t'] = np.nan
mir_target = mir['target'].values.astype(np.float64)
mir_ta = mir['t_abs'].values
mir_msk = mir['masked'].values
P(f"M1 months: {len(m_months)} (6 anchors + 12 scored); rows={len(mir):,} "
  f"(scored={int(mir_msk.sum()):,}, anchors={int((~mir_msk).sum()):,})")
P(f"M1 obs/cell = {(~mir_msk).sum()/15715:.2f} (test: 5.98); anchor gaps: "
  f"{[M_ANCHORS[i+1]-M_ANCHORS[i] for i in range(len(M_ANCHORS)-1)]} (test: [4,5,6,19,4])")

def ev_m1(**kw):
    p = run_masked(infra, mir, mir['TWS_t'], kw.pop('phi',0.80), **kw)
    mm = mir_msk & np.isfinite(p) & np.isfinite(mir_target)
    return float(np.sqrt(np.mean((p[mm]-mir_target[mm])**2))), p, mm

r_m1_bwd, p_m1_bwd, mm1 = ev_m1(use_denoise=False, use_bwd=True, lam=0.84)
r_m1_fwd, _, _ = ev_m1(use_denoise=False, use_bwd=False, lam=0.84)
r_m1_fwd74, _, _ = ev_m1(phi=0.74, use_denoise=False, use_bwd=False, lam=0.84)
P(f"M1 bwd   (phi=.80,lam=.84,no-dn): {r_m1_bwd:.4f}")
P(f"M1 fwd   (phi=.80):               {r_m1_fwd:.4f}")
P(f"M1 fwd   (phi=.74):               {r_m1_fwd74:.4f}")
P(f"M1 bwd advantage: {r_m1_fwd - r_m1_bwd:+.4f}   (P0 bwd advantage was {r_p0_fwd - r_p0_bwd:+.4f})")
P(f"GEOMETRY+DENSITY EFFECT (P0 - M1, bwd config): {r_m1_bwd - r_p0_bwd:+.4f}")

# per-run breakdown
def bwd_dist(m, anchors):
    nxt = [a for a in anchors if a > m]
    return (nxt[0]-(m+1)) if nxt else None

P("M1 per-run RMSE (bwd config):")
run_rmse = {}
for run_name in ['run1','run2','run3','run4']:
    months = [m for m, r in M_RUNS.items() if r == run_name]
    sel = mir_msk & np.isin(mir_ta, months) & np.isfinite(p_m1_bwd)
    rr = float(np.sqrt(np.mean((p_m1_bwd[sel]-mir_target[sel])**2)))
    run_rmse[run_name] = (int(sel.sum()), rr)
    P(f"  {run_name} (months {months}): n={int(sel.sum()):,} RMSE={rr:.4f}")
# same for fwd-only
pf = run_masked(infra, mir, mir['TWS_t'], 0.80, use_denoise=False, use_bwd=False, lam=0.84)
P("M1 per-run RMSE (fwd-only config):")
for run_name in ['run1','run2','run3','run4']:
    months = [m for m, r in M_RUNS.items() if r == run_name]
    sel = mir_msk & np.isin(mir_ta, months) & np.isfinite(pf)
    rr = float(np.sqrt(np.mean((pf[sel]-mir_target[sel])**2)))
    P(f"  {run_name}: n={int(sel.sum()):,} RMSE={rr:.4f}")

# per-month bwd-vs-fwd gain vs bwd distance (the decay curve)
P("\nM1 per-month: fwd RMSE vs bwd RMSE by bwd distance (gain decay curve):")
for m in sorted(M_RUNS):
    sel = mir_msk & (mir_ta == m) & np.isfinite(p_m1_bwd) & np.isfinite(pf)
    rf = float(np.sqrt(np.mean((pf[sel]-mir_target[sel])**2)))
    rb = float(np.sqrt(np.mean((p_m1_bwd[sel]-mir_target[sel])**2)))
    d = bwd_dist(int(m), M_ANCHORS)
    P(f"  {m//12}-{m%12+1:02d} bwd_dist={d}: fwd={rf:.4f} bwd={rb:.4f} gain={rf-rb:+.4f}")

# ---------- M2: longer-gap mirror (anchors Jan13,May13,Nov13,May14,Jul15; gap 14) ----------
log("\n=== M2: longer-gap mirror (next-anchor gap 14 instead of 11) ===")
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
M2_RUNS = {24157: 'run1', 24161: 'run2', 24162: 'run2', 24167: 'run2', 24168: 'run2',
           24173: 'run3', 24176: 'run3', 24177: 'run3', 24178: 'run3', 24179: 'run3',
           24180: 'run3', 24181: 'run3', 24182: 'run3', 24183: 'run3', 24187: 'run4'}
m2_months = set(M2_ANCHORS) | set(M2_RUNS)
mir2 = val[val['t_abs'].isin(m2_months)].copy()
mir2['masked'] = mir2['t_abs'].isin(M2_RUNS)
mir2['TWS_t'] = val_tws_visible.loc[mir2.index].values
mir2.loc[mir2['masked'], 'TWS_t'] = np.nan
mir2_target = mir2['target'].values.astype(np.float64)
mir2_ta = mir2['t_abs'].values
mir2_msk = mir2['masked'].values
P(f"M2 months: {len(m2_months)} (5 anchors + 15 scored); scored rows={int(mir2_msk.sum()):,}; "
  f"obs/cell={(~mir2_msk).sum()/15715:.2f}; gaps={[M2_ANCHORS[i+1]-M2_ANCHORS[i] for i in range(len(M2_ANCHORS)-1)]}")
p2_b = run_masked(infra, mir2, mir2['TWS_t'], 0.80, use_denoise=False, use_bwd=True, lam=0.84)
p2_f = run_masked(infra, mir2, mir2['TWS_t'], 0.80, use_denoise=False, use_bwd=False, lam=0.84)
mm2 = mir2_msk & np.isfinite(p2_b)
r_m2_bwd = float(np.sqrt(np.mean((p2_b[mm2]-mir2_target[mm2])**2)))
r_m2_fwd = float(np.sqrt(np.mean((p2_f[mm2]-mir2_target[mm2])**2)))
P(f"M2 bwd: {r_m2_bwd:.4f}   M2 fwd: {r_m2_fwd:.4f}   bwd advantage: {r_m2_fwd-r_m2_bwd:+.4f}")
P("M2 per-month gain curve:")
for m in sorted(M2_RUNS):
    sel = mir2_msk & (mir2_ta == m) & np.isfinite(p2_b) & np.isfinite(p2_f)
    rf = float(np.sqrt(np.mean((p2_f[sel]-mir2_target[sel])**2)))
    rb = float(np.sqrt(np.mean((p2_b[sel]-mir2_target[sel])**2)))
    d = bwd_dist(int(m), M2_ANCHORS)
    P(f"  {m//12}-{m%12+1:02d} bwd_dist={d}: fwd={rf:.4f} bwd={rb:.4f} gain={rf-rb:+.4f}")

# test-mix-weighted bwd advantage estimate from M2 gain curve
import collections
gain_by_d = {}
for m in sorted(M2_RUNS):
    sel = mir2_msk & (mir2_ta == m) & np.isfinite(p2_b) & np.isfinite(p2_f)
    rf = float(np.sqrt(np.mean((p2_f[sel]-mir2_target[sel])**2)))
    rb = float(np.sqrt(np.mean((p2_b[sel]-mir2_target[sel])**2)))
    d = bwd_dist(int(m), M2_ANCHORS)
    gain_by_d.setdefault(d, []).append((len(sel), rf-rb))
g = {d: np.average([x[1] for x in v], weights=[x[0] for x in v]) for d, v in gain_by_d.items()}
P(f"\ngain(bwd_dist) from M2: { {k: round(v,4) for k,v in sorted(g.items(), key=lambda kv: (kv[0] is None, kv[0]))} }")
# test masked-row bwd-dist mix (by month, ~equal rows): 2:2/12, 3:2/12, 4:1/12, 12..17:1/12 each, none:1/12
w = {2: 2/12, 3: 2/12, 4: 1/12, 'none': 1/12}
for d in range(12, 18): w[d] = 1/12
adv_a, adv_b = 0.0, 0.0
for d, wt in w.items():
    if d == 'none': continue
    if d in g:
        adv_a += wt*g[d]; adv_b += wt*g[d]
    else:  # d >= 13: (a) zero beyond d=12; (b) hold g(9) value (conservative)
        adv_a += wt*0.0
        adv_b += wt*max(g.get(9, 0.0), 0.0)
P(f"test-mix-weighted bwd advantage estimate: [{adv_a:+.4f} (gain=0 beyond d=12), {adv_b:+.4f} (gain(d>=13)=gain(9)>0)]")
P(f"vs P0-measured bwd advantage {r_p0_fwd-r_p0_bwd:+.4f} -> P0 protocol overstated bwd value by "
  f"{(r_p0_fwd-r_p0_bwd)-adv_a:.4f}..{(r_p0_fwd-r_p0_bwd)-adv_b:.4f}")


# P0 per-geometry: bwd distance classes
P("\nP0 honest rows by bwd distance (bwd config RMSE):")
for m in sorted(set(val_ta[honest].tolist())):
    sel = honest & (val_ta == m) & np.isfinite(p_p0_bwd)
    rr = float(np.sqrt(np.mean((p_p0_bwd[sel]-val_target[sel])**2)))
    P(f"  {m//12}-{m%12+1:02d} (bwd={bwd_dist(int(m), sorted(val_anchor_tas))}): n={int(sel.sum()):,} RMSE={rr:.4f}")

# ---------- M1d: mirror + richer D-hat (13 anchors) ----------
log("\n=== M1d: mirror rows, D-hat from 13 anchor fields (density isolation) ===")
EXTRA_DHAT = [24171, 24181, 24182, 24184, 24185]  # Apr14 Feb15 Mar15 May15 Jun15 (NOT scored in mirror)
mir_d = val[val['t_abs'].isin(m_months | set(EXTRA_DHAT))].copy()
mir_d['masked'] = mir_d['t_abs'].isin(M_RUNS)
mir_d['TWS_t'] = val_tws_visible.loc[mir_d.index].values   # restore pristine TWS
mir_d.loc[mir_d['masked'], 'TWS_t'] = np.nan
r_m1d_bwd = None
p_d = run_masked(infra, mir_d, mir_d['TWS_t'], 0.80, use_denoise=False, use_bwd=True, lam=0.84,
                 anchors_override=M_ANCHORS, extra_dhat_months=EXTRA_DHAT,
                 s_months=sorted(m_months))
sel_d = mir_d['masked'].values & np.isfinite(p_d) & np.isfinite(mir_d['target'].values)
r_m1d_bwd = float(np.sqrt(np.mean((p_d[sel_d]-mir_d['target'].values[sel_d])**2)))
P(f"M1d bwd (D-hat from {len(M_ANCHORS)+len(EXTRA_DHAT)} fields, filter anchors unchanged): {r_m1d_bwd:.4f}  "
  f"(M1 with 6-field D-hat: {r_m1_bwd:.4f}; P0 with 14-field: {r_p0_bwd:.4f})")
P(f"  -> D-hat density effect (M1d - M1): {r_m1d_bwd - r_m1_bwd:+.4f}; "
  f"geometry effect (P0 - M1d): {r_m1d_bwd - r_p0_bwd:+.4f}")
# ---------- denoiser check under mirror ----------
log("\n=== denoiser check under M1 (v17b said denoiser HURTS on P0) ===")
p_dn = run_masked(infra, mir, mir['TWS_t'], 0.80, use_denoise=True, use_bwd=True, lam=0.84)
mm_dn = mir_msk & np.isfinite(p_dn)
r_dn = float(np.sqrt(np.mean((p_dn[mm_dn]-mir_target[mm_dn])**2)))
P(f"M1 bwd + PC-denoiser: {r_dn:.4f} vs no-denoise {r_m1_bwd:.4f}  (P0: 0.6804 vs 0.6535)")

# ---------- phi grid under mirror ----------
log("\n=== phi grid under M1 (bwd) ===")
for phi in [0.70, 0.74, 0.80, 0.85, 0.90]:
    pp = run_masked(infra, mir, mir['TWS_t'], phi, use_denoise=False, use_bwd=True, lam=0.84)
    mmp = mir_msk & np.isfinite(pp)
    P(f"  phi={phi}: M1 bwd RMSE = {float(np.sqrt(np.mean((pp[mmp]-mir_target[mmp])**2))):.4f}")
P("\n=== phi grid under M1 (fwd-only) ===")
for phi in [0.70, 0.74, 0.80, 0.85, 0.90]:
    pp = run_masked(infra, mir, mir['TWS_t'], phi, use_denoise=False, use_bwd=False, lam=0.84)
    mmp = mir_msk & np.isfinite(pp)
    P(f"  phi={phi}: M1 fwd RMSE = {float(np.sqrt(np.mean((pp[mmp]-mir_target[mmp])**2))):.4f}")

# ---------- summary ----------
P("\n================ SUMMARY ================")
P(f"P0 honest CV (bwd-heavy v17b config): {r_p0_bwd:.4f}   | implied LB masked RMSE ~0.7355 -> gap +{0.7355-r_p0_bwd:.3f}")
P(f"M1 mirrored CV (same config):         {r_m1_bwd:.4f}   | gap +{0.7355-r_m1_bwd:.3f}")
P(f"M2 mirrored CV (gap-14 variant):       {r_m2_bwd:.4f}   (bwd adv {r_m2_fwd-r_m2_bwd:+.4f})")
P(f"protocol optimism explained by mirror: {r_m1_bwd-r_p0_bwd:+.4f} of the ~+0.082 total gap")
P(f"bwd-vs-fwd advantage: P0 {r_p0_fwd-r_p0_bwd:+.4f} -> M1 {r_m1_fwd-r_m1_bwd:+.4f} -> M2 {r_m2_fwd-r_m2_bwd:+.4f} "
  f"(test drift to explain: ~+0.027)")

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
print(f"\nsaved {OUT}")
