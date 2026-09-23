"""
AUDIT B - 03 (B4): floor decomposition recheck + test-era noise estimation + 2015-02 event forensics.
"""
import numpy as np, pandas as pd, time, sys
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import *
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

tr, te = load_cached()
train = build_train_df(tr)
n_cells = int(train['cc'].max())+1
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
cal = test.assign(m=test['TWS_t_masked'].astype(bool)).groupby(test['time'].dt.month)['m'].mean()

train['year'] = train['t_abs']//12
val = train[(train['year'] >= 2013) & (train['year'] <= 2015)].copy()
val['masked'] = val['t_abs'].apply(lambda t: (t%12+1)).map(cal).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val_target = val['target'].values.astype(np.float64)
val_msk = val['masked'].values; val_ta = val['t_abs'].values; val_cc = val['cc'].values
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val_ta])
honest = val_msk & ~shortcut
fit = train[train['year'] <= 2012].copy()
infra_cv = build_infra(fit, n_cells, 'cv')

# ================= (a) train-era noise std, method of moments =================
log("=== (a) train-era noise via detrended AR+noise MoM ===")
# per-cell detrended anomaly a_t = TWS - mu - trend, consecutive-present months only
F = np.full((len(np.sort(fit['t_abs'].unique())), n_cells), np.nan)
yms = np.sort(fit['t_abs'].unique()); ym_to_i = {int(v): i for i, v in enumerate(yms)}
tab = fit['t_abs'].values
F[fit['t_abs'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
mu = infra_cv['mu_c']; tb = infra_cv['tbar_c']; be = infra_cv['beta_c']
ta_arr = np.array([int(v) for v in yms], dtype=np.float64)
A_dt = F - mu[None,:] - (ta_arr[:,None]-tb[None,:])*be[None,:]
# full-train version too (for test-era mu)
infra_full = build_infra(train, n_cells, 'full')

def mom_noise(A, ta_arr, label):
    # per-cell r1, r2 using consecutive pairs (only pairs with t2=t1+1)
    T, n = A.shape
    r1n = np.zeros(n); r1d = np.zeros(n); r2n = np.zeros(n); r2d = np.zeros(n)
    v = np.zeros(n)
    ok = np.isfinite(A)
    for i in range(T):
        v += np.where(ok[i], (A[i]-0.0)**2, 0)   # anomalies are ~zero mean already
        cnt_ok = ok[i]
        if i+1 < T and ta_arr[i+1] == ta_arr[i]+1:
            m = ok[i] & ok[i+1]
            r1n += np.where(m, A[i]*A[i+1], 0); r1d += np.where(m, 1, 0)*m
        if i+2 < T and ta_arr[i+2] == ta_arr[i]+2:
            m = ok[i] & ok[i+2]
            r2n += np.where(m, A[i]*A[i+2], 0); r2d += np.where(m, 1, 0)*m
    nobs = ok.sum(axis=0)
    var_a = v/np.maximum(nobs,1)
    r1 = np.divide(r1n, np.maximum(r1d,1)); r2 = np.divide(r2n, np.maximum(r2d,1))
    good = (nobs >= 60) & (r1 > 0.05) & (r2 > 0.01) & (r1**2 > r2)
    # AR(1)+noise: r1 = phi*lam, r2 = phi^2*lam  => phi = r2/r1, lam = r1^2/r2
    phi_c = np.where(good, r2/np.maximum(r1,1e-9), np.nan)
    lam_c = np.where(good, r1**2/np.maximum(r2,1e-9), np.nan)
    phi_m = np.nanmedian(phi_c); lam_m = np.nanmedian(lam_c)
    var_m = np.nanmedian(var_a[good])
    noise_var = (1-lam_m)*var_m
    log(f"  [{label}] cells used={good.sum():,}  median r1={np.nanmedian(r1[good]):.3f} r2={np.nanmedian(r2[good]):.3f}")
    log(f"  [{label}] => phi={phi_m:.3f}  lam(fast share)={lam_m:.3f}  Var(anom)={var_m:.4f} (std {np.sqrt(var_m):.3f})")
    log(f"  [{label}] => NOISE STD = {np.sqrt(noise_var):.4f}   (team claim: 0.456)")
    # variogram cross-check: E[(a_t - a_{t+s})^2] = 2*noise + 2*Var(x)(1-phi^s)
    return dict(phi=phi_m, lam=lam_m, var=var_m, noise=np.sqrt(noise_var), r1=r1, r2=r2, var_a=var_a, good=good)

MoM_cv = mom_noise(A_dt, ta_arr, 'fit<=2012')

# ================= (d) 2015-02 event forensics =================
log("\n=== (d) honest-month error forensics (phi.80/lam.84) ===")
p_blend, DG = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=True, use_denoise=False, lam=0.84, diag=True)
meta = DG['meta']; rows = DG['rows_list']; starts = np.cumsum([0]+[len(r) for r in rows])
# slow proxy: val-era mean anchor anomaly
AFval = {a: DG['AF'][a] for a in DG['anchors']}
A_val = np.nanmean(np.array([AFval[a] for a in DG['anchors']]), axis=0)
log(f"  A_val (val-era slow anomaly proxy): std={np.nanstd(A_val):.4f}")
for j in range(len(meta)):
    m = int(meta['m'].iloc[j])
    s, e = starts[j], starts[j+1]
    rws = rows[j]; hon = honest[rws]
    if hon.sum() == 0: continue
    t_ = val_target[rws][hon]; mu_ = mu[val_cc[rws]][hon]
    dT_ = DG['Dtil'][s:e][hon]
    err = (mu_+dT_+DG['xf'][s:e][hon]+DG['wgt'][s:e][hon]*(np.nan_to_num(DG['xb'][s:e][hon])-DG['xf'][s:e][hon])) - t_
    derr = dT_ - A_val[val_cc[rws]][hon]
    fastpart = mu_+DG['xf'][s:e][hon] - (t_ - A_val[val_cc[rws]][hon])   # xhat - (x+n)
    log(f"  {m//12}-{m%12+1:02d}: errRMSE={np.sqrt(np.nanmean(err**2)):.4f}  errMEAN={np.nanmean(err):+.4f}  "
        f"DerrRMSE={np.sqrt(np.nanmean(derr**2)):.4f}  fast+noiseRMSE={np.sqrt(np.nanmean(fastpart**2)):.4f}  "
        f"corr(Derr,fast)={np.nan_to_num(np.corrcoef(derr[np.isfinite(derr)&np.isfinite(fastpart)], fastpart[np.isfinite(derr)&np.isfinite(fastpart)])[0,1]):+.3f}")

# global-mean fast anomaly around 2015 (common-mode shock?)
log("\n  global-mean detrended anomaly by month (val era, from truth):")
G = []
for m in np.sort(val['t_abs'].unique()):
    sel = val_ta == m
    a = (val_tws_visible.values[sel] - mu[val_cc[sel]])
    G.append((int(m), float(np.nanmean(a))))
gm = dict(G)
for m in sorted(gm):
    if 2014*12+9 <= m <= 2015*12+8:
        log(f"    {m//12}-{m%12+1:02d}: {gm[m]:+.3f}")

# ================= (c) D-tilde error on honest protocol =================
log("\n=== (c) D-tilde error vs val-era slow proxy ===")
derr_all = []
for j in range(len(meta)):
    m = int(meta['m'].iloc[j]); s, e = starts[j], starts[j+1]
    rws = rows[j]; hon = honest[rws]
    if hon.sum() == 0: continue
    derr_all.append(DG['Dtil'][s:e][hon] - A_val[val_cc[rws]][hon])
derr_all = np.concatenate(derr_all)
log(f"  RMSE(Dtil - A_val) over honest rows = {np.sqrt(np.nanmean(derr_all**2)):.4f}  (floor claim: 0.42)")
# but the true test analog: Dtil vs the ACTUAL slow state at tm (drifts). upper bound via anchor LOO on val:
log("  anchor-LOO D-tilde quality on val anchors (predict anchor a from other anchors+S+trendex):")
loo = []
for a in DG['anchors']:
    others = [x for x in DG['anchors'] if x != a]
    Dh_o = np.nanmean(np.array([AFval[x] for x in others]), axis=0)
    S_ = DG['S']
    trx = ((np.float64(a)-tb)*be).astype(np.float32)
    dt_ = 0.70*Dh_o + 0.45*S_ + 0.073*trx
    r = np.sqrt(np.nanmean((dt_ - (AFval[a]-0))**2))   # AFval includes fast+noise -> includes irreducible
    loo.append(r)
log(f"    LOO RMSE (Dtil vs raw anchor anomaly, includes fast+noise): mean={np.mean(loo):.4f}")

# ================= (b) TEST-era noise estimation =================
log("\n=== (b) test-era noise from anchor-pair correlations ===")
# test anchor fields (full-train mu)
tte, tmsk, tcc = te['t_abs'], te['masked'], te['cc']
tws = te['TWS']
anchors_t = [24188, 24192, 24197, 24203, 24222, 24226]
AFt = {}
for a in anchors_t:
    f = np.full(n_cells, np.nan, dtype=np.float32)
    sel = (tte == a) & (~tmsk)
    f[tcc[sel]] = tws[sel]
    AFt[a] = f - infra_full['mu_c']
log(f"  test anchor anomaly fields: std = {[round(float(np.nanstd(AFt[a])),3) for a in anchors_t]}")
# pairwise correlations by gap
pairs = []
ks = list(AFt)
for i in range(len(ks)):
    for j in range(i+1, len(ks)):
        g = ks[j]-ks[i]
        x, y = AFt[ks[i]], AFt[ks[j]]
        m = np.isfinite(x)&np.isfinite(y)
        r = np.corrcoef(x[m], y[m])[0,1]
        pairs.append((g, r, int(m.sum())))
pairs.sort()
log("  anchor pair correlations: " + ", ".join(f"g={g}:r={r:.3f}" for g, r, n in pairs))
# fit r(g) = A + B*phi^g (least squares over grid)
gs = np.array([p[0] for p in pairs], dtype=float); rs = np.array([p[1] for p in pairs])
best = None
for phi in np.arange(0.5, 0.99, 0.01):
    X = np.column_stack([np.ones_like(gs), phi**gs])
    coef, res_, *_ = np.linalg.lstsq(X, rs, rcond=None)
    pred = X@coef
    sse = float(np.sum((rs-pred)**2))
    if best is None or sse < best[0]:
        best = (sse, phi, coef)
sse, phi_t, coef_t = best
A_t, B_t = float(coef_t[0]), float(coef_t[1])
noise_share_t = 1 - A_t - B_t
log(f"  fit r(g)=A+B*phi^g: A={A_t:.3f} B={B_t:.3f} phi={phi_t:.2f} => noise share={noise_share_t:.3f}")
avg_var_t = np.mean([np.nanvar(AFt[a]) for a in anchors_t])
log(f"  test anchor anomaly var (mean)={avg_var_t:.4f} (std {np.sqrt(avg_var_t):.3f}) => implied TEST noise std = {np.sqrt(max(noise_share_t,0)*avg_var_t):.4f}")
# validate the estimator on train-era pseudo anchors (same gaps pattern)
log("  VALIDATION on train: pseudo-anchor sequences with gaps [4,5,6,19,4]:")
for start in range(24040, 24160-30):
    months = [start, start+4, start+9, start+15, start+34, start+38]
    if months[-1] > 24187: continue
    # need all present
    if any(m not in set(np.unique(tr['t_abs']).tolist()) for m in months): continue
    fields = []
    for mth in months:
        sel = tr['t_abs'] == mth
        f = np.full(n_cells, np.nan, dtype=np.float32)
        f[tr['cc'][sel]] = tr['TWS'][sel]
        fields.append(f - infra_full['mu_c'] - ((np.float64(mth)-infra_full['tbar_c'])*infra_full['beta_c']).astype(np.float32))
    prs = []
    for i in range(6):
        for j in range(i+1, 6):
            g = months[j]-months[i]
            x, y = fields[i], fields[j]
            m2 = np.isfinite(x)&np.isfinite(y)
            prs.append((g, np.corrcoef(x[m2], y[m2])[0,1]))
    gs2 = np.array([p[0] for p in prs]); rs2 = np.array([p[1] for p in prs])
    bst = None
    for phi in np.arange(0.5, 0.99, 0.01):
        X = np.column_stack([np.ones_like(gs2), phi**gs2])
        coef, *_ = np.linalg.lstsq(X, rs2, rcond=None)
        sse2 = float(np.sum((rs2-X@coef)**2))
        if bst is None or sse2 < bst[0]: bst = (sse2, phi, coef)
    if start % 12 == 0:
        log(f"    start={start//12}-{start%12+1:02d}: A={bst[2][0]:.3f} B={bst[2][1]:.3f} phi={bst[1]:.2f} noiseShare={1-bst[2][0]-bst[2][1]:.3f}")
log("  (train MoM noise share = 1-lam from (a); compare)")

# consecutive-anchor difference approach (test)
log("\n  consecutive-anchor difference variance (test):")
for i in range(len(anchors_t)-1):
    a, b = anchors_t[i], anchors_t[i+1]
    g = b-a
    x, y = AFt[a], AFt[b]
    m2 = np.isfinite(x)&np.isfinite(y)
    d = y[m2]-x[m2]
    log(f"    {a//12}-{a%12+1:02d} -> {b//12}-{b%12+1:02d} (g={g}): Var(d)={np.var(d):.4f} RMSE(d)={np.sqrt(np.var(d)):.4f} mean(d)={np.mean(d):+.4f}")
log("DONE")
