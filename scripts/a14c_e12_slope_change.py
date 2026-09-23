"""
Agent 14-c E1+E2: anchor anomaly table + slope-change (regime-change) fit.

Central hypothesis: generator continues each cell's train trend into the test era
with a REGIME CHANGE (slope factor r and/or offset at the boundary).
  A_j(c) = a_c + b_c*(t_j - t_mid) + fast + noise      (per-cell, 6 anchors)
  b_c  =? r * beta_c  (train slope)   -> r, R^2, ratio distribution
  a_c  =? alpha * trendex_c(t_mid) + off_c  (offset structure)

Also: variance decomposition of the slope field (signal vs estimation noise),
spatial smoothing of residual slope field, era-drift accounting.
Caches infrastructure to /home/z/my-project/data/a14c_cache.npz.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------------- train infrastructure ----------------
print("Loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_tr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxx_tr = np.nansum(td*td, axis=0)
beta_c = np.where(sxx_tr > 100, np.nansum(td*F64, axis=0)/np.where(sxx_tr>0,sxx_tr,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

full = ~np.isnan(F).any(axis=0)
full_idx = np.where(full)[0]
n_full = int(full.sum())

# detrended residual variance per cell (train) -> beta estimation noise
TD = t_abs_tr[:, None] - tbar_c[None, :]
A_dt = F64 - mu_c[None, :] - TD * beta_c[None, :]
sig2_tr = np.nanvar(A_dt, axis=0)          # per-cell detrended variance (fast+noise)
dof = np.maximum(ok_t.sum(axis=0) - 2, 1)
sig2_res_tr = np.nansum(np.where(ok_t, A_dt**2, np.nan), axis=0) / dof   # unbiased-ish
# per-cell beta noise variance
beta_noise2 = np.where(sxx_tr > 100, sig2_res_tr / np.where(sxx_tr>0, sxx_tr, 1), np.nan)

# PC bases (on full-history cells)
A_dtf = A_dt[:, full]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
_, _, Vt_dt = np.linalg.svd(A_dtf, full_matrices=False)
V_dt = Vt_dt[:200].T.astype(np.float32)     # detrended basis
A_an = (F64 - mu_c[None, :])[:, full]
A_an = A_an - A_an.mean(axis=0, keepdims=True)
_, _, Vt_an = np.linalg.svd(A_an, full_matrices=False)
V_an = Vt_an[:100].T.astype(np.float32)     # anomaly basis (trend-inclusive)
print(f"cells={n_cells}, full-history={n_full}, T={T}")

# ---------------- test anchors ----------------
print("Loading test...", flush=True)
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
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
A = np.array([AF[a] for a in anchors])      # (6, n_cells)
t_anc = np.array(anchors, dtype=np.float64)
t_mid = t_anc.mean()

print("\n================ E1: ANCHOR ANOMALY TABLE ================")
def corr2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0,1])
print(f"{'anchor':>10} {'t_abs':>7} {'dt_mid':>7} {'std':>7} {'mean':>8} {'nNaN':>5}  corr_to_others")
for j, a in enumerate(anchors):
    cors = [corr2(A[j], A[k]) for k in range(6) if k != j]
    print(f"{a:>10} {t_anc[j]:>7.0f} {t_anc[j]-t_mid:>7.1f} {np.nanstd(A[j]):>7.3f} {np.nanmean(A[j]):>+8.3f} {int((~np.isfinite(A[j])).sum()):>5}  {np.round(cors,3)}")
print(f"\ntrain beta_c: std={beta_c.std():.5f}/mo mean={beta_c.mean():+.6f}")
print(f"trendex at anchors (std across cells): " + ", ".join(f"{np.nanstd(trendex(a)):.3f}" for a in anchors))
print(f"corr(A_j, trendex_j): " + ", ".join(f"{corr2(A[j], trendex(a)):.3f}" for j, a in enumerate(anchors)))

# ---------------- E2: per-cell slope fit on 6 anchors ----------------
print("\n================ E2: SLOPE-CHANGE FIT ================")
X = np.column_stack([np.ones(6), t_anc - t_mid])          # design (6,2)
XtX_inv = np.linalg.inv(X.T @ X)
# per-cell fit for all cells with all 6 anchors finite
Aok = np.isfinite(A).all(axis=0)
a_int = np.full(n_cells, np.nan); b_sl = np.full(n_cells, np.nan)
Afull = A[:, Aok]
coefc = np.linalg.solve(X.T@X, X.T@Afull)                  # (2, n_ok)
a_int[Aok] = coefc[0]; b_sl[Aok] = coefc[1]
resid = Afull - (X @ coefc)
s2_c = (resid**2).sum(axis=0) / 4                          # per-cell resid var (fast+noise)
sig2_anc = float(np.median(s2_c))
Sxx = float(((t_anc - t_mid)**2).sum())
slope_noise2 = sig2_anc / Sxx
print(f"cells with all 6 anchors finite: {Aok.sum()}")
print(f"per-anchor residual var (fast+noise, median over cells): {sig2_anc:.4f} -> std {np.sqrt(sig2_anc):.3f}")
print(f"Sxx (anchor time spread) = {Sxx:.1f}  -> per-cell slope noise std = {np.sqrt(slope_noise2):.5f}/mo")
print(f"OLS slope field b: std={np.nanstd(b_sl):.5f}  (noise {np.sqrt(slope_noise2):.5f})")
print(f"  implied TRUE slope std = sqrt(var(b)-noise) = {np.sqrt(max(np.nanvar(b_sl)-slope_noise2,0)):.5f}/mo")
print(f"  train slope std beta = {beta_c.std():.5f}/mo")

def huber_regression(x, y, delta=2.0, iters=8):
    """y = s*x + c, Huber weights, returns (s, c, R2_raw, n_keep)"""
    ok = np.isfinite(x) & np.isfinite(y)
    x_, y_ = x[ok], y[ok]
    s, c = np.polyfit(x_, y_, 1)
    for _ in range(iters):
        e = y_ - (s*x_ + c)
        sc = 1.4826*np.median(np.abs(e - np.median(e))) + 1e-12
        u = np.abs(e)/(delta*sc)
        w = np.where(u <= 1, 1.0, 1.0/np.maximum(u, 1e-12))
        Wm = np.diag(w)  # avoid huge diag; use weighted normal equations
        Xw = np.column_stack([x_, np.ones(len(x_))])
        sw = np.sqrt(w)
        sol = np.linalg.solve((Xw*sw[:,None]).T@(Xw*sw[:,None]), (Xw*sw[:,None]).T@(y_*sw))
        s, c = sol[0], sol[1]
    e = y_ - (s*x_ + c)
    r2 = 1 - np.var(e)/np.var(y_)
    keep = (np.abs(e) <= delta*sc).mean()
    return s, c, r2, keep

ok = Aok & np.isfinite(beta_c) & (beta_noise2 > 0) & np.isfinite(b_sl)
x = beta_c[ok].astype(np.float64); y = b_sl[ok].astype(np.float64)
s_ols, c_ols, r2_ols, _ = huber_regression(x, y)
s_h, c_h, r2_h, keep_h = huber_regression(x, y, delta=2.0)
# trimmed: drop cells with tiny |beta| (ratio unstable) handled separately
# attenuation correction: beta measured with noise var beta_noise2
attn = 1.0/(1.0 + float(np.nanmean(beta_noise2[ok]))/np.var(x))
print(f"\n--- b_test (6-anchor OLS) on b_train (beta_c) ---")
print(f"OLS/huber slope r = {s_h:.4f} (raw OLS {s_ols:.4f}), intercept = {c_h:+.6f}, R^2 = {r2_h:.4f}, huber keep-frac = {keep_h:.2f}")
print(f"beta est-noise attenuation factor = {attn:.4f}  -> attenuation-corrected r = {s_h/attn:.4f}")
print(f"SE(r) approx = {np.sqrt((1-r2_h)/len(x))*np.std(y)/np.std(x):.4f}  (n={ok.sum()})")
print(f"=> test-era slope is {s_h/attn*100:.1f}% of train slope (r={s_h/attn:.3f})")

# ratio distribution (trim: require |beta| large, drop slope outliers)
sel = ok & (np.abs(beta_c) > np.nanquantile(np.abs(beta_c[ok]), 0.5))
ratio = b_sl[sel]/beta_c[sel]
lo, hi = np.nanquantile(ratio, [0.1, 0.9])
inl = ratio[(ratio >= lo) & (ratio <= hi)]
print(f"\nratio b/beta (|beta|>median, 10-90% trim): median={np.median(inl):.3f}  mean={inl.mean():.3f}  p25/p75={np.quantile(inl,0.25):.3f}/{np.quantile(inl,0.75):.3f}")
print(f"  NOTE: raw ratio is dominated by slope noise (noise/signal ~ {np.sqrt(slope_noise2)/ (s_h*np.std(x)):.1f}x)")

# variance decomposition of slope field
var_b_ols = float(np.nanvar(b_sl))
var_b_true = max(var_b_ols - slope_noise2, 0.0)
r_corr = s_h  # cov(b,beta)/var(beta)
var_expl_by_beta = (r_corr**2)*np.var(x)
print(f"\nvar(b_ols)={var_b_ols:.3e}  noise={slope_noise2:.3e}  var(b_true)~{var_b_true:.3e}")
print(f"var explained by r*beta = {var_expl_by_beta:.3e}  ({var_expl_by_beta/max(var_b_true,1e-12)*100:.1f}% of true slope var)")
print(f"INDEPENDENT slope variance (b_true - r*beta part): {max(var_b_true-var_expl_by_beta,0):.3e} -> std {np.sqrt(max(var_b_true-var_expl_by_beta,0)):.5f}/mo")

# ---------------- intercept / offset structure ----------------
print("\n--- intercept a_c (level at t_mid) vs trend continuation ---")
txm = trendex(int(round(t_mid)))   # trend-implied level at t_mid
ok2 = Aok & np.isfinite(txm)
s2, c2, r22, _ = huber_regression(txm[ok2].astype(np.float64), a_int[ok2].astype(np.float64))
# intercept noise var: sig2_anc * (XtX_inv[0,0])
int_noise2 = sig2_anc * float(XtX_inv[0,0])
var_a = float(np.nanvar(a_int)); var_a_true = max(var_a - int_noise2, 0)
print(f"a on trendex(t_mid): slope={s2:.3f} intercept={c2:+.4f} R^2={r22:.4f}")
print(f"var(a_ols)={var_a:.4f} noise={int_noise2:.4f} -> var(a_true)~{var_a_true:.4f} (std {np.sqrt(var_a_true):.3f})")
print(f"var(a_true) explained by alpha*trendex = {(s2**2)*np.nanvar(txm):.4f}; residual offset var ~ {max(var_a_true-(s2**2)*np.nanvar(txm),0):.4f} (std {np.sqrt(max(var_a_true-(s2**2)*np.nanvar(txm),0)):.3f})")

# off field: a - alpha*trendex(t_mid); smooth via PC projection
off_raw = a_int - s2*txm
off_full = np.where(Aok & np.isfinite(txm), off_raw, np.nan)
xg = off_full[full]
okg = np.isfinite(xg)
for name, Vb in [("anomaly-PC", V_an), ("detrended-PC", V_dt)]:
    for K in [10, 20, 50]:
        if Vb.shape[1] < K: continue
        VK = Vb[:, :K]
        proj = VK @ (VK.T @ np.where(okg, xg, 0.0))
        # noise captured by K-dim projection of pure noise: slope-var analog
        # off noise var int_noise2, isotropic -> K/N fraction
        exp_noise = int_noise2 * K / n_full
        print(f"  off-field {name} K={K}: proj var={np.nanvar(proj):.4f} (expected noise-in-subspace {exp_noise:.4f})")

# ---------------- era drift accounting ----------------
print("\n--- era drift: can trend continuation explain it? ---")
early_idx = [0,1,2,3]; late_idx = [4,5]
D_early = np.nanmean(A[early_idx], axis=0); D_late = np.nanmean(A[late_idx], axis=0)
o = np.isfinite(D_early) & np.isfinite(D_late) & np.isfinite(beta_c)
print(f"corr(D_early, D_late) = {corr2(D_early, D_late):.3f}   std(diff) = {np.std((D_early-D_late)[o]):.3f}")
t_e = t_anc[early_idx].mean(); t_l = t_anc[late_idx].mean()
pred_drift = (s_h*beta_c*(t_l - t_e))   # slope-model predicted era drift
diff = D_late - D_early
for nm, fld in [("raw diff", diff), ("diff - slope-model drift", diff - pred_drift)]:
    oo = o & np.isfinite(fld)
    print(f"  {nm}: std={np.std(fld[oo]):.3f} corr with beta = {corr2(fld, beta_c):.3f}")

# ---------------- spatial smoothing of residual slope field ----------------
print("\n--- residual slope field s = b - r*beta: does it have spatial structure? ---")
sres = b_sl - s_h*beta_c
sg = sres[full]
okg = np.isfinite(sg)
for name, Vb in [("anomaly-PC", V_an), ("detrended-PC", V_dt)]:
    for K in [10, 20, 50]:
        if Vb.shape[1] < K: continue
        VK = Vb[:, :K]
        proj = VK @ (VK.T @ np.where(okg, sg, 0.0))
        exp_noise = slope_noise2 * K / n_full
        print(f"  slope-resid {name} K={K}: proj var={np.nanvar(proj):.3e} (noise-in-subspace {exp_noise:.3e}) ratio={np.nanvar(proj)/exp_noise:.1f}")

# ---------------- save cache ----------------
np.savez_compressed(f'{DATA}/a14c_cache.npz',
    mu_c=mu_c, beta_c=beta_c, tbar_c=tbar_c, t_abs_tr=t_abs_tr, full=full,
    V_dt=V_dt, V_an=V_an, anchors=np.array(anchors), A=A, t_anc=t_anc,
    a_int=a_int, b_sl=b_sl, slope_noise2=slope_noise2, sig2_anc=sig2_anc,
    s_h=s_h, attn=attn, alpha_off=s2, XtX_inv=XtX_inv,
    beta_noise2=beta_noise2, t_mid=t_mid)
print("\ncache saved -> data/a14c_cache.npz")
