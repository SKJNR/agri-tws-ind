"""
AUDIT B2 (re-scoped) — item B3: FLOOR & TEST-ERA NOISE.
(a) train-era noise: ARMA(1,1)/AR(1)+white MoM on detrended fit-era panel (per-gap pooled covariances)
    -> var_f (fast), phi_f, var_e (white noise), fast innovation std. Which one is "0.456"?
(b) test-era noise from the 6 test anchors: pooled cross-cell covariances at 12 distinct gaps
    (15 anchor pairs), fit cov(g) = V_D + var_f*phi^g; var_e_test = V_anchor - V_D - var_f.
    Same estimator on train panel (V_D=0) for apples-to-apples.
(c) D-tilde error: (i) on mirrored M1 (val era, 6-anchor D-hat like test); (ii) directly on TEST
    anchors via leave-one-anchor-out D-tilde (honest, anchors visible). Decompose
    masked-err^2 = noise^2 + D-err^2 + fast-err^2; compare to claimed 0.456/0.42/0.37 -> 0.722.
Also: H/R/var_f calibration values in val (P0, 14 anchors) vs TEST (6 anchors) phase [B1 item d].
Cached npz only; no CSV loads. New file only.
"""
import numpy as np, pandas as pd, time, sys, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import load_cached, build_train_df, build_infra, run_masked2, COVS

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
OUT = '/home/z/my-project/download/auditB2_floor.txt'
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))

tr, te = load_cached()
train = build_train_df(tr)
train['year'] = train['t_abs'] // 12
n_cells = int(train['cc'].max()) + 1

fit = train[train['year'] <= 2012].copy()
log("building infra fit<=2012 and full...")
infra_fit = build_infra(fit, n_cells, 'fit')
infra_full = build_infra(train, n_cells, 'full')

# calendar mask fractions (test)
tmonth = te['t_abs'] % 12 + 1
cal = pd.Series(te['masked'].astype(float)).groupby(tmonth).mean()

# ================= (a) train-era noise: AR(1)+white MoM =================
def panel(df, infra):
    yms = np.sort(df['ym'].unique()); T = len(yms)
    ym_to_i = {int(v): i for i, v in enumerate(yms)}
    ta_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)
    F = np.full((T, n_cells), np.nan, dtype=np.float64)
    F[df['ym'].map(ym_to_i).values, df['cc'].values] = df['TWS_t'].values
    A = F - infra['mu_c'][None, :] - (ta_arr[:, None] - infra['tbar_c'][None, :])*infra['beta_c'][None, :]
    A = A - np.nanmean(A, axis=0, keepdims=True)          # center per cell
    return A, ta_arr

A_tr, ta_tr = panel(fit, infra_fit)
fullc = infra_fit['full']
A_tr = A_tr[:, fullc]
P(f"train MoM panel: {A_tr.shape[0]} months x {A_tr.shape[1]} full cells")

def pooled_cov_by_gap(A, ta_arr, gmax=18):
    T = len(ta_arr)
    out = {}
    for g in range(1, gmax+1):
        num, cnt = 0.0, 0
        for i in range(T):
            j = i + g
            if j >= T: break
            if ta_arr[j] - ta_arr[i] != g: continue
            x, y = A[i], A[j]
            ok = np.isfinite(x) & np.isfinite(y)
            num += np.sum(x[ok]*y[ok]); cnt += ok.sum()
        if cnt > 500:
            out[g] = num/cnt
    return out

cov_tr = pooled_cov_by_gap(A_tr, ta_tr)
V_tr = float(np.nanmean(np.nanvar(A_tr, axis=0, ddof=1)))
def lsq_nn(X, y):
    """Least squares with non-negative coefficients (project negative coefs to 0)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if (coef >= 0).all():
        return coef
    best = None
    for k in range(len(coef)):
        mask = np.ones(len(coef), bool); mask[k] = False
        c2, *_ = np.linalg.lstsq(X[:, mask], y, rcond=None)
        c = np.zeros(len(coef)); c[mask] = c2
        if (c >= 0).all():
            sse = float(np.sum((X@c - y)**2))
            if best is None or sse < best[0]: best = (sse, c)
    return coef if best is None else best[1]

def fit_ar(covs, with_D=False, phis=np.arange(0.50, 0.96, 0.01)):
    gs = np.array(sorted(covs.keys()), dtype=float)
    cv = np.array([covs[int(g)] for g in gs])
    best = None
    for phi in phis:
        b = phi**gs
        X = np.column_stack([np.ones_like(b), b]) if with_D else b[:, None]
        coef = lsq_nn(X, cv)
        pred = X @ coef
        sse = float(np.sum((cv - pred)**2))
        if best is None or sse < best[0]:
            best = (sse, phi, coef)
    return best

sse, phi_tr, coef_tr = fit_ar(cov_tr, with_D=False)
var_f_tr = float(coef_tr[0])
var_e_tr = max(V_tr - var_f_tr, 0.0)
P(f"\n(a) TRAIN-ERA (fit<=2012, detrended): V={V_tr:.4f} (std {np.sqrt(V_tr):.4f})")
P(f"    pooled cov rho(g): " + ", ".join(f"g={g}:{c/V_tr:.3f}" for g, c in list(cov_tr.items())[:8]))
P(f"    AR(1)+white MoM fit: phi_f={phi_tr:.2f}, var_f={var_f_tr:.4f} (lam={var_f_tr/V_tr:.3f}), var_e={var_e_tr:.4f}")
P(f"    -> sigma_e (white noise) = {np.sqrt(var_e_tr):.4f}; lam implied {var_f_tr/V_tr:.3f} (model uses 0.84)")
P(f"    -> fast innovation std sigma_eps = sqrt(var_f*(1-phi^2)) = {np.sqrt(var_f_tr*(1-phi_tr**2)):.4f}")
# same on FULL train era (2002-2015, detrended) for reference
A_full, ta_full = panel(train, infra_full)
A_full = A_full[:, infra_full['full']]
cov_full = pooled_cov_by_gap(A_full, ta_full)
V_full = float(np.nanmean(np.nanvar(A_full, axis=0, ddof=1)))
sseF, phi_F, coefF = fit_ar(cov_full, with_D=False)
var_f_F = float(coefF[0]); var_e_F = max(V_full - var_f_F, 0.0)
P(f"    (full-train 2002-15 refit: V={V_full:.4f}, phi={phi_F:.2f}, var_f={var_f_F:.4f}, var_e={var_e_F:.4f}, sigma_e={np.sqrt(var_e_F):.4f})")

# ================= (b) test-era noise from anchor covariances =================
P("\n(b) TEST-ERA noise via anchor-pair covariances")
tcc = te['cc'].values; tta = te['t_abs'].values; tmsk = te['masked'].astype(bool)
tmf = pd.Series(tmsk).groupby(tta).mean()
anchors_t = sorted(int(v) for v in tmf[tmf < 0.01].index)
P(f"    test anchors: {[(a, f'{a//12}-{a%12+1:02d}') for a in anchors_t]}")
AF_t = {}
for a in anchors_t:
    f = np.full(n_cells, np.nan, dtype=np.float64)
    sel = (tta == a) & (~tmsk)
    f[tcc[sel]] = te['TWS'].values[sel]
    AF_t[a] = f - infra_full['mu_c']        # anomaly vs full-train mu (incl. trend continuation = D)
fc_t = infra_full['full']
VA = np.mean([np.nanvar(AF_t[a][fc_t], ddof=1) for a in anchors_t])
pairs = []
for i in range(len(anchors_t)):
    for j in range(i+1, len(anchors_t)):
        g = anchors_t[j] - anchors_t[i]
        x, y = AF_t[anchors_t[i]][fc_t], AF_t[anchors_t[j]][fc_t]
        ok = np.isfinite(x) & np.isfinite(y)
        pairs.append((g, float(np.mean(x[ok]*y[ok]))))
P(f"    anchor anomaly variance V_A (mean over 6 anchors, full cells) = {VA:.4f} (std {np.sqrt(VA):.4f})")
P("    anchor-pair covariances by gap: " + ", ".join(f"g={g}:{c:.4f}" for g, c in sorted(pairs)))
cov_pairs = {}
for g, c in pairs: cov_pairs.setdefault(int(g), []).append(c)
cov_pairs = {g: float(np.mean(v)) for g, v in cov_pairs.items()}
sse_t, phi_t, coef_t = fit_ar(cov_pairs, with_D=True, phis=np.arange(0.30, 0.96, 0.01))
V_D_t, var_f_t = float(coef_t[0]), float(coef_t[1])
var_e_t = max(VA - V_D_t - var_f_t, 0.0)
P(f"    fit cov(g)=V_D+var_f*phi^g: V_D={V_D_t:.4f} (std {np.sqrt(V_D_t):.4f}), var_f={var_f_t:.4f}, phi={phi_t:.2f}")
P(f"    -> var_e_test = V_A - V_D - var_f = {var_e_t:.4f}  => sigma_e_test = {np.sqrt(var_e_t):.4f}")
P(f"    -> sigma_e_train (fit era) = {np.sqrt(var_e_tr):.4f}; ratio of variances test/train = {var_e_t/max(var_e_tr,1e-9):.2f}")
P(f"    (sanity: D-hat std measured elsewhere 0.901 -> V_D 0.81; our V_D={V_D_t:.3f})")

# same 3-param-free estimator on train panel restricted to same gaps? train has V_D=0:
P(f"    train fit at same gaps: var_f={var_f_tr:.4f} phi={phi_tr:.2f} vs test var_f={var_f_t:.4f} phi={phi_t:.2f}")

# direct gap-residual check (phi=0.80), raw and detrended, train vs test
P("\n    gap-residual variance check (r = A_b - phi^g A_a, phi=0.80):")
def rvar(fieldA, fieldB):
    x, y = fieldA[fc_t], fieldB[fc_t]
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.var(x[ok] - phi0*y[ok], ddof=1))
phi0 = 0.80
rows = []
for i in range(len(anchors_t)-1):
    a, b = anchors_t[i], anchors_t[i+1]
    g = b - a
    rt = rvar(AF_t[a], AF_t[b])
    # train-era equivalent at same gap: average over train month pairs
    num, cnt = 0.0, 0
    T = len(ta_tr)
    for ii in range(T):
        jj = ii + g
        if jj >= T: break
        if ta_tr[jj] - ta_tr[ii] != g: continue
        x, y = A_tr[ii], A_tr[jj]
        ok = np.isfinite(x) & np.isfinite(y)
        num += np.sum((y[ok] - phi0*x[ok])**2); cnt += ok.sum()
    rtr = num/max(cnt, 1)
    rows.append((g, rt, rtr))
    P(f"      gap {g:2d}: test-anchor var(r)={rt:.4f}  train var(r)={rtr:.4f}  excess={rt-rtr:+.4f}")

# ================= (c) D-tilde error =================
P("\n(c) D-tilde error")
# ---- (c1) mirrored M1 ----
val = train[(train['year'] >= 2013) & (train['year'] <= 2015)].copy()
val_msk = (val['t_abs'] % 12 + 1).map(cal).values > 0.5
val['masked'] = val_msk
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
M_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]
M_RUNS = {24157, 24161, 24162, 24167, 24168, 24173, 24176, 24177, 24178, 24179, 24180, 24187}
m_months = set(M_ANCHORS) | set(M_RUNS)
mir = val[val['t_abs'].isin(m_months)].copy()
mir['masked'] = mir['t_abs'].isin(M_RUNS)
mir['TWS_t'] = val_tws_visible.loc[mir.index].values
mir.loc[mir['masked'], 'TWS_t'] = np.nan
pred_m1, dg = run_masked2(infra_fit, mir, mir['TWS_t'], 0.80, use_denoise=False, use_bwd=True,
                          lam=0.84, dtil_w=(0.70, 0.45, 0.073), diag=True)
mm1 = mir['masked'].values & np.isfinite(pred_m1) & np.isfinite(mir['target'].values)
tgt = mir['target'].values.astype(np.float64)
mse_m1 = float(np.mean((pred_m1[mm1] - tgt[mm1])**2))
P(f"    M1 bwd RMSE replicated: {np.sqrt(mse_m1):.4f} (auditA: 0.6319)  n={int(mm1.sum()):,}")
P(f"    M1 calibration: H={dg['H']:.4f} R={dg['R']:.4f} var_f(calib)={dg['var_f']:.4f} varz={dg['varz']:.4f}")
dv = []
for a in dg['anchors']:
    r = dg['AF'][a] - dg['Dtil_fn'](a)
    v = float(np.nanvar(r[fc_t], ddof=1))
    dv.append(v)
    P(f"      anchor {a//12}-{a%12+1:02d}: var(AF-Dtil)={v:.4f} (RMSE {np.sqrt(v):.4f})")
var_AFmDtil = float(np.mean(dv))
D_err_m1 = max(var_AFmDtil - var_f_t - var_e_t, 0.0)   # subtract TEST-era fast+noise (val era ~ train regime; test-era used for LB floor)
D_err_m1_tr = max(var_AFmDtil - var_f_tr - var_e_tr, 0.0)
P(f"    mean var(AF-Dtil)={var_AFmDtil:.4f}; minus test fast+noise ({var_f_t+var_e_t:.4f}) -> D_err(M1)={np.sqrt(D_err_m1):.4f}")
P(f"    (using train-era fast+noise {var_f_tr+var_e_tr:.4f} instead -> D_err(M1)={np.sqrt(D_err_m1_tr):.4f})")

# ---- (c2) TEST anchors: leave-one-anchor-out D-tilde ----
# S (static cov field) from test covs via infra_full coef; trendex from infra_full
Zt = te['covs'].astype(np.float32)
okz = np.isfinite(Zt).all(axis=1)
cov_est = np.full(len(te), np.nan, dtype=np.float32)
cov_est[okz] = np.column_stack([Zt[okz], np.ones(okz.sum())]) @ infra_full['coef']
cov_field_t = {}
for m in np.unique(tta):
    selm = (tta == m) & okz
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[tcc[selm]] = cov_est[selm]
    cov_field_t[int(m)] = fm - infra_full['mu_c']
S_t = np.nanmean(np.array([cov_field_t[int(m)] for m in np.unique(tta)]), axis=0)
def trendex_full(t): return ((np.float64(t) - infra_full['tbar_c']) * infra_full['beta_c'])
dv_in, dv_loo = [], []
for a in anchors_t:
    others = [b for b in anchors_t if b != a]
    Dhat_loo = np.nanmean(np.array([AF_t[b] for b in others]), axis=0)
    Dtil_loo = 0.70*Dhat_loo + 0.45*S_t + 0.073*trendex_full(a)
    Dhat_in = np.nanmean(np.array([AF_t[b] for b in anchors_t]), axis=0)
    Dtil_in = 0.70*Dhat_in + 0.45*S_t + 0.073*trendex_full(a)
    v1 = float(np.nanvar((AF_t[a] - Dtil_in)[fc_t], ddof=1))
    v2 = float(np.nanvar((AF_t[a] - Dtil_loo)[fc_t], ddof=1))
    dv_in.append(v1); dv_loo.append(v2)
    P(f"      TEST anchor {a//12}-{a%12+1:02d}: var(AF-Dtil_all6)={v1:.4f}  var(AF-Dtil_LOO)={v2:.4f}")
var_loo = float(np.mean(dv_loo)); var_in = float(np.mean(dv_in))
D_err_loo = max(var_loo - var_f_t - var_e_t, 0.0)
P(f"    mean LOO var = {var_loo:.4f} (in-sample {var_in:.4f}) -> D_err(TEST, LOO) = {np.sqrt(D_err_loo):.4f}")

# ---- floor decomposition ----
P("\n=== FLOOR DECOMPOSITION (masked rows) ===")
fast_err_m1 = max(mse_m1 - D_err_m1 - var_e_t, 0.0)
P(f"    M1: total {np.sqrt(mse_m1):.4f} = noise {np.sqrt(var_e_t):.4f} + D-err {np.sqrt(D_err_m1):.4f} + fast-err {np.sqrt(fast_err_m1):.4f}")
P(f"        (squared: {mse_m1:.4f} = {var_e_t:.4f} + {D_err_m1:.4f} + {fast_err_m1:.4f}; claimed 0.722=sqrt(.456^2+.42^2+.37^2))")
floor_test = np.sqrt(var_e_t + D_err_loo + fast_err_m1)
P(f"    TEST-floor (test noise + TEST LOO D-err + M1 fast-err): {floor_test:.4f}")
# CV->LB scaling: v17b implied masked LB 0.7355 vs M1 0.6319 -> scale
scale = 0.7355/np.sqrt(mse_m1)
P(f"    with CV->LB masked scaling x{scale:.4f}: floor_LB ~ {floor_test*scale:.4f} vs implied masked 0.7355")

# ---- leader implications ----
P("\n=== LEADER (0.5596) implications ===")
for k0 in [0.6254, 0.62, 0.60, 0.58, 0.55, 0.52]:
    m2 = (0.5596**2 - 0.3348*k0**2)/0.6652
    if m2 > 0:
        P(f"    if k0={k0:.4f}: implied masked RMSE = {np.sqrt(m2):.4f}  (vs our floor {floor_test:.4f} / implied 0.7355)")
P(f"    noise share check: to reach masked 0.52 with D-err {np.sqrt(D_err_loo):.3f} and fast-err {np.sqrt(fast_err_m1):.3f}, noise would need std {np.sqrt(max(0.52**2 - D_err_loo - fast_err_m1, 0)):.3f}")

# ---- B1(d): H/R calibration val (14 anchors) vs test (6 anchors) ----
P("\n=== B1(d): H/R/var_f calibration, val-P0 vs TEST phase ===")
pred_p0, dg0 = run_masked2(infra_fit, val, val_tws_visible, 0.80, use_denoise=False, use_bwd=True,
                           lam=0.84, dtil_w=(0.70, 0.45, 0.073), diag=True)
P(f"    VAL  (P0, {len(dg0['anchors'])} anchors): H={dg0['H']:.4f} R={dg0['R']:.4f} var_f={dg0['var_f']:.4f}")
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val['t_abs'].values])
honest = val['masked'].values & ~shortcut
vT = val['target'].values.astype(np.float64)
hm = honest & np.isfinite(pred_p0) & np.isfinite(vT)
P(f"    P0 honest replication: {float(np.sqrt(np.mean((pred_p0[hm]-vT[hm])**2))):.4f} (v17b: 0.6535; n={int(hm.sum()):,})")
# test-phase calibration: build test frame and run
test_df = pd.DataFrame({'cc': tcc, 't_abs': tta, 'masked': tmsk, 'TWS_t': te['TWS'].astype('float32')})
for j, c in enumerate(COVS): test_df[c] = te['covs'][:, j].astype('float32')
test_vis = test_df['TWS_t'].copy()
test_df.loc[test_df['masked'], 'TWS_t'] = np.nan
_, dgt = run_masked2(infra_full, test_df, test_vis, 0.80, use_denoise=False, use_bwd=True,
                     lam=0.84, dtil_w=(0.70, 0.45, 0.073), diag=True)
P(f"    TEST ({len(dgt['anchors'])} anchors): H={dgt['H']:.4f} R={dgt['R']:.4f} var_f={dgt['var_f']:.4f}")

with open(OUT, 'w') as f: f.write('\n'.join(lines))
log(f"saved {OUT}")
