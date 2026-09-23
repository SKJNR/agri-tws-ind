"""
cv_lb_correlation.py — validate whether the 2013-2015 honest window rank-correlates
with the LB, using 5 reproducible variant configs we already submitted.

Configs evaluated (all reuse the v2 trendex infrastructure where applicable):
  v1a_decay       : pure empirical RH[h] decay, no covs, no D-tilde         (LB 0.8337)
  v1b_kalman_glb  : Kalman phi=0.80, D-tilde = 0.676 D-hat + 0.490 S        (LB 0.7152)
  v1c_kalman_era  : Kalman phi=0.85, era-interpolated D-tilde              (LB 0.7168)
  v2b_trendex     : Kalman phi=0.74, D-tilde = w1 D-hat + w2 S + w3 trendex (LB 0.7137)
  v3a_simple      : single-comp Kalman phi=0.97, lam=0.84, no D-tilde      (LB 0.7984)

Honest val protocol:
  - fit data      : train rows with year <= 2012
  - val data      : train rows with year in [2013, 2015]
  - val masking   : same calendar-month mask fraction as actual test (>0.5 = masked)
  - val target    : train['target'] = TWS_t at t+1 (the ground truth we're predicting)
  - infrastructure (mu_c, clim, cov regression, D-hat, S, W, trendex beta_c)
                    is fit on fit data ONLY — no leakage from val

Decision rule:
  rho >= 0.71 AND no top-3 rank inversion -> CV trustworthy
  0.40 <= rho < 0.71 OR top-3 inversion  -> partially trustworthy (use CV for big moves only)
  rho < 0.40                            -> CV broken; fix before next submission
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
RH = {1:0.874, 2:0.781, 3:0.712, 4:0.660, 5:0.623, 6:0.595, 7:0.574}
LAM_F = 0.84

LB = {
    'v1a_decay'      : 0.8337,
    'v1b_kalman_glb' : 0.7152,
    'v1c_kalman_era' : 0.7168,
    'v2b_trendex'    : 0.7137,
    'v3a_simple'     : 0.7984,
}

# ---------------- load train ----------------
print("Loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms_all = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms_all)}

# split fit / val
fit = train[train['time'].dt.year <= 2012].copy()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
print(f"fit: {len(fit):,} rows  ({int(fit['ym'].min())}..{int(fit['ym'].max())})", flush=True)
print(f"val: {len(val):,} rows  ({int(val['ym'].min())}..{int(val['ym'].max())})", flush=True)

# ---------------- load test (to extract masking pattern) ----------------
print("Loading test (for masking pattern only)...", flush=True)
test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
test_raw['masked'] = test_raw['TWS_t_masked'].astype(bool)
test_raw['cal_mon'] = test_raw['time'].dt.month
cal_mask_frac = test_raw.groupby('cal_mon')['masked'].mean()
print("Calendar-month mask fractions (from test):")
for m, f in cal_mask_frac.items():
    print(f"  month {m:2d}: {f:.3f}")

# ---------------- apply masking pattern to val ----------------
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
n_masked = int(val['masked'].sum())
n_unmasked = int((~val['masked']).sum())
print(f"val masked rows: {n_masked:,} ({n_masked/len(val)*100:.1f}%)  unmasked: {n_unmasked:,}")

# For masked rows, hide TWS_t (set to NaN) — Kalman must operate without it
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan

# We still need targets for ALL val rows (target = TWS_t at t+1, already in train)
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values
val_ta = val['t_abs'].values
val_msk = val['masked'].values

# ---------------- fit infrastructure on fit data only ----------------
print("\nFitting infrastructure on fit data (2002-2012)...", flush=True)
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

# global cov regression
Z = fit[COVS].values.astype('float32')
yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# per-cell OLS slope for trend extrapolation
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxy = np.nansum(td * F64, axis=0)
sxx = np.nansum(td * td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), sxy / np.where(sxx > 0, sxx, 1), 0.0).astype(np.float32)
print(f"beta_c: std={beta_c.std():.5f}/mo, mean={beta_c.mean():+.6f}, zeroed cells={int((beta_c==0).sum())}")

def trendex(t):
    return (np.float64(t) - tbar_c) * beta_c

# ---------------- build val "cov_field" + S + W from val covariates ----------------
print("Building cov_field / S / W on val...", flush=True)
Zv = val[COVS].values.astype('float32')
okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef

cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m_v}

# ---------------- val anchors (months with low mask fraction) ----------------
# Use threshold < 0.5 to match v3a's setup; v1/v2 use < 0.01 — but on val we have
# calendar-month-based masking so it'll be either ~0 or ~1 per month
mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v < 0.5].index)
print(f"val anchors (t_abs): {anchors_v}  ({len(anchors_v)} months)")

# ---------------- val anchor fields AF ----------------
AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]   # use the unmasked TWS_t
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)

# ---------------- D-tilde modes ----------------
# v1b global: 0.676 D-hat + 0.490 S (hardcoded from submission_v1.py)
def D_glb(t): return (0.676 * Dhat + 0.490 * S).astype(np.float32)

# v1c era-interpolated
early = [a for a in anchors_v if a < 2014*12 + 6]
late  = [a for a in anchors_v if a >= 2014*12 + 6]
if not early or not late:
    early = anchors_v[:len(anchors_v)//2]
    late  = anchors_v[len(anchors_v)//2:]
D_early = np.nanmean(np.array([AF[a] for a in early]), axis=0)
D_late  = np.nanmean(np.array([AF[a] for a in late]), axis=0)
c_early, c_late = float(np.mean(early)), float(np.mean(late))
def D_int(t):
    u = np.clip((t - c_early)/(c_late - c_early), 0.0, 1.0)
    return (0.676 * ((1-u)*D_early + u*D_late) + 0.490 * S).astype(np.float32)

# v2b trendex: LOO-fit weights w1, w2, w3
Xs, ys = [], []
for a in anchors_v:
    dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b != a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]]))
    ys.append(AF[a][ok])
Xw = np.vstack(Xs); yw = np.concatenate(ys)
w_ = np.linalg.solve(Xw.T@Xw + np.array([1e-3, 1e-3, 1e-3]), Xw.T@yw)
w1, w2, w3 = float(w_[0]), float(w_[1]), float(w_[2])
print(f"D-tilde (trendex) LOO weights: w1(D-hat)={w1:.3f}, w2(S)={w2:.3f}, w3(trendex)={w3:.3f}")
def D_trendex(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

# v3a simple: no D-tilde, single-component Kalman with cov obs
# (separate code path — handled below)

# ---------------- Kalman calibration (for v1b/v1c/v2b) ----------------
def calibrate(Dfun):
    cs, zs, vfs = [], [], []
    for a in anchors_v:
        dj = Dfun(a)
        ok = np.isfinite(W[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W[a][ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

# ---------------- Kalman prediction (two-component, used by v1b/v1c/v2b) ----------------
def kalman_predict(phi_f, Dfun):
    H, R, var_f = calibrate(Dfun)
    q = LAM_F*var_f*(1-phi_f**2)
    P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(val), np.nan, dtype=np.float64)
    for a in anchors_v:
        dja = Dfun(a)
        fa = AF[a] - dja
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        x0 = x.copy()
        # masked rows AT the anchor month itself (k=0 masked)
        sel0 = np.where((val_ta == a) & val_msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[val_cc[sel0]] + Dfun(a+1)[val_cc[sel0]] + phi_f*x0[val_cc[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x
            P = phi_f**2*P + q
            if m in W:
                wv = W[m]
                okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((val_ta == m) & val_msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in W:
                wv = W[tm]
                okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[val_cc[sel]] + Dfun(tm)[val_cc[sel]] + x2[val_cc[sel]]
    return pred

# ---------------- v1a pure decay prediction ----------------
def decay_predict():
    pred = np.full(len(val), np.nan, dtype=np.float64)
    anchors_arr = np.array(anchors_v)
    row_anchor = anchors_arr[np.searchsorted(anchors_arr, val_ta, side='right')-1]
    for a in anchors_v:
        sel = np.where((row_anchor == a) & val_msk)[0]
        if len(sel) == 0: continue
        h = (val_ta[sel] + 1 - a).astype(int)
        coef_h = np.array([RH.get(int(v), RH[7]) for v in h], dtype=np.float32)
        anom = AF[a][val_cc[sel]]
        pred[sel] = mu_c[val_cc[sel]] + coef_h*np.nan_to_num(anom)
    return pred

# ---------------- v3a single-component Kalman (no D-tilde, with cov obs) ----------------
def kalman_simple_predict(phi, lam):
    # calibration: cov obs noise R from fit data, on the fit period
    # Use same approach as submission_v3.py: R_h = var(TWS - cov_estimate) on val-anchor-month residuals
    var_tot = float(np.var(fit['TWS_t'].values))
    # Build cov estimate per cell on fit
    Z_fit = fit[COVS].values.astype('float32')
    z_fit = Z_fit @ coef[:5] + coef[5]
    # Honest R: fit cov-regression on 2002-2009, eval 2010-2012
    mf = (fit['time'].dt.year <= 2009).values
    Ah = np.column_stack([Z_fit[mf], np.ones(mf.sum())])
    ch = np.linalg.solve(Ah.T @ Ah + 10*np.eye(6), Ah.T @ fit['TWS_t'].values[mf])
    me = ~mf
    Ae = np.column_stack([Z_fit[me], np.ones(me.sum())])
    R_h = float(np.var(fit['TWS_t'].values[me] - Ae @ ch))
    print(f"  [v3a] var_tot={var_tot:.4f} R_h={R_h:.4f} phi={phi} lam={lam}")

    P_init = (1 - lam) * var_tot
    q = lam * var_tot * (1 - phi**2)
    pred = np.full(len(val), np.nan, dtype=np.float64)
    # Build cov estimate for val
    Zv_v = val[COVS].values.astype('float32')
    z_val = Zv_v @ coef[:5] + coef[5]
    # cell-indexed cov estimate per val month
    M_Z = {}
    M_ZOK = {}
    for m in np.sort(val['t_abs'].unique()):
        selm = (val_ta == m)
        fm = np.full(n_cells, np.nan, dtype=np.float32)
        ok = np.isfinite(z_val[selm])
        idx = val_cc[selm][ok]
        fm[idx] = z_val[selm][ok]
        M_Z[int(m)] = fm
        M_ZOK[int(m)] = np.isfinite(fm)
    for a in anchors_v:
        L_anchor = AF[a] + mu_c - mu_c   # = raw TWS at anchor - mu_c ... actually AF[a] = TWS_at_anchor - mu_c
        anom0 = AF[a]   # already an anomaly
        xh = np.where(np.isfinite(anom0), lam * anom0, 0.0).astype(np.float32)
        P = np.where(np.isfinite(anom0), P_init, 1e6).astype(np.float32)
        for k in range(1, 8):
            m = a + k
            if m not in M_Z: continue
            xh = phi * xh
            P = phi**2 * P + q
            z = M_Z[m]
            ok = M_ZOK[m] & (P < 1e5)
            K = np.where(ok, P / (P + R_h), 0.0).astype(np.float32)
            xh = np.where(ok, xh + K*(z - mu_c - xh), xh)
            P = np.where(ok, (1-K)*P, P)
            sel = np.where((val_ta == m) & val_msk)[0]
            if len(sel) == 0: continue
            pred[sel] = mu_c[val_cc[sel]] + phi * xh[val_cc[sel]]
    return pred

# ---------------- run all variants ----------------
print("\n--- Running variants on val ---", flush=True)
val_rmse = {}

print("[v1a_decay]")
p = decay_predict()
mask = val_msk & np.isfinite(p) & np.isfinite(val_target)
val_rmse['v1a_decay'] = float(np.sqrt(np.mean((p[mask]-val_target[mask])**2)))
print(f"  rows={mask.sum()}  RMSE={val_rmse['v1a_decay']:.4f}  LB={LB['v1a_decay']:.4f}")

print("[v1b_kalman_glb]  phi=0.80, global D-tilde")
p = kalman_predict(0.80, D_glb)
mask = val_msk & np.isfinite(p) & np.isfinite(val_target)
val_rmse['v1b_kalman_glb'] = float(np.sqrt(np.mean((p[mask]-val_target[mask])**2)))
print(f"  rows={mask.sum()}  RMSE={val_rmse['v1b_kalman_glb']:.4f}  LB={LB['v1b_kalman_glb']:.4f}")

print("[v1c_kalman_era]  phi=0.85, era-interp D-tilde")
p = kalman_predict(0.85, D_int)
mask = val_msk & np.isfinite(p) & np.isfinite(val_target)
val_rmse['v1c_kalman_era'] = float(np.sqrt(np.mean((p[mask]-val_target[mask])**2)))
print(f"  rows={mask.sum()}  RMSE={val_rmse['v1c_kalman_era']:.4f}  LB={LB['v1c_kalman_era']:.4f}")

print("[v2b_trendex]  phi=0.74, trendex D-tilde")
p = kalman_predict(0.74, D_trendex)
mask = val_msk & np.isfinite(p) & np.isfinite(val_target)
val_rmse['v2b_trendex'] = float(np.sqrt(np.mean((p[mask]-val_target[mask])**2)))
print(f"  rows={mask.sum()}  RMSE={val_rmse['v2b_trendex']:.4f}  LB={LB['v2b_trendex']:.4f}")

print("[v3a_simple]  phi=0.97, lam=0.84, no D-tilde")
p = kalman_simple_predict(0.97, 0.84)
mask = val_msk & np.isfinite(p) & np.isfinite(val_target)
val_rmse['v3a_simple'] = float(np.sqrt(np.mean((p[mask]-val_target[mask])**2)))
print(f"  rows={mask.sum()}  RMSE={val_rmse['v3a_simple']:.4f}  LB={LB['v3a_simple']:.4f}")

# ---------------- compute Spearman rho ----------------
print("\n--- CV vs LB rank correlation ---", flush=True)
tags = list(LB.keys())
cv_arr = np.array([val_rmse[t] for t in tags])
lb_arr = np.array([LB[t] for t in tags])
rho, pval = spearmanr(cv_arr, lb_arr)
print(f"\nSpearman rho = {rho:.4f}  (p-value = {pval:.4f})")
print(f"  n = {len(tags)}  (n=5 → critical rho at p=0.05 is ~0.90)")

# rank table
order = np.argsort(lb_arr)
print("\nRanked by LB (best=lowest first):")
print(f"  {'rank':<6}{'variant':<22}{'CV-RMSE':<10}{'LB':<10}{'CV rank':<10}")
cv_ranks = np.argsort(np.argsort(cv_arr)) + 1
lb_ranks = np.argsort(np.argsort(lb_arr)) + 1
for r, i in enumerate(order):
    print(f"  {r+1:<6}{tags[i]:<22}{cv_arr[i]:<10.4f}{lb_arr[i]:<10.4f}{cv_ranks[i]:<10}")

# top-3 inversion check
top3_lb = set(np.array(tags)[order[:3]])
top3_cv = set(np.array(tags)[np.argsort(cv_arr)[:3]])
overlap = len(top3_lb & top3_cv)
print(f"\nTop-3 LB set:   {sorted(top3_lb)}")
print(f"Top-3 CV set:   {sorted(top3_cv)}")
print(f"Top-3 overlap:  {overlap}/3")

# decision rule
if rho >= 0.71 and overlap >= 2:
    verdict = "TRUSTWORTHY — iterate on CV only, ship v4a/b/c tomorrow as planned"
elif rho >= 0.40:
    verdict = "PARTIALLY TRUSTWORTHY — use CV for big moves (>2% RMSE gap) only; reserve 1 LB slot per day for confirmation"
else:
    verdict = "BROKEN — DO NOT submit v4a/b/c tomorrow; fix the CV window first (likely val-period noise ratio doesn't match test)"
print(f"\nVerdict: {verdict}")

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.scatter(cv_arr, lb_arr, s=80, c='steelblue', edgecolors='k', zorder=3)
for i, t in enumerate(tags):
    ax.annotate(t, (cv_arr[i], lb_arr[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
# diagonal y=x reference
all_v = np.concatenate([cv_arr, lb_arr])
lo, hi = all_v.min()*0.95, all_v.max()*1.05
ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5, label='y=x (perfect CV↔LB)')
# linear fit
if len(tags) >= 2:
    b1, b0 = np.polyfit(cv_arr, lb_arr, 1)
    xs = np.array([cv_arr.min(), cv_arr.max()])
    ax.plot(xs, b1*xs + b0, 'r-', lw=1.2, alpha=0.7, label=f'linear fit (slope={b1:.2f})')
ax.set_xlabel('Local CV RMSE (2013-2015 honest window)')
ax.set_ylabel('LB RMSE (test period)')
ax.set_title(f'CV vs LB rank correlation: Spearman ρ = {rho:.3f}  (n={len(tags)})\nVerdict: {verdict[:60]}...' if len(verdict) > 60 else f'CV vs LB rank correlation: Spearman ρ = {rho:.3f}  (n={len(tags)})\nVerdict: {verdict}')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
fig.savefig(f'{DL}/cv_lb_correlation.png', dpi=130)
print(f"\nPlot saved: {DL}/cv_lb_correlation.png")

# save table
out = pd.DataFrame({
    'variant': tags,
    'cv_rmse_2013_15': cv_arr,
    'lb_rmse_test': lb_arr,
    'cv_rank': cv_ranks,
    'lb_rank': lb_ranks,
})
out['cv_lb_diff'] = out['cv_rmse_2013_15'] - out['lb_rmse_test']
out.to_csv(f'{DL}/cv_lb_correlation.csv', index=False)
print(f"Table saved: {DL}/cv_lb_correlation.csv")
print("\n--- final summary ---")
print(out.to_string(index=False))
print(f"\nSpearman rho = {rho:.4f}, p-value = {pval:.4f}")
print(f"Verdict: {verdict}")
