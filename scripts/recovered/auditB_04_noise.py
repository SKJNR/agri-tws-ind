"""
AUDIT B - 04: (a) fixed MoM noise estimator + ACF profile, (b) seasonality check,
(c) train-vs-test anchor-gap correlations & difference variances (test noise evidence).
"""
import numpy as np, pandas as pd, time, sys
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import *
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

tr, te = load_cached()
train = build_train_df(tr)
n_cells = int(train['cc'].max())+1
train['year'] = train['t_abs']//12
fit = train[train['year'] <= 2012]
infra_cv = build_infra(fit, n_cells, 'cv')

# ---------- build fit-era field matrix ----------
yms = np.sort(fit['t_abs'].unique()); T = len(yms)
ym_to_i = {int(v): i for i, v in enumerate(yms)}
ta_arr = np.array([int(v) for v in yms], dtype=np.float64)
F = np.full((T, n_cells), np.nan, dtype=np.float64)
F[fit['t_abs'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
mu = infra_cv['mu_c'].astype(np.float64)
A_dt = F - mu[None,:] - (ta_arr[:,None]-infra_cv['tbar_c'][None,:])*infra_cv['beta_c'][None,:].astype(np.float64)

# ---------- (a) per-cell ACF profile of detrended anomaly ----------
log("=== (a) per-cell ACF of detrended anomaly (consecutive-lag pairs only) ===")
maxlag = 6
ac = np.zeros((maxlag, n_cells)); an = np.zeros((maxlag, n_cells))
ok = np.isfinite(A_dt)
for k in range(1, maxlag+1):
    for i in range(T-k):
        if ta_arr[i+k] != ta_arr[i]+k: continue
        m = ok[i] & ok[i+k]
        ac[k-1] += np.where(m, A_dt[i]*A_dt[i+k], 0)
        an[k-1] += m
varc = (np.nansum(A_dt**2, axis=0))/np.maximum(ok.sum(axis=0), 1)
rk = ac/np.maximum(an, 1)/np.maximum(varc, 1e-9)
nobs = ok.sum(axis=0)
good = nobs >= 80
rk_med = np.nanmedian(rk[:, good], axis=1)
log("  median r_k (k=1..6): " + " ".join(f"{v:.3f}" for v in rk_med))
# AR(1)+noise => log(r_k) linear in k for k>=1
ks = np.arange(1, maxlag+1)
slope, intercept = np.polyfit(ks, np.log(np.maximum(rk_med, 1e-6)), 1)
phi_est = np.exp(slope); s_est = np.exp(intercept)
log(f"  log-linear fit: phi={phi_est:.3f}  s(fast share)={s_est:.3f} => noise share={1-s_est:.3f}")
log(f"  median per-cell Var(anom)={np.median(varc[good]):.4f} (std {np.sqrt(np.median(varc[good])):.3f})")
log(f"  => MoM noise std = {np.sqrt((1-s_est)*np.median(varc[good])):.4f}   (team claim 0.456)")
# also fit on lag 1..6 but allowing r0 = 1 to include noise: r_k = s*phi^k, k>=1
# per-cell noise share distribution
s_cells = np.exp(np.log(np.maximum(rk[0, good], 1e-6)) - np.log(np.maximum(rk[1, good], 1e-6))*0)  # placeholder
# use ratio-free: per-cell phi=r2/r1, s=r1/phi
phi_c = rk[1, good]/np.maximum(rk[0, good], 1e-9)
s_c = rk[0, good]/np.maximum(phi_c, 1e-9)
log(f"  per-cell: median phi={np.median(phi_c):.3f}, median s={np.median(s_c):.3f} (s>1 frac={np.mean(s_c>1)*100:.0f}%)")

# ---------- (b) seasonality check ----------
log("\n=== (b) seasonality: per-cell calendar-month means vs annual mean ===")
mon = (fit['t_abs']%12+1).values
anom = fit['TWS_t'].values - mu[fit['cc'].values]
dfm = pd.DataFrame({'cc': fit['cc'].values, 'mon': mon, 'a': anom})
mm = dfm.groupby(['cc','mon'])['a'].mean().unstack()
seas_amp = mm.std(axis=1)   # std across calendar months of monthly-mean anomaly
log(f"  per-cell seasonal amplitude (std of monthly means): median={seas_amp.median():.3f} p90={seas_amp.quantile(0.9):.3f}")
log(f"  median per-cell anomaly std (total) = {np.sqrt(np.median(varc[good])):.3f}")
log(f"  variance share of seasonality (approx, median cells): {seas_amp.median()**2/np.median(varc[good])*100:.1f}%")
# is the seasonal pattern persistent? split fit era in halves
h1 = fit[fit['t_abs'] <= np.median(ta_arr)]
h2 = fit[fit['t_abs'] > np.median(ta_arr)]
def month_means(d):
    m_ = (d['t_abs']%12+1).values
    a_ = d['TWS_t'].values - mu[d['cc'].values]
    dd = pd.DataFrame({'cc': d['cc'].values, 'mon': m_, 'a': a_})
    return dd.groupby(['cc','mon'])['a'].mean().unstack()
m1, m2 = month_means(h1), month_means(h2)
common = m1.index.intersection(m2.index)
c1 = m1.loc[common].values.flatten(); c2 = m2.loc[common].values.flatten()
okc = np.isfinite(c1)&np.isfinite(c2)
log(f"  persistence of monthly-mean anomaly across era halves: corr={np.corrcoef(c1[okc], c2[okc])[0,1]:.3f}")
log(f"  (if seasonality real & stable, corr should be high; monthly means avg ~%d obs each)" % (len(h1)//12//15715*1))

# ---------- (c) train pseudo-anchor r(g), Var(delta) vs test ----------
log("\n=== (c) anchor-gap correlations: train pseudo-anchors vs test ===")
present = set(int(v) for v in np.unique(tr['t_abs']))
# test anchors
anchors_t = [24188, 24192, 24197, 24203, 24222, 24226]
mu_full = None
infra_full = build_infra(train, n_cells, 'full')
mu_full = infra_full['mu_c'].astype(np.float64)
def field_at(m_abs, mu_ref, detrend_with=None):
    sel = tr['t_abs'] == m_abs
    f = np.full(n_cells, np.nan)
    f[tr['cc'][sel]] = tr['TWS'][sel]
    out = f - mu_ref
    if detrend_with is not None:
        out = out - ((m_abs-detrend_with['tbar_c'])*detrend_with['beta_c']).astype(np.float64)
    return out
AFt = {}
for a in anchors_t:
    selm = (te['t_abs'] == a) & (~te['masked'])
    f = np.full(n_cells, np.nan)
    f[te['cc'][selm]] = te['TWS'][selm]
    AFt[a] = f - mu_full   # NOT detrended (D includes trend continuation)

# train pseudo anchors: use months 2003..2014, mimic gaps [4,5,6] sets + long gaps
rng = np.random.default_rng(0)
train_pairs = {4: [], 5: [], 6: []}
train_dvar = {4: [], 5: [], 6: []}
starts = list(range(24040, 24160-40))
for st in starts:
    for g in (4, 5, 6):
        a, b = st, st+g
        if a not in present or b not in present: continue
        fa = field_at(a, mu_full, detrend_with=infra_full)
        fb = field_at(b, mu_full, detrend_with=infra_full)
        m2 = np.isfinite(fa)&np.isfinite(fb)
        if m2.sum() < 10000: continue
        train_pairs[g].append(np.corrcoef(fa[m2], fb[m2])[0,1])
        train_dvar[g].append(np.var(fb[m2]-fa[m2]))
for g in (4, 5, 6):
    log(f"  TRAIN detrended r(g={g}): mean={np.mean(train_pairs[g]):.3f} (n={len(train_pairs[g])})  Var(delta) mean={np.mean(train_dvar[g]):.3f}")
# test adjacent pairs
tp = {4: (24188, 24192), 5: (24192, 24197), 6: (24197, 24203), 4+24222: (24222, 24226)}
log("  TEST raw (D included) r(g): " + ", ".join(
    f"g={b-a}:{np.corrcoef(AFt[a][np.isfinite(AFt[a])&np.isfinite(AFt[b])], AFt[b][np.isfinite(AFt[a])&np.isfinite(AFt[b])])[0,1]:.3f}"
    for a, b in [(24188, 24192), (24192, 24197), (24197, 24203), (24222, 24226)]))
for a, b in [(24188, 24192), (24192, 24197), (24197, 24203), (24222, 24226)]:
    m2 = np.isfinite(AFt[a])&np.isfinite(AFt[b])
    log(f"  TEST Var(delta g={b-a}) = {np.var(AFt[b][m2]-AFt[a][m2]):.3f}")

# train RAW (not detrended) same-gap pairs for apples-to-apples with test (D-ish trend included)
train_pairs_raw = {4: [], 5: [], 6: []}
train_dvar_raw = {4: [], 5: [], 6: []}
for st in starts:
    for g in (4, 5, 6):
        a, b = st, st+g
        if a not in present or b not in present: continue
        fa = field_at(a, mu_full); fb = field_at(b, mu_full)
        m2 = np.isfinite(fa)&np.isfinite(fb)
        if m2.sum() < 10000: continue
        train_pairs_raw[g].append(np.corrcoef(fa[m2], fb[m2])[0,1])
        train_dvar_raw[g].append(np.var(fb[m2]-fa[m2]))
for g in (4, 5, 6):
    log(f"  TRAIN RAW r(g={g}): mean={np.mean(train_pairs_raw[g]):.3f}  Var(delta) mean={np.mean(train_dvar_raw[g]):.3f}")

# ---------- noise back-out from test r(g=4) under train-calibrated fast ----------
log("\n=== back out test noise from r(4) [train-calibrated fast params] ===")
phi = phi_est; s = s_est
varA_tr = np.median(varc[good])
varx_tr = s*varA_tr; varn_tr = (1-s)*varA_tr
log(f"  train: phi={phi:.3f} VarX={varx_tr:.3f} VarN={varn_tr:.3f} (noise std {np.sqrt(varn_tr):.3f})")
# test anchor var (raw, incl D)
varA_te = np.mean([np.nanvar(AFt[a]) for a in anchors_t])
r4_te = np.mean([np.corrcoef(AFt[24188][np.isfinite(AFt[24188])&np.isfinite(AFt[24192])], AFt[24192][np.isfinite(AFt[24188])&np.isfinite(AFt[24192])])[0,1],
                    np.corrcoef(AFt[24222][np.isfinite(AFt[24222])&np.isfinite(AFt[24226])], AFt[24226][np.isfinite(AFt[24222])&np.isfinite(AFt[24226])])[0,1]])
r4_tr = np.mean(train_pairs_raw[4])
log(f"  r(4): train(raw)={r4_tr:.3f}  test={r4_te:.3f}")
# model: r(g) = (VD + VX*phi^g)/(VD+VX+VN); D static over g=4
# train raw: D-part = residual slow wander; assume small: r_tr(g) ~ VX*phi^4/(VX+VN)
pred_r4_tr = varx_tr*phi**4/(varx_tr+varn_tr)
log(f"  predicted train r(4) from AR+noise model = {pred_r4_tr:.3f} (observed {r4_tr:.3f})")
log("DONE")
