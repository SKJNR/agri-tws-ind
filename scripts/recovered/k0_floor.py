"""K0 FLOOR MEASUREMENT: how much headroom does the pooled k0 model leave?

The current k0 model (linear + LGBM on [TWS_t anomaly, cov anomalies t, cov anomalies t+1])
is POOLED over all 15,715 cells. Per-cell AR(1) structure (phi median 0.823 on train)
suggests a per-cell model could be far better.

Tests on 2013-15 (fit <= 2012), all rows (no mask mimic):
  T1: persistence (pred = TWS_t)
  T2: global linear y ~ [x_t]                      (pooled AR)
  T3: per-cell AR(1): y = a_c + b_c x_t            (per-cell intercept+slope)
  T4: per-cell AR(1) + per-cell time weight? (skip)
  T5: T3 + covs(t+1) per-cell ridge correction
  T6: current k0-style global model with covs (reference)
Also: residual of T3 vs covs(t+1) correlation — do covs add anything beyond per-cell AR?
"""
import numpy as np, pandas as pd, time

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
n_cells = int(train['cc'].max())+1
log(f"train {len(train):,} rows, {n_cells} cells")

fit = train[train['time'].dt.year <= 2012]
ev = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]

# infra: mu_c, clim from fit
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float64')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).fillna(0).values.astype('float64')

# ---- T1 persistence ----
e = ev['target'].values - ev['TWS_t'].values
log(f"T1 persistence RMSE: {np.sqrt(np.nanmean(e**2)):.4f}")

# ---- per-cell AR(1): fit a_c + b_c * TWS_t on fit ----
# build per-cell time series matrices from fit
def percell_ar(fit_df):
    # returns a_c, b_c arrays
    sums = fit_df.groupby('cc').agg(n=('TWS_t','size'))
    # use covariance method per cell on consecutive months only (month gap check)
    fit_df = fit_df.sort_values(['cc','ym'])
    fit_df['x_prev'] = fit_df.groupby('cc')['TWS_t'].shift(1)
    fit_df['y_next'] = fit_df.groupby('cc')['target'].shift(-1)  # not needed
    # AR from (x_prev -> TWS_t) pairs where month gap == 1
    ym = fit_df['ym'].values; cc_ = fit_df['cc'].values
    xp = fit_df['x_prev'].values; xt = fit_df['TWS_t'].values
    def next_ym_ok(a, b):
        ya, ma = divmod(a, 100); yb, mb = divmod(b, 100)
        return (ya*12+ma) + 1 == (yb*12+mb)
    ok = np.isfinite(xp) & np.isfinite(xt)
    gap_ok = np.zeros(len(fit_df), dtype=bool)
    gap_ok[:-1] = [next_ym_ok(a, b) for a, b in zip(ym[:-1], ym[1:])]
    same_cell = np.zeros(len(fit_df), dtype=bool)
    same_cell[:-1] = (cc_[1:] == cc_[:-1])
    ok &= gap_ok & same_cell
    df = pd.DataFrame({'cc': cc_[ok], 'x': xp[ok], 'y': xt[ok]})
    g = df.groupby('cc')
    n = g.size(); sx = g['x'].sum(); sy = g['y'].sum(); sxx = (df['x']**2).groupby(df['cc']).sum(); sxy = (df['x']*df['y']).groupby(df['cc']).sum()
    n = n.reindex(range(n_cells), fill_value=0); sx = sx.reindex(range(n_cells), fill_value=0); sy = sy.reindex(range(n_cells), fill_value=0)
    sxx = sxx.reindex(range(n_cells), fill_value=0); sxy = sxy.reindex(range(n_cells), fill_value=0)
    mx = sx/np.maximum(n,1); my = sy/np.maximum(n,1)
    cov = sxy/np.maximum(n-1,1) - n/np.maximum(n-1,1)*mx*my
    var = sxx/np.maximum(n-1,1) - n/np.maximum(n-1,1)*mx*mx
    b = np.where((n >= 24) & (var > 1e-6), cov/np.maximum(var,1e-6), 0.8)
    a = np.where(n >= 24, my - b*mx, 0.0)
    return a, b, n.values

a_c, b_c, n_c = percell_ar(fit)
log(f"per-cell AR: b median={np.median(b_c[n_c>=24]):.3f}  p10={np.percentile(b_c[n_c>=24],10):.3f} p90={np.percentile(b_c[n_c>=24],90):.3f}")

# eval pairs: same-cell consecutive-month (x_t -> target)
ev_s = ev.sort_values(['cc','ym'])
x = ev_s['TWS_t'].values; y = ev_s['target'].values; cc_ = ev_s['cc'].values; ym = ev_s['ym'].values
def next_ym_ok(a, b):
    ya, ma = divmod(a, 100); yb, mb = divmod(b, 100)
    return (ya*12+ma) + 1 == (yb*12+mb)
ok = np.array([False] + [next_ym_ok(a,b) and (ca==cb) for a,b,ca,cb in zip(ym[:-1],ym[1:],cc_[:-1],cc_[1:])])
ok &= np.isfinite(x) & np.isfinite(y)
xv, yv, cv = x[ok], y[ok], cc_[ok]
log(f"eval consecutive pairs: {ok.sum():,}")

# T3 per-cell AR
p3 = a_c[cv] + b_c[cv]*xv
r3 = np.sqrt(np.mean((p3-yv)**2))
log(f"T3 per-cell AR(1) RMSE: {r3:.4f}")

# T2 pooled AR (global slope)
A = np.column_stack([xv, np.ones(len(xv))])
c2 = np.linalg.solve(A.T@A, A.T@yv)
log(f"T2 pooled AR RMSE:      {np.sqrt(np.mean((A@c2-yv)**2)):.4f}   slope={c2[0]:.3f}")

# residual of T3 vs covs(t+1): is there additional cov information?
ev_s2 = ev_s.iloc[1:][ok[1:]] if False else None
idx_ok = np.where(ok)[0]
ev_rows = ev_s.iloc[idx_ok]
res = yv - p3
covv = ev_rows[COVS].values
for j, c in enumerate(COVS):
    r = np.corrcoef(res, covv[:,j] - clim[cv, j])[0,1]
    print(f"  corr(AR residual, {c}(t) anomaly) = {r:+.4f}")
# next month covs
nxt_map = {}
ev3 = ev.copy()
for ym_ in ev3['ym'].unique():
    pass
# build next-month cov lookup
ev3 = ev3.sort_values(['cc','ym'])
for c in COVS:
    ev3[c+'_nxt'] = ev3.groupby('cc')[c].shift(-1)
sub = ev3.iloc[idx_ok]
for c in COVS:
    r = np.corrcoef(res, sub[c+'_nxt'].values - clim[cv, j] if False else sub[c+'_nxt'].values - clim[cv, COVS.index(c)])[0,1]
    print(f"  corr(AR residual, {c}(t+1) anomaly) = {r:+.4f}")

# T5: per-cell AR + global cov correction on residual
has_nxt = np.isfinite(sub[[c+'_nxt' for c in COVS]].values).all(axis=1)
Xc = np.column_stack([sub[c+'_nxt'].values - clim[cv, COVS.index(c)] for c in COVS] + [np.ones(len(sub))])
m5 = has_nxt & np.isfinite(Xc).all(axis=1)
Xm = Xc[m5]; rm = res[m5]
cr = np.linalg.solve(Xm.T@Xm + 1e-2*np.eye(6), Xm.T@rm)
p5 = p3[m5] + Xm@cr
log(f"T5 per-cell AR + covs(t+1) correction: {np.sqrt(np.mean((p5-yv[m5])**2)):.4f}  (rows {m5.sum():,})")

# T6: current k0-style: global linear on [x, covs t, covs t+1]
Xg = np.column_stack([xv - mu_c[cv]] + [sub[c].values - clim[cv, j] for j, c in enumerate(COVS)] + [sub[c+'_nxt'].values - clim[cv, j] for j, c in enumerate(COVS)] + [np.ones(len(sub))])
mg = np.isfinite(Xg).all(axis=1)
cg = np.linalg.solve(Xg[mg].T@Xg[mg] + 1e-3*np.eye(Xg.shape[1]), Xg[mg].T@yv[mg])
log(f"T6 global k0-style linear (all feats): {np.sqrt(np.mean((Xg[mg]@cg - yv[mg])**2)):.4f}")

# T7: per-cell AR + covs + per-cell AR residual shrinkage?
# quick: per-cell AR with b shrunk toward 0.82 median
for shrink in [0.0, 0.25, 0.5]:
    b_s = (1-shrink)*b_c + shrink*np.median(b_c[n_c>=24])
    p7 = a_c[cv] + b_s[cv]*xv
    print(f"  T7 b shrink {shrink:.2f}: {np.sqrt(np.mean((p7-yv)**2)):.4f}")
