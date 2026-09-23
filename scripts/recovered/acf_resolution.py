"""
Resolve the PC1-trend vs ACF-decay contradiction with brute-force calendar-correct ACF.

If PC1 (26.8% var) is a trend, pooled ACF must have a floor. Earlier index-based
ACF said r(24)=0.078, r(36)=0.004. Test on identical data:
  A) brute-force calendar-lag ACF via merge (gold standard, gaps handled)
  B) ACF of PC1-reconstructed field (same brute force)
  C) ACF after per-cell linear detrend
  D) where is the trend? per-cell slope distribution, var decomposition
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/acf_resolution.txt'
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['TWS_t'] = pd.to_numeric(train['TWS_t'], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
t_abs_vals = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms])
gaps = np.diff(t_abs_vals)
P(f"T={T} months, calendar span {t_abs_vals[-1]-t_abs_vals[0]+1}, gaps>1: {(gaps>1).sum()} instances")

ym_to_i = {int(v):i for i,v in enumerate(yms)}
F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
A = F - mu_c

# ---------- A) brute-force calendar-lag ACF via merge ----------
P("\n=== A) brute-force calendar-lag pooled ACF (merge) ===")
L = train[['cc','t_abs','TWS_t']].copy()
res = {}
for k in [1,2,3,6,12,18,24,36,48]:
    L2 = L.copy(); L2['t_abs'] = L2['t_abs'] + k
    mg = L.merge(L2, on=['cc','t_abs'], suffixes=('','_lag'))
    ok = mg['TWS_t'].notna() & mg['TWS_t_lag'].notna()
    x = mg['TWS_t'].values[ok]; y = mg['TWS_t_lag'].values[ok]
    # pooled over RAW values (not per-cell anomalies!) — check both
    r_raw = float(np.corrcoef(x, y)[0,1])
    # anomaly version: subtract per-cell mean
    ccm = mg['cc'].values[ok]
    xa = x - mu_c[ccm]; ya = y - mu_c[ccm]
    r_anom = float(np.corrcoef(xa, ya)[0,1])
    res[k] = (r_raw, r_anom, len(x))
    P(f"  lag {k:2d}: n={len(x):>9}, corr_raw={r_raw:+.4f}, corr_anom={r_anom:+.4f}")

# ---------- B) PC1 field ACF, same brute force ----------
P("\n=== B) PC1-reconstructed field, same brute-force ACF ===")
full = ~np.isnan(A).any(axis=0)
Af = A[:, full] - np.nanmean(A[:, full], axis=0, keepdims=True)
U, S_, Vt = np.linalg.svd(Af, full_matrices=False)
v_exp = S_**2/(S_**2).sum()
P(f"PC variance: PC1={v_exp[0]*100:.1f}%, PC2={v_exp[1]*100:.1f}%, PC3={v_exp[2]*100:.1f}%")
s1 = U[:,0]*S_[0]
P(f"PC1 score: std={s1.std():.2f}, corr(s1, t_index)={np.corrcoef(s1, np.arange(T))[0,1]:.4f}, "
  f"corr(s1, calendar_t)={np.corrcoef(s1, t_abs_vals.astype(float))[0,1]:.4f}")
# build PC1-only long-form (full cells)
cells_full = np.where(full)[0]
PC1_field = np.outer(s1, Vt[0])  # (T, n_full)
df1 = []
for ti in range(T):
    d = pd.DataFrame({'cc': cells_full, 't_abs': t_abs_vals[ti], 'v': PC1_field[ti]})
    df1.append(d)
df1 = pd.concat(df1, ignore_index=True)
for k in [1, 12, 24, 36]:
    L2 = df1.copy(); L2['t_abs'] = L2['t_abs'] + k
    mg = df1.merge(L2, on=['cc','t_abs'], suffixes=('','_lag'))
    r = float(np.corrcoef(mg['v'].values, mg['v_lag'].values)[0,1])
    P(f"  PC1-field lag {k:2d}: corr={r:+.4f} (n={len(mg)})")

# ---------- C) detrended ACF ----------
P("\n=== C) per-cell linear detrend (calendar time), then brute-force ACF ===")
tt = np.array(t_abs_vals, dtype=np.float64)
ttc = tt - tt.mean()
# OLS slope per full cell
Af64 = Af.astype(np.float64)
beta = (Af64 * ttc[:,None]).sum(axis=0) / (ttc**2).sum()
alpha = Af64.mean(axis=0)
A_dt = Af64 - np.outer(ttc, beta) - alpha[None,:]
var_before = float(Af64.var()); var_trend = float(np.outer(ttc,beta).var()); var_after = float(A_dt.var())
P(f"var: total={var_before:.4f}, trend={var_trend:.4f} ({var_trend/var_before*100:.1f}%), detrended={var_after:.4f}")
P(f"per-cell slope: std={beta.std():.5f}/month, mean={beta.mean():+.6f}; over 160 months: std*160={beta.std()*160:.3f}")
# detrended field back to long form
dfd = []
for ti in range(T):
    d = pd.DataFrame({'cc': cells_full, 't_abs': t_abs_vals[ti], 'v': A_dt[ti]})
    dfd.append(d)
dfd = pd.concat(dfd, ignore_index=True)
for k in [1, 6, 12, 24, 36]:
    L2 = dfd.copy(); L2['t_abs'] = L2['t_abs'] + k
    mg = dfd.merge(L2, on=['cc','t_abs'], suffixes=('','_lag'))
    r = float(np.corrcoef(mg['v'].values, mg['v_lag'].values)[0,1])
    P(f"  detrended lag {k:2d}: corr={r:+.4f}")

# ---------- D) the raw-value ACF puzzle ----------
P("\n=== D) diagnose: why did index-based anomaly ACF decay? ===")
# index-based (old method) on the SAME full-cell subset for direct comparison
for k in [12, 24, 36]:
    x = Af[:-k]; y = Af[k:]
    ok = np.isfinite(x) & np.isfinite(y)
    r_idx = float(np.corrcoef(x[ok].ravel(), y[ok].ravel())[0,1])
    P(f"  index-lag {k:2d} (old method, full cells): {r_idx:+.4f}")
# and calendar-true pairs only, same subset
for k in [12, 24, 36]:
    idx_pairs = []
    for ti in range(T):
        tj = np.where(t_abs_vals == t_abs_vals[ti] + k)[0]
        if len(tj) == 1:
            idx_pairs.append((ti, tj[0]))
    if not idx_pairs: continue
    xs = np.concatenate([Af[i] for i,j in idx_pairs])
    ys = np.concatenate([Af[j] for i,j in idx_pairs])
    ok = np.isfinite(xs) & np.isfinite(ys)
    r_cal = float(np.corrcoef(xs[ok], ys[ok])[0,1])
    P(f"  calendar-lag {k:2d} (exact pairs only, full cells): {r_cal:+.4f} ({len(idx_pairs)} month-pairs)")

# ---------- E) does the trend extend into the test era? ----------
P("\n=== E) train trend vs test-era anchor means ===")
# predicted drift from train trend at anchor times
for a_ym, a_t in [(201509, 24188), (201601, 24192), (201606, 24197), (201612, 24203), (201807, 24222), (201811, 24226)]:
    t_centered = a_t - tt.mean()
    pred_drift = t_centered * beta  # per-cell predicted anomaly from trend
    P(f"  {a_ym}: trend-predicted field std={pred_drift.std():.3f}, global mean={pred_drift.mean():+.4f}")
# actual anchor global means for reference: -0.19, -0.19, -0.32, -0.20, -0.03, -0.07
# corr of D-hat with trend pattern
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
test['TWS_t'] = pd.to_numeric(test['TWS_t'], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (test['t_abs'].values == a) & (~test['masked'].values)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[test['cc'].values[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
okD = np.isfinite(Dhat[full])
P(f"corr(D-hat, train trend pattern beta) = {np.corrcoef(Dhat[full][okD], beta[okD])[0,1]:+.4f}")
P(f"corr(D-hat, mu_c) = {np.corrcoef(Dhat[full][okD], mu_c[full][okD])[0,1]:+.4f}")
# what does the trend predict for D?
trend_at_anchors = np.array([(a - tt.mean()) for a in anchors])
pred_D = np.mean(np.outer(trend_at_anchors, beta), axis=0)
P(f"corr(D-hat, trend-extrapolated mean field) = {np.corrcoef(Dhat[full][okD], pred_D[okD])[0,1]:+.4f}")
P(f"std(trend-extrapolated D) = {pred_D[okD].std():.3f} vs std(D-hat) = {Dhat[full][okD].std():.3f}")

with open(OUT, 'w') as f:
    f.write('\n'.join(log))
P(f"\nsaved -> {OUT}")
