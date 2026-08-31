"""
PHASE 1 DIAGNOSTICS: Data Engineer + Statistician + Hydrologist + Spatial/GIS hats.
Comprehensive audit report -> /home/z/my-project/scripts/phase1_report.txt

Key questions:
  [DE]  ID structure, duplicates, target semantics, test/train alignment
  [STA] TWS distribution, per-cell AR(1) decay rates, ACF by region, seasonality of deltas
  [HYD] Is TWS an anomaly (TWSA)? Water balance check: delta vs P-proxy at proper lags
  [GIS] Neighbor correlations, spatial smoothness, regional clustering structure
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
R = []
def rep(s=''):
    print(s, flush=True)
    R.append(str(s))

rep("=" * 78)
rep("PHASE 1 COMPREHENSIVE DIAGNOSTIC REPORT")
rep("=" * 78)

# ============ [DATA ENGINEER HAT] ============
rep("\n" + "-" * 40 + "\n[1] DATA ENGINEER: integrity audit\n" + "-" * 40)
train = pd.read_csv(f'{DATA}/Train (1).csv')
test = pd.read_csv(f'{DATA}/Test (2).csv')
train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])

rep(f"Train: {train.shape}, Test: {test.shape}")
rep(f"Train dup IDs: {train['sample_id'].duplicated().sum()}, Test dup IDs: {test['ID'].duplicated().sum()}")
rep(f"Train nulls per col: {train.isnull().sum().sum()} total")
rep(f"Test nulls: TWS_t={test['TWS_t'].isnull().sum()} (masked), others={test.drop(columns=['TWS_t']).isnull().sum().sum()}")

# ID semantic check: does ID encode (time, lat, lon) exactly?
idt = test['ID'].str.extract(r'^(\d{8})_(-?\d+\.\d)_(-?\d+\.\d)$')
rep(f"Test IDs parse cleanly: {idt.notna().all().all()}")
mism = (pd.to_datetime(idt[0], format='%Y%m%d') != test['time']).sum() + \
       (idt[1].astype(float) != test['lat']).sum() + (idt[2].astype(float) != test['lon']).sum()
rep(f"ID<->columns mismatches: {mism}")

# train/test cell alignment
tr_cells = set(zip(train['lat'].round(1), train['lon'].round(1)))
te_cells = set(zip(test['lat'].round(1), test['lon'].round(1)))
rep(f"Cells: train={len(tr_cells)}, test={len(te_cells)}, test-not-in-train={len(te_cells - tr_cells)}")

# covariate continuity: are test covariate values consistent with train distribution? (drift check)
for c in ['SPEI_01_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']:
    rep(f"Drift {c}: train mean={train[c].mean():.3f} std={train[c].std():.3f} | "
        f"test mean={test[c].mean():.3f} std={test[c].std():.3f}")

# ============ [STATISTICIAN HAT] ============
rep("\n" + "-" * 40 + "\n[2] STATISTICIAN: distribution & time-series structure\n" + "-" * 40)
rep(f"TWS_t: mean={train['TWS_t'].mean():.4f} std={train['TWS_t'].std():.4f} "
    f"skew={train['TWS_t'].skew():.3f} kurt={train['TWS_t'].kurt():.3f}")
rep(f"target: mean={train['target'].mean():.4f} std={train['target'].std():.4f}")

# per-cell AR(1) on raw TWS (per-cell phi): TWS_{t+1} = a*TWS_t + b
train_sorted = train.sort_values(['lat', 'lon', 'time'])
g = train_sorted.groupby(['lat', 'lon'])
n_obs = g.size()
# per-cell OLS via sufficient stats
sx = g['TWS_t'].sum(); sy = g['target'].sum()
sxx = g.apply(lambda d: (d['TWS_t'] ** 2).sum(), include_groups=False)
sxy = g.apply(lambda d: (d['TWS_t'] * d['target']).sum(), include_groups=False)
n = n_obs.astype(float)
den = (n * sxx - sx.pow(2)).replace(0, np.nan)
num = (n * sxy - sx * sy)
phi = (num / den).rename('phi')
# align intercept computation
sy_al = sy.reindex(phi.index); sx_al = sx.reindex(phi.index); n_al = n.reindex(phi.index)
b_int_s = (sy_al - phi * sx_al) / n_al
rep(f"\nPer-cell AR(1) phi: mean={phi.mean():.3f} std={phi.std():.3f} "
    f"p10={phi.quantile(.1):.3f} p50={phi.quantile(.5):.3f} p90={phi.quantile(.9):.3f}")
rep(f"Cells with phi<0.5: {(phi<0.5).mean():.1%} | phi>0.95: {(phi>0.95).mean():.1%}")

# in-sample per-cell AR(1) RMSE (upper bound)
phi_by_cell = phi.reindex(pd.MultiIndex.from_arrays([train_sorted['lat'].round(1), train_sorted['lon'].round(1)])).values
b_by_cell = b_int_s.reindex(pd.MultiIndex.from_arrays([train_sorted['lat'].round(1), train_sorted['lon'].round(1)])).values
pred_in = phi_by_cell * train_sorted['TWS_t'].values + b_by_cell
rep(f"In-sample per-cell AR(1) RMSE (biased, upper bound): "
    f"{np.sqrt(np.nanmean((train_sorted['target'].values - pred_in) ** 2)):.4f}")

# anomaly autocorrelation decay (global): corr(anom_t, anom_{t-k})
cell_mean = g['TWS_t'].transform('mean')
anom = train_sorted['TWS_t'] - cell_mean
train_sorted['anom'] = anom.values
for k in [1, 2, 3, 6, 12]:
    lag = g['anom'].shift(k)
    ok = lag.notna()
    rep(f"ACF anom lag {k:2d}: {np.corrcoef(train_sorted['anom'][ok], lag[ok])[0,1]:.4f}")

# seasonality of delta
train_sorted['delta'] = train_sorted['target'] - train_sorted['TWS_t']
seas = train_sorted.groupby(train_sorted['time'].dt.month)['delta'].agg(['mean', 'std'])
rep("\nDelta by calendar month (mean +/- std):")
for m, row in seas.iterrows():
    rep(f"  month {m:2d}: {row['mean']:+.4f} +/- {row['std']:.4f}")

# variance components: how much variance is spatial vs temporal vs noise?
train_sorted['ym'] = train_sorted['time']
gm = train_sorted.groupby('time')['TWS_t'].transform('mean')
rep(f"\nVariance decomposition of TWS_t:")
rep(f"  total var: {train_sorted['TWS_t'].var():.4f}")
rep(f"  global-month var (shared): {gm.var():.4f} ({gm.var()/train_sorted['TWS_t'].var():.1%})")

# ============ [HYDROLOGIST HAT] ============
rep("\n" + "-" * 40 + "\n[3] HYDROLOGIST: water balance & TWSA verification\n" + "-" * 40)
# Is TWS an anomaly? If anomaly: per-cell long-run mean ~ 0
cm = train_sorted.groupby(['lat', 'lon'])['TWS_t'].mean()
rep(f"Per-cell means: mean={cm.mean():.4f} std={cm.std():.4f} | |mean|>0.5: {(cm.abs()>0.5).mean():.1%}")
rep("  -> if means were ~0 everywhere, TWS is a pure anomaly; here there IS a spatial "
    "mean structure (could be anomaly vs arbitrary reference period)")

# THE MYSTERY: delta vs SPEI correlation structure at correct lags
rep("\ndelta(t->t+1) vs covariates AT VARIOUS LAGS (using month-t row):")
for c in ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']:
    cc = np.corrcoef(train_sorted['delta'], train_sorted[c])[0, 1]
    rep(f"  corr(delta, {c} at t): {cc:+.4f}")

# lagged covariates: SPEI_01_{t+1} is NOT available, but what about SPEI_03? it embeds t-2..t
# check: does delta correlate with CHANGE in covariates?
for c in ['SPEI_01_t', 'SOIL_MOISTURE_t']:
    d_cov = g[c].diff()
    ok = d_cov.notna() & train_sorted['delta'].notna()
    rep(f"  corr(delta, Δ{c}): {np.corrcoef(train_sorted['delta'][ok], d_cov[ok])[0,1]:+.4f}")

# per-cell slope of delta on SPEI_01: is there cell-specific structure?
g2 = train_sorted.groupby(['lat', 'lon'])  # fresh groupby (sees 'delta' col)
sl_num = g2.apply(lambda d: (d['delta'] * d['SPEI_01_t']).sum(), include_groups=False) - \
         g2['delta'].sum() * g2['SPEI_01_t'].sum() / n
sl_den = g2.apply(lambda d: (d['SPEI_01_t'] ** 2).sum(), include_groups=False) - \
         g2['SPEI_01_t'].sum() ** 2 / n
slope = sl_num / sl_den.replace(0, np.nan)
rep(f"\nPer-cell slope(delta ~ SPEI_01): mean={slope.mean():+.4f} std={slope.std():.4f} "
    f"p10={slope.quantile(.1):+.4f} p90={slope.quantile(.9):+.4f}")

# KEY TEST: delta vs SPEI at FUTURE month (t+1) using train rows — is the signal there?
# build (cell, ym) -> SPEI_01 map, then for row at t, look up SPEI at t+1
sp_map = {}
for la, lo, t, s in zip(train['lat'], train['lon'], train['ym'] if 'ym' in train else train['time'].dt.strftime('%Y%m'), train['SPEI_01_t']):
    sp_map[(la, lo, str(t))] = s
# (too slow for 2M rows — use pivot instead)
pv = train.pivot_table(index=['lat', 'lon'], columns='time', values='SPEI_01_t')
tws_pv = train.pivot_table(index=['lat', 'lon'], columns='time', values='TWS_t')
tgt_pv = train.pivot_table(index=['lat', 'lon'], columns='time', values='target')
times = sorted(tws_pv.columns)
for i in range(len(times) - 1):
    if i == 0:
        delta_f = tws_pv[times[i + 1]] - tws_pv[times[i]]
        spei_next = pv[times[i + 1]]  # FUTURE covariate — signal ceiling test only!
        spei_cur = pv[times[i]]
    else:
        delta_f = pd.concat([delta_f, tws_pv[times[i + 1]] - tws_pv[times[i]]])
        spei_next = pd.concat([spei_next, pv[times[i + 1]]])
        spei_cur = pd.concat([spei_cur, pv[times[i]]])
ok = delta_f.notna() & spei_next.notna() & spei_cur.notna()
rep(f"\nSIGNAL CEILING TEST (train, using future SPEI — NOT allowed in prod, ceiling only):")
rep(f"  corr(delta_t->t+1, SPEI_01 at t+1): {np.corrcoef(delta_f[ok], spei_next[ok])[0,1]:+.4f}")
rep(f"  corr(delta_t->t+1, SPEI_01 at t):   {np.corrcoef(delta_f[ok], spei_cur[ok])[0,1]:+.4f}")
# same for soil moisture
sm_pv = train.pivot_table(index=['lat', 'lon'], columns='time', values='SOIL_MOISTURE_t')
sm_next = None
for i in range(len(times) - 1):
    s = sm_pv[times[i + 1]]
    sm_next = s if sm_next is None else pd.concat([sm_next, s])
ok2 = delta_f.notna() & sm_next.notna()
rep(f"  corr(delta_t->t+1, SOIL_M at t+1):  {np.corrcoef(delta_f[ok2], sm_next[ok2])[0,1]:+.4f}")

# ============ [SPATIAL/GIS HAT] ============
rep("\n" + "-" * 40 + "\n[4] SPATIAL/GIS: neighbor structure & regionalization\n" + "-" * 40)
# neighbor correlation of TWS anomalies at same month
cellmean = train.pivot_table(index=['lat', 'lon'], columns='time', values='TWS_t')
# anomaly per month: subtract per-cell monthly climatology
clim = cellmean.T.groupby(cellmean.T.index.month).transform('mean').T
anom_pv = cellmean - clim
# pick a sample of months, compute lag-1 neighbor correlation
t0, t1 = times[30], times[31]
a0, a1 = anom_pv[t0], anom_pv[t1]
rep(f"Anomaly field corr({times[30].date()}, {times[31].date()}) across cells: "
    f"{a0.corr(a1):.4f}  <- field-level month-to-month correlation")
# neighbor: correlate anom(cell, t) with anom(nearest lon neighbor, t)
idx = anom_pv.index.to_list()
loc = {(round(la,1), round(lo,1)): i for i, (la, lo) in enumerate(idx)}
cors_self, cors_nb = [], []
lats_arr = np.array([la for la, lo in idx]); lons_arr = np.array([lo for la, lo in idx])
for i in range(0, len(idx), 5):  # sample every 5th cell
    la, lo = idx[i]
    j = loc.get((round(la,1), round(lo + 1.0, 1)))
    if j is not None:
        c = anom_pv.iloc[i].corr(anom_pv.iloc[j])
        if not np.isnan(c):
            cors_nb.append(c)
rep(f"Same-month east-neighbor anomaly corr: mean={np.mean(cors_nb):.4f} (n={len(cors_nb)})")

# regional clustering by lat bands: predictability (AR1 phi) by latitude zone
phi_df = phi.rename('phi').reset_index()
phi_df['lat'] = phi_df['lat'].round(0)
zones = phi_df.groupby('lat')['phi'].agg(['mean', 'count']).query('count > 30')
rep("\nPer-cell AR(1) phi by latitude band (cells with >30 obs):")
rep(zones.round(3).to_string())

rep("\n" + "=" * 78)
rep("END OF PHASE 1 REPORT")

with open('/home/z/my-project/scripts/phase1_report.txt', 'w') as f:
    f.write('\n'.join(R))
print("\nReport saved.", flush=True)
