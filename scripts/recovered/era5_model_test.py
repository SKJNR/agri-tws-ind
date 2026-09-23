"""Prepare ERA5-enhanced model test harness — runs the moment era5_grid.parquet exists.

Design decisions (post-mortem informed):
  - SPARSE anchors only (gaps 4-8mo) — dense CV proved misleading
  - No PC-denoise, no backward pass — both hurt on sparse anchors
  - phi=0.74 baseline (v2b config = current best)
  - ERA5 features enter through THREE channels:
    1. cov_field replacement: build cov-est from ALL covs (old 5 + new ERA5) -> S, W
       (tests: better D-tracking via deep soil / snow)
    2. k=0 linear model: add ERA5(t) + ERA5(t+1) columns
    3. D-tilde regressors: add ERA5 static mean field as D predictor
  - Evaluate on 2013-15 val masked rows with sparse anchors; compare vs v2b baseline 0.6945

Usage: python3 scripts/era5_model_test.py   (waits for parquet if not ready)
"""
import os, time
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
PARQ = f'{DATA}/era5_grid.parquet'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

# wait for parquet
if not os.path.exists(PARQ):
    print("era5_grid.parquet not found — run aggregate_era5.py first")
    raise SystemExit(1)

era5 = pd.read_parquet(PARQ)
ERA5_VARS = [c for c in era5.columns if c not in ('ym','cc')]
print(f"era5 parquet: {len(era5):,} rows, vars: {ERA5_VARS}")
print(era5.head(3))

# quick sanity: coverage of train months
ym_min, ym_max = era5['ym'].min(), era5['ym'].max()
print(f"ym range: {ym_min}..{ym_max}, months: {era5['ym'].nunique()}, cells: {era5['cc'].nunique()}")

# correlation of ERA5 vars with TWS anomaly (fit period)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month

m = train.merge(era5, on=['cc','ym'], how='inner')
print(f"\nmerged train rows: {len(m):,}")
mu_c = m.groupby('cc')['TWS_t'].mean()
m['tws_anom'] = m['TWS_t'] - m['cc'].map(mu_c)
print("\ncorr(TWS anomaly, ERA5 var) on merged rows:")
for v in ERA5_VARS:
    sub = m[['tws_anom', v]].dropna()
    if len(sub) > 1000:
        r = np.corrcoef(sub['tws_anom'], sub[v])[0,1]
        print(f"  {v:<35} r = {r:+.4f}  (n={len(sub):,})")
    else:
        print(f"  {v:<35} insufficient overlap (n={len(sub)})")

# also: does deep soil (swvl4) predict the SLOW/D component better than shallow?
# test: corr of ERA5 static mean field with D-hat (to be run after anchor fields built)
print("\n--- static-field D-tracking test (train period) ---")
# per-cell ERA5 means
era5_mu = m.groupby('cc')[ERA5_VARS].mean()
tws_mu = m.groupby('cc')['TWS_t'].mean()
for v in ERA5_VARS:
    sub = pd.DataFrame({'e': era5_mu[v], 't': tws_mu}).dropna()
    if len(sub) > 100:
        r = np.corrcoef(sub['e'], sub['t'])[0,1]
        print(f"  corr(mu_{v}, mu_TWS) = {r:+.4f}")
