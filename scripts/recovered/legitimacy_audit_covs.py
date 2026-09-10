"""
Legitimacy audit: Is a public LB score of ~0.632 explainable by a LEGITIMATE
model that uses ONLY competition-provided data (covariates at target month)?

Method:
- Train.csv: target(t) = TWS at calendar month t+1 (proven exactly).
- Build features from covariates AT THE TARGET MONTH (t+1): SPEI_01/03/06/12, SOIL_MOISTURE.
- These are exactly the covariates the competition gives us for test month t+1
  in Test.csv (masking applies only to TWS_t, never to covariates).
- Honest validation on 2013-2015 (the established CV window from Task 12).
- Compare achievable RMSE to: our v18a (0.6937), v20c (0.6317), leader (0.5596).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge

print('Loading train...')
train = pd.read_csv('/home/z/my-project/data/Train (1).csv')
print('Train columns:', list(train.columns))
print('Train shape:', train.shape)

# Build calendar-correct next-month key
train['date'] = pd.to_datetime(train['time'])
train['y'] = train['date'].dt.year
train['m'] = train['date'].dt.month
train['t_abs'] = train['y'] * 12 + train['m'] - 1

# next calendar month t_abs+1; December->January handled automatically
train['t_next'] = train['t_abs'] + 1

# Covariates at the row's own month
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

# Create lookup of covariates by (cell, t_abs) for the NEXT month
cov_next = train[['lat', 'lon', 't_abs'] + COVS + ['TWS_t']].copy()
cov_next['t_lookup'] = cov_next['t_abs']
cov_next = cov_next.rename(columns={c: c + '_nx' for c in COVS})
cov_next['TWS_nx'] = cov_next.pop('TWS_t')

# merge: for each row, find the row of same cell at t_abs+1
train = train.merge(
    cov_next[['lat', 'lon', 't_lookup'] + [c + '_nx' for c in COVS] + ['TWS_nx']],
    left_on=['lat', 'lon', 't_next'], right_on=['lat', 'lon', 't_lookup'], how='left')

have_next = train['TWS_nx'].notna()
print(f'\nRows whose next calendar month exists in train: {have_next.sum()} / {len(train)} ({have_next.mean()*100:.1f}%)')

# sanity: target == TWS at next month (re-verify the identity on a sample)
chk = train[have_next]
print('corr(target, TWS_nx):', np.corrcoef(chk['target'], chk['TWS_nx'])[0, 1])

# ---- cell climatology (fit on <=2012 only, honest) ----
early = train[train['y'] <= 2012]
mu = early.groupby(['lat', 'lon'])['TWS_t'].mean().rename('mu_c')
train = train.merge(mu, on=['lat', 'lon'], how='left')
train['mu_c'] = train['mu_c'].fillna(0.0)

# ---- honest validation window: 2013-2015 rows whose t+1 exists ----
val = train[(train['y'] >= 2013) & (train['y'] <= 2015) & have_next.reindex(train.index, fill_value=False)]
fit = train[(train['y'] <= 2012) & have_next.reindex(train.index, fill_value=False)]
print(f'\nFit rows: {len(fit)}, Val rows: {len(val)}')

FEATS_NX = [c + '_nx' for c in COVS]

# ========== Model A: ridge on next-month covariates + climatology ==========
Xf = fit[FEATS_NX + ['mu_c']].values
yf = fit['target'].values
Xv = val[FEATS_NX + ['mu_c']].values
yv = val['target'].values

ridge = Ridge(alpha=1.0)
ridge.fit(Xf, yf)
pv = ridge.predict(Xv)
rmse_ridge = np.sqrt(np.mean((pv - yv) ** 2))
print(f'\n[A] Ridge(covs(t+1) + mu_c) val RMSE 2013-15: {rmse_ridge:.4f}')

# ========== Model B: LightGBM on next-month covariates ==========
Xf2 = fit[FEATS_NX + ['mu_c', 'lat', 'lon']].values
Xv2 = val[FEATS_NX + ['mu_c', 'lat', 'lon']].values
lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                         subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
                         n_jobs=-1, verbose=-1, random_state=42)
lgbm.fit(Xf2, yf)
pv2 = lgbm.predict(Xv2)
rmse_lgb = np.sqrt(np.mean((pv2 - yv) ** 2))
print(f'[B] LGBM(covs(t+1) + mu_c + geo) val RMSE 2013-15: {rmse_lgb:.4f}')

# ========== Model C: LGBM + current TWS (persistence channel, k=0 analogue) ==========
# includes TWS_t (current month) — only usable for unmasked rows
Xf3 = fit[FEATS_NX + ['mu_c', 'lat', 'lon', 'TWS_t']].values
Xv3 = val[FEATS_NX + ['mu_c', 'lat', 'lon', 'TWS_t']].values
lgbm2 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
                          n_jobs=-1, verbose=-1, random_state=42)
lgbm2.fit(Xf3, yf)
pv3 = lgbm2.predict(Xv3)
rmse_lgb2 = np.sqrt(np.mean((pv3 - yv) ** 2))
print(f'[C] LGBM(covs(t+1) + TWS_t + geo) val RMSE 2013-15: {rmse_lgb2:.4f}')

# ========== Reference points ==========
print('\n--- Reference points (RMSE, lower better) ---')
print('Persistence (TWS_t):           ', np.sqrt(np.mean((val['TWS_t'] - yv) ** 2)))
print('Climatology (mu_c):            ', np.sqrt(np.mean((val['mu_c'] - yv) ** 2)))
print('Our v18a public LB:             0.6937')
print('Our v20c public LB:             0.6317')
print('Leader (Aug 23) public LB:      0.5596')

# feature importance
imp = pd.Series(lgbm.feature_importances_, index=FEATS_NX + ['mu_c', 'lat', 'lon'])
print('\nLGBM feature importance (Model B):')
print(imp.sort_values(ascending=False).to_string())
