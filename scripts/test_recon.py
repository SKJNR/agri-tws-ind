"""
Quick test: per-cell linear reconstruction of TWS_t from covariates.
If accurate, use as feature for Model B (masked rows have covariates at month t!).
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
VAL_START = '2013-01-01'

train = pd.read_csv(f'{DATA}/Train (1).csv',
                    usecols=['time', 'lat', 'lon', 'TWS_t', 'SOIL_MOISTURE_t', 'SPEI_06_t', 'SPEI_12_t'])
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'SOIL_MOISTURE_t', 'SPEI_06_t', 'SPEI_12_t']:
    train[c] = pd.to_numeric(train[c], errors='coerce')
train['cell'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str))
train['cal_m'] = train['time'].dt.month

pre = train[train['time'] < VAL_START]
va = train[train['time'] >= VAL_START]

# per-cell OLS via sufficient statistics: TWS ~ 1 + SOIL + SPEI12 + SPEI06
def fit_predict(feats, pre, va):
    k = len(feats)
    cells_all, uniques = pd.factorize(train['cell'].values, sort=True)
    # rebuild codes for pre/va via merge on uniques
    cell_code = {c: i for i, c in enumerate(uniques)}
    codes_p = pre['cell'].map(cell_code).values
    codes_v = va['cell'].map(cell_code).values
    n_cells = len(uniques)
    Xp = np.column_stack([pre[f].values for f in feats] + [np.ones(len(pre))])
    yp = pre['TWS_t'].values
    XtX = np.zeros((n_cells, k + 1, k + 1))
    Xty = np.zeros((n_cells, k + 1))
    for i in range(k + 1):
        for j in range(i, k + 1):
            np.add.at(XtX[:, i, j], codes_p, Xp[:, i] * Xp[:, j])
        np.add.at(Xty[:, i], codes_p, Xp[:, i] * yp)
    for i in range(k + 1):
        for j in range(i):
            XtX[:, i, j] = XtX[:, j, i]
    XtX += np.eye(k + 1)[None] * 10.0
    coefs = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]
    Xv = np.column_stack([va[f].values for f in feats] + [np.ones(len(va))])
    pred = (Xv * coefs[codes_v]).sum(axis=1)
    rmse = np.sqrt(np.nanmean((va['TWS_t'].values - pred) ** 2))
    return rmse, pred, coefs, cell_code

for feats in [['SOIL_MOISTURE_t'],
              ['SOIL_MOISTURE_t', 'SPEI_12_t'],
              ['SOIL_MOISTURE_t', 'SPEI_12_t', 'SPEI_06_t']]:
    rmse, pred, coefs, code = fit_predict(feats, pre, va)
    print(f"TWS ~ {feats}: RMSE = {rmse:.4f} (TWS std = {va['TWS_t'].std():.4f})")

# reference: climatology RMSE on same val
cm = pre.groupby(['cell', 'cal_m'])['TWS_t'].mean()
va_clim = cm.reindex(pd.MultiIndex.from_arrays([va['cell'], va['cal_m']])).values
gm = pre.groupby('cal_m')['TWS_t'].mean()
va_clim = pd.Series(va_clim).fillna(va['cal_m'].map(gm)).values
print(f"climatology ref: {np.sqrt(np.nanmean((va['TWS_t'].values - va_clim)**2)):.4f}")
print(f"persistence impossible for masked (std ref): {va['TWS_t'].std():.4f}")
