"""Diagnose k=0 model: persistence vs full model on the SAME eval rows."""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
mu_c = train.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy()
Lk['t_next'] = Lk['t_abs'] + 1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values
yr = mgk['t_abs'].values // 12
x_p = (mgk['TWS_t'].values - mu_c[ccm]).astype(np.float32)   # persistence feature
yA = (mgk['target'].values - mu_c[ccm]).astype(np.float32)

for lo, hi, name in [(2010, 2016, '2010-15'), (2013, 2016, '2013-15'), (2002, 2010, '2002-09')]:
    ev = has_nxt & (yr >= lo) & (yr < hi)
    # persistence (slope fit on fit<=2012 subset of same feature)
    fit = has_nxt & (yr <= 2012)
    slope = float(np.sum(x_p[fit]*yA[fit])/np.sum(x_p[fit]**2))
    rmse_p = float(np.sqrt(np.mean((yA[ev] - slope*x_p[ev])**2)))
    # full model quick refit
    Xf = np.column_stack([x_p,
                          *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
                          *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
                          np.ones(len(mgk))]).astype(np.float32)
    Xf = np.nan_to_num(Xf, nan=0.0)
    w_rec = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
    sw = np.sqrt(w_rec)
    fitm = has_nxt & (yr <= 2012)
    A = Xf[fitm]*sw[fitm,None]
    cf = np.linalg.solve(A.T@A + 1e-3*np.eye(12), A.T@(yA[fitm]*sw[fitm]))
    rmse_f = float(np.sqrt(np.mean((Xf[ev]@cf - yA[ev])**2)))
    print(f"{name}: persistence slope={slope:.3f} RMSE={rmse_p:.4f} | full model RMSE={rmse_f:.4f} | n={int(ev.sum())}")
    print(f"   full-model coefs: TWS={cf[0]:.3f} covs_t={np.round(cf[1:6],3)} covs_t1={np.round(cf[6:11],3)}")
