"""C3 standalone: does ERA5 static field improve anchor-field (D) prediction?"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
ERA5_VARS = ['swvl1','swvl2','swvl3','swvl4','sd','tp','e','ro','t2m']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1

era5 = pd.read_parquet(f'{DATA}/era5_grid.parquet')
m = train.merge(era5, on=['cc','ym'], how='inner')

n_cells = int(m['cc'].max())+1
val = m[(m['t_abs'] >= 2013*12) & (m['t_abs'] < 2016*12)].copy()
fitp = m[m['t_abs'] < 2013*12]
mu_fit = fitp.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
E_static = fitp.groupby('cc')[ERA5_VARS].mean().reindex(range(n_cells)).values.astype(np.float32)

val_ta = val['t_abs'].values
existing = set(val_ta.tolist())
sparse = [mm for mm in [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6] if mm in existing]
print(f"sparse anchors: {sparse}", flush=True)

Fv = np.full((int(val_ta.max())+2, n_cells), np.nan, dtype=np.float32)
Fv[val_ta, val['cc'].values] = val['TWS_t'].values
AF = {a: Fv[a] - mu_fit for a in sparse}

# cov field S (fit-period regression, val-period fields)
Z = fitp[COVS].values.astype('float32'); yv = fitp['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Zval = val[COVS].values.astype('float32'); okv = np.isfinite(Zval).all(axis=1)
cov_est = np.full(len(val), np.nan, dtype=np.float32)
cov_est[okv] = np.column_stack([Zval[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for mm in np.sort(val['t_abs'].unique()):
    selm = (val_ta == mm) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val['cc'].values[selm]] = cov_est[selm]
    cov_field[mm] = fm - mu_fit
S = np.nanmean(np.array([cov_field[mm] for mm in sorted(cov_field.keys())]), axis=0)

# trendex from fit period (COMPACT: only months that exist in fit data)
fit_months = np.sort(fitp['t_abs'].unique())
fm_to_i = {int(v): i for i, v in enumerate(fit_months)}
Ftr = np.full((len(fit_months), n_cells), np.nan, dtype=np.float32)
Ftr[fitp['t_abs'].map(fm_to_i).values, fitp['cc'].values] = fitp['TWS_t'].values
t_axis = fit_months.astype(np.float64)
ok_t = np.isfinite(Ftr)
t_mat = np.where(ok_t, t_axis[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
F64 = Ftr.astype(np.float64)
sxy = np.nansum(td*F64, axis=0); sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx>100)&(ok_t.sum(axis=0)>=24), sxy/np.where(sxx>0,sxx,1), 0).astype(np.float32)
del F64, t_mat, td

def trendex(t): return ((np.float64(t) - tbar_c)*beta_c).astype(np.float32)

def loo_rmse(regressors, label):
    Xs, ys = [], []
    for a in sparse:
        dloo = np.nanmean(np.array([AF[b] for b in sparse if b != a]), axis=0)
        base = [dloo, S, trendex(a)]
        cols = [E_static[:, ERA5_VARS.index(r[1:])] for r in regressors]
        Xa = np.column_stack(base + cols)
        ok = np.isfinite(Xa).all(axis=1) & np.isfinite(AF[a])
        Xs.append(Xa[ok]); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(Xw.shape[1]), Xw.T@yw)
    pred = Xw @ w_
    rmse = float(np.sqrt(np.mean((pred-yw)**2)))
    print(f'{label:<45} LOO RMSE={rmse:.4f}', flush=True)
    return rmse

print('=== C3: anchor-field prediction with ERA5 static fields ===', flush=True)
loo_rmse([], 'D-hat + S + trendex (current)')
for r in ['Eswvl4','Esd','Etp','Ee','Et2m']:
    loo_rmse([r], f'+ {r}')
loo_rmse(['Eswvl4','Esd','Etp','Ee','Et2m'], '+ all 5 static')

# bonus: replace S with ERA5 static entirely (is ERA5 better than competition S?)
def loo_rmse2(use_era5_S, label):
    Xs, ys = [], []
    for a in sparse:
        dloo = np.nanmean(np.array([AF[b] for b in sparse if b != a]), axis=0)
        if use_era5_S:
            Xa = np.column_stack([dloo, E_static[:, 6], trendex(a)])  # E_static[:,6] = e
        else:
            Xa = np.column_stack([dloo, S, trendex(a)])
        ok = np.isfinite(Xa).all(axis=1) & np.isfinite(AF[a])
        Xs.append(Xa[ok]); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(Xw.shape[1]), Xw.T@yw)
    rmse = float(np.sqrt(np.mean((Xw@w_ - yw)**2)))
    print(f'{label:<45} LOO RMSE={rmse:.4f}', flush=True)

loo_rmse2(True, 'D-hat + E_e(static) + trendex [replace S]')
