"""2-b step 1: build field matrices and cache to npz. Also test-month structure."""
import numpy as np, pandas as pd, time as _t

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

t0 = _t.time()
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_tr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
CV = {}
for c in COVS:
    M = np.full((T, n_cells), np.nan, dtype=np.float32)
    M[train['ym'].map(ym_to_i).values, train['cc'].values] = train[c].values
    CV[c] = M

# cell coords
coords = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_c = coords['lat'].values.astype(np.float64)
lon_c = coords['lon'].values.astype(np.float64)

# per-cell OLS trend on calendar t
F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
den = np.nansum(td*td, axis=0)
beta_c = np.where(den > 100, np.nansum(td*F64, axis=0)/np.maximum(den,1e-9), 0.0)
mu_c = np.nanmean(F64, axis=0)
A_dt = F64 - mu_c[None,:] - (t_abs_tr[:,None]-tbar_c[None,:])*beta_c[None,:]  # detrended anomaly

np.savez_compressed('/home/z/my-project/scripts/tb_cache.npz',
    F=F, A_dt=A_dt, mu_c=mu_c, beta_c=beta_c, tbar_c=tbar_c,
    t_abs_tr=t_abs_tr, yms=yms, lat_c=lat_c, lon_c=lon_c,
    SPEI1=CV['SPEI_01_t'], SPEI3=CV['SPEI_03_t'], SPEI6=CV['SPEI_06_t'],
    SPEI12=CV['SPEI_12_t'], SM=CV['SOIL_MOISTURE_t'])

# ---------- test structure ----------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
g = test.groupby('ym').agg(n=('ID','size'), n_mask=('masked','sum'), t_abs=('t_abs','first'))
g['mask_frac'] = g['n_mask']/g['n']
print("TEST months (ym, t_abs, rows, mask_frac):")
print(g.to_string())
print(f"\ntrain cells: {n_cells}, train months: {T} ({yms[0]}..{yms[-1]})")
print(f"anomaly std (pooled, detrended): {np.nanstd(A_dt):.4f}")
print(f"elapsed {_t.time()-t0:.0f}s")
