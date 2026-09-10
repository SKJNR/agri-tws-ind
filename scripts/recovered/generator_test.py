"""
GENERATOR REVERSE-ENGINEERING:
1. AR(1) vs AR(2) vs AR(1)+seasonal — per-cell, fit pre-2013, eval 2013+
2. Innovation structure: temporal corr, per-year sigma_w (the floor)
3. Optimal achievable estimate per model class
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
VAL = pd.Timestamp('2013-01-01')

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target','SPEI_01_t'])
train['time'] = pd.to_datetime(train['time'])
train['cc'] = (train['lat'].round(1).astype(str)+'_'+train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = train['cc'].max()+1
train['m'] = train['time'].dt.month
train = train.sort_values(['cc','time']).reset_index(drop=True)

g = train.groupby('cc', sort=False)
# lags within cell (calendar-adjacent only! skip across gaps for AR fit integrity)
lag_ok = (g['time'].shift(1).dt.year*12+g['time'].shift(1).dt.month == train['time'].dt.year*12+train['time'].dt.month-1)
x0 = train['TWS_t'].values
x1 = train['target'].values          # x at t+1
xm1 = g['TWS_t'].shift(1).values     # x at t-1
ok2 = lag_ok.values & ~np.isnan(xm1)

pre = (train['time'] < VAL).values
va = ~pre
month = train['time'].dt.month.values

def fit_eval(feats_pre, y_pre, feats_va, y_va, name, ridge=1.0):
    A = np.column_stack([feats_pre, np.ones(len(feats_pre))])
    coef, *_ = np.linalg.lstsq(A + 0, y_pre, rcond=None)
    # ridge via normal equations
    AtA = A.T @ A + ridge*np.eye(A.shape[1])
    Aty = A.T @ y_pre
    coef = np.linalg.solve(AtA, Aty)
    p = np.column_stack([feats_va, np.ones(len(feats_va))]) @ coef
    rmse = np.sqrt(np.mean((y_va - p)**2))
    print(f"  {name:28s}: {rmse:.4f}")
    return rmse, p

print("=== AR structure test (global coefficients, all cells pooled, val=2013+) ===")
# AR(1)
fit_eval(x0[pre], x1[pre], x0[va], x1[va], "AR(1) global")
# AR(2)
m2 = pre & ok2
m2v = va & ok2
fit_eval(np.column_stack([x0, xm1])[m2], x1[m2],
         np.column_stack([x0, xm1])[m2v], x1[m2v], "AR(2) global")
# AR(1) + seasonal mean term (month of t+1)
m1next = (month % 12) + 1
fit_eval(np.column_stack([x0, (m1next == 1).astype(float), (m1next == 2).astype(float),
                          (m1next == 3).astype(float), (m1next == 4).astype(float),
                          (m1next == 5).astype(float), (m1next == 6).astype(float)]),
         x1, "placeholder", 0, 0) if False else None
seas = np.zeros((len(train), 11), dtype=np.float32)
for j, mm in enumerate(range(1, 12)):
    seas[:, j] = (m1next == mm).astype(np.float32)
fit_eval(np.column_stack([x0, seas])[pre], x1[pre],
         np.column_stack([x0, seas])[va], x1[va], "AR(1) + seasonal")
# AR(2) + seasonal
fit_eval(np.column_stack([x0, xm1, seas])[m2], x1[m2],
         np.column_stack([x0, xm1, seas])[m2v], x1[m2v], "AR(2) + seasonal")

# innovation temporal correlation (AR(2) signature)
phi_g = 0.77
innov = x1 - phi_g*x0
innov_lag = pd.Series(innov).shift(1)
adj = lag_ok.values & ~np.isnan(innov_lag)
c = np.corrcoef(innov[adj][1:], innov_lag.values[adj][1:])[0,1]
print(f"\ninnovation lag-1 corr (global phi): {c:+.4f}  (0 => pure AR(1))")

# per-year innovation std (floor per year): use per-cell AR(1) residuals
print("\n=== per-year sigma_w (AR(1) innovation std, floor estimate) ===")
train['innov'] = np.nan
# per-cell phi via sufficient stats on pre
gp = train[pre].groupby('cc')
n_ = gp.size().astype(float)
sx_ = gp['TWS_t'].sum(); sy_ = gp['target'].sum()
sxx_ = gp.apply(lambda d: (d['TWS_t']**2).sum(), include_groups=False)
sxy_ = gp.apply(lambda d: (d['TWS_t']*d['target']).sum(), include_groups=False)
idx = range(n_cells)
sx_ = sx_.reindex(idx).fillna(0); sy_ = sy_.reindex(idx).fillna(0)
sxx_ = sxx_.reindex(idx).fillna(0); sxy_ = sxy_.reindex(idx).fillna(0)
n_ = n_.reindex(idx).fillna(0)
den = (n_*sxx_ - sx_**2).replace(0, np.nan)
phi_cell = ((n_*sxy_ - sx_*sy_)/den)
phi_cell = phi_cell.fillna(0.77).clip(0.2, 0.99).values
train['phi_c'] = phi_cell[train['cc'].values]
train['innov'] = train['target'] - train['phi_c']*train['TWS_t']
by_year = train.groupby(train['time'].dt.year)['innov'].std()
print(by_year.round(3).to_string())
print(f"\nk=0 floor (val years 2013-15): {np.sqrt((by_year.loc[2013]**2+by_year.loc[2014]**2+by_year.loc[2015]**2)/3):.4f}")

# per-cell innovation std — heteroskedastic?
ic = train[pre].groupby('cc')['innov'].std()
print(f"per-cell innov std: mean={ic.mean():.3f} p10={ic.quantile(.1):.3f} p90={ic.quantile(.9):.3f}")
# spatial correlation of innovations (within same month) — should be high (0.98 earlier)
pv = train.pivot_table(index=['lat','lon'], columns='time', values='innov')
idxl = pv.index.to_list()
loc = {(round(la,1),round(lo,1)): i for i,(la,lo) in enumerate(idxl)}
cors = []
rng = np.random.RandomState(7)
for i in rng.choice(len(idxl), 300, replace=False):
    la, lo = idxl[i]
    j = loc.get((round(la,1), round(lo+1.0,1)))
    if j is not None:
        cc_ = pv.iloc[i].corr(pv.iloc[j])
        if not np.isnan(cc_): cors.append(cc_)
print(f"east-neighbor innov corr: {np.mean(cors):+.3f}")
