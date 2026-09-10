"""ERA5 value-test: 3 channels, honest 2013-15 eval.

C1: duplication check — is SOIL_MOISTURE_t just ERA5 swvl? is SPEI just tp?
C2: k=0 linear model — add ERA5(t), ERA5(t+1) features; honest eval (fit<=2012, eval 2013-15)
C3: anchor-field prediction — does ERA5 static/anomaly improve D-tilde (currently D-hat + S + trendex)?
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
ERA5_VARS = ['swvl1','swvl2','swvl3','swvl4','sd','tp','e','ro','t2m']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1

era5 = pd.read_parquet(f'{DATA}/era5_grid.parquet')
m = train.merge(era5, on=['cc','ym'], how='inner')

# ---------------- C1: duplication check ----------------
print("=== C1: duplication check (corr on merged rows) ===")
for ev in ['swvl1','swvl2','swvl3','swvl4','t2m']:
    sub = m[['SOIL_MOISTURE_t', ev]].dropna()
    r = np.corrcoef(sub['SOIL_MOISTURE_t'], sub[ev])[0,1]
    print(f"  corr(SOIL_MOISTURE_t, {ev}) = {r:+.4f}")
for sp in ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t']:
    sub = m[[sp, 'tp']].dropna()
    r = np.corrcoef(sub[sp], sub['tp'])[0,1]
    print(f"  corr({sp}, tp) = {r:+.4f}")
# within-cell (demeaned) correlation — the real duplication test
print("\n  within-cell (demeaned):")
for ev in ['swvl1','swvl2','swvl3','swvl4']:
    dm = m[['cc','SOIL_MOISTURE_t',ev]].dropna()
    dm['sm_d'] = dm['SOIL_MOISTURE_t'] - dm.groupby('cc')['SOIL_MOISTURE_t'].transform('mean')
    dm['ev_d'] = dm[ev] - dm.groupby('cc')[ev].transform('mean')
    r = np.corrcoef(dm['sm_d'], dm['ev_d'])[0,1]
    print(f"  corr(demeaned SOIL_MOISTURE_t, {ev}) = {r:+.4f}")

# ---------------- C2: k=0 linear model with ERA5 features ----------------
print("\n=== C2: k=0 model, honest eval (fit<=2012, eval 2013-15) ===")
# next-month ERA5 join
t_next = m['t_abs'].values + 1
ym_next = (t_next//12)*100 + (t_next%12) + 1
m['ym_next'] = ym_next
en = era5[['cc','ym']+ERA5_VARS].rename(columns={'ym':'ym_next', **{v: v+'_nxt' for v in ERA5_VARS}})
m2 = m.merge(en, on=['cc','ym_next'], how='left')
# also next-month competition covs
cov_next = m[['cc','ym']+COVS].rename(columns={'ym':'ym_next', **{c: c+'_nxt' for c in COVS}})
m2 = m2.merge(cov_next.drop_duplicates(['cc','ym_next']), on=['cc','ym_next'], how='left')

n_cells = int(m2['cc'].max())+1
mu_c = m2.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = m2.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype(np.float32)
eclim = m2.groupby('cc')[ERA5_VARS].mean().reindex(range(n_cells)).values.astype(np.float32)

def build_X(df, feats):
    cc = df['cc'].values
    cols = [df['TWS_t'].values - mu_c[cc]]
    for f in feats:
        if f.endswith('_nxt'):
            base = f[:-4]
            src = df[f].values
            if base in COVS:
                j = COVS.index(base)
                cols.append(src - clim[cc, j])
            else:
                j = ERA5_VARS.index(base)
                cols.append(src - eclim[cc, j])
        elif f in COVS:
            j = COVS.index(f)
            cols.append(df[f].values - clim[cc, j])
        else:
            j = ERA5_VARS.index(f)
            cols.append(df[f].values - eclim[cc, j])
    cols.append(np.ones(len(df)))
    X = np.column_stack(cols).astype(np.float32)
    return np.nan_to_num(X, nan=0.0)

def honest_eval(feats, label):
    sub = m2[m2['target'].notna() & m2['TWS_t'].notna()].copy()
    has_all = sub[feats].notna().all(axis=1).values
    sub = sub[has_all]
    yr = sub['t_abs'].values // 12
    fitm = yr <= 2012; evam = yr >= 2013
    X = build_X(sub, feats)
    y = (sub['target'].values - mu_c[sub['cc'].values]).astype(np.float32)
    w = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
    sw = np.sqrt(w)
    A = X[fitm]*sw[fitm,None]
    cf = np.linalg.solve(A.T@A + 1e-3*np.eye(X.shape[1]), A.T@(y[fitm]*sw[fitm]))
    pred = X[evam] @ cf
    rmse = float(np.sqrt(np.mean((pred - y[evam])**2)))
    print(f"  {label:<55} RMSE={rmse:.4f} (n={evam.sum():,})")
    return rmse

# baselines
r_base = honest_eval(COVS + [c+'_nxt' for c in COVS], 'baseline: TWS + covs(t) + covs(t+1)')
# + ERA5(t)
best_sets = []
for ev in ERA5_VARS:
    r = honest_eval(COVS + [c+'_nxt' for c in COVS] + [ev], f'+ {ev}(t)')
    best_sets.append((r, ev))
best_sets.sort()
top3 = [ev for _, ev in best_sets[:3]]
print(f"  top-3 single ERA5(t): {top3}")
honest_eval(COVS + [c+'_nxt' for c in COVS] + top3, f'+ top3 ERA5(t): {top3}')
honest_eval(COVS + [c+'_nxt' for c in COVS] + top3 + [v+'_nxt' for v in top3], f'+ top3 ERA5(t) + (t+1)')
honest_eval(COVS + [c+'_nxt' for c in COVS] + ERA5_VARS, '+ all 9 ERA5(t)')
honest_eval(COVS + [c+'_nxt' for c in COVS] + ERA5_VARS + [v+'_nxt' for v in ERA5_VARS], '+ all 9 ERA5(t) + (t+1)')

# ---------------- C3: anchor-field D-tilde enhancement ----------------
print("\n=== C3: anchor-field prediction with ERA5 (sparse-anchor protocol) ===")
# honest: use 2013-15 val window with sparse anchors
val = m2[(m2['t_abs'] >= 2013*12) & (m2['t_abs'] < 2016*12)].copy()
fitp = m2[m2['t_abs'] < 2013*12]
mu_fit = fitp.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype(np.float32)
# era5 static field (fit-period mean, per cell) — the "D-tracking static" candidate
E_static = fitp.groupby('cc')[ERA5_VARS].mean().reindex(range(n_cells)).values.astype(np.float32)

# sparse anchors: Jan 2013, Jul 2013, Jan 2014, Sep 2014, Jan 2015, Jul 2015 (existing months)
val_ta = val['t_abs'].values
existing = set(val_ta.tolist())
sparse = [mm for mm in [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6] if mm in existing]
print(f"sparse anchors: {sparse}")

# anchor fields
Fv = np.full((int(val_ta.max())+2, n_cells), np.nan, dtype=np.float32)
Fv[val_ta, val['cc'].values] = val['TWS_t'].values
AF = {a: Fv[a] - mu_fit for a in sparse}
Dhat = np.nanmean(np.array([AF[a] for a in sparse]), axis=0)

# existing S field (from covs, val period)
Z = fitp[COVS].values.astype('float32')
yv = fitp['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Zval = val[COVS].values.astype('float32')
okv = np.isfinite(Zval).all(axis=1)
cov_est = np.full(len(val), np.nan, dtype=np.float32)
cov_est[okv] = np.column_stack([Zval[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for mm in np.sort(val['t_abs'].unique()):
    selm = (val_ta == mm) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[val['cc'].values[selm]] = cov_est[selm]
    cov_field[mm] = fm - mu_fit
S = np.nanmean(np.array([cov_field[mm] for mm in sorted(cov_field.keys())]), axis=0)

# trendex
Ftr = np.full((2016*12, n_cells), np.nan, dtype=np.float32)
Ftr[fitp['t_abs'].values, fitp['cc'].values] = fitp['TWS_t'].values
t_axis = np.arange(Ftr.shape[0], dtype=np.float64)
ok_t = np.isfinite(Ftr)
t_mat = np.where(ok_t, t_axis[:, None], np.nan)
tbar_c = np.nanmean(t_mat[:2013*12], axis=0)
td = t_mat[:2013*12] - tbar_c[None, :]
F64 = Ftr[:2013*12].astype(np.float64)
sxy = np.nansum(td*F64, axis=0); sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx>100)&(ok_t[:2013*12].sum(axis=0)>=24), sxy/np.where(sxx>0,sxx,1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c)*beta_c).astype(np.float32)

# LOO regression: AF_a ~ D-hat_-a + S + trendex  vs  + E_static channels
def loo_rmse(regressors, label):
    Xs, ys = [], []
    for a in sparse:
        dloo = np.nanmean(np.array([AF[b] for b in sparse if b != a]), axis=0)
        base = [dloo, S, trendex(a)]
        cols = []
        for r in regressors:
            if r == 'Eswvl4': cols.append(E_static[:, 3])
            elif r == 'Esd': cols.append(E_static[:, 4])
            elif r == 'Etp': cols.append(E_static[:, 5])
            elif r == 'Ee': cols.append(E_static[:, 6])
            elif r == 'Ero': cols.append(E_static[:, 7])
            elif r == 'Et2m': cols.append(E_static[:, 8])
        Xa = np.column_stack(base + cols)
        ok = np.isfinite(Xa).all(axis=1) & np.isfinite(AF[a])
        Xs.append(Xa[ok]); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(Xw.shape[1]), Xw.T@yw)
    pred = Xw @ w_
    rmse = float(np.sqrt(np.mean((pred-yw)**2)))
    print(f"  {label:<45} LOO RMSE={rmse:.4f}")
    return rmse

loo_rmse([], 'D-hat + S + trendex (current)')
for r in ['Eswvl4','Esd','Etp','Ee','Et2m']:
    loo_rmse([r], f'+ {r}')
loo_rmse(['Eswvl4','Esd','Etp','Ee','Et2m'], '+ all 5 static')
