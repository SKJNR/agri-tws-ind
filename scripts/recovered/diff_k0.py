"""Diff my k0_linear vs the historical v4 recipe — coefficient level."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
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

# ===== historical mu_c: nanmean over field matrix =====
yms = np.sort(train['ym'].unique()); T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_hist = np.nanmean(F, axis=0)
clim_hist = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
# my mu_c: groupby mean
mu_mine = train.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).values.astype('float32')
print(f"mu diff: max|mu_hist-mu_mine| = {np.nanmax(np.abs(mu_hist-mu_mine)):.6f}  (nan cells hist: {np.isnan(mu_hist).sum()})")

# ===== historical coefficient fit (verbatim from submission_v4.py) =====
Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy()
Lk['t_next'] = Lk['t_abs'] + 1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values
yr = mgk['t_abs'].values // 12
Xf = np.column_stack([
    mgk['TWS_t'].values - mu_hist[ccm],
    *[mgk[c].values - clim_hist[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim_hist[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))
]).astype(np.float32)
yA = (mgk['target'].values - mu_hist[ccm]).astype('float32')
Xf = np.nan_to_num(Xf, nan=0.0)
w_rec = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype('float32')
sw = np.sqrt(w_rec)
selF = has_nxt_tr
A_f = Xf[selF]*sw[selF,None]
coefF_h = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
print("\nHISTORICAL coefF (12):", np.round(coefF_h, 4))

# ===== my version (no intercept, from diag script) =====
Xm = Xf[:, :11]
A_m = Xm[selF]*sw[selF,None]
coefF_m = np.linalg.solve(A_m.T@A_m + 1e-3*np.eye(11), A_m.T@(yA[selF]*sw[selF]))
print("MINE       coefF (11):", np.round(coefF_m, 4))
print("historical sans intercept:", np.round(coefF_h[:11], 4))

# ===== also: what did the historical actually PREDICT on test? reproduce and compare to v4a =====
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
msk = test['TWS_t_masked'].astype(bool).values
unm = ~msk
Lt = test[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = test[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
Xt = np.column_stack([
    mgt['TWS_t'].values - mu_hist[cct],
    *[mgt[c].values - clim_hist[cct, j] for j, c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim_hist[cct, j] for j, c in enumerate(COVS)],
    np.ones(len(mgt))
]).astype(np.float32)
Xt = np.nan_to_num(Xt, nan=0.0)
k0h = np.full(len(test), np.nan, dtype=np.float64)
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = unm & has_nxt_te & tw_ok
use_red  = unm & (~has_nxt_te) & tw_ok
colsR = [0,1,2,3,4,5,11]
# reduced model fit (historical): all rows, 7 cols incl intercept
A_r = Xf[:, colsR]*sw[:,None]
coefR_h = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))
k0h[use_full] = mu_hist[cct[use_full]] + Xt[use_full] @ coefF_h
k0h[use_red]  = mu_hist[cct[use_red]]  + Xt[use_red][:, colsR] @ coefR_h
bad = unm & np.isnan(k0h)
if bad.any(): k0h[bad] = mu_hist[cct][bad]

v4 = pd.read_csv(f'{DATA}/../download/submission_v4a.csv').set_index('ID')['Target'].values
tmap = dict(zip(test['ID'], range(len(test))))
sub_ids = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')['ID'].values
order = np.array([tmap[i] for i in sub_ids])
v4_k0 = v4[unm]; mine_k0 = k0h[order][unm]
print(f"\nreproduced-historical vs v4a.csv on k0 rows: corr={np.corrcoef(mine_k0, v4_k0)[0,1]:.5f}  max|diff|={np.abs(mine_k0-v4_k0).max():.6f}")
print(f"std: reproduced={mine_k0.std():.4f}  v4a.csv={v4_k0.std():.4f}")
