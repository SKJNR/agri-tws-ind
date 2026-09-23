"""Diagnose v17a k=0 anomaly: linear vs LGB component on test, vs v4a reference."""
import numpy as np, pandas as pd, warnings
import lightgbm as lgb
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
mu_c = train.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
unm = ~test['masked'].values

def feats(df, has_target):
    cols = ['cc','t_abs','TWS_t']+(['target'] if has_target else [])+COVS
    L = df[cols].copy(); L['t_next'] = L['t_abs']+1
    R = df[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
    mg = L.merge(R, on=['cc','t_next'], how='left')
    ccm = mg['cc'].values
    X = np.column_stack([
        mg['TWS_t'].values - mu_c[ccm],
        *[mg[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
        *[mg[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    ]).astype(np.float32)
    y = (mg['target'].values - mu_c[ccm]).astype('float32') if has_target else np.full(len(mg), np.nan, np.float32)
    yr = mg['t_abs'].values // 12
    return X, y, yr, mg['TWS_t'].notna().values

Xtr, ytr, yr, ok = feats(train, True)
ok &= np.isfinite(ytr)
Xtr, ytr, yr = Xtr[ok], ytr[ok], yr[ok]
wtr = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype('float32')
Xev, _, _, ok_ev = feats(test, False)
sel = unm & ok_ev

# linear (v4 recipe, no intercept)
has_nxt_tr = np.isfinite(Xtr[:, 6:11]).all(axis=1)
sw = np.sqrt(wtr); Xtr0 = np.nan_to_num(Xtr, nan=0.0)
A = Xtr0[has_nxt_tr]*sw[has_nxt_tr,None]
coefF = np.linalg.solve(A.T@A + 1e-3*np.eye(11), A.T@(ytr[has_nxt_tr]*sw[has_nxt_tr]))
colsR = np.array([0,1,2,3,4,5])
A_r = Xtr0[:, colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(6), A_r.T@(ytr*sw))
has_nxt_ev = np.isfinite(Xev[:, 6:11]).all(axis=1)
Xe = np.nan_to_num(Xev, nan=0.0)
p_lin = np.empty(len(Xev), np.float32)
p_lin[has_nxt_ev] = Xe[has_nxt_ev] @ coefF
p_lin[~has_nxt_ev] = Xe[~has_nxt_ev][:, colsR] @ coefR

# lgb
params = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
              min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              num_threads=2, seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
bst = lgb.train(params, lgb.Dataset(Xtr, label=ytr, weight=wtr), num_boost_round=400)
p_lgb = bst.predict(Xev)

# compare with v4a (known LB-era recipe output)
v4 = pd.read_csv(f'{DATA}/../download/submission_v4a.csv').set_index('ID')['Target'].values
sub_ids = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')['ID'].values
tmap = dict(zip(test['ID'], range(len(test))))
order = np.array([tmap[i] for i in sub_ids])
v4_k0 = v4[unm]; lin_k0 = p_lin[order][unm]; lgb_k0 = p_lgb[order][unm]
print(f"std: v4a={v4_k0.std():.4f}  linear={lin_k0.std():.4f}  lgb={lgb_k0.std():.4f}")
print(f"linear vs v4a: corr={np.corrcoef(lin_k0, v4_k0)[0,1]:.4f}  diff-RMSE={np.sqrt(((lin_k0-v4_k0)**2).mean()):.4f}")
print(f"lgb    vs v4a: corr={np.corrcoef(lgb_k0, v4_k0)[0,1]:.4f}  diff-RMSE={np.sqrt(((lgb_k0-v4_k0)**2).mean()):.4f}")
print(f"lgb    vs lin: corr={np.corrcoef(lgb_k0, lin_k0)[0,1]:.4f}  diff-RMSE={np.sqrt(((lgb_k0-lin_k0)**2).mean()):.4f}")
# feature distribution check on selected rows
print(f"\nfeature col0 (TWS anom): test k0 std={Xev[sel][:,0].std():.4f} vs train std={Xtr[:,0].std():.4f}")
for j, nm in enumerate(['SPEI1','SPEI3','SPEI6','SPEI12','SOIL']+['nxt_'+c[:5] for c in COVS]):
    print(f"  feat {j} ({nm:8s}): test k0 std={np.nanstd(Xev[sel][:,j]):.3f}  train std={np.nanstd(Xtr[:,j]):.3f}")
# where does lgb deviate most?
d = np.abs(lgb_k0 - lin_k0)
idx = np.argsort(-d)[:10]
print("\nlgb vs lin largest deviations (TWS anom feature):")
for i in idx:
    print(f"  TWS_anom={Xev[sel][i,0]:+.3f} lin={lin_k0[i]:+.3f} lgb={lgb_k0[i]:+.3f}")
