"""
Decisive legitimacy test: can ANCHOR-CALIBRATED per-cell cov->TWS mapping
explain masked-row RMSE ~0.62 (needed for public LB 0.6317) legitimately?

Test set has 6 fully-unmasked anchor months (2015-09, 2016-01, 2016-06,
2016-12, 2018-07, 2018-11) where BOTH TWS_t and covariates are visible.
If covariates encode the TWS latent state in the test era, a per-cell
cov->TWS regression calibrated on anchors could recover TWS at ANY test
month from that month's covariates -> including TARGET months of masked rows.

Protocol: Leave-One-Anchor-Out. Fit per-cell cov->TWS on 5 anchors,
predict the held-out anchor's TWS from its covariates. RMSE tells us the
achievable accuracy of TWS-from-covariates in the TEST ERA.
"""
import pandas as pd
import numpy as np

test = pd.read_csv('/home/z/my-project/data/Test (2).csv')
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

test['date'] = pd.to_datetime(test['time'])
test['y'] = test['date'].dt.year
test['m'] = test['date'].dt.month
test['t_abs'] = test['y'] * 12 + test['m'] - 1

# anchor months = fully unmasked
frac_masked = test.groupby('t_abs')['TWS_t_masked'].mean()
anchors = sorted(frac_masked[frac_masked < 0.5].index.tolist())
print('Anchor months (t_abs):', anchors)

# rows with visible TWS (anchors + partial-month cells)
vis = test[~test['TWS_t_masked'] & test['TWS_t'].notna()].copy()
print(f'Visible-TWS rows: {len(vis)} across months t_abs={sorted(vis.t_abs.unique())}')

# ---------- LOO over anchors: predict TWS at held-out anchor from its covs ----------
# per-cell linear map: TWS ~ a + b1*SPEI01 + b2*SPEI03 + b3*SPEI06 + b4*SPEI12 + b5*SOIL
# with >6 obs per cell impossible -> use POOLED global map + per-cell offset,
# and also a per-cell single-covariate (SOIL) map via shrinkage.
from sklearn.linear_model import Ridge

def loo_test(min_obs=4, use_per_cell_offset=True):
    rmses, ns = [], []
    for held in anchors:
        tr = vis[vis['t_abs'] != held]
        te = vis[vis['t_abs'] == held]
        # global ridge on covs
        Xtr, Xte = tr[COVS].values, te[COVS].values
        ytr, yte = tr['TWS_t'].values, te['TWS_t'].values
        r = Ridge(alpha=10.0).fit(Xtr, ytr)
        pred = r.predict(Xte)
        if use_per_cell_offset:
            # per-cell bias correction from training anchors
            tr2 = tr.copy(); tr2['resid'] = ytr - r.predict(Xtr)
            cell_bias = tr2.groupby(['lat','lon'])['resid'].mean()
            bias = te.set_index(['lat','lon']).index.map(cell_bias)
            bias = pd.Series(bias, index=te.index).fillna(0.0)
            # shrink the bias by number of obs
            cnt = tr2.groupby(['lat','lon'])['resid'].size()
            cn = te.set_index(['lat','lon']).index.map(cnt)
            cn = pd.Series(cn, index=te.index).fillna(0)
            shrink = cn / (cn + min_obs)
            pred = pred + bias * shrink
        rmses.append(np.sqrt(np.mean((pred - yte) ** 2)))
        ns.append(len(te))
    allr = np.sqrt(np.sum(np.array(rmses) ** 2 * ns) / np.sum(ns))
    return allr, rmses

g, per = loo_test(use_per_cell_offset=False)
print(f'\n[1] GLOBAL cov->TWS ridge (LOO anchor): RMSE = {g:.4f}')
g2, per2 = loo_test(use_per_cell_offset=True)
print(f'[2] + per-cell shrunk bias:             RMSE = {g2:.4f}')
print('    per-anchor RMSE:', [f'{x:.3f}' for x in per2])

# ---------- how good is the GLOBAL map per-anchor (no LOO, in-sample) ----------
r = Ridge(alpha=10.0).fit(vis[COVS].values, vis['TWS_t'].values)
pred = r.predict(vis[COVS].values)
print(f'\n[3] In-sample global map RMSE: {np.sqrt(np.mean((pred-vis.TWS_t.values)**2)):.4f}')

# ---------- correlation structure: does SOIL/SPEI track TWS in test era? ----------
for c in COVS:
    cc = np.corrcoef(vis[c].values, vis['TWS_t'].values)[0, 1]
    print(f'    corr({c}, TWS_t) at visible rows: {cc:+.4f}')

# ---------- spatial field correlation (month-mean fields) ----------
print('\n[4] Field-level: corr of monthly mean cov field vs monthly mean TWS field at anchors:')
for held in anchors:
    te = vis[vis['t_abs'] == held]
    for c in ['SOIL_MOISTURE_t', 'SPEI_12_t']:
        cc = np.corrcoef(te.groupby(['lat','lon'])[c].mean(), te.groupby(['lat','lon'])['TWS_t'].mean())[0,1]
        print(f'    anchor {held}: corr({c}, TWS) = {cc:+.4f}')

# ---------- KEY: train-era vs test-era per-cell cov->TWS strength ----------
print('\n[5] Per-cell (top-500 variance cells) cov->TWS correlation, train vs test anchors:')
train = pd.read_csv('/home/z/my-project/data/Train (1).csv',
                    usecols=['time','lat','lon','TWS_t','SOIL_MOISTURE_t','SPEI_12_t'])
train['y'] = pd.to_datetime(train['time']).dt.year
train['m'] = pd.to_datetime(train['time']).dt.month
train['t_abs'] = train['y']*12 + train['m'] - 1

cells = vis.groupby(['lat','lon'])['TWS_t'].std().sort_values(ascending=False).head(500).index
tr_corrs, te_corrs = [], []
for (la, lo) in cells:
    sub_tr = train[(train.lat==la)&(train.lon==lo)]
    if len(sub_tr) > 100:
        tr_corrs.append(np.corrcoef(sub_tr['SOIL_MOISTURE_t'], sub_tr['TWS_t'])[0,1])
    sub_te = vis[(vis.lat==la)&(vis.lon==lo)]
    if len(sub_te) >= 4:
        te_corrs.append(np.corrcoef(sub_te['SOIL_MOISTURE_t'], sub_te['TWS_t'])[0,1])
print(f'    train-era per-cell corr(SOIL,TWS): mean {np.mean(tr_corrs):+.4f} (n={len(tr_corrs)})')
print(f'    test-era  per-cell corr(SOIL,TWS): mean {np.mean(te_corrs):+.4f} (n={len(te_corrs)})')
