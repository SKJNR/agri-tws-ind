"""
Round 4 reconciliation experiments.

R1: target = TWS_t(t+1) exactness — airtight (Dec->Jan included, duplicates checked,
    probable join bugs reproduced: file-order shift, no-consec-check shift)
R2: cov->D coupling reconciliation:
    (a) leave-one-anchor-out corr(cov-field(m), D-hat_{-m})
    (b) static vs monthly decomposition of cov-field
    (c) their protocol repro: per-cell-demeaned covs -> D-hat
    (d) artifact checks: corr(D-hat, mu_c), corr(static part, mu_c)
    (e) raw-cov ridge -> D-hat_{-m} (my protocol, their ridge form)
    (f) decisive: does cov-field add incremental info over D-hat for anchor fields?
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/round4_reconciliation.txt'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

# ================= R1: target exactness =================
P("=== R1: target = TWS_t(t+1)? airtight test ===")
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target'])
tr['time'] = pd.to_datetime(tr['time'])
for c in ['TWS_t','target']:
    tr[c] = pd.to_numeric(tr[c], errors='coerce').astype('float32')
tr['cc'] = (tr['lat'].round(2).astype(str)+'_'+tr['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month
tr['t_abs'] = (tr['ym']//100)*12 + (tr['ym']%100) - 1

dup = int(tr.duplicated(subset=['cc','t_abs']).sum())
P(f"duplicated (cell, absolute-month) rows: {dup}")

L = tr[['cc','t_abs','TWS_t','target']]
R = tr[['cc','t_abs','TWS_t']].rename(columns={'t_abs':'t_next','TWS_t':'TWS_next'})
mg = L.merge(R, left_on=['cc','t_abs'], right_on=['cc','t_next'], how='inner')
P(f"consecutive-month pairs found (Dec->Jan included): {len(mg)} of {len(L)} rows")
ok = (mg['TWS_next'].notna() & mg['target'].notna()).values
d = (mg['TWS_next'].values[ok] - mg['target'].values[ok]).astype(np.float64)
r = float(np.corrcoef(mg['TWS_next'].values[ok], mg['target'].values[ok])[0,1])
P(f"MERGE-ON-NEXT-MONTH: corr = {r:.6f}, nonzero diffs = {int((np.abs(d)>1e-6).sum())} / {len(d)}, max|diff| = {np.abs(d).max():.8f}")
dec = ((mg['t_abs'] % 12) == 11) & ok   # December -> January pairs
P(f"  Dec->Jan pairs alone: {int(dec.sum())}, max|diff| among them = {np.abs(d[dec.values]).max():.8f}")

# bug repro 1: file-order positional shift(-1), no sort/groupby
raw = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['TWS_t','target'])
for c in ['TWS_t','target']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
nxt_pos = raw['TWS_t'].shift(-1)
okp = (nxt_pos.notna() & raw['target'].notna()).values
dp = (nxt_pos.values[okp] - raw['target'].values[okp]).astype(np.float64)
P(f"[bug repro 1] FILE-ORDER shift(-1): corr = {np.corrcoef(nxt_pos.values[okp], raw['target'].values[okp])[0,1]:.4f}, "
  f"max|diff| = {np.abs(dp).max():.2f}, mean|diff| = {np.abs(dp).mean():.3f}")

# bug repro 2: sort by (cell,time), shift(-1), NO consecutiveness check
ts = tr.sort_values(['cc','t_abs'])
nxt2 = ts.groupby('cc')['TWS_t'].shift(-1)
gap = (ts.groupby('cc')['t_abs'].shift(-1) - ts['t_abs'])
okg = (nxt2.notna() & ts['target'].notna()).values
dg = (nxt2.values[okg] - ts['target'].values[okg]).astype(np.float64)
P(f"[bug repro 2] SORT+SHIFT, no consec check: corr = {np.corrcoef(nxt2.values[okg], ts['target'].values[okg])[0,1]:.4f}, "
  f"max|diff| = {np.abs(dg).max():.2f}, mean|diff| = {np.abs(dg).mean():.3f}, "
  f"pairs with month-gap>1: {int((gap[okg]>1).sum())} ({100*(gap[okg]>1).mean():.2f}%)")

# file row order
head = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time'], nrows=200000)
t = pd.to_datetime(head['time'])
nm = t.dt.to_period('M').nunique()
P(f"[file order] first 200k rows span {nm} distinct months -> file is {'TIME-major' if nm < 40 else 'CELL-major'}")

# ================= R2: cov -> D reconciliation =================
P("\n=== R2: cov->D coupling reconciliation ===")
tr2 = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
tr2['time'] = pd.to_datetime(tr2['time'])
for c in ['TWS_t']+COVS:
    tr2[c] = pd.to_numeric(tr2[c], errors='coerce').astype('float32')
tr2['cc'] = (tr2['lat'].round(2).astype(str)+'_'+tr2['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = int(tr2['cc'].max())+1
mu_c = tr2.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')

# global cov regression (RAW covs, as in F1)
Z = tr2[COVS].values.astype(np.float32)
yv = tr2['TWS_t'].values.astype(np.float32)
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# train per-cell cov climatology (for the demeaned protocol)
clim = tr2.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

# test
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = tr2[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['masked'] = test['TWS_t_masked'].astype(bool)
mfrac = test.groupby('ym')['masked'].mean()
anchors = sorted(mfrac[mfrac < 0.01].index.tolist())
all_test_yms = sorted(test['ym'].unique())

# anchor anomaly fields
AF = {}
for a in anchors:
    sel = (test['ym']==a) & (~test['masked'])
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[test['cc'].values[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
AFm = np.array([AF[a] for a in anchors])
Dhat = np.nanmean(AFm, axis=0)
D_loo = {a: np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0) for a in anchors}

# cov fields per test month (raw protocol)
Zt = test[COVS].values.astype(np.float32)
okt = np.isfinite(Zt).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in all_test_yms:
    selm = (test['ym'].values == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[test['cc'].values[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c

# (a) LOO honest corr — the number that matters
P("\n(a) leave-one-anchor-out: corr(cov-field(m), D-hat_{-m}) across cells")
for a in anchors:
    cf = cov_field[a]; dl = D_loo[a]
    okc = np.isfinite(cf) & np.isfinite(dl)
    P(f"  {a}: {np.corrcoef(cf[okc], dl[okc])[0,1]:.4f}")

# (b) static vs monthly decomposition
P("\n(b) static vs monthly decomposition of cov-field")
S = np.nanmean(np.array([cov_field[m] for m in all_test_yms]), axis=0)
okS = np.isfinite(S) & np.isfinite(Dhat)
P(f"  corr(static part S, D-hat) = {np.corrcoef(S[okS], Dhat[okS])[0,1]:.4f}")
for a in anchors:
    W = cov_field[a] - S
    okw = np.isfinite(W) & np.isfinite(Dhat) & np.isfinite(AF[a])
    rW = float(np.corrcoef(W[okw], Dhat[okw])[0,1])
    rF = float(np.corrcoef(W[okw], (AF[a]-Dhat)[okw])[0,1])
    P(f"  {a}: corr(W_m, D-hat) = {rW:.4f} | corr(W_m, F_m - D-hat) [fast/drift tracking] = {rF:.4f}")

# (c) their protocol: per-cell DEmeaned covs -> ridge -> D-hat_{-m}
P("\n(c) per-cell-demeaned covs (train climatology removed) -> ridge -> D-hat_{-m}")
Ztest_dem = Zt - clim[test['cc'].values]
for a in anchors:
    selm = (test['ym'].values == a) & okt
    Xd = np.column_stack([Ztest_dem[selm], np.ones(selm.sum())])
    dl = D_loo[a][test['cc'].values[selm]]
    okd = np.isfinite(Xd).all(axis=1) & np.isfinite(dl)
    cf_ = np.linalg.solve(Xd[okd].T@Xd[okd] + 10*np.eye(6), Xd[okd].T@dl[okd])
    r_ = float(np.corrcoef(Xd[okd]@cf_, dl[okd])[0,1])
    P(f"  {a}: ridge corr (demeaned covs) = {r_:.4f}")

# (d) artifact checks
P("\n(d) artifact checks")
okd2 = np.isfinite(Dhat) & np.isfinite(mu_c)
P(f"  corr(D-hat, mu_c) across cells = {np.corrcoef(Dhat[okd2], mu_c[okd2])[0,1]:.4f}")
P(f"  corr(static part S, mu_c) = {np.corrcoef(S[okS], mu_c[okS])[0,1]:.4f}")

# (e) raw-cov ridge -> D-hat_{-m} (my signal, their ridge form)
P("\n(e) RAW covs -> ridge -> D-hat_{-m} (same ridge, no demeaning)")
for a in anchors:
    selm = (test['ym'].values == a) & okt
    Xr = np.column_stack([Zt[selm], np.ones(selm.sum())])
    dl = D_loo[a][test['cc'].values[selm]]
    okr = np.isfinite(Xr).all(axis=1) & np.isfinite(dl)
    cf_ = np.linalg.solve(Xr[okr].T@Xr[okr] + 10*np.eye(6), Xr[okr].T@dl[okr])
    r_ = float(np.corrcoef(Xr[okr]@cf_, dl[okr])[0,1])
    P(f"  {a}: ridge corr (raw covs) = {r_:.4f}")

# (f) decisive: incremental value of cov-field over D-hat for anchor fields
P("\n(f) decisive: predicting anchor field F_m from D-hat_{-m} vs + cov-field(m)")
base_rmse, combo_rmse, ba, bb = [], [], [], []
for a in anchors:
    dl = D_loo[a]; cf = cov_field[a]; Fm = AF[a]
    okf = np.isfinite(dl) & np.isfinite(cf) & np.isfinite(Fm)
    # fit (b1,b2) on other anchors
    Xs, ys = [], []
    for b in anchors:
        if b == a: continue
        dlb, cfb, Fb = D_loo[b], cov_field[b], AF[b]
        okb = np.isfinite(dlb) & np.isfinite(cfb) & np.isfinite(Fb)
        Xs.append(np.column_stack([dlb[okb], cfb[okb]])); ys.append(Fb[okb])
    X = np.vstack(Xs); y = np.concatenate(ys)
    w = np.linalg.solve(X.T@X + 1e-3*np.eye(2), X.T@y)
    base = float(np.sqrt(np.nanmean((Fm[okf]-dl[okf])**2)))
    combo = float(np.sqrt(np.nanmean((Fm[okf]-np.column_stack([dl[okf],cf[okf]])@w)**2)))
    base_rmse.append(base); combo_rmse.append(combo); ba.append(w[0]); bb.append(w[1])
    P(f"  {a}: RMSE(D-hat only) = {base:.4f} -> RMSE(+cov-field) = {combo:.4f}   weights (D, cov) = ({w[0]:.3f}, {w[1]:.3f})")
P(f"  MEAN: base = {np.mean(base_rmse):.4f}, combo = {np.mean(combo_rmse):.4f}, improvement = {np.mean(base_rmse)-np.mean(combo_rmse):.4f}")
P(f"  mean weights: D = {np.mean(ba):.3f}, cov = {np.mean(bb):.3f}")

# calibration numbers for V1b Kalman
P("\n(calib) var(F_m - D-hat) = fast+noise var to be tracked:")
res_var = np.nanmean([np.nanvar(AF[a]-Dhat) for a in anchors])
P(f"  var(F - D-hat) = {res_var:.4f} (std {np.sqrt(res_var):.4f})")
W_fast = []
for a in anchors:
    W = cov_field[a] - S
    okw = np.isfinite(W) & np.isfinite(AF[a]-Dhat)
    W_fast.append(float(np.nanvar((AF[a]-Dhat)[okw])))
P(f"  var(fast residual after best cov linear explanation) ~ {np.mean(W_fast)*(1-0.30):.4f} (rough)")

with open(OUT, 'w') as f:
    f.write('\n'.join(log))
P(f"\nsaved -> {OUT}")
