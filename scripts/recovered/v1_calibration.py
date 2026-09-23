"""
Final V1 calibration checks:
  F1: Do test-era covariate fields carry the persistent pattern D?
      - cov-based TWS estimate per test month; corr with D-hat across cells
      - does train-era cov R2 (~0.25) transfer to test era? (check at anchors)
  F2: Proper 3-param fit of the two-component anchor model:
      r(k) = vD + lam*phi^k*(1-vD)   on all 15 anchor pairs
      -> gives the V1 masked-row coefficients beta_h = lam*phi^h
  F3: z-drift structure: anchor global means over time (D evolution)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/v1_calibration.txt'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms)}
T = len(yms)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)

# global cov->TWS regression (fit all train, as used in v3)
Z = train[COVS].values.astype(np.float32)
yv = train['TWS_t'].values.astype(np.float32)
Z1 = np.column_stack([Z, np.ones(len(Z))])
ok = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[ok].T@Z1[ok] + 1e-2*np.eye(6), Z1[ok].T@yv[ok])
P(f"global cov->TWS regression coefficients: {np.round(coef,4)}")

# ---- test ----
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
la = codes['lat'].values; lo = codes['lon'].values
pos = {(int(round(float(la[i])*1000)), int(round(float(lo[i])*1000))): i for i in range(len(la))}
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
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)

# cov-estimated field per test month
Zt = test[COVS].values.astype(np.float32)
okt = np.isfinite(Zt).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt[okt], np.ones(okt.sum())]) @ coef
# per-month cov-estimated ANOMALY field
cov_field = {}
for m in all_test_yms:
    selm = (test['ym'].values == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[test['cc'].values[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c

# F1a: corr of each test month's cov-field with D-hat (across cells)
P("\n=== F1: do test covariates carry the persistent pattern D? ===")
P("corr(cov-field(m), D-hat) across cells, per test month:")
for m in all_test_yms:
    cf = cov_field[m]
    okc = np.isfinite(cf) & np.isfinite(Dhat)
    r = float(np.corrcoef(cf[okc], Dhat[okc])[0,1])
    # also corr of cov-field with own anchor's true field if anchor month
    extra = ""
    if m in AF:
        af = AF[m]
        oka = np.isfinite(cf) & np.isfinite(af)
        ra = float(np.corrcoef(cf[oka], af[oka])[0,1])
        extra = f" | corr(cov-field, actual anchor field) = {ra:.4f}"
    P(f"  {m}: corr(cov,D) = {r:.4f}{extra}")

# F1b: variance of D-hat explained by cov field at masked months
P("\nvar ratio: var(cov-field)/var(anchor fields) and cov-field std:")
stds = [np.nanstd(cov_field[m]) for m in all_test_yms]
P(f"cov-field std by month: min={np.min(stds):.3f}, max={np.max(stds):.3f}, mean={np.mean(stds):.3f}")
P(f"anchor field std: {np.nanmean([np.nanstd(AF[a]) for a in anchors]):.3f}; D-hat std: {np.nanstd(Dhat):.3f}")

# F2: two-component fit r(k) = vD + lam*phi^k*(1-vD)
P("\n=== F2: two-component anchor-pair fit ===")
pairs = []
for i,a in enumerate(anchors):
    for j,b in enumerate(anchors):
        if j <= i: continue
        k = (b//100-a//100)*12 + (b%100-a%100)
        if k <= 0: continue
        xa, xb = AF[a], AF[b]
        okp = np.isfinite(xa) & np.isfinite(xb)
        r = float(np.corrcoef(xa[okp], xb[okp])[0,1])
        pairs.append((k, r))
ks = np.array([p[0] for p in pairs], float)
rs = np.array([p[1] for p in pairs], float)
best = None
for vD in np.arange(0.25, 0.75, 0.005):
    for ph in np.arange(0.40, 0.96, 0.005):
        for lam in np.arange(0.60, 1.01, 0.01):
            pred = vD + lam*(ph**ks)*(1-vD)
            sse = np.sum((rs-pred)**2)
            if best is None or sse < best[0]:
                best = (sse, vD, ph, lam)
sse, vD, ph, lam = best
P(f"best fit: var(D)/var(anchor) = {vD:.3f}, phi_fast = {ph:.3f}, lambda_fast = {lam:.3f}, fit-rmse = {np.sqrt(sse/len(rs)):.4f}")
P(f"implied anchor coefficient r(h) = vD + lam*phi^h*(1-vD):")
for h in range(1, 8):
    P(f"  h={h}: {vD + lam*(ph**h)*(1-vD):.4f}")
P(f"compare v3a (phi=0.97, lam=0.84): " + ", ".join(f"{0.84*0.97**h:.3f}" for h in range(1,8)))

# F3: anchor global means over time (z-drift)
P("\n=== F3: D drift (anchor global means) ===")
for a in anchors:
    P(f"  {a}: global mean anomaly = {np.nanmean(AF[a]):+.4f}")
# 2015-16 era D vs 2018 era D
D_early = np.nanmean(np.array([AF[a] for a in anchors if a < 201700]), axis=0)
D_late = np.nanmean(np.array([AF[a] for a in anchors if a >= 201800]), axis=0)
okd = np.isfinite(D_early) & np.isfinite(D_late)
P(f"corr(D_early, D_late) across cells = {np.corrcoef(D_early[okd], D_late[okd])[0,1]:.4f}")
P(f"std(D_early) = {np.nanstd(D_early):.3f}, std(D_late) = {np.nanstd(D_late):.3f}, std(diff) = {np.nanstd(D_late-D_early):.3f}")

with open(OUT, 'w') as f:
    f.write('\n'.join(log))
P(f"\nsaved -> {OUT}")
