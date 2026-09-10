#!/usr/bin/env python3
"""
E1 BREAKTHROUGH TEST: Is D evolution driven by cumulative covariates?

Physics: TWS = total water storage. SPEI = standardized precip minus ET.
Cumulative water balance ~ storage change. If the generator (or real physics)
links D(c,t) - D(c,anchor) to cumulative SPEI/SOIL between anchor and t,
we can predict D continuously through the 2017 block (bwd gap 12-17 months,
50% of masked mass) -> could halve D error 0.42 -> podium LB.

Test A (test anchors, 15 pairs): dD(c) per cell vs mean covariates in (i,k].
Test B (train, high power): per-cell trend slope vs train-era mean covariates
       (train SLOW = trend; if trend ~ water balance, physics confirmed).
Test C (train window pairs): 2-yr slow changes vs cumulative covs.
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['t_abs'] = test['time'].str[:4].astype(int) * 12 + test['time'].str[5:7].astype(int) - 1
test['masked'] = test['TWS_t_masked'].astype(str).str.lower().isin(['true', '1'])
train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1

def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
for df in (train, test):
    df['cc'] = ckey(df['lat'], df['lon'])

base = train.groupby('cc').agg(mu=('TWS_t', 'mean')).reset_index()

# ================= TEST A: anchor-pair dD vs cumulative covariates =================
gm = test.groupby('t_abs').agg(n=('ID', 'count'), n_masked=('masked', 'sum'))
gm['vis'] = gm['n'] - gm['n_masked']
anchor_ts = sorted(gm[gm['vis'] > 1000].index.tolist())
print(f"Anchors: {[f'{t//12}-{t%12+1:02d}' for t in anchor_ts]}")

# covariate fields per test month (all rows visible for covs)
cov_month = test.groupby('t_abs')[COVS].mean()
print("\nGlobal monthly covariate means around anchors:")
print(cov_month.loc[anchor_ts].round(3).to_string())

# per-cell anchor anomaly fields
test_v = test[~test['masked']].merge(base, on='cc', how='left')
test_v['anom'] = test_v['TWS_t'] - test_v['mu']
anchor_fields = {t: test_v[test_v['t_abs'] == t].set_index('cc')['anom'] for t in anchor_ts}

# per-cell covariate means between month pairs
def cum_cov(i, k, mode='mean'):
    """mean covariate value over months in (i, k] at each cell"""
    months = [t for t in range(i + 1, k + 1) if t in test['t_abs'].unique()]
    sub = test[test['t_abs'].isin(months)]
    return sub.groupby('cc')[COVS].mean()

# static cov field S (test-era mean) for reference
S_static = test.groupby('cc')[COVS].mean()

print("\n===== TEST A: per-cell dD (anchor k - anchor i) vs mean covariates in (i,k] =====")
print(f"{'pair':>16} {'h':>3} | " + " | ".join(f"{c[:9]:>9}" for c in COVS))
results = []
for a in range(len(anchor_ts) - 1):
    i, k = anchor_ts[a], anchor_ts[a + 1]
    dD = (anchor_fields[k] - anchor_fields[i]).dropna()
    cc_common = cum_cov(i, k)
    cc_common = cc_common[cc_common.index.isin(dD.index)]
    dD_c = dD.loc[cc_common.index]
    cors = []
    for c in COVS:
        r = np.corrcoef(dD_c.values, cc_common[c].values)[0, 1]
        cors.append(r)
    results.append((i, k, cors, len(dD_c)))
    print(f"{i//12}-{i%12+1:02d}->{k//12}-{k%12+1:02d} {k-i:>3} | " +
          " | ".join(f"{r:>9.3f}" for r in cors))

# pooled: z-scored dD vs z-scored cov across all pairs
print("\n--- Pooled (z-scored within pair, all 5 consecutive pairs) ---")
all_zd, all_zc = [], {c: [] for c in COVS}
for i, k, cors, n in results:
    dD = (anchor_fields[k] - anchor_fields[i]).dropna()
    cc_common = cum_cov(i, k)
    cc_common = cc_common[cc_common.index.isin(dD.index)]
    dD_c = dD.loc[cc_common.index]
    zd = (dD_c - dD_c.mean()) / dD_c.std()
    all_zd.append(zd)
    for c in COVS:
        v = cc_common[c]
        all_zc[c].append((v - v.mean()) / v.std())
zd = pd.concat(all_zd)
print(f"n cells pooled: {len(zd):,}")
for c in COVS:
    zc = pd.concat(all_zc[c])
    r = np.corrcoef(zd.values, zc.values)[0, 1]
    print(f"  {c:16s}: pooled corr(dD, cov) = {r:+.3f}")

# Multi-cov regression on dD (consecutive pairs)
print("\n--- Ridge: dD ~ all covs (consecutive pairs, LOO by pair) ---")
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
X_all, y_all, pair_id = [], [], []
for pi, (i, k, cors, n) in enumerate(results):
    dD = (anchor_fields[k] - anchor_fields[i]).dropna()
    cc_common = cum_cov(i, k)
    cc_common = cc_common[cc_common.index.isin(dD.index)]
    dD_c = dD.loc[cc_common.index]
    X = cc_common[COVS].values
    Xs = (X - X.mean(0)) / X.std(0)
    y = ((dD_c - dD_c.mean()) / dD_c.std()).values
    X_all.append(Xs); y_all.append(y); pair_id.append(np.full(len(y), pi))
X_all = np.vstack(X_all); y_all = np.concatenate(y_all); pair_id = np.concatenate(pair_id)
# in-sample
for alpha in [0.1, 1.0]:
    m = Ridge(alpha=alpha).fit(X_all, y_all)
    r_in = np.corrcoef(m.predict(X_all), y_all)[0, 1]
    print(f"  Ridge(alpha={alpha}) in-sample R = {r_in:.3f}")
# LOO by pair
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
m = Ridge(alpha=1.0)
pred = cross_val_predict(m, X_all, y_all, groups=pair_id, cv=gkf)
r_loo = np.corrcoef(pred, y_all)[0, 1]
print(f"  Ridge LOO-by-pair R = {r_loo:.3f}  (var explained {r_loo**2*100:.0f}%)")

# ================= TEST B: train trend slope vs train-era covariates =================
print("\n===== TEST B: train per-cell trend slope vs train-era mean covariates =====")
tr = train[['cc', 't_abs', 'TWS_t'] + COVS].copy()
gb = tr.groupby('cc')
tr['t_c'] = tr['t_abs'] - gb['t_abs'].transform('mean')
tr['y_c'] = tr['TWS_t'] - gb['TWS_t'].transform('mean')
num = (tr['t_c'] * tr['y_c']).groupby(tr['cc']).sum()
den = (tr['t_c'] ** 2).groupby(tr['cc']).sum().clip(lower=1e-9)
slope = (num / den).rename('slope')
cov_mean_tr = tr.groupby('cc')[COVS].mean()
cmp = pd.concat([slope, cov_mean_tr], axis=1)
print(f"n cells: {len(cmp):,}; slope std: {cmp['slope'].std():.4f}/month")
for c in COVS:
    r = np.corrcoef(cmp['slope'].values, cmp[c].values)[0, 1]
    print(f"  corr(slope, mean {c}) = {r:+.3f}")
Xb = (cov_mean_tr.values - cov_mean_tr.values.mean(0)) / cov_mean_tr.values.std(0)
yb = (slope.values - slope.mean()) / slope.std()
mb = Ridge(alpha=1.0).fit(Xb, yb)
print(f"  Ridge(all covs) in-sample R = {np.corrcoef(mb.predict(Xb), yb)[0,1]:.3f} "
      f"(var explained {np.corrcoef(mb.predict(Xb), yb)[0,1]**2*100:.0f}%)")

# ================= TEST C: train 2-yr slow changes vs cumulative covs =================
print("\n===== TEST C: train 24-month anomaly change vs cumulative covariates =====")
# per-cell: mean anomaly over months [t+1, t+24] minus mean over [t-23, t]
# vs cumulative (sum) covariates over (t, t+24]
cells = base['cc'].values
tr_idx = train.set_index(['cc', 't_abs'])
# build per-cell arrays on the common t grid
t_min, t_max = train['t_abs'].min(), train['t_abs'].max()
print(f"train t_abs range: {t_min}..{t_max} ({t_max-t_min+1} months)")
piv_tws = train.pivot_table(index='cc', columns='t_abs', values='TWS_t')
piv_cov = {c: train.pivot_table(index='cc', columns='t_abs', values=c) for c in COVS}
rng = np.random.default_rng(42)
sample_cells = rng.choice(len(cells), size=min(4000, len(cells)), replace=False)
gap = 24
starts = list(range(t_min, t_max - 2 * gap + 1, 12))  # non-overlapping-ish windows
dD_list = {c: [] for c in COVS}
dD_target = []
for s in starts:
    e = s + gap
    if e + gap > t_max:
        continue
    # pivot columns have GAPS (some t_abs months absent from train) — select only
    # existing labels, else KeyError on non-contiguous ranges
    c1 = [t for t in range(s, e + 1) if t in piv_tws.columns]
    c2 = [t for t in range(e + 1, e + gap + 1) if t in piv_tws.columns]
    c3 = [t for t in range(s + 1, e + gap + 1) if t in piv_tws.columns]
    if len(c1) < gap * 0.8 or len(c2) < gap * 0.8 or len(c3) < gap * 1.6:
        continue  # skip windows with big month gaps
    a1 = piv_tws.iloc[sample_cells].loc[:, c1]
    a2 = piv_tws.iloc[sample_cells].loc[:, c2]
    d_target = a2.mean(axis=1).values - a1.mean(axis=1).values
    cum = piv_cov['SPEI_12_t'].iloc[sample_cells].loc[:, c3].mean(axis=1).values
    ok = ~np.isnan(d_target) & ~np.isnan(cum)
    dD_target.append(d_target[ok])
    dD_list['SPEI_12_t'].append(cum[ok])
dD_target = np.concatenate(dD_target)
cum12 = np.concatenate(dD_list['SPEI_12_t'])
r = np.corrcoef(dD_target, cum12)[0, 1]
print(f"n pairs: {len(dD_target):,}")
print(f"corr(24m anomaly change, mean SPEI_12 over window) = {r:+.3f}")
# also SOIL
print("(SPEI_12 is the top physics candidate; others checked in anchor test)")

print("\n===== VERDICT =====")
print("If pooled/LOO R > 0.5 in TEST A: cumulative covariates explain D evolution")
print("=> build D(c,t) = era-Dhat(c,t) + g(cum covs), the breakthrough lever.")
