"""Goldmine check: partial-month unmasked cells -> direct test-era lag-k correlations.
Settles lambda_test (noise fraction) and phi_test at k=1,2,3."""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['TWS_t'] = pd.to_numeric(train['TWS_t'], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
n_cells = int(train['cc'].max())+1
mu_c = train.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
test['TWS_t'] = pd.to_numeric(test['TWS_t'], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)

# unmasked observations per test month
obs = {}   # t_abs -> {cell: value}
for m, grp in test[~test['masked']].groupby('t_abs'):
    obs[int(m)] = dict(zip(grp['cc'].values, grp['TWS_t'].values))
print("unmasked cells per test month:")
for m in sorted(obs):
    print(f"  {m}: {len(obs[m])}")

months = sorted(obs)
print("\n=== consecutive/short-lag overlaps ===")
pairs_done = set()
for i, m1 in enumerate(months):
    for m2 in months[i+1:]:
        k = m2 - m1
        if k > 3 or k == 0: continue
        common = set(obs[m1]) & set(obs[m2])
        if len(common) < 8: continue
        cells = np.array(sorted(common))
        a1 = np.array([obs[m1][c] for c in cells]) - mu_c[cells]
        a2 = np.array([obs[m2][c] for c in cells]) - mu_c[cells]
        r = float(np.corrcoef(a1, a2)[0,1])
        print(f"  {m1} -> {m2} (k={k}): n={len(cells)}, r = {r:.4f}")

# also anchor -> partial (k=1)
print("\n=== anchor -> next-month partial overlaps ===")
for m1 in months:
    m2 = m1 + 1
    if m2 in obs and len(obs[m1]) > 1000 and len(obs[m2]) < 100:  # anchor -> partial
        common = set(obs[m1]) & set(obs[m2])
        if len(common) < 8: continue
        cells = np.array(sorted(common))
        a1 = np.array([obs[m1][c] for c in cells]) - mu_c[cells]
        a2 = np.array([obs[m2][c] for c in cells]) - mu_c[cells]
        r = float(np.corrcoef(a1, a2)[0,1])
        print(f"  {m1} -> {m2} (k=1): n={len(cells)}, r = {r:.4f}")

# same cells across ALL partial months? (fixed subset?)
print("\n=== cell overlap structure ===")
partial_months = [m for m in months if len(obs[m]) < 100]
if len(partial_months) >= 2:
    sets = [set(obs[m]) for m in partial_months]
    from itertools import combinations
    for (i, j) in combinations(range(len(partial_months)), 2):
        inter = len(sets[i] & sets[j])
        if inter > 0:
            print(f"  {partial_months[i]} & {partial_months[j]}: {inter} common cells")
    # union size
    print(f"  union of partial-month cells: {len(set().union(*sets))}")
