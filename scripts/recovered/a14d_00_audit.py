"""a14d_00_audit.py — precise test structure audit for the k=0 subproblem.

Questions answered:
 1. The 18 test months, per-month mask fractions, unmasked counts.
 2. The 6 anchors + which have a next-month row in Test (covs(t+1) availability).
 3. Classification of all 94,048 unmasked rows into k=0 classes.
 4. Train month presence / cells-per-month (for spatial feature availability).
 5. Grid geometry (lat/lon spacing) for the neighbor graph.
 6. Train target availability by month.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['masked'] = test['TWS_t_masked'].astype(bool)
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['cal_mon'] = test['time'].dt.month

print("=== TEST months ===")
g = test.groupby(['ym','t_abs','cal_mon']).agg(
    n=('ID','size'), n_masked=('masked','sum'), n_unmasked=('masked', lambda s: (~s).sum()))
g['mask_frac'] = g['n_masked']/g['n']
print(g.to_string())

test_months = set(test['t_abs'].unique())
print(f"\ntest months (t_abs): {sorted(test_months)}")

# next-month availability for unmasked rows
test['t_next'] = test['t_abs'] + 1
next_months = test_months  # covs(t+1) exists iff t+1 in test file
test['has_next'] = test['t_next'].map(lambda m: m in next_months)

unm = test[~test['masked']]
print(f"\n=== unmasked rows: {len(unm):,} of {len(test):,} ({len(unm)/len(test)*100:.1f}%) ===")
cls = unm.groupby(['t_abs','has_next']).size().rename('n').reset_index()
# map t_abs to ym label
ym_of = test.drop_duplicates('t_abs').set_index('t_abs')['ym'].to_dict()
cls['month'] = cls['t_abs'].map(ym_of)
print(cls.to_string())

print("\n=== unmasked rows WITH next-month covs:", int(unm['has_next'].sum()),
      f"({unm['has_next'].mean()*100:.1f}% of unmasked) ===")
print("=== unmasked rows WITHOUT next-month covs:", int((~unm['has_next']).sum()), "===")

# anchors = months ~fully unmasked
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"\nanchors: {[ym_of[a] for a in anchors]}")

# partial months: unmasked cells pool
part = mfrac[(mfrac >= 0.01) & (mfrac < 0.99)].index
pool = set(test[(test['t_abs'].isin(part)) & (~test['masked'])]['ID'])
print(f"partial months: {len(part)}; unique unmasked pool rows: {len(pool)}")
# pool cells that also have next month in file
pm_rows = test[(test['t_abs'].isin(part)) & (~test['masked'])]
print(f"partial-month unmasked rows with next-month in file: {int(pm_rows['has_next'].sum())}")

# calendar-month mask fractions (the pattern used by cv_lb_correlation)
cal = test.groupby('cal_mon')['masked'].mean().sort_index()
print("\ncalendar-month mask fractions:")
print(cal.round(3).to_string())
print("=> unmasked calendar months (frac<=0.5):", sorted(cal[cal <= 0.5].index.tolist()))

# ---------------- train ----------------
print("\n=== TRAIN ===")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
print(f"rows: {len(train):,}  months present: {train['ym'].nunique()}  "
      f"range {train['ym'].min()}..{train['ym'].max()}")
per = train.groupby('ym').agg(n=('TWS_t','size'), n_target=('target', lambda s: s.notna().sum()))
full_n = per['n'].max()
print(f"cells per month: min={per['n'].min()} max={per['n'].max()} mode={per['n'].mode()[0]}")
print(f"months with < max cells: {(per['n'] < full_n).sum()} of {len(per)}")
print(f"rows with target: {train['target'].notna().sum():,}")
# months where next month missing in train (=> no target)
yms = np.sort(train['ym'].unique())
t_abs_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms])
gaps = [(int(yms[i]), int(yms[i+1])) for i in range(len(yms)-1) if t_abs_arr[i+1] != t_abs_arr[i]+1]
print(f"train month gaps: {len(gaps)}")
print(f"months whose next month absent: {len([1 for i in range(len(t_abs_arr)-1) if t_abs_arr[i+1] != t_abs_arr[i]+1])+1}")

# grid geometry
lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
print(f"\nlat: {len(lats)} unique, [{lats[0]}, {lats[-1]}], diffs unique: {np.unique(np.round(np.diff(lats),3))[:5]}")
print(f"lon: {len(lons)} unique, [{lons[0]}, {lons[-1]}], diffs unique: {np.unique(np.round(np.diff(lons),3))[:5]}")
print(f"n cells (lat,lon pairs): {train.groupby(['lat','lon']).ngroups}")

# val-window anchor structure check (2013-2015)
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]
vmonths = np.sort(val['ym'].unique())
print(f"\nval months present (2013-15): {len(vmonths)}: {[int(v) for v in vmonths]}")
