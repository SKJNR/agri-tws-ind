"""PHASE 0 for v8: state verification.
1. Train/Test date ranges (resolve the T=138 puzzle)
2. LEAK CHECK: do any test targets (cell, t+1) exist in Train.csv?  <-- the leader-gap suspect
3. Anchor months, test month list, masking fractions
4. Package availability (lightgbm)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])

tr_months = sorted(train['time'].dt.strftime('%Y-%m').unique())
te_months = sorted(test['time'].dt.strftime('%Y-%m').unique())
print(f"TRAIN months: {len(tr_months)}  {tr_months[0]} .. {tr_months[-1]}")
print(f"TEST  months: {len(te_months)}  {te_months[0]} .. {te_months[-1]}")
print("test months:", te_months)
gaps = [m for m in pd.period_range(tr_months[0], tr_months[-1], freq='M').strftime('%Y-%m') if m not in tr_months]
print("gaps inside train span:", gaps)
ov = set(tr_months) & set(te_months)
print("OVERLAP train∩test months:", sorted(ov) if ov else "NONE")

# ---- masking structure ----
test['masked'] = test['TWS_t_masked'].astype(bool)
mfrac = test.groupby(test['time'].dt.strftime('%Y-%m'))['masked'].agg(['mean','count'])
print("\nper-month mask fraction:")
print(mfrac.to_string())

# ---- LEAK CHECK: test target month = t+1; is (cell, t+1) observed in train? ----
# key: (lat, lon, year, month)
def keyify(df, col):
    return set(zip(df['lat'].round(2), df['lon'].round(2),
                   df['time'].dt.year, df['time'].dt.month))
tr_keys = keyify(train, 'TWS_t')

test['tgt_year'] = test['time'].dt.year + (test['time'].dt.month == 12).astype(int)
test['tgt_month'] = test['time'].dt.month % 12 + 1
tgt_keys = list(zip(test['lat'].round(2), test['lon'].round(2), test['tgt_year'], test['tgt_month']))
test['tgt_in_train'] = [k in tr_keys for k in tgt_keys]
print(f"\nLEAK CHECK: test rows whose TARGET (t+1) exists in Train: {test['tgt_in_train'].sum()} / {len(test)}")
if test['tgt_in_train'].any():
    sub = test[test['tgt_in_train']]
    print(sub.groupby(sub['time'].dt.strftime('%Y-%m')).size())
    # how many of those are masked rows?
    print("  of which masked:", sub['masked'].sum(), " unmasked:", (~sub['masked']).sum())

# also: do TEST cells' TWS_t at anchor months coincide with train values (sanity of synthetic gen)?
print("\npackages:", end=' ')
try:
    import lightgbm; print(f"lightgbm={lightgbm.__version__}", end=' ')
except Exception as e:
    print(f"lightgbm MISSING ({e})", end=' ')
try:
    import sklearn; print(f"sklearn={sklearn.__version__}")
except Exception as e:
    print(f"sklearn MISSING ({e})")

# quick LB arithmetic recap
print("\n--- implied component RMSE ---")
for name, lb in [('v2b', 0.7137), ('v6c(best)', 0.7030)]:
    for k0 in [0.64, 0.62, 0.60]:
        m2 = (lb**2 - 0.335*k0**2)/0.665
        print(f"{name}: LB={lb}, assumed k0={k0} -> implied masked RMSE={np.sqrt(m2):.4f}")
