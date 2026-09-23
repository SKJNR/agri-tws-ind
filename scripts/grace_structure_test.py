"""STRUCTURAL TEST: competition month coverage vs real GRACE month availability.

If competition train-gaps + test-anchor months == GravIS/COST-G valid (non-NaN) months,
the competition TWS is a real GRACE product and we can identify WHICH ONE.
"""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

ds = xr.open_dataset(f'{S}/gravis_tws_grid.nc')
tws = ds['tws'].values.astype(np.float64)   # (256, 180, 360)
tv = pd.to_datetime(ds['time'].values)
yms_gravis = [t.year*100+t.month for t in tv]
# valid months = any finite data over land
valid = []
for i in range(len(tv)):
    f = tws[i]
    land = f[np.isfinite(f)]
    valid.append(land.size > 1000)  # has real data
gravis_valid = set(np.array(yms_gravis)[valid])
gravis_all = set(yms_gravis)
print(f"GravIS months total: {len(gravis_all)}, valid: {len(gravis_valid)}")

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time'])
te['time'] = pd.to_datetime(te['time'])
te_ym = set(te['time'].dt.year*100 + te['time'].dt.month)
tr_ym = set(tr['ym'].unique())

# full calendar 2002-04..2018-12
full = set()
for y in range(2002, 2019):
    for m in range(1, 13):
        ym = y*100+m
        if ym >= 200204 and ym <= 201812:
            full.add(ym)

comp_months = tr_ym | te_ym
print(f"\ncompetition months (train+test): {len(comp_months)}")
print(f"calendar 200204..201812: {len(full)}")

print("\n--- months in calendar but NOT in competition ---")
missing_comp = sorted(full - comp_months)
print(missing_comp)
print(f"\n--- GravIS invalid months within 200204..201812 ---")
gravis_missing = sorted(full - gravis_valid)
print(gravis_missing)
print(f"\n--- overlap between comp-missing and gravis-missing ---")
print(sorted(set(missing_comp) & set(gravis_missing)))
print(f"comp-missing: {len(missing_comp)}, gravis-missing: {len(gravis_missing)}, shared: {len(set(missing_comp)&set(gravis_missing))}")

# reverse: months in comp but gravis-invalid?
print("\n--- competition months where GravIS is INVALID ---")
print(sorted(comp_months & set(gravis_missing)))

# also: test anchor months vs gravis validity
anchors = [201509, 201601, 201606, 201612, 201807, 201811]
masked = [201602, 201603, 201607, 201608, 201609, 201701, 201702, 201703, 201704, 201705, 201706, 201812]
print("\nanchors in gravis_valid:", {a: a in gravis_valid for a in anchors})
print("masked in gravis_valid:", {a: a in gravis_valid for a in masked})
