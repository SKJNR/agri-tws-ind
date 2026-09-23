"""
test_month_table.py — definitive per-t_abs structure of Test.csv.

For each t_abs month in the test: n rows, n masked, mask fraction, n cells.
Plus: which months our pipeline uses as anchors (mfrac<0.01), and which are
"near-anchors" (0.01 <= mfrac < 0.5) that we might be wasting.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['masked'] = test['TWS_t_masked'].astype(bool)
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1

g = test.groupby('t_abs').agg(n=('masked','size'), n_masked=('masked','sum')).reset_index()
g['frac'] = g['n_masked']/g['n']
g['ym_str'] = (g['t_abs']//12).astype(str) + '-' + ((g['t_abs']%12)+1).astype(str).str.zfill(2)
print(f"{'month':<9}{'t_abs':>7}{'n_rows':>9}{'masked':>9}{'frac':>8}  role")
anchors = []
for _, r in g.sort_values('t_abs').iterrows():
    if r['frac'] < 0.01: role = 'ANCHOR (we use)'
    elif r['frac'] < 0.5: role = f"NEAR-ANCHOR? {int(r['n']-r['n_masked']):,} visible cells UNUSED"
    else: role = 'masked month'
    if r['frac'] < 0.01: anchors.append(int(r['t_abs']))
    print(f"{r['ym_str']:<9}{r['t_abs']:>7}{int(r['n']):>9,}{int(r['n_masked']):>9,}{r['frac']:>8.3f}  {role}")

print(f"\nanchors our pipeline uses: {anchors}")
print(f"gaps between anchors: {np.diff(anchors).tolist()}")
tot = len(test)
print(f"\ntotal rows {tot:,} | masked {int(test['masked'].sum()):,} ({test['masked'].mean()*100:.1f}%)")
# visible cells in near-anchor months
na = g[(g['frac'] >= 0.01) & (g['frac'] < 0.5)]
if len(na):
    print(f"near-anchor months: {len(na)}, total visible cells there: {int((na['n']-na['n_masked']).sum()):,}")
else:
    print("no near-anchor months (all months either <1% or >50% masked)")
