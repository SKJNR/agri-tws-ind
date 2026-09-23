"""
test_mask_structure.py — inspect the REAL test masking pattern at partial months.

Questions:
  M1: at partial months (cal 1,6,7,9,12), WHICH cells are masked? random 50% or structured?
  M2: same cells masked every partial month, or different?
  M3: spatial pattern: masked cells clustered (blocks/stripes) or scattered?
  M4: how many test rows are masked-at-partial-month (the interp-eligible pool)?
  M5: neighbor visibility: for a masked cell at a partial month, is at least one
      of its 4/8 neighbors visible?
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['masked'] = test['TWS_t_masked'].astype(bool)
test['cal_mon'] = test['time'].dt.month
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['lat','lon','TWS_t'])
cells = train[['lat','lon']].drop_duplicates().reset_index(drop=True)
cells['cc'] = (cells['lat'].round(2).astype(str)+'_'+cells['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = len(cells)
lats = np.sort(cells['lat'].unique()); lons = np.sort(cells['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)
pos = {}
for _, r in cells.iterrows():
    i, j = lat_i[r['lat']], lon_i[r['lon']]
    cc_grid[i, j] = r['cc']; pos[(i,j)] = r['cc']

test = test.merge(cells, on=['lat','lon'], how='left')
assert test['cc'].notna().all()
test['cc'] = test['cc'].astype('int32')
ta = test['t_abs'].values; msk = test['masked'].values; cc = test['cc'].values
mon = test['cal_mon'].values

partial = [1, 6, 7, 9, 12]
full = [2, 3, 4, 5, 8]
anchor = [11]

# M1: fractions
print("M1: mask fraction by calendar month:")
for m in sorted(test['cal_mon'].unique()):
    f = msk[mon == m].mean()
    print(f"  month {m:2d}: {f:.3f}  (n={int((mon==m).sum()):,})")

# M2: consistency across partial months
print("\nM2: same cells masked at every partial month?")
masks = {}
for m in partial:
    rows = mon == m
    fm = np.zeros(n_cells, dtype=bool)
    fm[cc[rows]] = msk[rows]
    masks[m] = fm
    print(f"  month {m}: masked {fm.sum():,}/{n_cells}")
base = masks[partial[0]]
for m in partial[1:]:
    inter = (base & masks[m]).sum()
    print(f"  overlap month {partial[0]}&{m}: {inter:,} ({inter/max(base.sum(),1)*100:.1f}% of first)")

# M3: spatial structure of masked cells (month 1 example)
print("\nM3: spatial pattern (month 1):")
fm = masks[1].astype(np.float32)
g = np.full((NI, NJ), np.nan, dtype=np.float32)
for (i, j), c in pos.items():
    g[i, j] = fm[c]
# neighbor same-state fraction: for masked cells, how many of 4-neighbors also masked?
same, tot = 0, 0
gm = g == 1
gv = g == 0
for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
    gs = np.roll(gm, dj, axis=1)
    if di > 0: gs = np.vstack([np.zeros((di, NJ), bool), gs[:-di]])
    if di < 0: gs = np.vstack([gs[-di:], np.zeros((-di, NJ), bool)])
    ok = gm & gs
    same += ok.sum(); tot += gm.sum()
print(f"  P(4-neighbor also masked | cell masked): {same/tot:.3f}  (random 50% would be 0.5)")
# row/col pattern check
print(f"  masked fraction by lon parity: even {gm[:, 0::2].mean():.3f} vs odd {gm[:, 1::2].mean():.3f}")
print(f"  masked fraction by lat parity: even {gm[0::2, :].mean():.3f} vs odd {gm[1::2, :].mean():.3f}")

# M4: pool sizes
print("\nM4: test row pools:")
n_partial_masked = int((msk & np.isin(mon, partial)).sum())
n_full_masked = int((msk & np.isin(mon, full)).sum())
n_k0 = int((~msk).sum())
print(f"  masked @ partial months (INTERP-ELIGIBLE): {n_partial_masked:,} ({n_partial_masked/len(test)*100:.1f}%)")
print(f"  masked @ full months:                     {n_full_masked:,} ({n_full_masked/len(test)*100:.1f}%)")
print(f"  unmasked (k=0):                            {n_k0:,} ({n_k0/len(test)*100:.1f}%)")

# M5: neighbor visibility for masked cells at partial months
print("\nM5: neighbor visibility (month 1):")
gv_grid = g == 0
vis_cnt = np.zeros((NI, NJ), dtype=np.int32)
for di, dj in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
    gs = np.roll(gv_grid, dj, axis=1)
    if di > 0: gs = np.vstack([np.zeros((di, NJ), bool), gs[:-di]])
    if di < 0: gs = np.vstack([gs[-di:], np.zeros((-di, NJ), bool)])
    vis_cnt += gs.astype(np.int32)
masked_cells = np.where(fm.astype(bool))[0]
# map cc -> grid
cc_to_ij = {c: ij for ij, c in pos.items()}
cnts = [vis_cnt[cc_to_ij[c]] for c in masked_cells]
cnts = np.array(cnts)
print(f"  visible 8-neighbors for masked cells: mean {cnts.mean():.2f}, min {cnts.min()}, P10 {np.percentile(cnts,10):.0f}, zero {int((cnts==0).sum()):,}/{len(cnts):,}")
print("\nDone.", flush=True)
