"""
AUDIT 15-c part 1: GRID FORENSICS + ANOMALY SWEEP (checklist items 1 and 6)
Writes only stdout. No deliverables.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
pd.set_option('display.width', 200)

# ---------- load minimal columns ----------
print("loading train lat/lon/time/TWS/masked-relevant cols...", flush=True)
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target'])
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
ss = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')

print(f"train rows={len(tr):,}  test rows={len(te):,}  ss rows={len(ss):,}")

# ================= 1. GRID FORENSICS =================
print("\n=== 1. GRID FORENSICS ===")
lats_tr = np.sort(tr['lat'].unique()); lons_tr = np.sort(tr['lon'].unique())
print(f"unique lat: {len(lats_tr)}  range [{lats_tr.min()}, {lats_tr.max()}]")
print(f"unique lon: {len(lons_tr)}  range [{lons_tr.min()}, {lons_tr.max()}]")
dlat = np.diff(lats_tr); dlon = np.diff(lons_tr)
print(f"lat spacing: unique diffs={np.unique(np.round(dlat,6))}  (min={dlat.min():.6f}, max={dlat.max():.6f})")
print(f"lon spacing: unique diffs={np.unique(np.round(dlon,6))}  (min={dlon.min():.6f}, max={dlon.max():.6f})")
print(f"lat monotonic in file? {tr['lat'].is_monotonic_increasing if False else 'n/a (row order)'}")

cells_tr = tr.groupby(['lat','lon']).size()
print(f"unique (lat,lon) cells in train: {len(cells_tr)}")
cells_te = te.groupby(['lat','lon']).size()
print(f"unique (lat,lon) cells in test:  {len(cells_te)}")
print(f"test cells absent from train: {len(set(cells_te.index) - set(cells_tr.index))}")
print(f"train cells absent from test:  {len(set(cells_tr.index) - set(cells_te.index))}")

# grid regularity: full product vs present cells
print(f"full lat x lon product would be {len(lats_tr)*len(lons_tr):,} cells -> {len(cells_tr):,} present "
      f"({100*len(cells_tr)/(len(lats_tr)*len(lons_tr)):.1f}%)")

# cells per lat band
per_lat = cells_tr.groupby(level=0).size()
print(f"\ncells per lat band: min={per_lat.min()}, max={per_lat.max()}, mean={per_lat.mean():.1f}")
print("  lat values with <max cell count (first 10):")
short = per_lat[per_lat < per_lat.max()]
print(short.head(10).to_string())
print(f"  (total lat bands with missing lon points: {len(short)})")

# global land check
land_05 = 148000  # ~global land count at 0.5deg (rough)
print(f"\n0.25deg global grid = {360/0.25*180/0.25:,.0f} cells; 0.5deg = {720*360:,}; 1deg = {360*180:,}")
print(f"our {len(cells_tr):,} cells -> consistent with 0.5deg land mask ({100*len(cells_tr)/259200:.1f}% of global 0.5deg)")
print(f"lat spans both hemispheres: {lats_tr.min()} to {lats_tr.max()}; lon full wrap: {lons_tr.min()} to {lons_tr.max()}")

# duplicate (lat,lon) rows within a single month? (structural duplicates)
dup_check = tr.duplicated(subset=['time','lat','lon']).sum()
print(f"\nduplicate (time,lat,lon) rows in TRAIN: {dup_check}")
dup_check_te = te.duplicated(subset=['time','lat','lon']).sum()
print(f"duplicate (time,lat,lon) rows in TEST:  {dup_check_te}")

# ---------------- test row ordering ----------------
print("\n--- TEST ROW ORDERING ---")
te2 = te.copy(); te2['time'] = pd.to_datetime(te2['time'])
te2['ym'] = te2['time'].dt.year*100 + te2['time'].dt.month
for name, keys in [('time,lat,lon', ['ym','lat','lon']),
                   ('time,lon,lat', ['ym','lon','lat']),
                   ('time,lat,lon (raw time)', ['time','lat','lon'])]:
    idx = np.lexsort(tuple(te2[k].values for k in reversed(keys)))
    ok = (idx == np.arange(len(te2))).all()
    print(f"  sorted by {name:26s}: {ok}")
# confirm ID order == test row order (cheap re-verify)
print(f"  SS ID order == test row order: {(ss['ID'].values == te.index.values.astype(str)).all() if False else 'skip (prior finding)'}")
# SS rows vs test rows one-to-one via position
print(f"  SS row count == test row count: {len(ss)==len(te)}")

# month blocks contiguous?
first_of_block = te2['ym'].ne(te2['ym'].shift()).cumsum()
print(f"  month blocks contiguous (18 blocks): {first_of_block.max() == te2['ym'].nunique()}")
print(f"  test months: {te2['ym'].nunique()}  from {te2['ym'].min()} to {te2['ym'].max()}")

# rows per month
rpm = te2.groupby('ym').size()
print(f"  rows per test month: min={rpm.min()}, max={rpm.max()}, n_months={len(rpm)}")
print(f"  months with <15715 rows: {(rpm<15715).sum()} -> {[f'{m}:{n}' for m,n in rpm[rpm<15715].items()]}")

# ================= 6. ANOMALY SWEEP =================
print("\n=== 6. ANOMALY SWEEP ===")

# (a) mask fraction per test month
te2['masked'] = te2['TWS_t_masked'].astype(bool)
mfrac = te2.groupby('ym')['masked'].agg(['mean','size'])
print("\nmask fraction per test month:")
for ym, row in mfrac.iterrows():
    print(f"  {int(ym)//100}-{int(ym)%100:02d}: {row['mean']:.4f}  (n={int(row['size']):,})")

# anchors
anch = mfrac[mfrac['mean'] < 0.01].index.tolist()
partial = mfrac[(mfrac['mean'] >= 0.01) & (mfrac['mean'] < 0.98)].index.tolist()
full_masked = mfrac[mfrac['mean'] >= 0.98].index.tolist()
print(f"\nanchors (mask<1%): {anch}")
print(f"partial months (1%-98%): {len(partial)}")
print(f"fully-masked months (>=98%): {len(full_masked)}")

# (b) TWS_t present at masked rows / missing at unmasked rows
tws_nan = te2['TWS_t'].isna() | (te2['TWS_t'].astype(str) == 'na')
# read TWS_t numerically
te3 = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['TWS_t','TWS_t_masked'])
te3['TWS_num'] = pd.to_numeric(te3['TWS_t'], errors='coerce')
msk = te3['TWS_t_masked'].astype(bool)
nan_tws = te3['TWS_num'].isna()
print(f"\nTWS_t NaN at MASKED rows:    {int((nan_tws & msk).sum()):,} / {int(msk.sum()):,}")
print(f"TWS_t NaN at UNMASKED rows:  {int((nan_tws & ~msk).sum()):,} / {int((~msk).sum()):,}  (should be 0)")
print(f"TWS_t PRESENT at masked rows: {int((~nan_tws & msk).sum()):,}  (should be 0)")

# (c) covariate NaN counts per column per file
covs = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
trc = pd.read_csv(f'{DATA}/Train (1).csv', usecols=covs)
tec = pd.read_csv(f'{DATA}/Test (2).csv', usecols=covs)
print("\ncovariate NaN counts:")
for c in covs:
    print(f"  {c:16s} train={trc[c].isna().sum():,}  test={tec[c].isna().sum():,}")
print(f"  target NaN in train: {tr['target'].isna().sum():,}")
print(f"  TWS_t  NaN in train: {tr['TWS_t'].isna().sum():,}")

# (d) train months present / gaps
tr2 = tr.copy(); tr2['time'] = pd.to_datetime(tr2['time'])
tr2['ym'] = tr2['time'].dt.year*100 + tr2['time'].dt.month
yms = np.sort(tr2['ym'].unique())
print(f"\ntrain months: {len(yms)} from {yms.min()} to {yms.max()}")
# expected months
all_yms = []
y, m = divmod(int(yms.min()), 100)
ye, me_ = divmod(int(yms.max()), 100)
cur = y*12+m-1; end = ye*12+me_-1
while cur <= end:
    all_yms.append((cur//100)*100 + (cur%100)+1); cur += 1
gaps = sorted(set(all_yms) - set(yms.tolist()))
print(f"expected months {len(all_yms)}; missing (gap) months: {len(gaps)} -> {gaps}")

# (e) train rows per month / cells per month distribution
trpm = tr2.groupby('ym').size()
print(f"train rows/month: min={trpm.min():,} max={trpm.max():,}")

# (f) rounded-key collision check for pos{} dict (audit item 2c support)
codes = tr[['lat','lon']].drop_duplicates()
keys = [(int(round(float(a)*1000)), int(round(float(b)*1000))) for a,b in zip(codes['lat'], codes['lon'])]
print(f"\npos{{}} key check: {len(keys)} distinct (lat,lon) -> {len(set(keys))} distinct rounded keys  "
      f"(collisions: {len(keys)-len(set(keys))})")
tkeys = [(int(round(float(a)*1000)), int(round(float(b)*1000))) for a,b in zip(te['lat'], te['lon'])]
tr_keyset = set(keys)
print(f"test rounded keys not in train keyset: {len(set(tkeys) - tr_keyset)}")

print("\nDONE part 1")
