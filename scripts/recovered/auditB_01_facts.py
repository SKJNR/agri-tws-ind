"""
AUDIT B - 01: cache data + verify structural facts assumed by rebuild_v17b.py.
Saves auditB_train.npz / auditB_test.npz for later scripts.
"""
import numpy as np, pandas as pd, time
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------- load train ----------
log("loading train...")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
log(f"train {len(train):,} rows, {n_cells} cells, months {train['t_abs'].min()}..{train['t_abs'].max()}")

# duplicates per (cc, t_abs)?
dup = train.duplicated(['cc','t_abs']).sum()
log(f"duplicate (cc,t_abs) rows in train: {dup}")

# rows per month over time (era structure)
rpm = train.groupby('t_abs').size()
log(f"rows/month: min={rpm.min()} max={rpm.max()} mean={rpm.mean():.0f}")
log(f"rows/month 2013-2015: {rpm[(rpm.index>=2013*12)&(rpm.index<=2015*12+11)].describe().to_dict()}")

# NaN census
for c in ['TWS_t','target']+COVS:
    log(f"  train {c}: NaN={train[c].isna().sum():,}")

# ---------- verify target == TWS(t+1) exactly ----------
log("verifying target == TWS(t+1)...")
key = train['cc'].astype('int64')*100000 + train['t_abs'].astype('int64')
twsp = train['TWS_t'].values
# build map (cc, t_abs) -> TWS
srt = np.lexsort((train['t_abs'].values, train['cc'].values))
k_sorted = train['cc'].values[srt].astype(np.int64)*100000 + train['t_abs'].values[srt].astype(np.int64)
v_sorted = twsp[srt]
lookup_k = train['cc'].values.astype(np.int64)*100000 + (train['t_abs'].values.astype(np.int64)+1)
idx = np.searchsorted(k_sorted, lookup_k)
idx_c = np.clip(idx, 0, len(k_sorted)-1)
found = (k_sorted[idx_c] == lookup_k) & np.isfinite(v_sorted[idx_c])
tgt = train['target'].values
ok = found & np.isfinite(tgt)
diff = np.abs(tgt[ok] - v_sorted[idx_c][ok])
log(f"  target vs TWS(t+1): n={ok.sum():,}  max|diff|={diff.max():.6g}  nonzero={(diff>1e-6).sum()}")
frac_missing_next = 1 - found.mean()
log(f"  rows whose t+1 not in train: {frac_missing_next*100:.2f}% (these have NaN target or target outside file)")

# ---------- load test ----------
log("loading test...")
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
assert (test['cc'] >= 0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
log(f"test {len(test):,} rows; masked={test['masked'].sum():,} ({test['masked'].mean()*100:.2f}%)")
log(f"test TWS NaN agreement with masked flag: {((test['TWS_t'].isna()) == test['masked']).all()}")
log(f"test cov NaN: {[int(test[c].isna().sum()) for c in COVS]}")
log(f"test duplicate (cc,t_abs): {test.duplicated(['cc','t_abs']).sum()}")

# cells in test not in train? (already asserted) / cells in train not in test:
log(f"cells in test: {test['cc'].nunique()}, cells in train: {n_cells}")

# per-t_abs census of test
tmf = test.groupby('t_abs')['masked'].agg(['size','sum','mean'])
anchors = sorted(int(v) for v in tmf[tmf['mean']<0.01].index)
log(f"TEST anchors (frac<0.01): {[(a, f'{a//12}-{a%12+1:02d}') for a in anchors]}")
log(f"TEST anchor gaps: {[anchors[i+1]-anchors[i] for i in range(len(anchors)-1)]}")

# calendar-month pooled mask fractions (drives val protocol!)
cal = test.groupby(test['time'].dt.month)['masked'].mean()
log("cal-month pooled mask fractions (val protocol threshold 0.5):")
for m, f in cal.items():
    log(f"  month {m:2d}: {f:.4f}  -> val {'MASKED' if f>0.5 else 'unmasked'}")
missing_cal = set(range(1,13)) - set(cal.index.tolist())
log(f"calendar months absent from test entirely: {sorted(missing_cal)} (-> NaN frac -> val unmasked)")

# ---------- val protocol replication (v17b exact logic) ----------
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val['masked'] = val['time'].dt.month.map(cal).values > 0.5
val_ta = val['t_abs'].values
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val_ta])
honest = val['masked'].values & ~shortcut
log(f"VAL: rows={len(val):,} masked={val['masked'].sum():,} shortcut={shortcut.sum():,} honest={honest.sum():,} (v17b claimed 93,829)")
vm = val[val['masked']]
log(f"VAL masked months: {sorted(set(zip(vm['t_abs']//12, vm['t_abs']%12+1)))}")
vh = val[honest]
log(f"VAL honest months: {sorted(set(zip(vh['t_abs']//12, vh['t_abs']%12+1)))}")
# anchor months in val
va = sorted(val_anchor_tas)
log(f"VAL anchor months (n={len(va)}): {[(t//12, t%12+1) for t in va]}")
# anchor gaps in val
log(f"VAL anchor gaps: {[va[i+1]-va[i] for i in range(len(va)-1)]}")

# ---------- k0 rows: covs(t+1) availability in TEST ----------
unm = ~test['masked'].values
key_t = test['cc'].values.astype(np.int64)*100000 + test['t_abs'].values.astype(np.int64)
srt_t = np.lexsort((test['t_abs'].values, test['cc'].values))
k_t_sorted = key_t[srt_t]
has_next = np.zeros(len(test), dtype=bool)
lookup = test['cc'].values.astype(np.int64)*100000 + (test['t_abs'].values.astype(np.int64)+1)
idx_t = np.searchsorted(k_t_sorted, lookup)
idx_tc = np.clip(idx_t, 0, len(k_t_sorted)-1)
has_next = (k_t_sorted[idx_tc] == lookup)
log(f"TEST k0 (unmasked) rows: {unm.sum():,}; with (cell,t+1) present in test: {has_next[unm].sum():,} (worklog claimed 62,576)")
log(f"TEST k0 rows lacking covs(t+1): {(~has_next[unm]).sum():,} (prompt claimed ~31,472)")

# direct-copy surface (v17b copy_vals logic)
vis = test.loc[unm & np.isfinite(test['TWS_t'].values), ['cc','t_abs','TWS_t']]
vis_map = {(int(c), int(t)): v for c, t, v in zip(vis['cc'], vis['t_abs'], vis['TWS_t'])}
keys = list(zip(test['cc'].tolist(), (test['t_abs']+1).tolist()))
copy_vals = np.array([vis_map.get(k, np.nan) for k in keys], dtype=np.float64)
n_copy = int(np.isfinite(copy_vals).sum())
n_copy_masked = int((np.isfinite(copy_vals) & test['masked'].values).sum())
n_copy_k0 = int((np.isfinite(copy_vals) & unm).sum())
log(f"direct-copyable rows: {n_copy} (masked: {n_copy_masked}, k0: {n_copy_k0})  [v17b docstring said ~379]")

# masked-row gap structure on TEST (to nearest unmasked obs of same cell, within test)
# build per-cell sorted arrays of unmasked t_abs
cell_unm = {}
tcc = test['cc'].values; tta = test['t_abs'].values; tmsk = test['masked'].values
for c, t, m in zip(tcc, tta, tmsk):
    if not m:
        cell_unm.setdefault(int(c), []).append(int(t))
for c in cell_unm: cell_unm[c] = np.sort(np.array(cell_unm[c]))
hb = np.full(len(test), -1); hf = np.full(len(test), -1)
for i, (c, t, m) in enumerate(zip(tcc, tta, tmsk)):
    if not m: continue
    arr = cell_unm.get(int(c))
    if arr is None: continue
    j = np.searchsorted(arr, t)
    if j > 0: hb[i] = t - arr[j-1]
    if j < len(arr): hf[i] = arr[j] - t
log("TEST masked-row: gap to prev unmasked (t-units): " + str(dict(pd.Series(hb[hb>=0]).value_counts().sort_index())))
log("TEST masked-row: gap to next unmasked (t-units): " + str(dict(pd.Series(hf[hf>=0]).value_counts().sort_index())))

# ---------- save cache ----------
np.savez_compressed('/home/z/my-project/scripts/auditB_train.npz',
    cc=train['cc'].values.astype(np.int32), t_abs=train['t_abs'].values.astype(np.int32),
    TWS=train['TWS_t'].values.astype(np.float32), target=train['target'].values.astype(np.float32),
    covs=train[COVS].values.astype(np.float32))
np.savez_compressed('/home/z/my-project/scripts/auditB_test.npz',
    cc=test['cc'].values.astype(np.int32), t_abs=test['t_abs'].values.astype(np.int32),
    TWS=test['TWS_t'].values.astype(np.float32), masked=test['masked'].values.astype(bool),
    covs=test[COVS].values.astype(np.float32))
# cell coordinates table
np.savez_compressed('/home/z/my-project/scripts/auditB_cells.npz',
    lat=codes['lat'].values.astype(np.float32), lon=codes['lon'].values.astype(np.float32))
log("cached npz files saved.")
log("DONE")
