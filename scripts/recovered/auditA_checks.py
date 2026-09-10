"""
AUDIT A3/A4 — leakage surface checks + submission format audit
A3: test-side k0 covs(t+1) availability, direct-copy surface over ALL rows,
    the 417 exception cells, val-side k0 covs(t+1) availability.
A4: submission_v17a/b vs SampleSubmission (bit-exact IDs/order, NaN/inf, ranges)
    + 1000-row spot check of target(t)=TWS_t(t+1) on train.
Outputs: /home/z/my-project/download/auditA_checks.txt
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
OUT = '/home/z/my-project/download/auditA_checks.txt'
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))

# ---------------- A4: submission format ----------------
P("=== A4: SUBMISSION FORMAT AUDIT ===")
samp = pd.read_csv(f'{DATA}/SampleSubmission (4).csv', dtype=str, keep_default_na=False)
P(f"SampleSubmission: {len(samp):,} rows, cols={list(samp.columns)}, "
  f"unique IDs={samp['ID'].nunique():,}, Target all-zero={ (samp['Target']=='0').all() }")
for tag in ['v17a', 'v17b']:
    sub = pd.read_csv(f'{DL}/submission_{tag}.csv', dtype={'ID':str}, keep_default_na=False)
    same_cols = list(sub.columns) == ['ID','Target']
    id_match = (sub['ID'].values == samp['ID'].values).all() if len(sub)==len(samp) else False
    tgt = pd.to_numeric(sub['Target'], errors='coerce')
    n_nan = int(tgt.isna().sum()); n_inf = int(np.isinf(tgt.values).sum())
    P(f"submission_{tag}: rows={len(sub):,} cols_ok={same_cols} ID_set&order_bit_exact={id_match} "
      f"NaN={n_nan} inf={n_inf} min={tgt.min():.3f} max={tgt.max():.3f} mean={tgt.mean():.4f} std={tgt.std():.4f} "
      f"|Target|>5: {(tgt.abs()>5).sum()}")
    # duplicate IDs?
    P(f"  duplicate IDs: {int(sub['ID'].duplicated().sum())}   ID dtype/whitespace anomalies: "
      f"{int((sub['ID'] != sub['ID'].str.strip()).sum())}")

# ---------------- A4b: target identity spot check (1000 random train rows) ----------------
P("\n=== A4b: target(t)=TWS_t(t+1) SPOT CHECK (1000 random train rows) ===")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target'])
train['time'] = pd.to_datetime(train['time'])
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
key = (np.round(train['lat']*10).astype(np.int64)+1000)*100000 + (np.round(train['lon']*10).astype(np.int64)+5000)
train['cc'] = key
rng = np.random.default_rng(42)
idx = rng.choice(len(train), 1000, replace=False)
smp = train.iloc[idx]
lookup = {(int(c), int(t)): v for c, t, v in zip(train['cc'], train['t_abs'], train['TWS_t'])}
hits, miss, mism = 0, 0, 0
for c, t, tg in zip(smp['cc'], smp['t_abs'], smp['target']):
    v = lookup.get((int(c), int(t)+1), None)
    if v is None: miss += 1
    elif abs(v - tg) < 1e-6: hits += 1
    else: mism += 1
P(f"1000 random rows: exact match={hits}, no (cell,t+1) row in train={miss}, MISMATCH={mism}")

# ---------------- A3: test-side leakage surface ----------------
P("\n=== A3: TEST-SIDE LEAKAGE SURFACE ===")
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['masked'] = test['TWS_t_masked'].astype(bool)
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['cc'] = (np.round(test['lat']*10).astype(np.int64)+1000)*100000 + (np.round(test['lon']*10).astype(np.int64)+5000)

# direct copy over ALL rows
vis = set(zip(test.loc[~test['masked'],'cc'].tolist(), test.loc[~test['masked'],'t_abs'].tolist()))
pres = set(zip(test['cc'].tolist(), test['t_abs'].tolist()))
n_copy_all = sum(1 for c, t in zip(test['cc'], test['t_abs']) if (int(c), int(t)+1) in vis)
n_pres_all = sum(1 for c, t in zip(test['cc'], test['t_abs']) if (int(c), int(t)+1) in pres)
P(f"ALL rows with (cell,t+1) VISIBLE in test (direct answer read): {n_copy_all:,}")
P(f"ALL rows with (cell,t+1) PRESENT (any state; covs(t+1) available): {n_pres_all:,}")

# k0 rows: covs(t+1) availability by month
unm = test[~test['masked']]
avail = unm.apply(lambda r: (int(r['cc']), int(r['t_abs'])+1) in pres, axis=1)
P(f"\nk0 (unmasked) rows: {len(unm):,}; covs(t+1) available: {int(avail.sum()):,} "
  f"({avail.mean()*100:.1f}%)  [worklog claim: 62,576/94,048 = 66.5%]")
P("k0 covs(t+1) availability by month:")
for m, grp in unm.groupby('t_abs'):
    av = sum(1 for c, t in zip(grp['cc'], grp['t_abs']) if (int(c), int(t)+1) in pres)
    P(f"  {int(m)//12}-{int(m)%12+1:02d}: {av:,}/{len(grp):,}")

# the exception rows (unmasked inside full-mask months)
gt = test.groupby('t_abs')['masked'].mean()
fullmask_months = [int(t) for t in gt.index[gt >= 0.99]]
exc = test[(test['t_abs'].isin(fullmask_months)) & (~test['masked'])]
P(f"\nexception rows (unmasked inside >=99%-masked months): {len(exc):,} "
  f"across {exc['t_abs'].nunique()} months, {exc['cc'].nunique()} distinct cells")
exc_cells = set(exc['cc'].tolist())
# consecutive-visibility overlaps (k=1): is any cell visible at BOTH m and m+1?
vis_by_m = {int(m): set(test.loc[(test['t_abs']==m) & (~test['masked']), 'cc'].tolist())
            for m in sorted(test['t_abs'].unique())}
P("consecutive-month VISIBLE-cell overlaps (k=1 direct-copy surface for k0 rows):")
for m in sorted(vis_by_m):
    if m+1 in vis_by_m:
        ov = vis_by_m[m] & vis_by_m[m+1]
        P(f"  {m//12}-{m%12+1:02d} -> {(m+1)//12}-{(m+1)%12+1:02d}: |vis(m)|={len(vis_by_m[m]):,} "
          f"|vis(m+1)|={len(vis_by_m[m+1]):,} overlap={len(ov)}")
# exception cells visible at anchors?
anchor_months = [int(t) for t in gt.index[gt < 0.01]]
anch_vis = set(test.loc[(test['t_abs'].isin(anchor_months)) & (~test['masked']), 'cc'].tolist())
P(f"exception cells also visible at >=1 anchor month: {len(exc_cells & anch_vis)}/{len(exc_cells)}")

# ---------------- A3: val-side k0 eval realism ----------------
P("\n=== A3: VAL k0 EVAL SET REALISM (P0 protocol) ===")
valt = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','TWS_t_masked'])
# recompute cal mask fractions
test_m = pd.to_datetime(valt['time']).dt.month
valt['masked'] = valt['TWS_t_masked'].astype(bool)
cal_frac = valt.groupby(test_m)['masked'].mean()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val['masked'] = val['time'].dt.month.map(cal_frac).values > 0.5
val_keys = set(zip(val['cc'].tolist(), val['t_abs'].tolist()))
vu = val[~val['masked']]
av = sum(1 for c, t in zip(vu['cc'], vu['t_abs']) if (int(c), int(t)+1) in val_keys)
P(f"val unmasked (k0-eval) rows: {len(vu):,}; covs(t+1) available within val: {av:,} ({av/len(vu)*100:.1f}%) "
  f"[test: 66.5%]")
# distribution of k0 eval rows by calendar month vs test
P(f"val k0 rows by calendar month: {dict(sorted(vu['time'].dt.month.value_counts().items()))}")
P(f"test k0 rows by calendar month: {dict(sorted(unm['time'].dt.month.value_counts().items()))}")

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
print(f"\nsaved {OUT}")
