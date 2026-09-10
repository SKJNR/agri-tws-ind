"""
AUDIT 15-c part 2: MERGE CORRECTNESS in submission_v8.py (checklist item 2)
(a) train k0 merge on ['cc','t_next'] — Dec->Jan calendar correctness
(b) test covs(t+1) merge — 2018-12 / missing-next-month handling; full vs reduced counts
(c) pos{} rounding collision check (done in part 1, re-asserted here)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked']+COVS)
for df in (tr, te): df['time'] = pd.to_datetime(df['time'])

# cc codes exactly as in submission_v8.py
tr['cc'] = (tr['lat'].round(2).astype(str)+'_'+tr['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
codes = tr[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
te['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(te['lat'],te['lon'])]
print("(c) pos{} dict: unmatched test rows:", int((te['cc']<0).sum()), " (assert in v8 passes:", bool((te['cc']>=0).all()), ")")
print("    distinct train (lat,lon) ->", len(pos), "keys (0 collisions iff equal to 15715)")

for df in (tr, te):
    ym = df['time'].dt.year*100 + df['time'].dt.month
    df['t_abs'] = (ym//100)*12 + (ym%100) - 1

# ================= (a) TRAIN k0 merge =================
print("\n=== (a) train k0 merge on ['cc','t_next'] ===")
Lk = tr[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = tr[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
print("Rk uniqueness on (cc,t_next):", Rk.duplicated(subset=['cc','t_next']).sum(), "dup rows (must be 0 for 1:1 merge)")
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
print(f"row count Lk={len(Lk):,} -> merged={len(mgk):,}  (inflation = {len(mgk)-len(Lk):,}, must be 0)")

# verify every matched pair: merged row real t_abs == t_next (i.e. exact calendar next month)
matched = mgk[COVS[0]+'_nxt'].notna()
print(f"matched (has covs_nxt) rows: {int(matched.sum()):,} / {len(mgk):,}")

# Dec->Jan pairs in the merge
dec = (mgk['t_abs']%12==11)
print(f"Dec rows in Lk: {int(dec.sum()):,}; of those, matched to next Jan: {int((dec & matched).sum()):,}")
dj = mgk[dec & matched]
ok_cal = ((dj['t_next']%12)==0).all()  # next month is January
print(f"all Dec->t_next%12==0 (Jan): {bool(ok_cal)}")

# strongest possible check: for every matched row, pull the actual next-month TWS via
# independent calendar arithmetic and confirm target == TWS(next month) [known identity]
sub = mgk[mgk['target'].notna() & matched]
lookup = tr.set_index(['cc','t_abs'])['TWS_t']
key = pd.MultiIndex.from_arrays([sub['cc'], sub['t_next']])
tws_next = lookup.reindex(key)
diff = (sub['target'].values - tws_next.values)
print(f"target == TWS_t(t_next) on merged pairs: nonzero diffs = {int((np.abs(diff)>1e-6).sum()):,} / {len(sub):,}"
      f"  max|diff| = {np.nanmax(np.abs(diff)):.2e}")

# unmatched analysis: why unmatched? (gap month globally, cell missing that month, or last month)
unm = mgk[~matched]
tr_months = set(tr['t_abs'].unique())
unm['next_exists_globally'] = unm['t_next'].isin(tr_months)
print(f"\nunmatched rows: {len(unm):,}")
print(f"  next month absent from train entirely (gap/last month): {int((~unm['next_exists_globally']).sum()):,}")
print(f"  next month exists globally but cell missing it:        {int(unm['next_exists_globally'].sum()):,}")
has_nxt_tr = matched.values
print(f"has_nxt_tr (full-model train rows): {int(has_nxt_tr.sum()):,}  reduced: {int((~has_nxt_tr).sum()):,}")

# ================= (b) TEST covs(t+1) merge =================
print("\n=== (b) test covs(t+1) merge ===")
te['masked'] = te['TWS_t_masked'].astype(bool)
Lt = te[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = te[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c:c+'_nxt' for c in COVS}})
print("Rt uniqueness on (cc,t_next):", Rt.duplicated(subset=['cc','t_next']).sum(), "dup rows")
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
print(f"row count Lt={len(Lt):,} -> merged={len(mgt):,} (inflation 0 required)")
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
tw_ok = np.isfinite(mgt['TWS_t'].values)
msk = mgt['TWS_t_masked'].astype(bool).values if 'TWS_t_masked' in mgt else None
# recompute mask aligned to merged order (merge preserves left order — verify)
assert (mgt['cc'].values == te['cc'].values).all() and (mgt['t_abs'].values == te['t_abs'].values).all()
msk = te['masked'].values
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
neither  = (~msk) & ~tw_ok
print(f"unmasked rows total: {int((~msk).sum()):,}")
print(f"  use_full  (full k0 model, covs t+1):  {int(use_full.sum()):,}")
print(f"  use_red   (reduced k0 model):         {int(use_red.sum()):,}")
print(f"  neither (TWS_t NaN though unmasked!): {int(neither.sum()):,}")

# 2018-12 specifically
dec18 = mgt['t_abs'].values == (2018*12+11)
print(f"\n2018-12 rows: {int(dec18.sum()):,}; with t_next matched: {int((dec18 & has_nxt_te).sum())} (must be 0 — 2019-01 not in test)")
print(f"2018-12 unmasked rows: {int((dec18 & ~msk).sum())} -> all go to reduced model: {int((dec18 & use_red).sum())}")

# which unmasked rows lack t_next and why
unm_te = mgt[use_red]
by_month = unm_te.groupby('t_abs').size()
print("\nunmasked (k=0) rows WITHOUT covs(t+1), by month (t_abs -> ym):")
for t, n in by_month.items():
    print(f"  t_abs={int(t)} ({int(t)//12}-{int(t)%12+1:02d}): {int(n):,}")
print("\nunmasked (k=0) rows WITH covs(t+1), by month:")
for t, n in mgt[use_full].groupby('t_abs').size().items():
    print(f"  t_abs={int(t)} ({int(t)//12}-{int(t)%12+1:02d}): {int(n):,}")

# verify matched test pairs are exact calendar next (t_next == actual row t_abs)
mm = mgt[has_nxt_te]
chk = Rt.set_index(['cc','t_next']).index
print(f"\nsanity: t_next - t_abs == 1 for all matched test rows: {bool(((mm['t_next']-mm['t_abs'])==1).all())}")

# masked rows with next month present (used for nothing in k0, but check structure)
print(f"masked rows with has_nxt_te: {int((msk & has_nxt_te).sum()):,} (k0 features unused there)")

print("\nDONE part 2")
