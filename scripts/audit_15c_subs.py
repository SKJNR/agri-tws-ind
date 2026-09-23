"""
AUDIT 15-c part 3: SUBMISSION MECHANICS (item 3) + PREDICTION FORENSICS (item 4)
Reloads v8a/v8b/v8c (+v2b, v4a for forensics), verifies mechanics, analyzes diffs.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'; DL = '/home/z/my-project/download'

ss = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month
te['t_abs'] = (te['ym']//100)*12 + (te['ym']%100) - 1
msk = te['TWS_t_masked'].astype(bool).values
print(f"test rows={len(te):,}; masked={msk.sum():,} ({msk.mean():.1%}); unmasked={(~msk).sum():,}")
print(f"ID order == test row order: {(ss['ID'].values == pd.Series(np.arange(len(te)))).all() if False else 'check below'}")
# ID structural check: SS ids unique & count
print(f"SS: {len(ss):,} rows, {ss['ID'].nunique():,} unique IDs, cols={list(ss.columns)}")

# ================= 3. SUBMISSION MECHANICS =================
print("\n=== 3. SUBMISSION MECHANICS ===")
subs = {}
for tag in ['v8a','v8b','v8c','v2b','v4a']:
    o = pd.read_csv(f'{DL}/submission_{tag}.csv')
    subs[tag] = o
    idm = (o['ID'].values == ss['ID'].values).all()
    nn = o['Target'].isna().sum()
    t = o['Target'].values.astype(np.float64)
    t32 = t.astype(np.float32).astype(np.float64)
    rep_err = np.abs(t - t32)
    print(f"{tag}: rows={len(o):,} IDs_match={idm} NaN={nn} "
          f"range=[{t.min():.4f},{t.max():.4f}] mean={t.mean():.4f} std={t.std():.4f} "
          f"| float32 max_rep_err={rep_err.max():.3e} mean={rep_err.mean():.3e} "
          f"cols={list(o.columns)}")

# ================= 4. PREDICTION FORENSICS =================
print("\n=== 4. PREDICTION FORENSICS (per-row diffs) ===")
a = subs['v8a']['Target'].values
b2 = subs['v2b']['Target'].values
b4 = subs['v4a']['Target'].values
for name, d in [('v8a-v2b', a-b2), ('v8a-v4a', a-b4), ('v4a-v2b', b4-b2)]:
    print(f"{name}: mean={d.mean():+.4f} std={d.std():.4f} max|d|={np.abs(d).max():.4f} "
          f"p50|d|={np.median(np.abs(d)):.4f} p90|d|={np.percentile(np.abs(d),90):.4f} corr={np.corrcoef([a if name.startswith('v8a') else (b2 if 'v4a' in name else b4)][0], [b2 if 'v2b' in name else b4][0])[0,1]:.4f}")

ym = te['ym'].values
months = np.sort(te['ym'].unique())
print("\ndiff stats by test month (v8a-v2b):")
print(f"{'ym':>7} {'n':>7} {'mask%':>6} {'mean_d':>8} {'std_d':>7} {'p90|d|':>7} {'std_pred':>8} {'std_v2b':>8}")
for m in months:
    sel = ym == m
    d = (a-b2)[sel]
    print(f"{int(m):>7} {sel.sum():>7} {msk[sel].mean():>6.3f} {d.mean():>+8.4f} {d.std():>7.4f} "
          f"{np.percentile(np.abs(d),90):>7.4f} {a[sel].std():>8.4f} {b2[sel].std():>8.4f}")

print("\ndiff stats by mask status:")
for lbl, sel in [('masked', msk), ('unmasked', ~msk)]:
    d = (a-b2)[sel]
    print(f"  {lbl:8s} n={sel.sum():>7,} mean_d={d.mean():+.5f} std_d={d.std():.5f} max|d|={np.abs(d).max():.4f} "
          f"rmse_d={np.sqrt((d**2).mean()):.5f}")

# correlation of prediction magnitude with |diff|
anom = np.abs(a)  # raw magnitude
print("\ncorr(|v8a pred|, |v8a-v2b|) overall:", f"{np.corrcoef(anom, np.abs(a-b2))[0,1]:.4f}")
print("corr(|v8a pred|, |v8a-v4a|):", f"{np.corrcoef(anom, np.abs(a-b4))[0,1]:.4f}")
# by mask status
for lbl, sel in [('masked', msk), ('unmasked', ~msk)]:
    print(f"  corr(|pred|,|v8a-v2b|) {lbl:8s}: {np.corrcoef(anom[sel], np.abs(a-b2)[sel])[0,1]:.4f}")

# horizon analysis on masked rows
anchors = [201509, 201601, 201606, 201612, 201807, 201811]
anch_tab = np.array([(y//100)*12 + (y%100) - 1 for y in anchors])
ta = te['t_abs'].values
target_month = ta + 1  # target is TWS at next month
prev_anch = np.full(len(te), -1, dtype=np.int64)
for i in range(len(te)):
    idx = np.searchsorted(anch_tab, target_month[i], side='right') - 1
    prev_anch[i] = anch_tab[idx] if idx >= 0 else -1
horizon = target_month - prev_anch
msk_rows = np.where(msk)[0]
print("\nmasked-row |v8a-v2b| diff by horizon (months since prev anchor):")
print(f"{'h':>3} {'n':>8} {'mean|d|':>8} {'rmse_d':>8} {'std_d':>7} {'std_pred':>8}")
for h in np.unique(horizon[msk_rows]):
    sel = msk_rows[horizon[msk_rows] == h]
    d = (a-b2)[sel]
    print(f"{int(h):>3} {len(sel):>8,} {np.abs(d).mean():>8.4f} {np.sqrt((d**2).mean()):>8.4f} {d.std():>7.4f} {a[sel].std():>8.4f}")

# 2018-12 block anomaly check (fwd-only, no future anchor)
print("\nblock prediction std (v8a) by month — anomaly check for fwd-only 2018-12:")
for m in months:
    sel = ym == m
    print(f"  {int(m)}: masked_rows={int((sel&msk).sum()):>6,} pred_std_masked={a[sel&msk].std():.4f}" if (sel&msk).sum()>0 else f"  {int(m)}: masked_rows=0")

d18 = (ym == 201812) & msk
other = (ym != 201812) & msk
print(f"\n2018-12 masked pred std: {a[d18].std():.4f} (n={d18.sum():,})")
print(f"other masked pred std:   {a[other].std():.4f} (n={other.sum():,})")
# long-gap block 2017-01..06 (horizon 2..7 from 2016-12 anchor)
blk17 = (ym >= 201701) & (ym <= 201706) & msk
print(f"2017-01..06 masked pred std: {a[blk17].std():.4f} (n={blk17.sum():,})")
# what fraction of v8a-v2b RMSE change is in which block?
rmse_d_by_block = {
    '2016-02..2016-09 (after Jan-16 anchor)': (ym >= 201602) & (ym <= 201609) & msk,
    '2017-01..2017-06 (long gap)': blk17,
    '2018-12 (fwd-only)': d18,
}
tot = ((a-b2)**2).sum()
for k, sel in rmse_d_by_block.items():
    print(f"  share of sum d^2 in {k}: {((a-b2)[sel]**2).sum()/tot*100:.1f}%  (rows {sel.sum():,}, {sel.sum()/msk.sum()*100:.1f}% of masked)")

# unmasked-row diff concentration (k0 model changed in v8)
print(f"\nunmasked sum d^2 share: {((a-b2)[~msk]**2).sum()/tot*100:.1f}% of total")

print("\nDONE part 3")
