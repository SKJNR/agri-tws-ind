"""a14b_recon.py — quick structural reconnaissance for the factor-Kalman build.
Checks: month lists, gaps, fit/val split, val anchor months, full-cell counts, test months.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'])
train['time'] = pd.to_datetime(train['time'])
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
train['t_abs'] = (train['ym'] // 100) * 12 + (train['ym'] % 100) - 1
n_cells = train[['lat', 'lon']].drop_duplicates().shape[0]
print(f"train rows={len(train):,}  cells={n_cells}  months={train['ym'].nunique()}")

yms = np.sort(train['ym'].unique())
t_abs_m = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms])
print(f"t_abs range {t_abs_m.min()}..{t_abs_m.max()}  ({t_abs_m.max()-t_abs_m.min()+1} slots, {len(t_abs_m)} present)")
gaps = np.diff(t_abs_m)
print("gap sizes:", dict(zip(*np.unique(gaps, return_counts=True))))
# where are the big gaps?
for i, g in enumerate(gaps):
    if g > 1:
        print(f"  gap {g} after {yms[i]} -> {yms[i+1]}")

# per-month row counts
cnt = train.groupby('ym').size()
print(f"rows/month: min={cnt.min()} max={cnt.max()} n==15715: {(cnt == 15715).sum()}/{len(cnt)}")

# fit / val split
fit_mask = train['time'].dt.year <= 2012
val_mask = (train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)
print(f"\nfit rows={fit_mask.sum():,} months={train.loc[fit_mask,'ym'].nunique()}")
print(f"val rows={val_mask.sum():,} months={train.loc[val_mask,'ym'].nunique()}")
val_yms = np.sort(train.loc[val_mask, 'ym'].unique())
val_tabs = [(int(v)//100)*12 + (int(v)%100) - 1 for v in val_yms]
print("val months:", [f"{v//100}-{v%100:02d}" for v in val_yms])
vg = np.diff(val_tabs)
print("val month gaps:", dict(zip(*np.unique(vg, return_counts=True))))

# test months / anchors
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time', 'TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
test['ym'] = test['time'].dt.year * 100 + test['time'].dt.month
test['t_abs'] = (test['ym'] // 100) * 12 + (test['ym'] % 100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
mfrac = test.groupby('t_abs')['masked'].mean().sort_index()
print("\ntest months (t_abs, ym, mask_frac):")
for t, f in mfrac.items():
    ym = int(test.loc[test['t_abs'] == t, 'ym'].iloc[0])
    print(f"  {t}  {ym//100}-{ym%100:02d}  frac={f:.3f}  n={int((test['t_abs']==t).sum())}")

# calendar-month mask fractions (for CV protocol)
test['cal_mon'] = test['time'].dt.month
cal = test.groupby('cal_mon')['masked'].mean()
print("\ncalendar-month mask fractions:", {int(k): round(v, 3) for k, v in cal.items()})

# val anchors under CV protocol
val = train.loc[val_mask].copy()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal).values > 0.5
mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v < 0.5].index)
print(f"\nval masked rows: {int(val['masked'].sum()):,} ({val['masked'].mean()*100:.1f}%)")
print(f"val anchors ({len(anchors_v)}): {anchors_v}")
mm = sorted(set(val.loc[val['masked'], 't_abs']).tolist())
print(f"val masked months ({len(mm)}): {mm}")

# full-cell counts
tr_cc = (train['lat'].round(2).astype(str) + '_' + train['lon'].round(2).astype(str)).astype('category')
train['cc'] = tr_cc.cat.codes.astype('int32')
for name, m in [('fit', fit_mask), ('full-train', np.ones(len(train), bool))]:
    sub = train.loc[m]
    cntc = sub.groupby('cc').size()
    n_months = sub['ym'].nunique()
    print(f"{name}: months={n_months} cells with all months present={int((cntc == n_months).sum())}/{n_cells}")

# target NaN rate in val
vt = train.loc[val_mask, 'target']
print(f"\nval target NaN: {int(vt.isna().sum()):,}/{len(vt):,}")
