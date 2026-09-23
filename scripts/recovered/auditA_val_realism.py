"""
AUDIT A2 — VAL-PROTOCOL REALISM AUDIT (Train + Test, structure only)
Recomputes the honest-CV val structure (fit<=2012, val=2013-15, calendar mask >0.5
exactly as rebuild_v17b.py does) and compares observation density / anchor geometry /
horizon distributions against the real Test structure from A1.
Outputs: /home/z/my-project/download/auditA_val_realism.txt
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/auditA_val_realism.txt'
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))

# ---------- TEST side (cheap reload) ----------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['masked'] = test['TWS_t_masked'].astype(bool)
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
cal_mask_frac = test.groupby(test['time'].dt.month)['masked'].mean()   # same object builder uses

gt = test.groupby('t_abs').agg(rows=('masked','size'), nmask=('masked','sum'))
gt['frac'] = gt['nmask']/gt['rows']
t_anchors = sorted(int(t) for t in gt.index[gt['frac'] < 0.01])
t_masked_months = sorted(int(t) for t in gt.index[(gt['frac'] >= 0.01)])
P("=== TEST STRUCTURE (recomputed; matches A1 census) ===")
P(f"calendar-month mask fractions used by builder: { {int(k): round(v,4) for k,v in cal_mask_frac.items()} }")
P(f"anchors: {[(a, f'{a//12}-{a%12+1:02d}') for a in t_anchors]}  gaps={[t_anchors[i+1]-t_anchors[i] for i in range(len(t_anchors)-1)]}")
P(f"masked months ({len(t_masked_months)}): {[(m, f'{m//12}-{m%12+1:02d}') for m in t_masked_months]}")
t_obs = int((~test['masked']).sum()); t_ncell = test[['lat','lon']].drop_duplicates().shape[0]
P(f"test unmasked obs: {t_obs:,} over {t_ncell} cells = {t_obs/t_ncell:.2f} obs/cell")

P("\ntest masked-month geometry (target month tm=m+1):")
for m in t_masked_months:
    prev_a = max((a for a in t_anchors if a < m), default=None)
    next_a = min((a for a in t_anchors if a > m), default=None)
    fwd = (m+1-prev_a) if prev_a is not None else None
    bwd = (next_a-(m+1)) if next_a is not None else None
    P(f"  {m//12}-{m%12+1:02d}: rows={int(gt.loc[m,'nmask']):>6,}  fwd(anchor->tm)={fwd}  bwd(next_anchor->tm)={bwd}")

# ---------- TRAIN side ----------
P("\n=== TRAIN CENSUS ===")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
ncell_tr = train[['lat','lon']].drop_duplicates().shape[0]
P(f"train rows: {len(train):,}  cells: {ncell_tr}  months {train['ym'].min()}..{train['ym'].max()}")
gm = train.groupby('t_abs').size()
span = np.arange(train['t_abs'].min(), train['t_abs'].max()+1)
absent_tr = [int(t) for t in span if t not in gm.index]
P(f"present months: {len(gm)} of {len(span)} in span; absent ({len(absent_tr)}): "
  f"{[(t, f'{t//12}-{t%12+1:02d}') for t in absent_tr]}")
sizes = gm.values
P(f"rows/present-month: min={sizes.min():,} p50={int(np.percentile(sizes,50)):,} max={sizes.max():,} (ncell={ncell_tr})")
part = {int(t): int(gm.loc[t]) for t in gm.index if gm.loc[t] < ncell_tr - 50}
P(f"PARTIAL train months (<ncell-50 rows): {len(part)}: {part}")

# ---------- VAL protocol structure (exactly as rebuild_v17b.py) ----------
P("\n=== VAL PROTOCOL STRUCTURE (fit<=2012, val=2013-15, cal_mask_frac>0.5 -> all-cells masked) ===")
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val['masked'] = val['time'].dt.month.map(cal_mask_frac).values > 0.5
gv = val.groupby('t_abs').agg(rows=('masked','size'), nmask=('masked','sum'))
gv['frac'] = gv['nmask']/gv['rows']
v_present = sorted(int(t) for t in gv.index)
v_anchors = sorted(int(t) for t in gv.index[gv['frac'] < 0.01])
v_masked = sorted(int(t) for t in gv.index[gv['frac'] >= 0.01])
P(f"val present months: {len(v_present)} of 36")
P(f"val ANCHOR months ({len(v_anchors)}): {[(a, f'{a//12}-{a%12+1:02d}') for a in v_anchors]}")
if len(v_anchors) > 1:
    P(f"val anchor gaps: {[v_anchors[i+1]-v_anchors[i] for i in range(len(v_anchors)-1)]}")
P(f"val MASKED months ({len(v_masked)}): {[(m, f'{m//12}-{m%12+1:02d}') for m in v_masked]}")
P(f"absent val months (of 36): {[(t, f'{t//12}-{t%12+1:02d}') for t in span if 2013*12 <= t <= 2015*12+11 and t not in v_present]}")
v_unmask_rows = int((~val['masked']).sum())
P(f"val unmasked rows: {v_unmask_rows:,} -> obs/cell = {v_unmask_rows/ncell_tr:.2f} (test: {t_obs/t_ncell:.2f}) "
  f"RATIO = {(v_unmask_rows/ncell_tr)/(t_obs/t_ncell):.2f}x")
P(f"val masked rows: {int(val['masked'].sum()):,} (test: 186,913)")

# per-cell obs density distribution
upc = val[~val['masked']].groupby(['lat','lon']).size()
P(f"val unmasked obs/cell dist: mean={upc.mean():.2f} p5={np.percentile(upc,5):.0f} p50={np.percentile(upc,50):.0f} "
  f"p95={np.percentile(upc,95):.0f} max={upc.max()}")
rpc = val.groupby(['lat','lon']).size()
P(f"val rows/cell dist: mean={rpc.mean():.2f} p50={np.percentile(rpc,50):.0f} max={rpc.max()}")

# honest rows & geometry
val_ta = val['t_abs'].values
shortcut = np.array([(t+1) in set(v_anchors) for t in val_ta])
honest = val['masked'].values & ~shortcut
P(f"\nval honest rows (masked & t+1 not an anchor month): {int(honest.sum()):,}")
hm = val[honest]
geo_counts = {}
for m, grp in hm.groupby('t_abs'):
    prev_a = max((a for a in v_anchors if a < m), default=None)
    next_a = min((a for a in v_anchors if a > m), default=None)
    fwd = (m+1-prev_a) if prev_a is not None else None
    bwd = (next_a-(m+1)) if next_a is not None else None
    P(f"  honest month {m//12}-{m%12+1:02d}: rows={len(grp):>6,}  fwd(anchor->tm)={fwd}  bwd(next_anchor->tm)={bwd}")
    geo_counts[(fwd,bwd)] = geo_counts.get((fwd,bwd),0) + len(grp)
P(f"val honest geometry mix (fwd,bwd)->rows: {geo_counts}")

# pooled horizon distributions for val honest rows (per-cell, like A1 test calc)
cell_unm = val[~val['masked']].groupby(['lat','lon'])['t_abs'].apply(lambda s: np.array(sorted(s)))
cu = {k: np.array(v) for k, v in cell_unm.items()}
hb = []; hf = []
for (la, lo), t in zip(hm[['lat','lon']].values, hm['t_abs'].values):
    arr = cu.get((la, lo))
    if arr is None or len(arr)==0:
        hb.append(-99); hf.append(-99); continue
    j = np.searchsorted(arr, t)
    hb.append(t - arr[j-1] if j > 0 else -99)
    hf.append(arr[j] - t if j < len(arr) else -99)
hb = np.array(hb); hf = np.array(hf)
P(f"\nval honest h_back (t - last unmasked, per-cell): {dict(sorted(pd.Series(hb[hb>0]).value_counts().items()))}  no-prior={int((hb<0).sum())}")
P(f"val honest h_fwd (next unmasked - t, per-cell): {dict(sorted(pd.Series(hf[hf>0]).value_counts().items()))}  no-next={int((hf<0).sum())}")

# ---------- D-hat anchor count & spacing ----------
P("\n=== D-HAT INPUT COMPARISON ===")
P(f"val: D-hat averaged over {len(v_anchors)} anchor fields spanning {v_anchors[-1]-v_anchors[0]} months")
P(f"test: D-hat averaged over {len(t_anchors)} anchor fields spanning {t_anchors[-1]-t_anchors[0]} months")

# ---------- ERA STATS ----------
P("\n=== ERA STATS (train, per-cell demeaned anomaly) ===")
mu = train.groupby(['lat','lon'])['TWS_t'].transform('mean')
train['anom'] = train['TWS_t'] - mu
per_year = train.groupby(train['time'].dt.year).agg(n=('anom','size'), std_anom=('anom','std'))
# persistence: RMSE of anom(t) vs anom(t-1) same cell (calendar-correct via sort+shift with cell check)
tr_sorted = train.sort_values(['lat','lon','t_abs'])
a = tr_sorted['anom'].values
prev_a = np.concatenate([[np.nan], a[:-1]])
same_cell = (tr_sorted['lat'].values[1:] == tr_sorted['lat'].values[:-1]) & \
            (tr_sorted['lon'].values[1:] == tr_sorted['lon'].values[:-1])
gap = np.concatenate([[99], tr_sorted['t_abs'].values[1:] - tr_sorted['t_abs'].values[:-1]])
ok = same_cell & (gap[1:] == 1)
yr = tr_sorted['time'].dt.year.values
pers = {}
for y in range(2003, 2016):
    m = ok & (yr[1:] == y)
    pers[y] = float(np.sqrt(np.mean((a[1:][m] - prev_a[1:][m])**2)))
for y in range(2002, 2016):
    P(f"  {y}: n={int(per_year.loc[y,'n']):>8,}  std_anom={per_year.loc[y,'std_anom']:.4f}  "
      f"persistence_RMSE={pers.get(y, float('nan')):.4f}")

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
print(f"\nsaved {OUT}")
