"""THE PHYSICS TEST: can the ERA5 water budget integrate TWS from anchors?

Generator insight: TWS(t) - TWS(anchor) should = sum of monthly (P - E - R) over the gap
(ECMWF sign convention: e is NEGATIVE for evaporation, so budget = tp + e - ro).

If corr holds, this is a fundamentally better masked-row predictor than phi^k decay:
  phi=0.74 decays to 0.16 by k=6; budget integration RETAINS accumulated signal.

Tests on TRAIN (where we know TWS at every month):
  T1: corr(dTWS_1mo, budget_1mo)          — same-month change tracking
  T2: corr(dTWS_k, cumsum budget_k) for k=1..8 — integration quality
  T3: RMSE of TWS(t) = TWS(anchor) + cumsum budget, vs pure persistence decay
  T4: per-cell regression slope of dTWS on budget (is it 1.0 like true physics?)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1

era5 = pd.read_parquet(f'{DATA}/era5_grid.parquet')
era5['budget'] = era5['tp'] + era5['e'] - era5['ro']   # m/day (ECMWF convention)
# scale to monthly total: x days in month
era5['y'] = era5['ym'] // 100
era5['m'] = era5['ym'] % 100
days = pd.Series(era5['ym']).map(lambda ym: {
    1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31
}[ym % 100]).values.astype(np.float32)
era5['budget_m'] = era5['budget'].values * days   # monthly water budget (m)

m = train.merge(era5[['cc','ym','budget_m','tp','e','ro','sd','swvl4','t2m']], on=['cc','ym'], how='inner')
print(f"merged: {len(m):,} rows ({len(m)/len(train)*100:.1f}% of train)")

# also merge next month budget (for target = TWS(t+1))
era5n = era5.copy()
era5n['t_abs'] = (era5n['ym']//100)*12 + (era5n['ym']%100) - 1
era5n2 = era5[['cc','t_abs' if 't_abs' in era5 else 'ym']].copy() if False else None
# build next-month key directly
# proper: ym_next from t_abs+1
t_abs_next = m['t_abs'].values + 1
ym_next = (t_abs_next//12)*100 + (t_abs_next%12) + 1
m['ym_next'] = ym_next
m = m.merge(era5[['cc','ym','budget_m']].rename(columns={'ym':'ym_next','budget_m':'budget_m_next'}), on=['cc','ym_next'], how='left')

# ---------------- T1: same-month change tracking ----------------
print("\n--- T1: corr(TWS_t - TWS_{t-1}, budget_m(t)) — need prev TWS; use target instead ---")
# target(t) = TWS(t+1); so target - TWS_t = change over next month
sub = m[['cc','target','TWS_t','budget_m_next','budget_m']].dropna()
d1 = (sub['target'] - sub['TWS_t']).values
r = np.corrcoef(d1, sub['budget_m_next'].values)[0,1]
print(f"corr(dTWS_1mo, budget(t+1)) = {r:+.4f}  (n={len(sub):,})")
# also current month budget (should be less relevant for next-month change)
r0 = np.corrcoef(d1, sub['budget_m'].values)[0,1]
print(f"corr(dTWS_1mo, budget(t))   = {r0:+.4f}")

# scale check: regression slope
slope = np.polyfit(sub['budget_m_next'].values, d1, 1)[0]
print(f"regression slope dTWS ~ a*budget(t+1): a = {slope:.3f} (physics says ~1.0 if units match)")

# ---------------- T2: k-month integration ----------------
print("\n--- T2/T3: anchor + budget integration on TRAIN ---")
# build cell x month matrices for a few anchor choices
yms = np.sort(m['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms)}
T = len(yms)
n_cells = int(m['cc'].max())+1

F = np.full((T, n_cells), np.nan, dtype=np.float32)
B = np.full((T, n_cells), np.nan, dtype=np.float32)
F[m['ym'].map(ym_to_i).values, m['cc'].values] = m['TWS_t'].values
B[m['ym'].map(ym_to_i).values, m['cc'].values] = m['budget_m'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms])

# choose anchors: Jan of each year 2004..2013 (train era, has D? no — train has trend+fast, no D.
# but budget integration should still work for the fast+trend part)
results = []
for k in [1,2,3,4,5,6,7,8]:
    cs, n = [], 0
    persist_err, budget_err, count = 0.0, 0.0, 0
    for anchor_ym in [2004*100+1, 2006*100+1, 2008*100+1, 2010*100+1, 2012*100+1]:
        ai = ym_to_i[anchor_ym]
        ti = ai + k
        if ti >= T: continue
        # target month must exist
        F_t = F[ti]; F_a = F[ai]
        # cumsum budget from anchor+1..t (the months elapsed)
        Bc = np.nansum(B[ai+1:ti+1], axis=0)   # sum of budgets over intervening months
        ok = np.isfinite(F_t) & np.isfinite(F_a) & np.isfinite(Bc)
        if ok.sum() < 100: continue
        persist_err += np.sum((F_a[ok] - F_t[ok])**2)
        budget_err += np.sum((F_a[ok] + Bc[ok] - F_t[ok])**2)
        count += ok.sum()
        cs.append(np.corrcoef(Bc[ok], (F_t - F_a)[ok])[0,1])
    if count > 0:
        results.append((k, np.mean(cs), np.sqrt(persist_err/count), np.sqrt(budget_err/count)))
        print(f"  k={k}: corr(cumbudget, dTWS)={np.mean(cs):+.4f} | persistence RMSE={np.sqrt(persist_err/count):.4f} "
              f"| anchor+budget RMSE={np.sqrt(budget_err/count):.4f} (n={count:,})")

print("\n--- T4: per-cell slope distribution (dF over 1 month vs budget) ---")
sub2 = m[['cc','t_abs','TWS_t','budget_m_next','ym_next']].dropna(subset=['budget_m_next'])
# get TWS at ym_next for slope calc
tw_next = m[['cc','ym_next','TWS_t']].rename(columns={'ym_next':'ym2','TWS_t':'TWS_next'})
# careful: TWS at ym_next — pull from F matrix instead
slopes = []
for cc in range(0, n_cells, 50):
    rows = sub2[sub2['cc']==cc]
    for _, row in rows.iterrows():
        pass
    break
# simpler: vectorized 1-month pairs using merge
pairs = m[['cc','ym','TWS_t','budget_m']].merge(
    m[['cc','ym','TWS_t']].rename(columns={'ym':'ym_next2'}), on='cc', how='inner')
# this is getting complicated; do it directly with the matrices
dF = F[1:] - F[:-1]      # change from month i to i+1
Bm = B[1:]                # budget of month i+1
ok = np.isfinite(dF) & np.isfinite(Bm)
# overall slope
x = Bm[ok]; y = dF[ok]
a, b = np.polyfit(x, y, 1)
print(f"overall 1-mo slope: dF = {a:.3f}*budget + {b:.5f}   (r={np.corrcoef(x,y)[0,1]:+.4f}, n={ok.sum():,})")

# per-cell slopes for a sample of cells
cell_ids = np.where(ok.sum(axis=0) > 60)[0]
sl = []
for c in cell_ids[::20]:
    xx = Bm[ok[:, c], c]; yy = dF[ok[:, c], c]
    if len(xx) > 30 and xx.std() > 0:
        sl.append(np.polyfit(xx, yy, 1)[0])
sl = np.array(sl)
print(f"per-cell slopes (n={len(sl)}): mean={sl.mean():.3f}, median={np.median(sl):.3f}, std={sl.std():.3f}, "
      f"frac in [0.5,2.0]={np.mean((sl>0.5)&(sl<2.0))*100:.0f}%")
