"""R1 fixed: correct next-month merge (L.t_abs+1 == R.t_abs), Dec->Jan included."""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target'])
tr['time'] = pd.to_datetime(tr['time'])
for c in ['TWS_t','target']:
    tr[c] = pd.to_numeric(tr[c], errors='coerce').astype('float32')
tr['cc'] = (tr['lat'].round(2).astype(str)+'_'+tr['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month
tr['t_abs'] = (tr['ym']//100)*12 + (tr['ym']%100) - 1

L = tr[['cc','t_abs','TWS_t','target']].copy()
L['t_next'] = L['t_abs'] + 1                      # look for next month's row
R = tr[['cc','t_abs','TWS_t']].rename(columns={'t_abs':'t_next','TWS_t':'TWS_next'})
mg = L.merge(R, on=['cc','t_next'], how='left')   # next month's TWS for same cell

found = mg['TWS_next'].notna() & mg['target'].notna()
print(f"rows with a genuine next-month row + target: {int(found.sum())} / {len(mg)}")
d = (mg['TWS_next'].values[found] - mg['target'].values[found]).astype(np.float64)
r = float(np.corrcoef(mg['TWS_next'].values[found], mg['target'].values[found])[0,1])
print(f"corr(TWS_t(t+1), target) = {r:.6f}")
print(f"nonzero |diff|>1e-6: {int((np.abs(d)>1e-6).sum())} / {len(d)}")
print(f"max|diff| = {np.abs(d).max():.8f}   mean|diff| = {np.abs(d).mean():.8f}")
dec = (mg['t_abs'].values[found] % 12) == 11
print(f"Dec->Jan pairs: {int(dec.sum())}, max|diff| among them = {np.abs(d[dec]).max():.8f}")

# exact replication code for the other AI
print("""
REPLICATION CODE (pandas):
  tr['t_abs'] = tr['ym']//100*12 + tr['ym']%100 - 1
  L = tr[['cc','t_abs','TWS_t','target']].copy(); L['t_next'] = L['t_abs']+1
  R = tr[['cc','t_abs','TWS_t']].rename(columns={'t_abs':'t_next','TWS_t':'TWS_next'})
  mg = L.merge(R, on=['cc','t_next'], how='left')
  (mg['TWS_next'] - mg['target']).abs().max()   # -> 0.0
""")
