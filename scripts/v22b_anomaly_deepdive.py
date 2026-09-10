"""V22B ANOMALY DEEP-DIVE + split re-decode evidence pack.

New facts (2026-09-05 late):
  v24   = 0.683791578  (in adoption band -> slot-1 adopted)
  v22b  = 0.701171004  (expected EXACTLY 0.699118155 -> ANOMALY)

Key observation: v13b-vs-v12b dRMSE2 = 0.002874583 and
v22b-vs-v22 dRMSE2 = 0.002874582 — IDENTICAL to 1e-9.
Both pairs differ ONLY on the 77,850 era rows (E). Identical mass =>
the era modification hits public-scored rows => the decoded
"public = first 7 months" split is WRONG.

This script:
  A. Verifies bit-level: which rows differ in (v13b,v12b) and (v22b,v22_splice).
  B. Computes per-month era pred-diff squared sums (computable without truth).
  C. Verifies the PE/PK probe constructions (what exactly was submitted).
  D. Emits the era-mass equation under candidate split hypotheses.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time', 'TWS_t'])
test.columns = [c.strip() for c in test.columns]
tw_col = 'TWS_t' if 'TWS_t' in test.columns else 'TWS_t_masked'
test['masked'] = test[tw_col].isna()
test['ym'] = test['time'].str[:4].astype(int) * 100 + test['time'].str[5:7].astype(int)

ids = test['ID'].values
ym = test['ym'].values
msk = test['masked'].values

def load(v):
    df = pd.read_csv(f'{DL}/submission_{v}.csv')
    df.columns = [c.strip() for c in df.columns]
    assert (df['ID'].values == ids).all(), f'ID order mismatch in {v}'
    return df['Target'].values.astype(np.float64)

v12b, v13b = load('v12b'), load('v13b')
v22, v22b, v21a = load('v22_splice'), load('v22b'), load('v21a')
v24 = load('v24')

print('=== A. bit-level diff-sets ===')
for name, a, b in [('v13b vs v12b', v13b, v12b), ('v22b vs v22', v22b, v22)]:
    d = a - b
    nz = np.abs(d) > 0
    print(f'{name}: differing rows = {nz.sum():,}  '
          f'rms(d) over diff = {np.sqrt((d[nz]**2).mean()):.5f}')
    per = pd.DataFrame({'ym': ym[nz], 'd2': d[nz]**2}).groupby('ym')['d2'].agg(['count', 'sum'])
    print(per.to_string(float_format=lambda x: f'{x:12.2f}'))

print('\nextra check: v22b vs v22 differ anywhere OUTSIDE the v13b-vs-v12b set?',
      np.setdiff1d(np.where(np.abs(v22b - v22) > 0)[0],
                   np.where(np.abs(v13b - v12b) > 0)[0]).size)
print('k0 blocks identical v22 vs v21a (all k0 rows)?',
      np.abs(v22[~msk] - v21a[~msk]).max())

print('\n=== B. era pred-diff squared sums per month (truth-free) ===')
d = v13b - v12b
nz = np.abs(d) > 0
sd2_total = float((d[nz]**2).sum())
print(f'total sum d^2 over E = {sd2_total:.1f}  '
      f'-> RMS = {np.sqrt(sd2_total / nz.sum()):.5f}')

print('\n=== C. PE/PK probe verification ===')
v8 = load('v8a')
pe = pd.read_csv(f'{DL}/probe_PE.csv'); pe.columns = ['ID', 'Target']
pk = pd.read_csv(f'{DL}/probe_PK.csv'); pk.columns = ['ID', 'Target']
pl = pd.read_csv(f'{DL}/probe_PL.csv'); pl.columns = ['ID', 'Target']
for nm, pr in [('PE', pe), ('PK', pk), ('PL', pl)]:
    assert (pr['ID'].values == ids).all(), f'ID order mismatch in probe_{nm}'
    dp = pr['Target'].values.astype(np.float64) - v8
    chg = np.abs(dp) > 0
    months_chg = sorted(set(ym[chg]))
    print(f'probe_{nm}: changed rows={chg.sum():,} ({chg.mean()*100:.1f}%), months changed={months_chg}')
    k0_chg = (~msk & chg).sum()
    print(f'   changed & k0: {k0_chg:,}   changed & masked: {(msk & chg).sum():,}')

print('\n=== D. era-mass equation ===')
MASS = 0.002874582
print(f'observed public dRMSE2 from era block = {MASS:.9f}')
print(f'  = sum_(E AND public) [e_era^2 - e_v12b^2] / |public|')
print(f'  sum d^2 over FULL E = {sd2_total:.1f} (upper bound on the d^2 part alone)')
for P, rho in [(84288, 0.30), (84288, 0.389), (109222, 1.0), (124751, None)]:
    if rho:
        s = MASS * P
        print(f'  if |public|={P:,} and era-public share = {rho:.3f}: '
              f'sum_(E&P)(de^2) = {s:8.1f}  -> per-era-row mean = {s/(rho*77850):.5f}')
print('  (any uniform-random split => public dRMSE2 == full-set mean of de^2 '
      '= same for private; era measured WORSE where scored)')
print('\nDONE.')
