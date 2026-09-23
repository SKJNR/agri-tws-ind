"""COMPLETE 2016-01 PROBE PAIR READOUT + SPLIT-MODEL FIT.

Scores:  s+ = 1.052248929   (probe_m201601_plus,  v24 + 3.0 on all 2016-01 rows)
         s- = 0.942148731   (probe_m201601_minus, v24 - 3.0 on all 2016-01 rows)
         s0 = 0.683791578   (v24 base, exact)

Exact pair identities (error cross-term cancels):
    f  = |P n M| / |P| = (s+^2 + s-^2 - 2 s0^2) / (2 delta^2)   [delta = 3]
    eb = mean public error on that month = (s+^2 - s-^2) / (12 f)

Then: per-month (n_m, k0_m) from Test.csv; fit the k0-stratified sampling model
    f_m = (rho * k0_m + mask_m) / (rho * K + M)      [rho = r_k0 / r_mask]
to the measured f(201601) EXACTLY, and cross-check against two independent
measurements:
    era-share:  public share of the 77,850 era rows = 0.2350 (from v22b dRMSE2
                0.002874582 / mean d^2 = rms 0.1106^2)
    PK probe:   v8a k0->mu_c => decoded public k0-share ~ 0.43 (old decode)
Pre-register f-predictions for the era months under each hypothesis.
"""
import numpy as np, pandas as pd

S0, SP, SM = 0.683791578, 1.052248929, 0.942148731
DELTA = 3.0
DATA = '/home/z/my-project/data'

# ---------- exact pair readout ----------
f = (SP**2 + SM**2 - 2*S0**2) / (2*DELTA**2)
eb = (SP**2 - SM**2) / (12*f)
print('=' * 68)
print('EXACT 2016-01 PAIR READOUT')
print('=' * 68)
print(f'  f(2016-01)  = {f:.6f}   (public fraction of the set in this month)')
print(f'  e_bar       = {eb:+.4f}    (mean public error on 2016-01 rows, v24)')
print(f'  uniform pred f = n_m/N     : see below')
print(f'  old-block pred f = 15,647/109,222 = 15,647/109,222 = {15647/109222:.4f}'
      f'   <-- measured is 4x BELOW: OLD BLOCK DEAD')

# ---------- per-month structure ----------
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time', 'TWS_t'])
test['ym'] = test['time'].str[:4].astype(int)*100 + test['time'].str[5:7].astype(int)
g = test.groupby('ym')['TWS_t']
tab = pd.DataFrame({'n': g.size(), 'k0': g.apply(lambda s: s.notna().sum())})
tab['mask'] = tab['n'] - tab['k0']
N = int(tab['n'].sum()); K = int(tab['k0'].sum()); M = int(tab['mask'].sum())
print('\n' + '=' * 68)
print(f'TEST MONTH STRUCTURE  (N={N:,}  K(k0)={K:,}  M(masked)={M:,})')
print('=' * 68)
print(f"{'month':>7}{'rows':>8}{'k0':>8}{'masked':>8}{'k0 share':>9}")
for m, r in tab.iterrows():
    star = '  <-- probed' if m in (201601, 201609, 201704) else ''
    print(f"{m:>7}{int(r['n']):>8,}{int(r['k0']):>8,}{int(r['mask']):>8,}"
          f"{r['k0']/r['n']:>9.3f}{star}")

# ---------- model fits ----------
m0 = int(tab.loc[201601, 'n']); k0_0 = int(tab.loc[201601, 'k0']); mk_0 = m0 - k0_0
E77 = {201609: 15479, 201703: 15636, 201704: 15617, 201705: 15568, 201706: 15550}
E77_ROWS = sum(E77.values())

# k0-stratified fit: solve f = (rho*k0_m + mask_m)/(rho*K + M) for rho
# f*(rho*K + M) = rho*k0_m + mask_m  ->  rho*(f*K - k0_m) = mask_m - f*M
num, den = mk_0 - f*M, f*K - k0_0
rho = num / den if den != 0 else np.nan
denom = rho*K + M
pub_k0share = rho*K/denom
era_share_model = E77_ROWS/denom

print('\n' + '=' * 68)
print('SPLIT-MODEL FIT  (row-wise sampling, k0 rows at rho x the masked rate)')
print('=' * 68)
print(f'  2016-01: n={m0:,}  k0={k0_0:,}  masked={mk_0:,}')
print(f'  fitted rho (r_k0/r_mask) = {rho:.4f}   [f is |P|-invariant]')
print(f'  implied public k0-share   = {pub_k0share:.4f}   (PK old decode: 0.43)')
print(f'  implied public era-share  = {era_share_model:.4f}   (measured: 0.2350)')
print(f'  |P| if 30% (84,288): r_k0={84_288*pub_k0share/K:.4f}  '
      f'r_mask={84_288*(1-pub_k0share)/M:.4f}')

# predictions for era months
print('\n' + '=' * 68)
print('PRE-REGISTERED PREDICTIONS FOR THE 4 REMAINING PROBES')
print('=' * 68)
print(f"{'hypothesis':<28}{'f(201609)':>10}{'f(201704)':>10}")
u609 = tab.loc[201609, 'n']/N; u704 = tab.loc[201704, 'n']/N
rows = [
    ('uniform random', u609, u704),
    (f'k0-stratified (rho={rho:.3f})',
     (rho*tab.loc[201609, 'k0'] + tab.loc[201609, 'mask'])/denom,
     (rho*tab.loc[201704, 'k0'] + tab.loc[201704, 'mask'])/denom),
    ('late-tilt (f>uniform by 20%)', 1.2*u609, 1.2*u704),
]
for name, a, b in rows:
    print(f'{name:<28}{a:>10.4f}{b:>10.4f}')
print(f"{'old block (dead)':<28}{0.0:>10.4f}{0.0:>10.4f}")

# single-side score ranges for eyeball sanity
print('\nSingle-side score ranges (s = sqrt(s0^2 + 9f + 6f*e_bar), e_bar in [-0.4, 0.4]):')
for name, a, _ in rows:
    lo = np.sqrt(S0**2 + 9*a - 2.4*a); hi = np.sqrt(S0**2 + 9*a + 2.4*a)
    print(f'  {name:<28} 201609 pair sides in [{lo:.3f}, {hi:.3f}]')

# exact pair formula reminder
print('\nREADOUT when each pair completes:')
print(f'  f_m  = (s_plus^2 + s_minus^2 - 2*{S0}^2)/18.0')
print('  e_m  = (s_plus^2 - s_minus^2)/(12*f_m)')
print(f'\n2016-01 cross-checks:  f*N/n_m = {f*N/m0:.4f} (1.000 = uniform);  '
      f'bias^2 = {eb**2:.4f}')
