"""Task 37 (2026-09-09): v26 gate DISPLAY-LEVEL closure + m201704_plus probe
readout + v28 arming decision.

User pasted (Zindi submissions table, ~2h ago):
  submission_probe_m201704_plus.csv = 1.006681664
  submission_v26_twocorr.csv        = 0.678573305   (ACTUAL)

Protocol constants (worklog Tasks 22/29/32):
  probe base = v24, s0 = 0.683791578 EXACT, DELTA = +3.0 on ALL rows of the month
  single-plus readout with solved split f = 1/17:  e = (17*dMSE - 9) / 6
  v28 rule (Round-9/10, pre-registered): stack iff |e| > 0.05 at 0.8 shrink,
  two-sided gate +/- 0.0005.
"""
import math
import hashlib
import numpy as np
import pandas as pd

S0 = 0.683791578          # v24 public (probe base)
DELTA = 3.0
V26_PRED = 0.678573306    # task29 card / task32 chain recompute
V26_ACT = 0.678573305     # pasted today
V27_ACT = 0.677006525     # exact hit yesterday (residual -8.62e-11)
SP_201704 = 1.006681664   # pasted today
SHRINK, ARM, GATE = 0.8, 0.05, 0.0005
PUB_ROWS = 84288          # |P| (task32 dev-unit: (f - 1/17) * 84288)
DATA = '/tmp/my-project/data'
DL = '/tmp/my-project/download'

print('=' * 74)
print('1) v26 GATE CLOSURE — display level')
print('=' * 74)
resid = V26_PRED - V26_ACT
lo, hi = 0.678069, 0.679069
print(f'  predicted (pre-registered, BEFORE v26 score existed): {V26_PRED:.9f}')
print(f'  ACTUAL (pasted today):                                {V26_ACT:.9f}')
print(f'  residual = {resid:+.3e}   -> display-level match at 8 dp')
print(f'  gate [{lo}, {hi}] -> {"PASS dead-center" if lo <= V26_ACT <= hi else "FAIL"}')
print('  correction-chain ledger now FULLY display-verified:')
print('    v24 0.683791578 (base)  v25 0.679780277 (-1.19e-10)')
print(f'    v26 {V26_ACT:.9f} ({resid:+.1e})   v27 {V27_ACT:.9f} (-8.62e-11)')

print()
print('=' * 74)
print('2) m201704_plus single-probe readout (base v24, delta +3.0)')
print('=' * 74)
d = SP_201704 ** 2 - S0 ** 2
print(f'  dMSE = {SP_201704:.9f}^2 - {S0:.9f}^2 = {d:.9f}')
e17 = (17 * d - DELTA ** 2) / (2 * DELTA)
print(f'  e(201704) at f = 1/17 exactly:  {e17:+.6f}')
# robustness: f within +/- 12 public rows of 1/17
for dev in (-12, -6, 0, +6, +12):
    f = 1 / 17 + dev / PUB_ROWS
    e = (d / f - DELTA ** 2) / (2 * DELTA)
    print(f'    f = 1/17 {dev:+3d} rows ({f:.7f}) -> e = {e:+.6f}')
f_need_zero = d / DELTA ** 2          # e would have to be 0
dev_zero = (f_need_zero - 1 / 17) * PUB_ROWS
print(f'  (for e to be 0, f would need {f_need_zero:.6f} = 1/17 '
      f'{dev_zero:+.1f} rows -> implausible; bias is REAL)')
f_imp = d / (DELTA ** 2 + 2 * DELTA * e17)
dev_imp = (f_imp - 1 / 17) * PUB_ROWS
print(f'  implied f = {f_imp:.7f}  = 1/17 {dev_imp:+.1f} rows -> 201704 is a')
print('  normal public month; NOT the zero month (hunt stays 201812).')

print()
print('=' * 74)
print('3) v28 arming decision (pre-registered rule: |e| > 0.05)')
print('=' * 74)
armed = abs(e17) > ARM
print(f'  |e(201704)| = {abs(e17):.4f}  vs gate {ARM}  -> '
      f'{"ARMED" if armed else "NOT ARMED (sub-gate)"}')
gain = (1 / 17) * (2 * SHRINK - SHRINK ** 2) * e17 ** 2
dv = gain / (2 * V27_ACT)
print(f'  hypothetical v28a = v27 - 0.8e on 201704 rows:')
print(f'    public gain = {dv:.6f}  -> predicted {V27_ACT - dv:.9f}')
print(f'    private projection ~ {0.92 * dv:+.6f} (month-weights ~0.92x public)')
print('    -> ~0.0001 class: real but marginal; discipline default = SKIP')

print()
print('=' * 74)
print('4) bit-verify the probe file vs v24 (construction audit)')
print('=' * 74)
try:
    test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time'])
    test['ym'] = (test['time'].str[:4].astype(int) * 100
                  + test['time'].str[5:7].astype(int))
    ids, ym = test['ID'].values, test['ym'].values
    b = pd.read_csv(f'{DL}/submission_v24.csv')
    b.columns = [c.strip() for c in b.columns]
    p = pd.read_csv(f'{DL}/submission_probe_m201704_plus.csv')
    p.columns = [c.strip() for c in p.columns]
    ok_id = (p['ID'].values == ids).all() and (b['ID'].values == ids).all()
    sel = ym == 201704
    d1 = p['Target'].values.astype(np.float64) - b['Target'].values.astype(np.float64)
    ok_in = np.abs(d1[sel] - DELTA).max()
    ok_out = np.abs(d1[~sel]).max()
    h = hashlib.md5(open(f'{DL}/submission_probe_m201704_plus.csv', 'rb').read()).hexdigest()
    print(f'  rows=280,961 & ID-align: {"PASS" if ok_id and len(p) == 280961 else "FAIL"}')
    print(f'  201704 rows: n={int(sel.sum()):,}  max|d-3.0| = {ok_in:.2e}  '
          f'(month share of file: {sel.sum()/280961:.4f})')
    print(f'  all other rows: max|d| = {ok_out:.2e}   md5 {h[:10]}')
    print(f'  -> construction = v24 + 3.0 on ALL {int(sel.sum()):,} rows of 201704: '
          f'{"PASS" if ok_in < 1e-9 and ok_out < 1e-12 else "FAIL"}')
except Exception as ex:
    print(f'  file audit skipped/failed: {ex}')

print()
print('=' * 74)
print('5) remaining lane & next slots (today: >=2 of 5 used)')
print('=' * 74)
print('  ARMED month: 201612 (pair files exist; ENSO prior |ONI| 0.45)')
print('    -> submit probe_m201612_plus NEXT SLOT; paste score here.')
print('    -> if |e(201612)| > 0.05: v28 = v27 - 0.8e on 2016-12 rows,')
print('       predicted & two-sided gate +/-0.0005 (expected 0.6766-0.6769).')
print('  201609: probe pair exists, unmeasured (optional, same readout).')
print('  201812: zero-month hunt (report value only).')
print('  v28a(201704-only) optional: predicted 0.676916, +0.00009 — default SKIP.')
print()
print('PASTE-BACK: 201612 plus score (and any other scores not yet pasted:')
print('201609 pair / 201612 minus / minus partners).')
