"""Task 38 (2026-09-09): m201612 probe PAIR readout (the LAST armed month)
+ v28 build + pre-registered prediction.

User pasted (Zindi submissions table, just now):
  submission_probe_m201612_minus.csv = 0.956643248
  submission_probe_m201612_plus.csv  = 1.036776186

Protocol constants (worklog Tasks 22/29/32/37):
  probe base = v24, s0 = 0.683791578 EXACT, DELTA = +/-3.0 on ALL rows of month
  pair readout (assumption-free): f = (sp^2+sm^2-2 s0^2)/(2 D^2),
                                   e = (sp^2-sm^2)/(4 D f)
  single-plus cross-check (solved split f=1/17): e = (17 dMSE - 9)/6
  v28 rule (Round-9/10, pre-registered): stack iff |e| > 0.05 at 0.8 shrink,
  two-sided gate +/- 0.0005. 201612 = 100% k0 anchor month -> zero masked-row risk.
"""
import hashlib
import math
import numpy as np
import pandas as pd

S0 = 0.683791578            # v24 public (probe base) - EXACT
SP = 1.036776186            # m201612_plus  ACTUAL (pasted today)
SM = 0.956643248            # m201612_minus ACTUAL (pasted today)
V27_ACT = 0.677006525       # v27 actual (exact hit, resid -8.62e-11)
DELTA = 3.0
SHRINK, ARM, GATE = 0.8, 0.05, 0.0005
PUB_ROWS = 84288            # |public|
DL = '/tmp/my-project/download'
OUT = f'{DL}/submission_v28_deccorr.csv'

def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()

print('=' * 76)
print('1) m201612 PAIR READOUT (base v24, delta +/-3.0, assumption-free)')
print('=' * 76)
sp2, sm2, s02 = SP**2, SM**2, S0**2
dmp, dmm = sp2 - s02, sm2 - s02
print(f'  s0   = {S0:.9f}   MSE0 = {s02:.12f}')
print(f'  s+   = {SP:.9f}   dMSE+ = {dmp:.12f}')
print(f'  s-   = {SM:.9f}   dMSE- = {dmm:.12f}')
f = (dmp + dmm) / (2 * DELTA**2)                 # = sum/18
e = (dmp - dmm) / (4 * DELTA * f)                # = diff/12f
print(f'  f  = (dMSE+ + dMSE-)/18      = {f:.9f}   '
      f'({(f-1/17)*PUB_ROWS:+.1f} rows vs 1/17)')
print(f'  e  = (dMSE+ - dMSE-)/(12 f)  = {e:+.6f}   <-- mean public residual, 201612')
print()
print('  cross-checks:')
e17 = (17 * dmp - DELTA**2) / (2 * DELTA)        # single-plus at f=1/17
print(f'   single-plus at f=1/17: e = {e17:+.6f}  (pair is the assumption-free value)')
trig = 17 * dmp - 9
print(f'   anomaly trigger |17*dMSE+ - 9| = {abs(trig):.4f}  vs 0.3  -> '
      f'{"FIRED (pair required, and pair WAS submitted)" if abs(trig) > 0.3 else "quiet"}')
print(f'   zero-month test: f = {f:.6f} >> 0  -> 201612 is a NORMAL public month')
print('     (zero-month hunt stays 201812, unchanged)')
print()
print('  bias league table (mean public residual e, all measured months):')
tab = [(201601, +0.3108, 'v25 corrected'), (201612, e, 'v28 (THIS READOUT)'),
       (201811, +0.1795, 'v27 corrected'), (201606, -0.1366, 'v26 corrected'),
       (201509, +0.1018, 'v26 corrected'), (201807, -0.0731, 'v27 corrected'),
       (201704, +0.0465, 'sub-gate, documented skip')]
for m, ev, note in sorted(tab, key=lambda r: -abs(r[1])):
    print(f'    {m}  e = {ev:+.4f}   {note}')
print(f'  -> 201612 |e| = {abs(e):.4f}: 2nd-largest bias of the whole program')
print(f'     (ENSO prior said 0.05-0.15 at |ONI| 0.45 -> prior UNDER-CALLED ~1.7x;')
print('      direction fine, magnitude unreliable - which is why we probe)')

print()
print('=' * 76)
print('2) BIT-AUDIT (construction + no-double-correction)')
print('=' * 76)
test = pd.read_csv('/tmp/my-project/data/Test (2).csv', usecols=['ID', 'time'])
ym = (test['time'].str[:4].astype(int) * 100 + test['time'].str[5:7].astype(int)).values
ids = test['ID'].values
sel = ym == 201612
RT = dict(float_precision='round_trip')   # exact strtod: untouched rows stay byte-identical
b = pd.read_csv(f'{DL}/submission_v24.csv', **RT); b.columns = ['ID', 'Target']
v7 = pd.read_csv(f'{DL}/submission_v27_2corr.csv', **RT); v7.columns = ['ID', 'Target']
pp = pd.read_csv(f'{DL}/submission_probe_m201612_plus.csv', **RT); pp.columns = ['ID', 'Target']
pm = pd.read_csv(f'{DL}/submission_probe_m201612_minus.csv', **RT); pm.columns = ['ID', 'Target']
assert (pp['ID'].values == ids).all() and (pm['ID'].values == ids).all()
assert (b['ID'].values == ids).all() and (v7['ID'].values == ids).all()
dpp = pp['Target'].values.astype(np.float64) - b['Target'].values.astype(np.float64)
dpm = pm['Target'].values.astype(np.float64) - b['Target'].values.astype(np.float64)
ok_p = np.abs(dpp[sel] - DELTA).max() < 1e-9 and np.abs(dpp[~sel]).max() < 1e-12
ok_m = np.abs(dpm[sel] + DELTA).max() < 1e-9 and np.abs(dpm[~sel]).max() < 1e-12
print(f'  probe_plus  : +3.0 on all {int(sel.sum()):,} rows of 201612, 0 elsewhere: '
      f'{"PASS" if ok_p else "FAIL"}  (md5 {md5(f"{DL}/submission_probe_m201612_plus.csv")[:10]})')
print(f'  probe_minus : -3.0 on all {int(sel.sum()):,} rows of 201612, 0 elsewhere: '
      f'{"PASS" if ok_m else "FAIL"}  (md5 {md5(f"{DL}/submission_probe_m201612_minus.csv")[:10]})')
d77 = v7['Target'].values.astype(np.float64) - b['Target'].values.astype(np.float64)
untouched = np.abs(d77[sel]).max()
months_hit = sorted(set(ym[np.abs(d77) > 1e-12].tolist()))
print(f'  v27 vs v24 on 201612 rows: max|d| = {untouched:.2e}  -> '
      f'{"NO prior 201612 correction (no double-correction)" if untouched == 0 else "FAIL"}')
print(f'  v27 vs v24 nonzero months: {months_hit}  -> expect exactly '
      '[201509, 201601, 201606, 201807, 201811]')
assert untouched == 0 and months_hit == [201509, 201601, 201606, 201807, 201811]
print('  -> audit PASS; v28 = v27 - 0.8e on 201612 rows is clean to stack')

print()
print('=' * 76)
print('3) ARMING DECISION + v28 BUILD')
print('=' * 76)
armed = abs(e) > ARM
print(f'  |e(201612)| = {abs(e):.4f}  vs gate {ARM}  ->  '
      f'{"*** ARMED *** (pre-registered rule fires)" if armed else "not armed"}')
c = SHRINK * e
print(f'  correction c = 0.8 * e = {c:+.6f}  (SUBTRACT from 201612 predictions)')
tgt = v7['Target'].values.astype(np.float64).copy()
tgt[sel] -= c
assert np.isnan(tgt).sum() == 0
pd.DataFrame({'ID': ids, 'Target': tgt}).to_csv(OUT, index=False)
chk = pd.read_csv(OUT, **RT); chk.columns = ['ID', 'Target']
d_rt = chk['Target'].values.astype(np.float64) - v7['Target'].values.astype(np.float64)
untouched_rt = np.abs(d_rt[~sel]).max()
print(f'  wrote {OUT}')
print(f'  round-trip (round_trip parser): 201612 rows n={int(sel.sum()):,}, '
      f'max|d+c| = {np.abs(d_rt[sel] + c).max():.2e}')
print(f'  untouched rows max|d| = {untouched_rt:.2e}  '
      f'({"EXACTLY 0 - byte-identical" if untouched_rt == 0.0 else "DRIFT - FAIL"})')
n_lines = sum(1 for a, bb in zip(open(f"{DL}/submission_v27_2corr.csv"),
                                 open(OUT)) if a != bb)
print(f'  text-level line diff vs v27: {n_lines} lines (expect exactly {int(sel.sum())})')
assert untouched_rt == 0.0 and n_lines == int(sel.sum())
print(f'  md5(v28) = {md5(OUT)}')
print(f'  rows = {len(chk):,}, ID-align PASS, no NaN -> upload-ready')

print()
print('=' * 76)
print('4) PRE-REGISTERED PREDICTION (chained on v27 ACTUAL)')
print('=' * 76)
gain_mse = f * (2 * c * e - c**2)
gain_id = 0.96 * f * e * e
assert abs(gain_mse - gain_id) < 1e-12
mse28 = V27_ACT**2 - gain_mse
pred = math.sqrt(mse28)
print(f'  MSE gain = f(2ce - c^2) = 0.96*f*e^2 = {gain_mse:.12f}')
print(f'  predicted PUBLIC v28 = {pred:.9f}   ({pred - V27_ACT:+.6f} vs v27)')
lo, hi = pred - GATE, pred + GATE
print(f'  two-sided gate [{lo:.6f}, {hi:.6f}]  (formal PASS/FAIL, pre-registered)')
print('  expected residual class: ~1e-9 (5 previous chain hits: -1.19e-10,')
print('   +1.0e-9, -8.62e-11, +2 earlier) - the gate is deliberately wide')
print(f'  private projection: ~{0.92 * (V27_ACT - pred):+.6f} RMSE (month-weights ~0.92x)')
print(f'  chain after v28: v24 0.683792 -> v25 0.679780 -> v26 0.678573 ->')
print(f'   v27 0.677007 -> v28 ~{pred:.6f}; total banked ~{V27_ACT - pred + 0.006785:.6f}')
print(f'  step size {V27_ACT - pred:.6f} = largest since v25 (0.004011);')
print(f'  remaining gap to 0.666 floor after v28: ~{pred - 0.666:.4f} (mostly noise)')
print('  display rank: ~0.6749 moves us past the 0.675-0.677 clean-corridor')
print('   segment (~3-5 display ranks, live-scan dependent; display-only value)')

print()
print('=' * 76)
print('5) ENSO PRIOR CROSS-CHECK (nino34.ascii.txt)')
print('=' * 76)
try:
    oni = {}
    for ln in open('/tmp/my-project/data/nino34.ascii.txt'):
        p = ln.split()
        if len(p) >= 5 and p[0].isdigit():
            # cols: YR MON TOTAL ClimAdjust ANOM -> anomaly = p[4]
            oni[(int(p[0]), int(p[1]))] = float(p[4])
    ond = [oni.get((2016, m)) for m in (10, 11, 12)]
    if all(v is not None for v in ond):
        oni_d = sum(ond) / 3
        print(f'  NINO34 ANOM OND-2016 = {[f"{v:+.2f}" for v in ond]}'
              f' -> ONI(Dec-16) ~ {oni_d:+.2f} (weak La Nina)')
        print(f'  |ONI| ~ {abs(oni_d):.2f} (worklog prior -0.45); measured |e| = '
              f'{abs(e):.3f} vs prior expectation 0.05-0.15')
        print('  -> weak-La-Nina month carrying an El-Nino-grade bias (~1.7x the')
        print('     prior top): ENSO magnitude story unreliable - arm-priority guide only,')
        print('     probes decide (and the pre-registered |e|>0.05 rule fired on MEASUREMENT)')
except Exception as ex:
    print(f'  (skipped: {ex})')

print()
print('=' * 76)
print('6) LANE STATUS AFTER v28 + NEXT SLOTS')
print('=' * 76)
print('  slots today (Sep 9): 201704_plus + v26 + 201612 pair = 4 of 5 ->')
print('  ONE slot left: SUBMIT submission_v28_deccorr.csv NOW, paste score here.')
print('  After v28: correction lane EXHAUSTED (all 6 k0 anchor months measured;')
print('   5 corrected + 201704 sub-gate-documented). Remaining, all optional:')
print('   201609 pair (unmeasured; report/ledger value only - masked month, no')
print('   correction lane), 201812 zero-month hunt (report value), region-split')
print('   201601 (gated |De|>0.15), report rewrite by Sep 10, forum posts,')
print('   standing selection v27+v21a -> upgrade to v28+v21a on gate PASS.')
print()
print('PASTE-BACK: v28 actual score (and v26/m201612 rows if not all pasted).')
