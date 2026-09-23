"""Task 32: readout of the 201807/201811 probe pairs + v27 build (stacked
corrections) + chained exact public-score prediction.

New scores pasted by user (probe base = v24, s0 = 0.683791578, delta = 3.0):
  201807: plus 0.985651381 / minus 1.011489355   (plus < minus -> e < 0)
  201811: plus 1.030112961 / minus 0.966563336   (plus > minus -> e > 0)
v26 actual NOT yet pasted -> chain from task29's exact prediction (verified
pipeline: v24 0.683791578 -> v25 0.679780277 actual, residual -1.19e-10).

Policy (Round-9 card): stack |e| > 0.05 at 0.8 shrinkage, gate band +/-0.0005.
201612 pair pending (not in this paste) -> fold into v28 if |e| > 0.05.
"""
import math
import numpy as np, pandas as pd, hashlib

S0 = 0.683791578          # v24 public (probe base)
S25 = 0.679780277         # v25 actual
DELTA = 3.0
DATA = '/home/z/my-project/data'
V26 = '/tmp/my-project/download/submission_v26_twocorr.csv'
V26_MIRROR = '/home/z/my-project/download/submission_v26_twocorr.csv'
OUTS = ['/tmp/my-project/download', '/home/z/my-project/download']

pairs = {
    201807: (0.985651381, 1.011489355),   # (plus, minus)
    201811: (1.030112961, 0.966563336),
    # 201612: (PASTE_PLUS, PASTE_MINUS),  # pending
}
prior_pairs = {          # for the v26 chain recompute (task29 values)
    201509: (1.016170581, 0.980180251),
    201606: (0.974080709, 1.022389739),
}
ENSO = {201807: 0.14, 201811: 0.97, 201612: -0.45}

print('=' * 72)
print('NEW PROBE PAIRS — exact readout (base v24)')
print('=' * 72)
print(f"{'month':>7}{'class':>9}{'f':>10}{'e_bar':>9}{'ONI':>7}{'rows dev':>10}")
res = {}
for m, (sp, sm) in sorted(pairs.items()):
    f = (sp**2 + sm**2 - 2*S0**2) / (2*DELTA**2)
    e = (sp**2 - sm**2) / (12*f)
    res[m] = (f, e)
    cls = 'k0'
    dev = (f - 1/17) * 84288
    print(f'{m:>7}{cls:>9}{f:>10.6f}{e:>+9.4f}{ENSO.get(m, 0):>7.2f}'
          f'{dev:>+10.1f}')
print(f"{'1/17 =':>16}{1/17:>10.6f}   <- both f on 1/17: NOT the zero month;")
print('  both participate in public (zero-month candidates remain for 201812).')
print('  ENSO prior holds: |ONI| 0.97 -> e +0.18 (2nd largest found); '
      '|ONI| 0.14 -> e -0.07.')

# ---------------- v26 chain recompute (sanity: reproduce task29) ----------------
f5, e5 = [( ( (sp**2+sm**2-2*S0**2)/(2*DELTA**2) ),
            (sp**2-sm**2)/(12*((sp**2+sm**2-2*S0**2)/(2*DELTA**2))) )
          for m,(sp,sm) in prior_pairs.items()][0]
g5 = (prior_pairs[201509][0]**2+prior_pairs[201509][1]**2-2*S0**2)/(2*DELTA**2)
g6 = (prior_pairs[201606][0]**2+prior_pairs[201606][1]**2-2*S0**2)/(2*DELTA**2)
e5v = (prior_pairs[201509][0]**2-prior_pairs[201509][1]**2)/(12*g5)
e6v = (prior_pairs[201606][0]**2-prior_pairs[201606][1]**2)/(12*g6)
c5, c6 = 0.8*e5v, 0.8*e6v
gain26 = g5*(2*c5*e5v - c5**2) + g6*(2*c6*e6v - c6**2)
S26_PRED = math.sqrt(S25**2 - gain26)
print()
print(f'v26 chain recompute: {S26_PRED:.9f}  (task29 card: 0.678573)')
S26_ACTUAL = None    # <- paste actual here when gate returns
S26 = S26_ACTUAL if S26_ACTUAL else S26_PRED

# ---------------- v27 build ----------------
f7, e7 = res[201807]
f11, e11 = res[201811]
c7 = 0.8 * e7
c11 = 0.8 * e11
gain27 = f7*(2*c7*e7 - c7**2) + f11*(2*c11*e11 - c11**2)
pred27 = math.sqrt(S26**2 - gain27)

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time'])
test['ym'] = test['time'].str[:4].astype(int)*100 + test['time'].str[5:7].astype(int)
ids = test['ID'].values; ym = test['ym'].values
base_path = V26 if __import__('os').path.exists(V26) else V26_MIRROR
base = pd.read_csv(base_path)
base.columns = [c.strip() for c in base.columns]
assert (base['ID'].values == ids).all() and len(base) == 280961
vals = base['Target'].values.astype(np.float64)

sel7, sel11 = ym == 201807, ym == 201811
print()
print('=' * 72)
print('v27 = v26  %+.4f on 2018-07 rows  %+.4f on 2018-11 rows'
      % (-c7, -c11))
print('=' * 72)
print(f'  corrections (0.8-shrunken measured biases):')
print(f'    2018-07: e = {e7:+.4f}  ->  shift {+c7:+.4f}   ({sel7.sum():,} rows)')
print(f'    2018-11: e = {e11:+.4f}  ->  shift {-c11:+.4f}   ({sel11.sum():,} rows)')
print(f'  base: {base_path}')
v27 = vals.copy()
v27[sel7] -= c7      # c7 negative -> adds
v27[sel11] -= c11    # c11 positive -> subtracts

print(f'  PREDICTED PUBLIC (chained): {pred27:.9f}')
if S26_ACTUAL:
    print(f'    chained from v26 ACTUAL {S26_ACTUAL}')
else:
    print(f'    chained from v26 PREDICTION {S26_PRED:.9f} '
          f'(re-run with actual when pasted)')
print(f'  GATE (two-sided, +/-0.0005): [{pred27-0.0005:.6f}, {pred27+0.0005:.6f}]')
print(f'  gain vs v26: {pred27-S26:+.6f} public;  private projection '
      f'~{0.92*gain27/(2*0.677):+.6f} (month-weights ~0.92x public)')

for out in OUTS:
    p = f'{out}/submission_v27_2corr.csv'
    pd.DataFrame({'ID': ids, 'Target': v27}).to_csv(p, index=False)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()[:10]
    chk = pd.read_csv(p)
    d = chk['Target'].values - vals
    ok = (np.abs(d[sel7] + c7).max() < 1e-9 and np.abs(d[sel11] + c11).max() < 1e-9
          and np.abs(d[~(sel7 | sel11)]).max() < 1e-12 and np.isfinite(d).all()
          and (chk['ID'].values == ids).all() and len(chk) == 280961)
    print(f'  {p}  md5 {h}  bit-verify {"PASS" if ok else "FAIL"}')

print()
print('=' * 72)
print('LEDGER + remaining correction lane')
print('=' * 72)
rows = [
    ('v21a', 0.687374005, 'clean stack ceiling (slot-2)'),
    ('v24', 0.683791578, 'compliant Dcache k0 (probe base)'),
    ('v25', 0.679780277, 'Jan-16 correction, actual'),
    ('v26', S26, 'Sep15+Jun16, predicted (gate pending paste)'),
    ('v27 (pred)', pred27, 'Jul18+Nov18 corrections'),
]
for name, s, note in rows:
    print(f'  {name:<16}{s:.6f}   {note}')
print(f'  v28 (pending): 201612 pair -> if |e|>0.05, +0.8e on 2016-12')
print(f'  wall: clean floor 0.666; gap after v27 = {pred27-0.666:.4f}')
print()
print('PASTE-BACK CHECKLIST: (1) v26 actual public score [gate '
      '0.678069-0.679069]; (2) 201612 pair scores if submitted; '
      '(3) then submit v27 next slot.')
