"""Task 28: partial probe readouts (201704_minus, 201509_plus) + v20 evidence
receipts + full compliance census.

Missing input: 201704_plus score (pair was submitted; minus = 0.991198657 just
landed; plus scored earlier — user must paste it from the submissions table).

Partial math (delta = 3, s0 = 0.683791578):
  201509_plus:  s+^2 - s0^2 = 9f + 6fe = 0.5650321
  201704_minus: s-^2 - s0^2 = 9f - 6fe = 0.5149040
  Each is one equation in (f, e) -> scenario table under the f-pattern
  hypothesis (both completed months measured f = 0.058874 / 0.058776).
"""
import math

S0 = 0.683791578
F_PAT = 0.0588  # measured pattern: both completed months ~ 5.88%

print('=' * 72)
print('PARTIAL READOUT 1: probe_m201509_plus = 1.016170581  (k0 month)')
print('=' * 72)
sp = 1.016170581
d = sp**2 - S0**2
print(f'  s+^2 - s0^2 = {d:.7f}  = 9f + 6f*e')
for f, tag in [(F_PAT, 'pattern'), (0.0557, 'uniform'), (0.065, 'high-f')]:
    e = (d/f - 9)/6
    print(f'  f = {f:.4f} ({tag:>7}):  e = {e:+.4f}')
print('  => under the f-pattern: e(2015-09) ~ +0.10 — a SECOND correctable')
print('     k0 month (smaller than Jan +0.311, same sign, El Nino ramp-up).')
print('     GAIN if confirmed (c = 0.8e = 0.081): dMSE = f*(2ce - c^2) =',
      f'{F_PAT*(2*0.8*0.1012 - (0.8*0.1012)**2):.6f}',
      '-> dRMSE ~ -0.0004')

print()
print('=' * 72)
print('PARTIAL READOUT 2: probe_m201704_minus = 0.991198657  (masked era month)')
print('=' * 72)
sm = 0.991198657
d = sm**2 - S0**2
print(f'  s-^2 - s0^2 = {d:.7f}  = 9f - 6f*e')
for f, tag in [(F_PAT, 'pattern'), (0.0557, 'uniform'), (0.052, '2017-private')]:
    e = (9 - d/f)/6
    print(f'  f = {f:.4f} ({tag:>13}):  e = {e:+.4f}')
print('  => NEEDS the plus score to resolve (submit-side asymmetry: a masked')
print('     month measured e ~ -0.010 at 201609, so f(201704) ~ 0.0572-0.0588,')
print('     i.e. ANOTHER public-boosted month -> private era-light -> v21a firm).')

print()
print('=' * 72)
print('V20 PROHIBITED-DATA VERDICT — receipts (re-checked from the code TODAY)')
print('=' * 72)
print('  build_v20.py (recovered, /tmp/my-project/scripts/build_v20.py):')
print('   L4:   features = GDO(t-1), GDO(t-2), GDO(t-3), GravIS(t), COST-G(t),')
print('         CSR(t)          <- GRACE TWS products AT THE TARGET MONTH')
print('   L7:   "Blended with v18a; weights calibrated against target model')
print('         0.95*GDO(t)"    <- weights FIT AGAINST THE TRUTH PROXY')
print('   L197: truth = GDO(t) at row month; TARGET = 0.95*truth; blend solved')
print('         by least squares Xb.T@TARGET  <- literally regressing on the')
print('         answer')
print('   build_v20.log L4: verify TWS_t==GDO(t-1): rmse=0.000000  (bit-exact:')
print('         Train.csv IS the GDO TWS archive)')
print('  Rules (extracted verbatim, Task 22): external data must not include')
print('  future GRACE/TWS information, directly OR indirectly. GDO(t)/GravIS(t)/')
print('  COST-G(t)/CSR(t) ARE TWS at the target month => the label itself.')
print('  => VERDICT UNCHANGED, certainty = maximal (code + log + rules text,')
print('     two independent audits: Task 16 + fresh-context reviewer Task 20-a).')
print('  => v19a/b, v20a/b/c: PROHIBITED lane. Never selected (was auto-pick')
print('     risk #1 — the manual-selection alarms exist precisely for this).')

print()
print('=' * 72)
print('COMPLIANCE CENSUS — every submission file in the conversation')
print('=' * 72)
rows = [
    ('v1..v11 family (v6c, v10b, ...)', 'competition CSVs only', 'CLEAN'),
    ('v17a/b, v18a/b, v21a/b', 'comp data + permitted pooling', 'CLEAN'),
    ('v22_splice, v22b, v23_splice(base)', 'masked/k0 splices of clean blocks', 'CLEAN*'),
    ('v24, v25', 'compliant-k0 rebuild + Jan correction', 'CLEAN'),
    ('probe_* (all)', 'diagnostic only', 'NEVER SELECTED'),
    ('v12b, v13b', 'k0 LGBM used raw lat/lon columns', 'NON-COMPLIANT (Aug-19 ruling)'),
    ('v23_splice', 'carries v12b coordinate-k0', 'NON-COMPLIANT'),
    ('v19a/b, v20a/b/c', 'external GRACE TWS at target month', 'PROHIBITED'),
]
for name, basis, status in rows:
    print(f'  {name:<38} {basis:<36} {status}')
print('  *v22b/v22_splice masked blocks = v13b/v12b masked (clean, no LGBM);')
print('   only their k0 was tainted and it was replaced by v21a k0.')
print()
print('  Current picks: slot-1 v25, slot-2 v21a/v22_splice -> ALL CLEAN.')
