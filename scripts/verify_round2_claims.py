#!/usr/bin/env python3
"""
TASK 20-c VERIFICATION: independently check the Round-2 reviewer's claims
before acting on them (decision loop: reviewer verdicts also get verified).

Claims to check:
  C1. submission_v23_splice.csv = v21a on masked rows + v12b on k0 rows,
      bit-exact, full 280,961 rows, valid format.
  C2. v12b == v10b bit-exact on ALL masked rows (reviewer's decode premise).
  C3. The k0 delta algebra: v10b 0.699997 vs v12b 0.695357 with identical
      masked => all delta on public k0 rows; recompute.
  C4. v22_splice = v12b masked + v21a k0 (also check content).
  C5. Public/private k0+masked fractions (reviewer: public k0 46,939/109,222).
"""
import numpy as np
import pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time', 'TWS_t'])
test.columns = [c.strip() for c in test.columns]
tw_col = 'TWS_t' if 'TWS_t' in test.columns else 'TWS_t_masked'
test['masked'] = test[tw_col].isna()
test['t_abs'] = test['time'].str[:4].astype(int) * 12 + test['time'].str[5:7].astype(int)
LO, HI = 2015 * 12 + 9, 2016 * 12 + 8
test['is_public'] = test['t_abs'].between(LO, HI)


def load(v):
    df = pd.read_csv(f'{DL}/submission_{v}.csv')
    df.columns = ['ID', 'Target']
    return df


m = test[['ID', 'masked', 'is_public']].copy()
for v in ('v21a', 'v12b', 'v10b', 'v23_splice', 'v22_splice'):
    m = m.merge(load(v), on='ID', suffixes=('', f'_{v}'))
m = m.rename(columns={'Target': 'Target_v21a'})

print("=" * 74)
print("C5: segment fractions")
print("=" * 74)
pub = m[m['is_public']]
n_pub = len(pub)
print(f"public rows: {n_pub:,}  k0 {int((~pub['masked']).sum()):,} "
      f"({(~pub['masked']).mean():.4%})  masked {int(pub['masked'].sum()):,} "
      f"({pub['masked'].mean():.4%})")

print()
print("=" * 74)
print("C1: v23_splice content — v21a on masked, v12b on k0?")
print("=" * 74)
for seg, mask in (('masked', m['masked']), ('k0', ~m['masked'])):
    d23_21 = (m.loc[mask, 'Target_v23_splice'] - m.loc[mask, 'Target_v21a']).abs().max()
    d23_12 = (m.loc[mask, 'Target_v23_splice'] - m.loc[mask, 'Target_v12b']).abs().max()
    print(f"{seg:>6} rows: max|v23-v21a|={d23_21:.2e}  max|v23-v12b|={d23_12:.2e}  "
          f"n={int(mask.sum()):,}")
n23 = len(load('v23_splice'))
print(f"v23 rows: {n23:,}  (expect 280,961)  IDs match test: "
      f"{set(load('v23_splice')['ID']) == set(test['ID'])}")

print()
print("=" * 74)
print("C2: v12b vs v10b on masked rows (bit-exact claim)")
print("=" * 74)
for seg, mask in (('masked', m['masked']), ('k0', ~m['masked'])):
    d = (m.loc[mask, 'Target_v12b'] - m.loc[mask, 'Target_v10b']).abs()
    print(f"{seg:>6} rows: max|v12b-v10b|={d.max():.2e}  "
          f"n changed>1e-9: {(d > 1e-9).sum():,}  n={int(mask.sum()):,}")

print()
print("=" * 74)
print("C3: k0 delta algebra (public rows only — LB is public-only)")
print("=" * 74)
LB = {'v10b': 0.699997215, 'v12b': 0.695357171, 'v18a': 0.693738722,
      'v21a': 0.687374005}
f_k0_pub = (~pub['masked']).mean()
f_m_pub = pub['masked'].mean()
print(f"public k0 fraction: {f_k0_pub:.6f}  masked fraction: {f_m_pub:.6f}")
d_mse = LB['v12b'] ** 2 - LB['v10b'] ** 2
d_k0_sq = d_mse / f_k0_pub
print(f"v10b->v12b total MSE delta: {d_mse:+.6f}  =>  k0-sq delta: {d_k0_sq:+.6f}")
print(f"(reviewer claimed 0.015068)")

# v18a vs v12b k0 (different k0, different masked -> 2 eq, 2 unknowns per pair
# needs an assumption; instead do v18a vs v21a: same masked, k0 differs)
d_mse2 = LB['v18a'] ** 2 - LB['v21a'] ** 2
d_k0_sq2 = d_mse2 / f_k0_pub
print(f"v21a->v18a k0-sq delta (same masked): {d_k0_sq2:+.6f}")

# absolute decode under masked-m assumptions, public window
print("\nAbsolute public k0 RMSE decode (assumed public masked m):")
print(f"{'m':>6} | {'K_v21a (a15)':>12} {'K_v12b (Dcache)':>15} {'verdict':>28}")
for m_assumed in (0.70, 0.71, 0.72, 0.73):
    resid = LB['v21a'] ** 2 - f_m_pub * m_assumed ** 2
    K21 = np.sqrt(resid / f_k0_pub) if resid > 0 else np.nan
    resid12 = LB['v12b'] ** 2 - f_m_pub * m_assumed ** 2
    K12 = np.sqrt(resid12 / f_k0_pub) if resid12 > 0 else np.nan
    diff = K21 - K12
    verdict = ("Dcache BETTER by " + f"{-diff:.3f}" if diff < -0.002 else
               ("a15 BETTER by " + f"{diff:.3f}" if diff > 0.002 else "approximately EQUAL"))
    print(f"{m_assumed:>6} | {K21:>12.4f} {K12:>15.4f} {verdict:>28}")

print()
print("=" * 74)
print("C4: v22_splice content — v12b on masked, v21a on k0?")
print("=" * 74)
for seg, mask in (('masked', m['masked']), ('k0', ~m['masked'])):
    d22_21 = (m.loc[mask, 'Target_v22_splice'] - m.loc[mask, 'Target_v21a']).abs().max()
    d22_12 = (m.loc[mask, 'Target_v22_splice'] - m.loc[mask, 'Target_v12b']).abs().max()
    print(f"{seg:>6} rows: max|v22-v21a|={d22_21:.2e}  max|v22-v12b|={d22_12:.2e}")

print()
print("=" * 74)
print("IMPLIED v23 PUBLIC SCORE (pre-registered band check)")
print("=" * 74)
# v23 = v21a masked + v12b k0. If masked block contributes what it does in v21a
# and k0 block what it does in v12b:
#   S23^2 = S_v21a^2 - f_k0*K_a15^2 + f_k0*K_dc^2 = S_v21a^2 + f_k0*(K_dc^2-K_a15^2)
# From v12b vs v10b (same masked): K_dc^2 = K_v8^2 - 0.015065
# From v18a vs v21a (same masked): K_v18^2 = K_a15^2 + 0.020456
# If v18-k0 == v8-k0 lineage (both lin/lgb): K_dc^2 = K_a15^2 + 0.020456 - 0.015065
#   => K_dc^2 - K_a15^2 = +0.005391 => S23^2 = S_v21a^2 + 0.4297*0.005391
if True:
    dk = 0.020456 - 0.015065
    s23 = np.sqrt(LB['v21a'] ** 2 + f_k0_pub * dk)
    print(f"if v18a-k0 == v10b-k0 lineage (v8 lin/lgb):")
    print(f"  K_dc^2 - K_a15^2 = +{dk:.6f}  =>  S23 ~ {s23:.4f}")
    print(f"  (WORSE than v21a by {s23 - LB['v21a']:.4f} — a15 k0 wins, keep v21a)")
    print()
    print(f"if v12b's Dcache-k0 is instead v8-lineage-independent, the only")
    print(f"  resolution is the EXPERIMENT: submit v23, read the score.")
    print(f"  Decision rule (pre-registered): adopt v23 iff S23 < 0.687374.")
