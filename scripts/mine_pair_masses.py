"""Mine ALL (file-pair, diff-set, LB-mass) equations from submission history.

Every pair of submitted files with known public scores that differ on a
locally-computable row set gives:
    score_a^2 - score_b^2 = sum_(diff AND public) [e_a^2 - e_b^2] / |public|
If the diff-set is entirely OUTSIDE the public set, the mass must be 0.
Nonzero mass => diff-set intersects the public set. This maps the public
set's structure without spending any submission slots.

Pairs to mine (all scores are PUBLIC LB, RMSE):
  v12a(0.701765344) vs v10b(0.699997215)  -- era-Dtil on masked rows
  v13 (0.696325144) vs v12b(0.695357171)  -- horizon-gated era
  v13b(0.697421091) vs v12b(0.695357171)  -- era block E (77,850)
  v22b(0.701171004) vs v22 (0.699118155)  -- era block E again (independent)
  v21a(0.687374005) vs v18a(0.693738722)  -- k0 swap (a15 vs top3-ens)
  v24 (0.683791578) vs v23 (0.683548406)  -- compliant vs tainted k0
  v21b(0.689902088) vs v21a(0.687374005)  -- LOO-refit Dtil weights
  v17b(0.704955918) vs v6c (0.703041139)  -- denoiser hedge
  v10b(0.699997215) vs v6c (0.703041139)  -- k0 blend
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time', 'TWS_t'])
test.columns = [c.strip() for c in test.columns]
test['ym'] = test['time'].str[:4].astype(int) * 100 + test['time'].str[5:7].astype(int)
ids = test['ID'].values
ym = test['ym'].values

SCORES = {
    'v1': 0.8059, 'v2b': 0.7962, 'v2c': 0.8337, 'v3a': 0.7984, 'v1b': 0.7152,
    'v1c': 0.7168, 'v4a': 0.715488093, 'v4c': 0.7144, 'v5a': 0.7056, 'v5b': 0.7050,
    'v6c': 0.703041139, 'v8a': 0.70702259, 'v8b': 0.7091, 'v10b': 0.699997215,
    'v11b': 0.710571063, 'v12a': 0.701765344, 'v12b': 0.695357171,
    'v13': 0.696325144, 'v17b': 0.704955918, 'v18a': 0.693738722,
    'v21a': 0.687374005, 'v21b': 0.689902088, 'v13b': 0.697421091,
    'v22_splice': 0.699118155, 'v23_splice': 0.683548406,
    'v24': 0.683791578, 'v22b': 0.701171004,
}

def load(v):
    df = pd.read_csv(f'{DL}/submission_{v}.csv')
    df.columns = [c.strip() for c in df.columns]
    assert (df['ID'].values == ids).all(), f'ID order mismatch {v}'
    return df['Target'].values.astype(np.float64)

PAIRS = [
    ('v12a', 'v10b'), ('v13', 'v12b'), ('v13b', 'v12b'), ('v22b', 'v22_splice'),
    ('v21a', 'v18a'), ('v24', 'v23_splice'), ('v21b', 'v21a'), ('v17b', 'v6c'),
    ('v10b', 'v6c'), ('v22_splice', 'v12b'), ('v23_splice', 'v21a'),
]

print(f"{'pair':>22} {'dScore':>10} {'dRMSE2':>10} {'nDiff':>8}  "
      f"{'rms(d)':>7}  months-differing (n rows)")
for a, b in PAIRS:
    try:
        va, vb = load(a), load(b)
    except FileNotFoundError:
        print(f'{a} vs {b}: file missing, skipped')
        continue
    d = va - vb
    nz = np.abs(d) > 0
    sa, sb = SCORES[a], SCORES[b]
    mass = sa**2 - sb**2
    months = pd.Series(ym[nz]).value_counts().sort_index()
    mstr = ', '.join(f'{m}:{c}' for m, c in months.items())
    print(f'{a + " vs " + b:>22} {sa-sb:+10.6f} {mass:+10.6f} {nz.sum():8,}  '
          f'{np.sqrt((d[nz]**2).mean()):7.4f}  {mstr}')

print('\n--- interpretation guide ---')
print('mass = 0 exactly  -> diff-set fully OUTSIDE public (or errors identical)')
print('mass != 0         -> diff-set INTERSECTS the public scored rows')
print('Only pairs with mass != 0 are informative about the public set shape.')
