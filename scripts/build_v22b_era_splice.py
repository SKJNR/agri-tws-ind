"""SUBMISSION V22b — the era-treatment private bet, made selectable (Task 21).

Found by the Round-3 GM red-team audit: v13b is bit-identical to v12b on ALL
109,222 public rows (verified in Task 17's C2 check) and differs only on
77,850 private masked rows (era-inclusive Dtil on 2016-09, 2017-03..06 —
the no-backward-pass block that dominates the private round).

v22b = v13b's masked block + v21a's k0 block (both compliant):
  - public rows: bit-identical to v22_splice => public score EXACTLY
    0.699118155 (predetermined, zero public risk);
  - private rows: carries the era-treatment bet (val-measured -0.0153 RMSE on
    real masked rows, z~45) instead of v12b's static masked block.

Pre-registered rules (before submission):
  - Score MUST equal 0.699118155 exactly. Any deviation = live C2-class
    file-processing anomaly -> investigate; keep v22_splice as slot-2.
  - If exact: v22b becomes the SLOT-2 default (supersedes v22_splice;
    v22_splice stays the named fallback).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time', 'TWS_t'])
test.columns = [c.strip() for c in test.columns]
tw_col = 'TWS_t' if 'TWS_t' in test.columns else 'TWS_t_masked'
test['masked'] = test[tw_col].isna()
test['t_abs'] = test['time'].str[:4].astype(int) * 12 + test['time'].str[5:7].astype(int)
LO, HI = 2015 * 12 + 9, 2016 * 12 + 8
pub = test['t_abs'].between(LO, HI).values
msk = test['masked'].values

def load(v):
    df = pd.read_csv(f'{DL}/submission_{v}.csv')
    df.columns = ['ID', 'Target']
    return df

ids = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')[['ID']]
v13b, v21a, v22 = load('v13b'), load('v21a'), load('v22_splice')
assert (v13b['ID'].values == ids['ID'].values).all()
assert (v21a['ID'].values == ids['ID'].values).all()
assert (v22['ID'].values == ids['ID'].values).all()

m13b = v13b['Target'].values.astype(np.float64)
m21a = v21a['Target'].values.astype(np.float64)

pred = np.where(msk, m13b, m21a)  # masked <- v13b (era bet); k0 <- v21a (a15)
out = ids.copy()
out['Target'] = pred.astype(np.float32)
out.to_csv(f'{DL}/submission_v22b.csv', index=False)

# ---------------- verification ----------------
print("=== v22b verification ===")
v22b = load('v22b'); v22b_v = v22b['Target'].values.astype(np.float64)
v22_v = v22['Target'].values.astype(np.float64)
checks = [
    ('rows 280,961', len(v22b) == 280961),
    ('ID order == SampleSubmission', (v22b['ID'].values == ids['ID'].values).all()),
    ('all finite', bool(np.isfinite(v22b_v).all())),
    ('public ALL rows bit-exact vs v22_splice', float(np.abs(v22b_v[pub] - v22_v[pub]).max()) == 0.0),
    ('k0 rows bit-exact vs v21a (all)', float(np.abs(v22b_v[~msk] - m21a[~msk]).max()) == 0.0),
    ('masked rows bit-exact vs v13b (all)', float(np.abs(v22b_v[msk] - m13b[msk]).max()) == 0.0),
]
for name, okk in checks:
    print(f"  [{'PASS' if okk else 'FAIL'}] {name}")

d_priv_msk = v22b_v[~pub & msk] - v22_v[~pub & msk]
n_diff = int((np.abs(d_priv_msk) > 1e-9).sum())
print(f"  private masked rows differing from v22_splice: {n_diff:,} "
      f"(expect ~77,850 = the v13b era block)")
print(f"  public score predetermined: 0.699118155 (bit-identical public rows)")
print("DONE.")
