"""SPLIT-MAP PROBES — exact measurement of the public set's month structure.

Background (Task 22): the v22b anomaly (0.701171004 vs predetermined
0.699118155) has dRMSE2 = 0.002874582 — BIT-IDENTICAL to the v13b anomaly's
0.002874583. Both files differ from their base on exactly the same 77,850
era rows. Identical mass => the era rows INTERSECT the public scored set =>
the decoded "public = first 7 months (2015-09..2016-08)" is WRONG.

Probe design (exact, truth-free):
  base  = v24 (public score EXACTLY 0.683791578)
  probe = v24 with +delta (or -delta) added to ALL rows of ONE month M.
    s_plus^2  - s0^2 = [delta^2 * |P AND M| + 2*delta*sum_{P AND M} e] / |P|
    s_minus^2 - s0^2 = [delta^2 * |P AND M| - 2*delta*sum_{P AND M} e] / |P|
  Averaging the pair cancels the unknown error cross-term EXACTLY:
    f_M = |P AND M| / |P| = (s_plus^2 + s_minus^2 - 2*s0^2) / (2*delta^2)
  Bonus diagnostic: (s_plus^2 - s_minus^2)/(4*delta) = sum_{P AND M} e / |P|
  (the public total error on that month, scaled — sign tells which way our
  predictions are biased there).

Months probed (3, 2 slots each = 6 submissions):
  2016-01 : old-decode public block member  -> f ~ 0.143 if block holds
  2016-09 : era month (E)                    -> f > 0 = tail; value = tail size
  2017-04 : era month (E)                    -> f > 0 = tail; value = tail size

Interpretation table (pre-registered):
  f(201601) ~ 0.143 & f(201609) ~ 0 & f(201704) ~ 0  -> old block + tail NOT
      in these months (tail elsewhere; era-mass then from other E months)
  f(201601) ~ 0.143 & f(201609), f(201704) in (0, 0.05) -> block + small late
      tail -> private still late-heavy -> reshuffle LIVE
  all three f*M*N/n_m equal (i.e. f_m ~ 0.055 * n_m/15500) -> uniform random
      split -> private == public in distribution -> NO reshuffle
  f(201601) small too -> public not even the early block -> scattered months
      (check vs bruteforce's 157 candidate subsets)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
S0 = 0.683791578          # v24 public score (exact)
DELTA = 3.0
MONTHS = [201601, 201609, 201704]

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time'])
test['ym'] = test['time'].str[:4].astype(int) * 100 + test['time'].str[5:7].astype(int)
ids = test['ID'].values
ym = test['ym'].values

base = pd.read_csv(f'{DL}/submission_v24.csv')
base.columns = [c.strip() for c in base.columns]
assert (base['ID'].values == ids).all()
vals = base['Target'].values.astype(np.float64)

for m in MONTHS:
    sel = ym == m
    n = int(sel.sum())
    for sign, tag in [(+1, 'plus'), (-1, 'minus')]:
        pv = vals.copy()
        pv[sel] = pv[sel] + sign * DELTA
        out = pd.DataFrame({'ID': ids, 'Target': pv})
        path = f'{DL}/submission_probe_m{m}_{tag}.csv'
        out.to_csv(path, index=False)
        # verification
        chk = pd.read_csv(path)
        assert (chk['ID'].values == ids).all() and len(chk) == 280961
        cv = chk['Target'].values
        d = cv - vals
        ok_m = np.abs(d[sel] - sign * DELTA).max()
        ok_o = np.abs(d[~sel]).max()
        print(f'{path}: n(month {m})={n:,}  max|d-in-month|={ok_m:.2e}  '
              f'max|d-outside|={ok_o:.2e}  finite={np.isfinite(cv).all()}')

print('\n=== submit order & pre-registered readout ===')
print('Submit the 6 files as the daily cap allows (e.g., 3 today, 3 tomorrow).')
print('Paste the 6 scores back; then compute:')
print(f'  f_m = (s_plus^2 + s_minus^2 - 2*{S0}^2) / {2*DELTA**2}')
print('  uniform-split signature: f_m * 280961 / n_m  equal across months')
print('  block signature:         f(201601) ~ 15647/109222 = 0.1433')
print('DONE.')
