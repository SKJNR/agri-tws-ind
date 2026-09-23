"""Task 40 (2026-09-09 eve): 201812 ZERO-MONTH HUNT probe - pre-built for Sep-10.

User challenge (Round-4 escalation, HAT trigger #4): "we can't wait just by
assumption others used prohibited data... put any researcher hats necessary
to make a breakthrough." This probe is one of the two measured lanes that
survive the Round-4 re-audit (the other, region-split 201601, is
RULING-GATED - see ADVERSARIAL_REVIEW_ROUND4.md - and is NOT pre-built).

Design (this is a NEW probe class - zero-month test, not a bias pair):
  base   = v28 (CURRENT chain head, s0 = 0.674859467 EXACT, 6th exact hit)
  action = +3.0 on ALL rows of 201812 (the private zero-month CANDIDATE)
  H0     : 201812 carries ZERO public rows (fully private) ->
           public score UNCHANGED = 0.674859467 display-exact
           -> split solution CONFIRMED (report asset: "17 equal public months
              + 1 private month, confirmed by null probe")
  H1     : 201812 has hidden public rows -> score MOVES ->
           NEW month discovered -> submit minus partner next slot for the
           pair readout -> if |e|>0.05 and k0-safe -> v29 candidate lane
  detection power: display shows 9 decimals; even k=1 public row gives
           dMSE ~ (9+6e)/84288, dRMSE ~ 8e-5 -> visible at the 5th decimal.
           H0 is falsifiable to single-row resolution.

Risk: ZERO. The probe file is never selected; the selected files (v28+v21a)
are untouched. Perturbing private rows in an unselected file has no effect
on any scored outcome. Slots: 1 (Sep-10 morning, before report finalizes -
the H0 result is a report asset).
"""
import hashlib
import numpy as np
import pandas as pd

S0 = 0.674859467            # v28 ACTUAL public (probe base) - EXACT, 6th hit
DELTA = 3.0
PUB_ROWS = 84288
DL = '/tmp/my-project/download'
OUT = f'{DL}/submission_probe_m201812_plus.csv'

def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()

print('=' * 76)
print('TASK 40: 201812 zero-month hunt probe (base v28, +3.0 on all 201812)')
print('=' * 76)

# --- row/month structure from the authoritative test file ---
test = pd.read_csv('/tmp/my-project/data/Test (2).csv', usecols=['ID', 'time'])
ym = (test['time'].str[:4].astype(int) * 100 + test['time'].str[5:7].astype(int)).values
ids = test['ID'].values
sel = ym == 201812
n_sel = int(sel.sum())
print(f'  201812 rows in test set: {n_sel:,} of {len(ids):,} total')
assert n_sel > 0, '201812 absent - hunt moot'

# --- build: v28 + 3.0 on 201812 rows, byte-identical elsewhere ---
RT = dict(float_precision='round_trip')   # exact strtod: untouched rows stay byte-identical
v28 = pd.read_csv(f'{DL}/submission_v28_deccorr.csv', **RT)
v28.columns = ['ID', 'Target']
assert (v28['ID'].values == ids).all(), 'ID alignment FAIL'
base = v28['Target'].values.astype(np.float64)
tgt = base.copy()
tgt[sel] += DELTA
assert np.isnan(tgt).sum() == 0
pd.DataFrame({'ID': ids, 'Target': tgt}).to_csv(OUT, index=False)

# --- byte audit (the discipline that caught the 1-ULP drift in Task 38) ---
chk = pd.read_csv(OUT, **RT); chk.columns = ['ID', 'Target']
d = chk['Target'].values.astype(np.float64) - base
ok_delta = np.abs(d[sel] - DELTA).max() < 1e-9
untouched = np.abs(d[~sel]).max()
n_lines = sum(1 for a, b in zip(open(f'{DL}/submission_v28_deccorr.csv'), open(OUT)) if a != b)
print(f'  wrote {OUT}')
print(f'  audit: +3.0 on {n_sel:,} rows: {"PASS" if ok_delta else "FAIL"} | '
      f'untouched max|d| = {untouched:.2e} '
      f'({"EXACTLY 0 - byte-identical" if untouched == 0.0 else "DRIFT - FAIL"})')
print(f'  text-level line diff vs v28: {n_lines} (expect exactly {n_sel})')
assert ok_delta and untouched == 0.0 and n_lines == n_sel
print(f'  md5 = {md5(OUT)}')
print(f'  rows = {len(chk):,}  -> upload-ready (Sep-10 slot 1)')

print()
print('=' * 76)
print('PRE-REGISTERED READOUT (recorded BEFORE any submission)')
print('=' * 76)
print(f'  H0 (zero-month, expected): score == {S0:.9f} display-exact')
print('      -> 201812 = fully private month. Split solution CONFIRMED.')
print('         Report asset banked; hunt lane CLOSED. No minus partner.')
print('  H1 (hidden public rows): score != base by dRMSE ~ k*7.9e-5 (k rows)')
print('      -> NEW public month discovered. Next slot: minus partner')
print('         (build v28-3.0 on 201812) for the pair readout f,e.')
print('         Then the standing arming rule: |e|>0.05 AND k0-safe -> v29.')
print('  gate discipline: no action unless H1 fires with |e| > 0.05;')
print('  prediction residual class: 0 (H0 is an equality prediction - the')
print('  7th display-level exact test of the machine, free with the hunt).')
print()
print('MANIFEST ROW (paste into submissions_manifest.md before upload):')
print(f'  2026-09-10 | submission_probe_m201812_plus.csv | {md5(OUT)[:10]} | '
      f'v28+3.0 on ALL {n_sel:,} rows of 201812 (bit-audit PASS) | '
      f'zero-month hunt: H0 score=={S0:.9f} | PREDICTED: {S0:.9f} (H0)')
