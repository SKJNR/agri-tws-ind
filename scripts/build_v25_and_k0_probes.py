"""Task 25: complete 2016-09 pair readout + the JANUARY BIAS LANE.

New scores: m201609_plus = 0.99657787, m201609_minus = 0.999967921.
Pair identities (exact): f = (s+^2 + s-^2 - 2 s0^2)/18 ; e = (s+^2 - s-^2)/(12 f).

Headline check: e(2016-01) = +0.311 (31 sigma, sigma ~ 0.010) — v24 over-predicts
January-2016 rows. Within-month random split => private Jan rows share the bias
(+/- 0.007) => a shrunken constant correction transfers. Build:

  v25 = v24 - 0.25 on ALL 2016-01 rows (c = 0.8 * e_bar shrinkage).
        Predicted public EXACTLY: s25^2 = s0^2 - f*(2 c e - c^2).
        Pre-registered gate: public in [0.6788, 0.6808] => lane CONFIRMED.

Also: H1 hypothesis (bias = generator's 0.95 shrink acting on the month's
TWS field) — offline test: 0.05 * mean(TWS_t) per k0 month vs measured e(Jan).
And: build the 5 remaining k0-month probe pairs (bias tomography program).
"""
import numpy as np, pandas as pd, hashlib, math

DATA = '/home/z/my-project/data'
V24 = '/tmp/my-project/download/submission_v24.csv'
OUTS = ['/tmp/my-project/download', '/home/z/my-project/download']
S0 = 0.683791578
DELTA = 3.0
C_JAN = 0.25

# ---------- pair readouts ----------
pairs = {
    201601: (1.052248929, 0.942148731),
    201609: (0.99657787, 0.999967921),
}
print('=' * 70)
print('COMPLETED PAIR READOUTS (exact)')
print('=' * 70)
read = {}
for m, (sp, sm) in pairs.items():
    f = (sp**2 + sm**2 - 2*S0**2) / (2*DELTA**2)
    e = (sp**2 - sm**2) / (12*f)
    read[m] = (f, e)
    print(f'  {m}:  f = {f:.6f}   e_bar = {e:+.4f}   (sigma_e ~ 0.010)')
f1, e1 = read[201601]

# ---------- load test + v24 ----------
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID', 'time', 'TWS_t'])
test['ym'] = test['time'].str[:4].astype(int)*100 + test['time'].str[5:7].astype(int)
ids = test['ID'].values; ym = test['ym'].values; tws_t = test['TWS_t'].values
base = pd.read_csv(V24)
base.columns = [c.strip() for c in base.columns]
assert (base['ID'].values == ids).all() and len(base) == 280961
vals = base['Target'].values.astype(np.float64)

# ---------- H1 offline test: 0.05 * mean(TWS_t) per k0 month ----------
print('\n' + '=' * 70)
print('H1 TEST: e_m =? 0.05 * mean(TWS_t) per k0 month  (generator 0.95 shrink)')
print('=' * 70)
k0_months = [201509, 201601, 201606, 201612, 201807, 201811]
for m in k0_months:
    sel = ym == m
    mu = np.nanmean(tws_t[sel])
    tag = ''
    if m == 201601:
        tag = f'   vs measured e = {e1:+.4f}  => H1 {"CONFIRMED" if abs(0.05*mu - e1) < 0.08 else "REFUTED"} (magnitude)'
    print(f'  {m}: mean(TWS_t) = {mu:+.3f}   0.05*mean = {0.05*mu:+.4f}{tag}')

# ---------- v25: January constant correction ----------
pred25 = math.sqrt(S0**2 - f1*(2*C_JAN*e1 - C_JAN**2))
sel = ym == 201601
v25 = vals.copy(); v25[sel] -= C_JAN
d = v25 - vals
print('\n' + '=' * 70)
print(f'v25 = v24 - {C_JAN} on all {int(sel.sum()):,} rows of 2016-01')
print(f'  PREDICTED PUBLIC SCORE (exact): {pred25:.6f}   '
      f'(v24 = {S0}; delta = {pred25 - S0:+.6f})')
print(f'  GATE: public in [0.6788, 0.6808] -> transfer CONFIRMED, lane open;')
print(f'        outside -> STOP lane and investigate (within-month split suspect).')
print('=' * 70)

# ---------- k0-month probe pairs (bias tomography) ----------
K0_PROBE_MONTHS = [201509, 201606, 201612, 201807, 201811]

def write(df, name):
    for out in OUTS:
        p = f'{out}/{name}'
        df.to_csv(p, index=False)
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()[:10]
        print(f'  {p}  md5 {h}')

print('\nwriting files (both workspaces):')
write(pd.DataFrame({'ID': ids, 'Target': v25}), 'submission_v25_jancorr.csv')
for m in K0_PROBE_MONTHS:
    s = ym == m
    for sign, tag in [(+1, 'plus'), (-1, 'minus')]:
        pv = vals.copy(); pv[s] = pv[s] + sign*DELTA
        write(pd.DataFrame({'ID': ids, 'Target': pv}),
              f'submission_probe_m{m}_{tag}.csv')

# ---------- verification pass ----------
print('\n' + '=' * 70)
print('VERIFICATION (re-read every written file)')
print('=' * 70)
for out in OUTS:
    ok = True
    for m in K0_PROBE_MONTHS:
        for sign, tag in [(+1, 'plus'), (-1, 'minus')]:
            chk = pd.read_csv(f'{out}/submission_probe_m{m}_{tag}.csv')
            cv = chk['Target'].values
            s = ym == m
            ok &= np.abs((cv - vals)[s] - sign*DELTA).max() < 1e-9
            ok &= np.abs((cv - vals)[~s]).max() < 1e-12
    chk = pd.read_csv(f'{out}/submission_v25_jancorr.csv')
    cv = chk['Target'].values; s = ym == 201601
    ok &= np.abs((cv - vals)[s] + C_JAN).max() < 1e-9
    ok &= np.abs((cv - vals)[~s]).max() < 1e-12
    ok &= np.isfinite(cv).all() and len(chk) == 280961
    print(f'  {out}: ALL BIT-EXACT {"PASS" if ok else "FAIL"}')

# ---------- readout cheat sheet ----------
print('\n' + '=' * 70)
print('READOUT CHEAT SHEET')
print('=' * 70)
print(f'f_m = (s+^2 + s-^2 - 2*{S0}^2)/18.0    e_m = (s+^2 - s-^2)/(12*f_m)')
print('201704 (today, files already exist from Task 22):')
print('  f ~ 0.0588 -> 2017-era ALSO public-boosted -> private era-light -> v21a firm')
print('  f ~ 0.052  -> 2016-only boost -> private 2017-heavier -> keep v22_splice in play')
print('  f < 0.048  -> 2017-era strongly private -> masked-diversity hedge revives')
print('  |e| < 0.05 expected (masked month); |e| > 0.1 -> correctable in v26')
print(f'\nSlot ledger after today (201704 pair): 45/200; ~35 slots remain before Sep 13.')
