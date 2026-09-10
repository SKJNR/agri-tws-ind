#!/usr/bin/env python3
"""
BREAKTHROUGH ANALYSIS — after v18a LB 0.693738722 (new team best)

Q from user: why only minor tweaks, not a jump to 0.50-0.55 (leader territory)?

Part 1: Exact math of where 0.6937 comes from and what each error component costs.
Part 2: THE UNEXPLOITED DATA: partial months in Test have 2-65 visible cells each.
        Those are FREE observations of the D field at months BETWEEN anchors.
        If D(c,t) evolves with a GLOBAL pattern alpha(t)/beta(t), partial months
        pin it -> could halve D error for the 2017 block (50% of masked mass).
Part 3: Measure whether that global D-evolution signal is actually there.
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
print("=" * 70)
print("PART 1: THE MATH OF 0.6937 — why tweaks, not breakthrough")
print("=" * 70)

# LB decomposition (established: LB^2 = 0.6652*masked^2 + 0.3348*k0^2)
LB = 0.693738722
for k0 in [0.625, 0.640, 0.650]:
    m = np.sqrt((LB**2 - 0.3348 * k0**2) / 0.6652)
    print(f"  k0={k0:.3f} -> implied masked RMSE = {m:.4f}")

# Component budget (from harness floor analysis)
noise, d_err, f_err = 0.456, 0.42, 0.37
print(f"\n  Our masked floor decomposition (quadrature):")
print(f"    target noise {noise}^2 = {noise**2:.3f}  ({noise**2/0.517*100:.0f}% of var)")
print(f"    D-tilde err  {d_err}^2 = {d_err**2:.3f}  ({d_err**2/0.517*100:.0f}% of var)")
print(f"    fast err     {f_err}^2 = {f_err**2:.3f}  ({f_err**2/0.517*100:.0f}% of var)")
print(f"    total = sqrt(0.517) = {np.sqrt(noise**2+d_err**2+f_err**2):.3f}  vs actual masked 0.719")

# What the leader has
print(f"\n  Leader 0.5596 implies (at their k0):")
for k0L in [0.50, 0.55, 0.58, 0.60]:
    mL = np.sqrt((0.5596**2 - 0.3348 * k0L**2) / 0.6652)
    ndL = np.sqrt(max(mL**2 - noise**2, 0))
    print(f"    k0={k0L:.2f} -> masked={mL:.3f} -> their D+fast err = {ndL:.3f} (ours: 0.556)")

# Scenario table: what each improvement buys
print(f"\n  SCENARIO: LB if we cut D error (k0 fixed 0.640, fast 0.37, noise 0.456):")
for d_new in [0.42, 0.35, 0.30, 0.25, 0.20, 0.10]:
    m_new = np.sqrt(noise**2 + d_new**2 + f_err**2)
    lb_new = np.sqrt(0.6652 * m_new**2 + 0.3348 * 0.640**2)
    print(f"    D err {d_new:.2f} -> masked {m_new:.3f} -> LB {lb_new:.4f}"
          f"  ({'TOP-3' if lb_new < 0.6234 else 'TOP-10' if lb_new < 0.671 else 'off-podium'})")
print(f"\n  => HALVING D error (0.42->0.21) = LB ~0.624 = podium territory.")
print(f"  => Noise alone caps masked at 0.456 -> absolute LB floor ~0.50.")
print(f"  => Leader at 0.5596 is ~0.06 above the theoretical floor: near-optimal.")

print()
print("=" * 70)
print("PART 2: PARTIAL MONTHS — the free D observations nobody is using")
print("=" * 70)

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['t_abs'] = test['time'].str[:4].astype(int) * 12 + test['time'].str[5:7].astype(int) - 1
test['masked'] = test['TWS_t_masked'].astype(str).str.lower().isin(['true', '1'])
print(f"Test rows: {len(test):,}; masked: {test['masked'].sum():,} ({test['masked'].mean()*100:.1f}%)")

# Month-level structure
gm = test.groupby('t_abs').agg(n=('ID', 'count'), n_masked=('masked', 'sum'))
gm['vis'] = gm['n'] - gm['n_masked']
gm['mask_frac'] = gm['n_masked'] / gm['n']
gm['ym'] = [(t // 12, t % 12 + 1) for t in gm.index]

print("\nAll 40 test months (vis = visible rows):")
for t, row in gm.iterrows():
    tag = ""
    if row['mask_frac'] < 0.01: tag = "<== ANCHOR (fully visible)"
    elif 0 < row['vis'] <= 100: tag = "<== PARTIAL (free D obs!)"
    elif row['mask_frac'] > 0.99: tag = "(fully masked)"
    y, m = row['ym']
    print(f"  {y}-{m:02d}: n={row['n']:,} vis={row['vis']:>5} ({(1-row['mask_frac'])*100:5.1f}% visible) {tag}")

partials = gm[(gm['vis'] > 0) & (gm['vis'] < 1000)]  # partial = mostly-masked month with a visible pool
print(f"\nPartial months found: {len(partials)}")
print(f"Total free visible rows in partial months: {partials['vis'].sum():.0f}")

# Anchor months
anchors = gm[gm['mask_frac'] < 0.01]
print(f"Anchor months: {len(anchors)} -> {[(f'{y}-{m:02d}') for y, m in anchors['ym']]}")
print(f"Gaps between anchors (months): {np.diff(anchors.index).tolist()}")

print()
print("=" * 70)
print("PART 3: IS THERE A GLOBAL D-EVOLUTION SIGNAL IN PARTIAL MONTHS?")
print("=" * 70)

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1

# Cell code: round coords to 0.1 grid key
def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)

train['cc'] = ckey(train['lat'], train['lon'])
test['cc'] = ckey(test['lat'], test['lon'])

# mu_c and trend slope from train (leak-free: train only)
base = train.groupby('cc').agg(mu=('TWS_t', 'mean'), n=('TWS_t', 'count')).reset_index()
# per-cell linear trend slope
tr = train[['cc', 't_abs', 'TWS_t']].copy()
tr['y'] = tr['TWS_t']
gb = tr.groupby('cc')
# slope via cov(t, y)/var(t) — vectorized
tr['t_c'] = tr['t_abs'] - gb['t_abs'].transform('mean')
tr['y_c'] = tr['y'] - gb['y'].transform('mean')
slopes = (tr.groupby('cc').apply(lambda g: (g['t_c'] * g['y_c']).sum() / max((g['t_c']**2).sum(), 1e-9),
                                 include_groups=False)).rename('slope').reset_index()
base = base.merge(slopes, on='cc')
print(f"Train cells: {len(base):,}; mu_c range [{base['mu'].min():.2f}, {base['mu'].max():.2f}]")

# Anchor fields: mean TWS anomaly per cell at each anchor month (D observations)
anchor_ts = sorted(anchors.index.tolist())
test_v = test[~test['masked']].copy()  # all visible rows (anchors + partials)
test_v = test_v.merge(base[['cc', 'mu', 'slope']], on='cc', how='left')
test_v['anom'] = test_v['TWS_t'] - test_v['mu']
print(f"\nVisible test rows: {len(test_v):,}; cells with train history: {test_v['cc'].isin(base['cc']).sum():,}")

# D-hat per anchor: anomaly minus trend-extrapolation? Use raw anomaly as D+fast proxy.
# fast is mean-reverting ~0 mean across cells at a month? No — fast field has spatial structure
# (PC1-100=98.8%), but its GLOBAL mean is ~0. So cross-cell regression of anomaly on
# static predictors separates D-shape from fast-noise.
anchor_anom = {}
for t in anchor_ts:
    sub = test_v[test_v['t_abs'] == t]
    anchor_anom[t] = sub.set_index('cc')['anom']
    print(f"  anchor {t//12}-{t%12+1:02d}: {len(sub):,} cells, anom std {sub['anom'].std():.3f}, mean {sub['anom'].mean():+.3f}")

# D-hat field = mean of anchor anomalies (static shape)
all_anchor = pd.DataFrame(anchor_anom)
D_hat = all_anchor.mean(axis=1)
print(f"\nD-hat (mean of 6 anchor anomaly fields): std {D_hat.std():.3f}")

# PARTIAL MONTHS: the test
print("\n--- Partial months: anomaly vs D-hat regression ---")
# For each partial month: visible cells' anomaly y_c. Model: anom(c,t) = a(t)*D_hat(c) + b(t) + fast(c,t)
# If D shape is stable (a~1, b~0), no evolution info. If a(t)/b(t) drift, we can interpolate.
rows = []
for t in partials.index.tolist():
    sub = test_v[test_v['t_abs'] == t]
    if len(sub) < 3:
        continue
    sub = sub[sub['cc'].isin(D_hat.index)]
    if len(sub) < 3:
        continue
    x = D_hat.loc[sub['cc']].values
    y = sub['anom'].values
    # regression y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    pred = A @ coef
    r2 = 1 - ((y - pred)**2).sum() / max(((y - y.mean())**2).sum(), 1e-9)
    resid_std = (y - pred).std()
    # also: simple mean anomaly (level drift)
    rows.append(dict(t_abs=t, ym=f"{t//12}-{t%12+1:02d}", n=len(sub),
                     alpha=a, beta=b, r2=r2, resid_std=resid_std,
                     mean_anom=y.mean()))
pm = pd.DataFrame(rows)
print(pm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# Compare with what anchors say: fit same regression at anchors for reference
print("\n--- Same regression at ANCHOR months (reference: alpha should be ~1, beta ~0) ---")
arows = []
for t in anchor_ts:
    sub = test_v[test_v['t_abs'] == t]
    sub = sub[sub['cc'].isin(D_hat.index)]
    x = D_hat.loc[sub['cc']].values
    y = sub['anom'].values
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    arows.append(dict(t_abs=t, ym=f"{t//12}-{t%12+1:02d}", n=len(sub), alpha=a, beta=b,
                      mean_anom=y.mean()))
apd = pd.DataFrame(arows)
print(apd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# KEY TEST: partial months between Jun-16 anchor (t=24196?) and Jul-18 anchor — the 2017 block
# Do partial months in 2017 show alpha/beta drift vs anchors?
print("\n--- Verdict: is there GLOBAL D-evolution signal usable for the 2017 block? ---")
if len(pm):
    pm17 = pm[(pm['t_abs'] > anchor_ts[3]) & (pm['t_abs'] < anchor_ts[4])]
    print(f"Partial months inside the big 2017 gap: {len(pm17)}")
    if len(pm17):
        print(pm17[['ym', 'n', 'alpha', 'beta', 'mean_anom']].to_string(index=False))
    print(f"\nAnchor alpha range: [{apd['alpha'].min():.3f}, {apd['alpha'].max():.3f}], "
          f"beta range: [{apd['beta'].min():.3f}, {apd['beta'].max():.3f}]")
    print(f"Partial alpha range: [{pm['alpha'].min():.3f}, {pm['alpha'].max():.3f}], "
          f"beta range: [{pm['beta'].min():.3f}, {pm['beta'].max():.3f}]")
    # Noise floor: with n~30 cells and fast std ~0.5, alpha SE ~ 0.5/(sqrt(30)*std(D_hat)) ~ 0.18
    n_typ = pm['n'].median()
    alpha_se = 0.5 / (np.sqrt(n_typ) * D_hat.std())
    print(f"Alpha measurement noise (n={n_typ:.0f}, fast std 0.5): SE ~ {alpha_se:.3f}")
    spread_a = pm['alpha'].std()
    print(f"Partial alpha std: {spread_a:.3f} vs SE {alpha_se:.3f} -> "
          f"{'SIGNAL PRESENT (std >> SE)' if spread_a > 2.5*alpha_se else 'signal within noise'}")
    spread_b = pm['beta'].std()
    beta_se = 0.5 / np.sqrt(n_typ)
    print(f"Partial beta std: {spread_b:.3f} vs SE {beta_se:.3f} -> "
          f"{'SIGNAL PRESENT' if spread_b > 2.5*beta_se else 'signal within noise'}")

pm.to_csv('/home/z/my-project/download/partial_month_analysis.csv', index=False)
print("\nSaved: download/partial_month_analysis.csv")
