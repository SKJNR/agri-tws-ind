"""Task 46 (2026-09-11): test the OTHER AI's "Gap Idea #2" — horizon-adaptive
spatial smoothing — on the honest 2013-15 replay harness. Measure, don't argue.

CONTEXT (external AI's claim, this time grounded in our real files):
  - tb13_out.txt: large spatial scales persist more (>24deg band h1 r=0.771
    vs <2deg 0.611) -> at long horizon h the surviving signal is large-scale
    -> smoothing sigma should INCREASE with h.
  - production build_v21_phaseC.py uses FIXED smooth_sigma=2.0 deg for all h.
  - claimed: "h>=4 rows are 62.5% of private masked rows"; expected gain
    0.002-0.005 private LB.

PRE-REGISTERED DECISION GATES (recorded before any run):
  Gate A (harness): best adaptive sigma(h) must beat flat 2.0deg baseline by
    >= 0.005 cal-wtd RMSE (Xres units) — the dilution correction (test masked
    error includes D-tilde error which smoothing cannot touch; harness has no
    D) means a smaller harness gain would translate to <0.002 public, below
    the noise-free detection we demand for a selection swap.
  Gate B (LB, only if Gate A passes): v30 public must be <= v21a_public -
    0.0015 (0.685874) to swap final slot-2. Otherwise keep v21a, tombstone.

HONEST LIMITATIONS (logged before the run):
  L1: harness = train era, no D offset. Absolute gains OVERSTATE test gains
      (D-tilde error dilutes; fast-state share of masked MSE < 1).
  L2: val anchors give h=2..6; test max h=7 (per solved calendar). h=7 uses
      the h>=6 sigma by pre-registered extension.
  L3: harness kernel = grid gaussian_filter (cells); production = cKDTree
      sphere kernel (deg). Cell size computed from the actual grid; effect
      sizes comparable, exact values differ slightly.
"""
import numpy as np, pandas as pd, sys
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(0)
CACHE = '/tmp/my-project/scripts/tb_cache.npz'
TEST_CSV = '/tmp/my-project/data/Test (2).csv'
PHI, LAM, BWD_MAX_GAP = 0.82, 0.80, 8          # mid production config

d = np.load(CACHE)
F, yms, t_abs_tr = d['F'].astype(np.float64), d['yms'], d['t_abs_tr']
lat_c, lon_c = d['lat_c'], d['lon_c']
SPEI1, SPEI3, SPEI6, SPEI12, SM = [d[k].astype(np.float64) for k in
                                   ['SPEI1', 'SPEI3', 'SPEI6', 'SPEI12', 'SM']]
T, n_cells = F.shape
COVS = {'SPEI_01': SPEI1, 'SPEI_03': SPEI3, 'SPEI_06': SPEI6,
        'SPEI_12': SPEI12, 'SM': SM}

print('=' * 78)
print('TASK 46: horizon-adaptive smoothing — honest harness test (pre-registered)')
print('=' * 78)

# ============ PART 1: real test-set h distribution (verify 62.5% claim) ====
print('\n[PART 1] test masked-row horizon distribution (from Test (2).csv)')
test = pd.read_csv(TEST_CSV, usecols=['time', 'TWS_t_masked'])
test['ym'] = test['time'].str[:4].astype(int) * 100 + test['time'].str[5:7].astype(int)
t_abs = (test['ym'] // 100) * 12 + (test['ym'] % 100) - 1
masked = test['TWS_t_masked'].astype(bool).values
mfrac = pd.Series(masked).groupby(t_abs).mean()
anchors_te = sorted(int(m) for m, f in mfrac.items() if f < 0.01)
anch_arr = np.array(anchors_te)
h_fwd = np.full(len(test), -1, dtype=int)
sel_m = masked
ta_m = t_abs[sel_m].values
idx = np.searchsorted(anch_arr, ta_m, side='right') - 1
h_valid = idx >= 0
h_fwd[sel_m] = np.where(h_valid, ta_m - anch_arr[np.maximum(idx, 0)] + 1, -1)
n_mask = int(sel_m.sum())
dist = pd.Series(h_fwd[sel_m]).value_counts().sort_index()
print(f'  k0 anchor months (test): {anchors_te}')
print(f'  masked rows total: {n_mask:,}')
for h, c in dist.items():
    if h > 0:
        print(f'    h={h:2d}: {c:6,} rows  ({100*c/n_mask:5.1f}%)')
hge4 = int((h_fwd[sel_m] >= 4).sum()); hge3 = int((h_fwd[sel_m] >= 3).sum())
print(f'  -> h>=3: {100*hge3/n_mask:.1f}%   h>=4: {100*hge4/n_mask:.1f}% '
      f'(other-AI claim: 62.5% h>=4)')
bwd_cov = 0
for m in np.unique(ta_m):
    nxt = anch_arr[anch_arr > m]
    if len(nxt) and (int(nxt[0]) - (int(m) + 1)) <= BWD_MAX_GAP:
        bwd_cov += int(((ta_m == m)).sum())
print(f'  rows with a backward anchor within {BWD_MAX_GAP} months: '
      f'{bwd_cov:,} ({100*bwd_cov/n_mask:.1f}%)')
W_H = {int(h): int(c) for h, c in dist.items() if h > 0}   # row weights

# ============ PART 2: harness (production-like scalar Kalman) ==============
t_fit_end = int(np.searchsorted(yms, 201301))
t_tr = t_abs_tr[:t_fit_end]
mu = np.nanmean(F[:t_fit_end], axis=0)
td = t_tr - t_tr.mean()
beta = (td[:, None] * (F[:t_fit_end] - mu)).sum(0) / np.sum(td ** 2)
tbar = t_tr.mean()
Xres = F - mu[None, :] - np.outer(t_abs_tr - tbar, beta)

# composite cov field W (tb12's ridge weights, refit here for self-containment)
cov_devs = {nm: M - np.nanmean(M[:t_fit_end], axis=0, keepdims=True)
            for nm, M in COVS.items()}
Y = Xres[:t_fit_end]
A_ = np.stack([cov_devs[nm][:t_fit_end] for nm in COVS], axis=0)
ok = np.isfinite(Y) & np.all(np.isfinite(A_), axis=0)
Av, Yv = A_[:, ok], Y[ok]
w_cov = np.linalg.solve(Av @ Av.T + 1e-6 * np.eye(5), Av @ Yv)
Wcomp = sum(w_cov[k] * cov_devs[nm] for k, nm in enumerate(COVS))

# scalar Kalman params from fit era (production formula)
var_f = float(np.nanvar(Xres[:t_fit_end]))
c_ = float(np.mean([np.cov(Wcomp[t][np.isfinite(Wcomp[t]) & np.isfinite(Xres[t])],
                            Xres[t][np.isfinite(Wcomp[t]) & np.isfinite(Xres[t])])[0, 1]
                     for t in range(t_fit_end)]))
varz = float(np.mean([np.nanvar(Wcomp[t]) for t in range(t_fit_end)]))
H = c_ / (LAM * var_f); R = max(varz - c_ * c_ / (LAM * var_f), 1e-4)
q = LAM * var_f * (1 - PHI ** 2); P0 = LAM * (1 - LAM) * var_f
print(f'\n[PART 2] harness: phi={PHI} lam={LAM} H={H:.3f} R={R:.3f} '
      f'var_f={var_f:.3f} (val Xres std={np.nanstd(Xres[t_fit_end:]):.3f})')

# grid + smoother (cell size in deg)
lats = np.sort(np.unique(lat_c)); lons = np.sort(np.unique(lon_c))
nl, no = len(lats), len(lons)
cell_deg = float(np.median(np.diff(lats)))
lat_i = {v: i for i, v in enumerate(lats)}; lon_i = {v: i for i, v in enumerate(lons)}
lat_idx = np.array([lat_i[v] for v in lat_c]); lon_idx = np.array([lon_i[v] for v in lon_c])
Wg = np.zeros((nl, no)); Wg[lat_idx, lon_idx] = 1.0
def gsmooth(field, sigma_deg):
    s = sigma_deg / cell_deg
    G = np.full((nl, no), np.nan); G[lat_idx, lon_idx] = field
    Gf = np.where(np.isnan(G), 0.0, G)
    num = gaussian_filter(Gf, s, mode='constant')
    den = gaussian_filter(Wg, s, mode='constant')
    vals = num[lat_idx, lon_idx] / np.maximum(den[lat_idx, lon_idx], 1e-9)
    return np.where(np.isfinite(field), vals, np.nan)
print(f'  grid {nl}x{no}, cell={cell_deg:.2f} deg; production sigma=2.0deg '
      f'= {2.0/cell_deg:.2f} cells')

# val protocol (tb12 anchors, test-calendar-like gaps 4,5,6,5)
anchor_yms_val = [201309, 201401, 201406, 201412, 201505]
ym_arr = yms.astype(int)
anchor_t = [int(np.searchsorted(ym_arr, y)) for y in anchor_yms_val]
val_months = [t for t in range(t_fit_end, T) if t not in anchor_t]

def fwd_state(i, tm):
    x = np.where(np.isfinite(Xres[i]), LAM * np.nan_to_num(Xres[i]), 0.0)
    P = np.where(np.isfinite(Xres[i]), P0, var_f).copy()
    for m in range(i + 1, tm + 1):
        x = PHI * x; P = PHI ** 2 * P + q
        if np.isfinite(Wcomp[m]).any():
            w_ = Wcomp[m]; okw = np.isfinite(w_)
            Kg = np.where(okw, P * H / (H * H * P + R), 0.0)
            x = np.where(okw, x + Kg * (np.nan_to_num(w_) - H * x), x)
            P = np.where(okw, (1 - Kg * H) * P, P)
    return x, P

def bwd_state(k, tm):
    x = np.where(np.isfinite(Xres[k]), LAM * np.nan_to_num(Xres[k]), 0.0)
    P = np.where(np.isfinite(Xres[k]), P0, var_f).copy()
    for m in range(k - 1, tm - 1, -1):
        x = PHI * x; P = PHI ** 2 * P + q
        if np.isfinite(Wcomp[m]).any():
            w_ = Wcomp[m]; okw = np.isfinite(w_)
            Kg = np.where(okw, P * H / (H * H * P + R), 0.0)
            x = np.where(okw, x + Kg * (np.nan_to_num(w_) - H * x), x)
            P = np.where(okw, (1 - Kg * H) * P, P)
    return x, P

anch_v = np.array(anchor_t)
def predict_month(m, sigma_rule):
    tm = m + 1
    i = int(anch_v[anch_v <= m][-1])
    xf, Pf = fwd_state(i, tm)
    later = anch_v[anch_v > m]
    if len(later) and (int(later[0]) - tm) <= BWD_MAX_GAP:
        xb, Pb = bwd_state(int(later[0]), tm)
        wgt = (1.0 / np.maximum(Pf, 1e-6)) / (1.0 / np.maximum(Pf, 1e-6) +
                                              1.0 / np.maximum(Pb, 1e-6))
        x = wgt * xf + (1 - wgt) * xb
    else:
        x = xf
    h = tm - i
    sig = sigma_rule(h)
    if sig and sig > 0:
        x = gsmooth(x, sig)
    return x

def evaluate(sigma_rule, label):
    errs = {}
    for m in val_months:
        if m < anch_v[0] or m + 1 >= T:      # need an anchor before month m
            continue
        i = int(anch_v[anch_v <= m][-1])
        h = (m + 1) - i
        xh = predict_month(m, sigma_rule)
        x = Xres[m + 1] if m + 1 < T else None
        if x is None:
            continue
        o = np.isfinite(x) & np.isfinite(xh)
        if o.sum() == 0:
            continue
        e = (xh[o] - x[o]) ** 2
        errs.setdefault(h, []).append(e.mean())
    per_h = {h: float(np.sqrt(np.mean(v))) for h, v in sorted(errs.items())}
    # cal-wtd with TRUE test row weights (Part 1)
    tot_w = sum(W_H.get(h, 0) for h in per_h)
    cal = float(np.sqrt(sum(W_H.get(h, 0) * per_h[h] ** 2 for h in per_h) / tot_w))
    hs = ' '.join(f'h{h}={per_h[h]:.3f}' for h in per_h)
    print(f'  {label:34s}: {hs} | cal-wtd={cal:.4f}')
    return cal, per_h

print('\n--- sigma sweep (flat) ---')
results = {}
for s in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    results[f'flat {s:.1f}'] = evaluate((lambda ss: (lambda h: ss))(s),
                                        f'flat sigma={s:.1f}deg' +
                                        (' (prod)' if s == 2.0 else ''))
print('\n--- sigma sweep (horizon-adaptive, pre-registered 3-piece rule) ---')
for s1, s2, s3 in [(2.0, 2.0, 4.0), (2.0, 3.0, 4.0), (2.0, 3.0, 5.0),
                   (2.0, 4.0, 6.0), (1.5, 3.0, 6.0), (2.0, 3.0, 6.0),
                   (2.5, 3.5, 5.0), (2.0, 2.0, 3.0), (2.0, 4.0, 4.0),
                   (3.0, 4.0, 6.0), (1.0, 3.0, 5.0), (2.0, 5.0, 8.0)]:
    rule = (lambda a, b, c: (lambda h: a if h <= 3 else (b if h <= 5 else c)))(s1, s2, s3)
    results[f'ada {s1}/{s2}/{s3}'] = evaluate(
        rule, f'adaptive sigma h<=3:{s1} h4-5:{s2} h>=6:{s3}')

base = results['flat 2.0'][0]
best_name = min(results, key=lambda k: results[k][0])
best = results[best_name][0]
print('\n' + '=' * 78)
print(f'VERDICT: baseline flat-2.0 cal-wtd = {base:.4f}')
print(f'         best = {best_name} cal-wtd = {best:.4f}  '
      f'(gain {base - best:+.4f})')
print(f'GATE A (>= 0.005 gain): {"PASS -> build v30 probe" if base - best >= 0.005 else "FAIL -> tombstone idea #2, no submission"}')
print('=' * 78)
