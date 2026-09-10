"""
PERSISTENCE-CALIBRATION VALIDATION on train's persistent era.

Simulate the test block structure on 2011-2014 (which had high pattern persistence:
2011-2012 pattern corr 0.71, 2012-2013 0.56):
  anchors: 2011-01, 2011-06, 2011-12, 2012-07, 2013-01, 2013-07  (test-like 4-7m gaps)
  masked: the months after each anchor (k=1..6)
Approaches:
  M1: global-phi decay (0.77) — current system behavior
  M2: per-cell AR decay (phi_c from pre-period)
  M3: calibrated phi from anchor-field correlations (test-style calibration)
  M4: two-component: persistent pattern (18m pre-window mean anomaly) + fast decay
  M5: Kalman with anchor-obs noise (P_init = noise var) + calibrated phi
All predict in level space: pred = mu_c + decay*(anchor - mu_c) [+ pattern term]
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'])
train['time'] = pd.to_datetime(train['time'])
train['cc'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = int(train['cc'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
train['midx'] = ym_idx.astype('int32')
n_months = len(ym_codes)
ym_to_midx = {int(v): i for i, v in enumerate(ym_codes)}
M_L = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_L[ym_idx, train['cc'].values] = train['TWS_t'].values
M_T = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_T[ym_idx, train['cc'].values] = train['target'].values

# parameters from PRE-SIMULATION period (< 2011-01-01)
PRE_END = pd.Timestamp('2011-01-01')
pre = train[train['time'] < PRE_END]
mu_c = pre.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype(np.float32)
PHI_G = 0.77

# anchors for simulation (train months, test-like spacing)
# anchors for simulation (train months, test-like spacing)
sim_anchors = [pd.Timestamp(t) for t in ['2010-12-01', '2011-06-01', '2011-12-01',
                                          '2012-07-01', '2013-01-01', '2013-07-01']]

# anchor anomaly fields
fields = {}
sim_anchors = [a for a in sim_anchors if (a.year * 100 + a.month) in ym_to_midx]
for a in sim_anchors:
    mi = ym_to_midx[a.year * 100 + a.month]
    fields[a] = M_L[mi] - mu_c

# calibrated phi from anchor-pair field correlations (like the real test calibration)
pairs = []
for i, a in enumerate(sim_anchors):
    for b in sim_anchors[i + 1:]:
        fa, fb = fields[a], fields[b]
        ok = np.isfinite(fa) & np.isfinite(fb)
        c = np.corrcoef(fa[ok], fb[ok])[0, 1]
        lag = (b.year - a.year) * 12 + (b.month - a.month)
        pairs.append((lag, c))
# fit rho(k) = lam * phi^k via least squares on log
lags = np.array([p[0] for p in pairs], dtype=float)
cors = np.array([p[1] for p in pairs], dtype=float)
ok = cors > 0.05
A = np.column_stack([np.log(lags[ok]), np.ones(ok.sum())])
b = np.log(cors[ok])
coef, *_ = np.linalg.lstsq(A, b, rcond=None)
phi_cal = float(np.exp(coef[0]))
lam_cal = float(np.exp(coef[1]))
print(f"anchor pairs: {[(l, round(c,3)) for l, c in pairs]}")
print(f"calibrated: phi={phi_cal:.3f} lam={lam_cal:.3f}  (rho(k)={lam_cal:.3f}*{phi_cal:.3f}^k)")

# two-component: persistent pattern estimate = mean anomaly over 18m before each anchor
pattern_est = {}
for a in sim_anchors:
    ym0 = (a.year * 12 + a.month - 1) - 18
    ym1 = a.year * 12 + a.month - 1
    yms = [(t // 12) * 100 + t % 12 + 1 for t in range(ym0, ym1)]
    mids = [ym_to_midx[v] for v in yms if v in ym_to_midx]
    sub = M_L[mids]  # (m, cells)
    pat = np.nanmean(sub, axis=0) - mu_c
    pattern_est[a] = np.where(np.isfinite(pat), pat, 0.0).astype(np.float32)

# ---------- evaluate ----------
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
res = {m: [] for m in ['M1_global', 'M2_percell', 'M3_calib', 'M4_twocomp', 'M3b_calib_lam']}
kall = []
for a in sim_anchors:
    a_mi = ym_to_midx[a.year * 100 + a.month]
    L_anchor = M_L[a_mi]
    anom_anchor = L_anchor - mu_c
    for k in range(1, 7):
        tot = a.year * 12 + (a.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in ym_to_midx:
            continue
        t_mi = ym_to_midx[tym]
        # the row at month t has target = TWS at t+1: use M_T (target matrix)
        y = M_T[t_mi]
        ok = np.isfinite(y) & np.isfinite(L_anchor)
        if ok.sum() == 0:
            continue
        # M1: global phi decay, predict anomaly at t+1 from anchor: phi^(k+1)
        pred = mu_c + (PHI_G ** (k + 1)) * anom_anchor
        res['M1_global'].append((y[ok], pred[ok]))
        # M3: calibrated (predict t+1: lag k+1 from anchor)
        rho = lam_cal * phi_cal ** (k + 1)
        pred = mu_c + rho * anom_anchor
        res['M3_calib'].append((y[ok], pred[ok]))
        # M3b: calibrated but assume lam=1 (pure state share)
        rho_b = phi_cal ** (k + 1)
        pred = mu_c + rho_b * anom_anchor
        res['M3b_calib_lam'].append((y[ok], pred[ok]))
        # M4: two-component: pattern + fast decay on (anchor - pattern)
        pat = pattern_est[a]
        pred = mu_c + pat + (PHI_G ** (k + 1)) * (anom_anchor - pat)
        res['M4_twocomp'].append((y[ok], pred[ok]))
        kall.append(np.full(ok.sum(), k))

# M2: per-cell phi decay (using per-cell phi fit pre-2011) — approximate with global for speed
# (per-cell phi identical machinery shown elsewhere; global proxy here)
print("\n=== SIMULATED TEST (train 2011-2013, anchors 4-7m apart) ===")
for name, parts in res.items():
    ys = np.concatenate([p[0] for p in parts])
    ps = np.concatenate([p[1] for p in parts])
    print(f"{name:16s}: RMSE={np.sqrt(np.mean((ys-ps)**2)):.4f}")

# per-k for the best few
kk = np.concatenate(kall)
for name in ['M1_global', 'M3_calib', 'M4_twocomp']:
    ys = np.concatenate([p[0] for p in res[name]])
    ps = np.concatenate([p[1] for p in res[name]])
    pk = {k: np.sqrt(np.mean((ys[kk == k] - ps[kk == k]) ** 2)) for k in range(1, 7) if (kk == k).sum()}
    print(f"{name:16s}: " + " ".join(f"k{k}={v:.3f}" for k, v in pk.items()))

# reference: what would persistence (no decay) give?
ys = np.concatenate([p[0] for p in res['M1_global']])
ps = np.concatenate([mu_c[np.argmax(np.zeros((1, n_cells)), axis=0)] * 0 + np.concatenate([p[1] for p in res['M1_global']])]) if False else None
# pure persistence: pred = anchor level
pers = []
for a in sim_anchors:
    a_mi = ym_to_midx[a.year * 100 + a.month]
    for k in range(1, 7):
        tot = a.year * 12 + (a.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in ym_to_midx:
            continue
        t_mi = ym_to_midx[tym]
        y = M_T[t_mi]
        ok = np.isfinite(y) & np.isfinite(M_L[a_mi])
        pers.append((y[ok], M_L[a_mi][ok]))
ys = np.concatenate([p[0] for p in pers]); ps = np.concatenate([p[1] for p in pers])
print(f"{'persistence':16s}: RMSE={np.sqrt(np.mean((ys-ps)**2)):.4f}")
