#!/usr/bin/env python3
"""
NOISE FLOOR MEASUREMENT — the decisive strategic experiment.

Our whole error budget assumes target noise std = 0.456 (estimated from TRAIN
ACF factoring, lambda=0.84). The leader's 0.5596 implies D+fast error ~0.26-0.33,
which is barely possible WITH our noise floor but easy WITHOUT it.

Direct measurement via spatial statistics:
- D field: static, spatially smooth (corr 0.84 at 5 deg)
- fast field: spatially smooth (corr 0.97-0.99 at 1 deg)
- noise: WHITE in space

=> variance of adjacent-cell anomaly differences ~= 2*sigma^2_noise + tiny
   (fast-field adjacent contribution ~ 2*var_fast*(1-r_adj) ~ 0.01-0.03)

Measure at: 6 test anchors vs a sample of train months. If sigma_test <
sigma_train, our Kalman R / shrinkage is miscalibrated and there is a large
immediate win available.

Also: measure covariate field smoothness (are SPEI/SOIL noise-free?).
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['t_abs'] = test['time'].str[:4].astype(int) * 12 + test['time'].str[5:7].astype(int) - 1
test['masked'] = test['TWS_t_masked'].astype(str).str.lower().isin(['true', '1'])

def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
train['cc'] = ckey(train['lat'], train['lon'])
test['cc'] = ckey(test['lat'], test['lon'])

# ---- cell coordinate map (1-degree grid; build adjacency from the test grid) ----
grid = test[['cc', 'lat', 'lon']].drop_duplicates('cc').set_index('cc').sort_index()
lat = grid['lat'].values; lon = grid['lon'].values
n_cells = len(grid)
print(f"Cells: {n_cells}; lat [{lat.min()}, {lat.max()}], lon [{lon.max()}, {lon.min()}]")

# adjacency: cells differing by exactly 1 degree in lat or lon (rook neighbors)
from scipy.spatial import cKDTree
pts = np.column_stack([lat, lon])
tree = cKDTree(pts)
pairs = tree.query_pairs(r=1.5, output_type='ndarray')  # includes diag ~1.41
d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
adj = pairs[d < 1.05]  # rook + slight tolerance
adj_diag = pairs[(d > 1.3) & (d < 1.5)]
print(f"Rook-adjacent pairs: {len(adj):,}; diagonal pairs: {len(adj_diag):,}")

# ---- helper: noise from adjacent diffs of a field (indexed by cc) ----
def noise_from_adj(field, adj_pairs):
    """field: pd.Series indexed by cc. Returns per-pair diff variance."""
    v = field.reindex(grid.index).values
    if np.isnan(v).any():
        m = ~np.isnan(v)
        ok = m[adj_pairs[:, 0]] & m[adj_pairs[:, 1]]
        ap = adj_pairs[ok]
    else:
        ap = adj_pairs
    diff = v[ap[:, 0]] - v[ap[:, 1]]
    return np.var(diff), len(ap)

mu_c = train.groupby('cc')['TWS_t'].mean()

# ---- TEST ANCHORS ----
print("\n===== TEST anchor fields: adjacent-diff noise estimate =====")
test_v = test[~test['masked']].copy()
test_v['anom'] = test_v['TWS_t'] - test_v['cc'].map(mu_c)
gm = test.groupby('t_abs').agg(n=('ID', 'count'), nm=('masked', 'sum'))
anchor_ts = sorted(gm[(gm['n'] - gm['nm']) > 1000].index.tolist())
test_noise = []
for t in anchor_ts:
    f = test_v[test_v['t_abs'] == t].set_index('cc')['anom']
    var_diff, np_ = noise_from_adj(f, adj)
    sigma = np.sqrt(var_diff / 2)
    test_noise.append(sigma)
    print(f"  anchor {t//12}-{t%12+1:02d}: var(adj diff)={var_diff:.4f} -> sigma_noise ~= {sigma:.3f}  (n_pairs={np_})")
print(f"  TEST anchor noise: mean {np.mean(test_noise):.3f}  range [{min(test_noise):.3f}, {max(test_noise):.3f}]")

# ---- TRAIN reference (sample of months across eras) ----
print("\n===== TRAIN months (reference): adjacent-diff noise estimate =====")
train['anom'] = train['TWS_t'] - train['cc'].map(mu_c)
rng = np.random.default_rng(0)
all_t = sorted(train['t_abs'].unique())
sample_t = [all_t[i] for i in rng.choice(len(all_t), size=24, replace=False)]
train_noise = []
for t in sample_t:
    f = train[train['t_abs'] == t].set_index('cc')['anom']
    var_diff, np_ = noise_from_adj(f, adj)
    sigma = np.sqrt(var_diff / 2)
    train_noise.append((t, sigma))
tn = np.array([s for _, s in train_noise])
print(f"  TRAIN noise: mean {tn.mean():.3f}  std {tn.std():.3f}  range [{tn.min():.3f}, {tn.max():.3f}]")
# by era
early = np.array([s for t, s in train_noise if t < 24080])
late = np.array([s for t, s in train_noise if t >= 24080])
print(f"  early era (<2008): mean {early.mean():.3f}; late era: mean {late.mean():.3f}")

print(f"\n  >>> VERDICT: test sigma {np.mean(test_noise):.3f} vs train sigma {tn.mean():.3f}")
print(f"  >>> ratio = {np.mean(test_noise)/tn.mean():.3f} "
      f"({'TEST IS CLEANER - noise floor assumption WRONG' if np.mean(test_noise) < 0.9*tn.mean() else 'consistent with train floor'})")

# ---- Covariate smoothness: are SPEI/SOIL noisy? ----
print("\n===== Covariate field smoothness (test months, visible rows) =====")
for c in ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']:
    t0 = anchor_ts[0]
    f = test[test['t_abs'] == t0].set_index('cc')[c]
    var_diff, np_ = noise_from_adj(f, adj)
    sigma_c = np.sqrt(var_diff / 2)
    print(f"  {c:16s}: sigma_white(adj) = {sigma_c:.3f}   field std = {f.std():.3f}")

# ---- Cross-check: adjacent-diff on anchor-pair DIFFERENCE fields (2 sigma) ----
print("\n===== Cross-check: anchor-pair difference fields (var should be 2*sigma^2) =====")
anchor_fields = {t: test_v[test_v['t_abs'] == t].set_index('cc')['anom'] for t in anchor_ts}
for a in range(len(anchor_ts) - 1):
    i, k = anchor_ts[a], anchor_ts[a + 1]
    df_ = anchor_fields[k] - anchor_fields[i]
    var_diff, np_ = noise_from_adj(df_, adj)
    sigma_pair = np.sqrt(var_diff / 2)  # = sqrt(2)*sigma_noise
    print(f"  {i//12}-{i%12+1:02d} -> {k//12}-{k%12+1:02d}: implied sigma_noise = {sigma_pair/np.sqrt(2):.3f}")

# ---- What our model assumes vs measured ----
print("\n===== IMPLICATION =====")
sig_tr = tn.mean(); sig_te = np.mean(test_noise)
for label, sig in [("train-based (current assumption)", 0.456),
                   ("measured TRAIN", sig_tr), ("measured TEST anchors", sig_te)]:
    # masked floor with current D+fast error 0.556
    m = np.sqrt(sig**2 + 0.556**2)
    # optimal if we could halve D+fast
    m2 = np.sqrt(sig**2 + 0.28**2)
    lb = lambda mm: np.sqrt(0.6652 * mm**2 + 0.3348 * 0.640**2)
    print(f"  sigma={sig:.3f} ({label:32s}): masked now ~{m:.3f} (LB {lb(m):.4f}); "
          f"if D+fast halved: {m2:.3f} (LB {lb(m2):.4f})")
