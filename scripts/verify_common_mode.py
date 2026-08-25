"""
VERIFY OTHER AI'S COMMON-MODE CLAIMS + establish the true decay profile.

Claims under test (their Round-3 message):
  C1: neighbor-cell AR(1) residual correlation r = 0.9578
  C2: PC1 = 9.1% of residual variance; PC1-5 = 28.1%
  C3: common mode std = 0.103 (3.6% of residual variance)
  C4: common mode explains the long-horizon correlation floor

New measurements:
  N1: direct train field ACF k=1..12 (empirical decay profile)
  N2: r(k) = a + b*phi^k fit on 15 anchor pairs (floor + fast phi)
  N3: anchor-pair correlation with/without global-mean removal
  N4: z(anchor) values; z(t) dynamics in train
  N5: single-month spatial autocorrelation of residual maps
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/common_mode_verification.txt'
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

# ================= load train =================
P("=== loading train ===")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['TWS_t'] = pd.to_numeric(train['TWS_t'], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms)}
T = len(yms)
P(f"train: {T} months x {n_cells} cells")

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
P(f"field missing frac: {np.mean(np.isnan(F)):.4f}")

# grid step detection
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
la = codes['lat'].values; lo = codes['lon'].values
ulat = np.unique(la); ulo = np.unique(lo)
step_lat = float(np.median(np.diff(ulat))); step_lon = float(np.median(np.diff(ulo)))
P(f"grid: {len(ulat)} lats (step {step_lat}), {len(ulo)} lons (step {step_lon})")

# anomaly + pooled phi
mu_c = np.nanmean(F, axis=0)
A = F - mu_c
X, Y = A[:-1], A[1:]
okm = np.isfinite(X) & np.isfinite(Y)
phi = float((X[okm]*Y[okm]).sum() / (X[okm]*X[okm]).sum())
P(f"pooled train phi = {phi:.4f}")

# ================= N1: direct train field ACF =================
P("\n=== N1: train field ACF (pooled over t and cells) ===")
for k in range(1, 13):
    x, y = A[:-k], A[k:]
    ok = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[ok].ravel(), y[ok].ravel())[0,1])
    P(f"  r({k:2d}) = {r:.4f}   [old formula 0.839*0.89^{k} = {0.839*0.89**k:.4f}]")

# ================= residuals =================
E = A[1:] - phi*A[:-1]                      # correct: per-cell demeaned
G = F - float(np.nanmean(F))
E2 = G[1:] - phi*G[:-1]                     # artifact version: NO per-cell demeaning
P(f"\nAR(1) residual std = {np.nanstd(E):.4f}  (their claim: 0.543)")

# ================= C1: neighbor correlations =================
pos = {(int(round(float(la[i])*1000)), int(round(float(lo[i])*1000))): i for i in range(len(la))}
dsl = int(round(step_lat*1000)); dso = int(round(step_lon*1000))
ia, ib = [], []
for i in range(len(la)):
    for dk, dj in [(0, dso), (dsl, 0)]:
        j = pos.get((int(round(float(la[i])*1000))+dk, int(round(float(lo[i])*1000))+dj))
        if j is not None and j > i:
            ia.append(i); ib.append(j)
ia = np.array(ia); ib = np.array(ib)
P(f"neighbor pairs (E/N at grid step): {len(ia)}")

def pooled(E_):
    x = E_[:, ia]; y = E_[:, ib]
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok].ravel(), y[ok].ravel())[0,1]), int(ok.sum())

r_art, _ = pooled(E2)
r_tru, _ = pooled(E)
P(f"C1 ARTIFACT version (residuals NOT per-cell demeaned): pooled neighbor corr = {r_art:.4f}   <- reproduces their 0.9578?")
P(f"C1 CORRECT version (per-cell demeaned):                 pooled neighbor corr = {r_tru:.4f}")

# per-pair time-series correlation on correct residuals
mu_E = np.nanmean(E, axis=0); sd_E = np.nanstd(E, axis=0)
Z = (E - mu_E) / np.where(sd_E > 0, sd_E, 1.0)
prod = Z[:, ia] * Z[:, ib]
per_pair = np.nanmean(np.where(np.isfinite(prod), prod, np.nan), axis=0)
P(f"C1 per-pair time-series corr (correct resid): mean = {np.nanmean(per_pair):.4f}, median = {np.nanmedian(per_pair):.4f}")

# ================= N5: single-month spatial autocorr of residual maps =================
P("\n=== N5: single-month spatial autocorr of residual map (east-neighbor) ===")
for t in [50, 100, 150]:
    if t < E.shape[0]:
        f = E[t]
        x = f[ia]; y = f[ib]
        ok = np.isfinite(x) & np.isfinite(y)
        P(f"  month idx {t}: spatial autocorr = {np.corrcoef(x[ok], y[ok])[0,1]:.4f}")

# ================= C2: PCA of residual field =================
full_cells = ~np.isnan(E).any(axis=0)
P(f"\ncells with full residual history: {full_cells.sum()} / {n_cells}")
Ef = E[:, full_cells]
Ef = Ef - Ef.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Ef, full_matrices=False)
v = S**2 / (S**2).sum()
P(f"C2 PCA of CORRECT residuals: PC1={v[0]*100:.1f}%, PC2={v[1]*100:.1f}%, PC3={v[2]*100:.1f}%, PC1-5={v[:5].sum()*100:.1f}%, PC1-10={v[:10].sum()*100:.1f}%")
E2f = E2[:, full_cells]
E2f = E2f - E2f.mean(axis=0, keepdims=True)
_, S2, _ = np.linalg.svd(E2f, full_matrices=False)
v2 = S2**2/(S2**2).sum()
P(f"C2 PCA of ARTIFACT residuals (not demeaned): PC1={v2[0]*100:.1f}%, PC1-5={v2[:5].sum()*100:.1f}%")
s1 = U[:,0]*S[0]
P(f"PC1 score: std={s1.std():.4f}, lag1 ac={np.corrcoef(s1[:-1],s1[1:])[0,1]:.3f}, lag12 ac={np.corrcoef(s1[:-12],s1[12:])[0,1]:.3f}")

# ================= C3: global mean of residuals =================
gm = np.nanmean(E, axis=1)
P(f"\nC3 global monthly mean of AR residuals: std={np.nanstd(gm):.4f}  (their claim: 0.103)")
P(f"C3 variance fraction of global mean: {np.nanvar(gm)/np.nanvar(E)*100:.2f}%  (their claim: 3.6%)")
z_raw = np.nanmean(A, axis=1)
P(f"N4 z(t)=global mean of raw anomalies: std={np.nanstd(z_raw):.4f}, "
  f"lag1={np.corrcoef(z_raw[:-1],z_raw[1:])[0,1]:.3f}, "
  f"lag6={np.corrcoef(z_raw[:-6],z_raw[6:])[0,1]:.3f}, "
  f"lag12={np.corrcoef(z_raw[:-12],z_raw[12:])[0,1]:.3f}")

# ================= anchors =================
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
test['TWS_t'] = pd.to_numeric(test['TWS_t'], errors='coerce').astype('float32')
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['masked'] = test['TWS_t_masked'].astype(bool)
mf = test.groupby('ym')['masked'].mean()
anchors = sorted(mf[mf < 0.01].index.tolist())
P(f"\nstrict fully-unmasked anchors: {anchors}")

anch = {}
for a in anchors:
    sel = (test['ym']==a) & (~test['masked'])
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[test['cc'].values[sel]] = test['TWS_t'].values[sel]
    anch[a] = fa - mu_c

P("z(anchor) = global mean of anchor anomaly field:")
for a in anchors:
    P(f"  {a}: {np.nanmean(anch[a]):+.4f}")

# ================= N3: GM-removal test =================
P("\n=== N3: pair corr original vs after removing global mean (common mode) ===")
pairs = []
for i,a in enumerate(anchors):
    for j,b in enumerate(anchors):
        if j <= i: continue
        k = (b//100-a//100)*12 + (b%100-a%100)
        if k <= 0: continue
        xa, xb = anch[a], anch[b]
        ok = np.isfinite(xa) & np.isfinite(xb)
        r0 = float(np.corrcoef(xa[ok], xb[ok])[0,1])
        xad = xa - np.nanmean(xa); xbd = xb - np.nanmean(xb)
        r1 = float(np.corrcoef(xad[ok], xbd[ok])[0,1])
        pairs.append((a,b,k,r0,r1))
        P(f"  {a}->{b} (k={k:2d}): r={r0:.4f} -> GM-removed {r1:.4f}  (drop {r0-r1:+.4f})")

# ================= N2: r(k) = a + b*phi^k fit =================
ks = np.array([p[2] for p in pairs], dtype=float)
rs = np.array([p[3] for p in pairs], dtype=float)
best = None
for a_ in np.arange(0.0, 0.5, 0.005):
    for ph in np.arange(0.60, 0.99, 0.005):
        w = ph ** ks
        b_ = np.sum((rs - a_)*w)/np.sum(w*w)
        sse = np.sum((rs - a_ - b_*w)**2)
        if best is None or sse < best[0]:
            best = (sse, a_, b_, ph)
sse, a_, b_, ph = best
P(f"\nN2 floor fit r(k)=a+b*phi^k: floor a={a_:.3f}, b={b_:.3f}, phi={ph:.3f}, fit-rmse={np.sqrt(sse/len(rs)):.4f}")
P(f"   implied r(1..7): " + ", ".join(f"{a_+b_*ph**k:.3f}" for k in range(1,8)))
best1 = None
for lam in np.arange(0.50, 1.0, 0.005):
    for ph1 in np.arange(0.60, 0.999, 0.005):
        pred = lam*ph1**ks
        sse1 = np.sum((rs-pred)**2)
        if best1 is None or sse1 < best1[0]: best1 = (sse1, lam, ph1)
P(f"   pure single-exponential fit: lam={best1[1]:.3f}, phi={best1[2]:.3f}, fit-rmse={np.sqrt(best1[0]/len(rs)):.4f}")

with open(OUT, 'w') as f:
    f.write('\n'.join(log))
P(f"\nsaved -> {OUT}")
