"""2-b step 9: FIXED-BASIS multi-scale model (Gaussian spatial bands) vs per-cell AR(1).
No PCA overfitting: bands are fixed Gaussian smoothings; per-band pooled phi from train era."""
import numpy as np
from scipy.ndimage import gaussian_filter
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, yms, t_abs_tr, lat_c, lon_c = d['F'].astype(np.float64), d['yms'], d['t_abs_tr'], d['lat_c'], d['lon_c']
full = ~np.isnan(F).any(axis=0)
Ff = F[:, full]
T, n_full = Ff.shape
lat_f, lon_f = lat_c[full], lon_c[full]
lats = np.sort(np.unique(lat_f)); lons = np.sort(np.unique(lon_f))
nl, no = len(lats), len(lons)
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
GRID = np.full((nl, no), np.nan)
lat_idx = np.array([lat_i[v] for v in lat_f]); lon_idx = np.array([lon_i[v] for v in lon_f])
GRID[lat_idx, lon_idx] = np.arange(n_full)
print(f"grid: {nl} x {no} = {nl*no}, cells: {n_full}")

t_split = int(np.searchsorted(yms, 201101))
t_tr = t_abs_tr[:t_split]; tbar = t_tr.mean()
mu = Ff[:t_split].mean(0)
td = t_tr - tbar
beta = (td[:,None]*(Ff[:t_split]-mu)).sum(0)/(td**2).sum()
R_tr = Ff[:t_split] - mu[None,:] - np.outer(td, beta)
R_te = Ff[t_split:] - mu[None,:] - np.outer(t_abs_tr[t_split:]-tbar, beta)
t_te = t_abs_tr[t_split:]

# Gaussian smoothing on grid (NaN-aware)
W = np.zeros((nl,no)); W[lat_idx, lon_idx] = 1.0
def gsmooth(field, sigma_cells):
    G = np.full((nl,no), np.nan); G[lat_idx, lon_idx] = field
    Gf = np.where(np.isnan(G), 0.0, G)
    num = gaussian_filter(Gf, sigma_cells, mode='constant')
    den = gaussian_filter(W, sigma_cells, mode='constant')
    out = np.full(n_full, np.nan)
    ok = den[lat_idx, lon_idx] > 1e-6
    vals = np.where(ok, num[lat_idx, lon_idx]/np.maximum(den[lat_idx,lon_idx],1e-9), 0.0)
    out[:] = vals
    return out

SIGMAS = [1.5, 3.0, 6.0, 12.0]
def bands(field):
    sm = [gsmooth(field, s) for s in SIGMAS]
    b = [field - sm[0]]                       # band0: <~2deg (fastest)
    for k in range(len(SIGMAS)-1):
        b.append(sm[k] - sm[k+1])             # intermediate bands
    b.append(sm[-1])                          # band4: >~12 deg (slowest)
    return b

# per-band phi from train era (pooled, variance-weighted across cells)
print("\nband variance share & pooled lag-1 phi (train era):")
Btr = [np.zeros_like(R_tr) for _ in range(5)]
for t in range(t_split):
    bb = bands(R_tr[t])
    for k in range(5): Btr[k][t] = bb[k]
phi_b = []; var_b = []
for k in range(5):
    X = Btr[k]
    v = X.var(0).mean()
    r1 = np.corrcoef(X[:-1].ravel(), X[1:].ravel())[0,1]
    phi_b.append(r1); var_b.append(v)
    print(f"  band{k} (sigma {SIGMAS[k] if k<4 else 'inf'}{'-'+str(SIGMAS[k]) if k>0 and k<4 else ''}): var={v:.4f} ({v/sum(var_b)*100:.1f}%), phi1={r1:.3f}, phi1^7={r1**7:.3f}")
phi_b = np.array(phi_b)

# per-cell AR1 baseline
phi_cell = np.array([np.corrcoef(R_tr[:-1,j],R_tr[1:,j])[0,1] for j in range(n_full)])

n_te = R_te.shape[0]
res = {name: {h:[] for h in range(1,8)} for name in ['cellAR1','band','bandphi_fit_te']}
for i in range(1, n_te-8):
    x = R_te[i-1]
    bb = bands(x)
    for h in range(1,8):
        tgt = R_te[i-1+h]
        pred = phi_cell**h * x
        res['cellAR1'][h].append(pred - tgt)
        pred = sum((phi_b[k]**h)*bb[k] for k in range(5))
        res['band'][h].append(pred - tgt)
# band phi re-estimated on eval (upper bound, in-sample)
Bte = [np.zeros_like(R_te) for _ in range(5)]
for t in range(n_te):
    bb = bands(R_te[t])
    for k in range(5): Bte[k][t] = bb[k]
phi_te = [np.corrcoef(Bte[k][:-1].ravel(), Bte[k][1:].ravel())[0,1] for k in range(5)]
print("eval-window band phis (in-sample upper bound):", np.round(phi_te,3))
for i in range(1, n_te-8):
    x = R_te[i-1]; bb = bands(x)
    for h in range(1,8):
        tgt = R_te[i-1+h]
        pred = sum((phi_te[k]**h)*bb[k] for k in range(5))
        res['bandphi_fit_te'][h].append(pred - tgt)

print(f"\ntarget residual std (eval): {R_te.std():.4f}")
for name in res:
    out = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res[name].items()}
    print(f"  {name:14s}: " + " ".join(f"h{h}={out[h]:.4f}" for h in range(1,8)))
o_c = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res['cellAR1'].items()}
o_b = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res['band'].items()}
print(f"  band gain vs cellAR1: " + " ".join(f"h{h}={o_c[h]-o_b[h]:+.4f}" for h in range(1,8)))
