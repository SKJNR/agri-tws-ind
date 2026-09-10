"""2-b step 13: val-window (2013-15) band-wise lag-h ACF / regression = test-era-appropriate
persistence calibration. Quantify per-band-profile persistence vs scalar-profile persistence
on the masked-row protocol (anchors: 201311,201401,201406,201412,201505)."""
import numpy as np
from scipy.ndimage import gaussian_filter
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, yms, t_abs_tr, lat_c, lon_c = d['F'].astype(np.float64), d['yms'], d['t_abs_tr'], d['lat_c'], d['lon_c']
SPEI1, SPEI3, SPEI6, SPEI12, SM = [d[k].astype(np.float64) for k in ['SPEI1','SPEI3','SPEI6','SPEI12','SM']]
T, n_cells = F.shape
t_fit_end = int(np.searchsorted(yms, 201301))
t_tr = t_abs_tr[:t_fit_end]
mu = np.nanmean(F[:t_fit_end], axis=0)
td = t_tr - t_tr.mean()
beta = (td[:,None]*(F[:t_fit_end]-mu)).sum(0)/np.sum(td**2)
tbar = t_tr.mean()
Xres = F - mu[None,:] - np.outer(t_abs_tr - tbar, beta)

lats = np.sort(np.unique(lat_c)); lons = np.sort(np.unique(lon_c))
nl, no = len(lats), len(lons)
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
lat_idx_all = np.array([lat_i[v] for v in lat_c]); lon_idx_all = np.array([lon_i[v] for v in lon_c])
Wg = np.zeros((nl,no)); Wg[lat_idx_all, lon_idx_all] = 1.0
def gsmooth(field, sigma_cells):
    G = np.full((nl,no), np.nan); G[lat_idx_all, lon_idx_all] = field
    Gf = np.where(np.isnan(G), 0.0, G)
    num = gaussian_filter(Gf, sigma_cells, mode='constant')
    den = gaussian_filter(Wg, sigma_cells, mode='constant')
    vals = num[lat_idx_all, lon_idx_all]/np.maximum(den[lat_idx_all,lon_idx_all],1e-9)
    return np.where(np.isfinite(field), vals, np.nan)
SIG = [1.5, 3.0, 6.0, 12.0]
def bands_of(field):
    sm = [gsmooth(field, s) for s in SIG]
    b = [field - sm[0]]
    for k in range(3): b.append(sm[k]-sm[k+1])
    b.append(sm[-1])
    return np.array(b)
BAND_NAMES = ['<2deg','2-4','4-9','9-24','>24']

# band fields for all months (val window focus)
Bval = np.stack([bands_of(Xres[t]) for t in range(t_fit_end, T)])
n_val = Bval.shape[0]
print(f"val months: {n_val} ({yms[t_fit_end]}..{yms[-1]})")

# band lag-h regression coefficients (slope) and corr in the VAL window (in-sample calibration)
print("\nVAL-window band lag-h corr (r) [test-era-appropriate persistence]:")
print("  band    " + " ".join(f"h={h}   " for h in range(1,8)))
for k in range(5):
    x = Bval[:,k,:]
    row = f"  {BAND_NAMES[k]:6s} "
    for h in range(1,8):
        a = x[:-h]; b = x[h:]
        o = np.isfinite(a)&np.isfinite(b)
        r = np.corrcoef(a[o], b[o])[0,1]
        row += f"{r:+.3f} "
    print(row)
# scalar (all cells pooled) lag-h corr for comparison
x = Xres[t_fit_end:]
row = "  scalar "
for h in range(1,8):
    a = x[:-h]; b = x[h:]
    o = np.isfinite(a)&np.isfinite(b)
    r = np.corrcoef(a[o], b[o])[0,1]
    row += f"{r:+.3f} "
print(row)

# ---------- masked-row protocol evaluation ----------
ym_arr = yms.astype(int)
anchor_t = [int(np.searchsorted(ym_arr, y)) for y in [201311, 201401, 201406, 201412, 201505]]
print("\nanchors:", [(int(ym_arr[t]), t) for t in anchor_t])
def last_anchor(t):
    la = [a for a in anchor_t if a <= t]
    return max(la) if la else None

# methods: predict Xres(t) from bands at last anchor
def pred_scalar(t, la, rho):
    h = t - la
    return rho(h) * Xres[la]
def pred_band(t, la, rho_b):
    h = t - la
    B = bands_of(Xres[la])
    return sum(rho_b(k, h)*B[k] for k in range(5))

# calibrate rho (scalar) and rho_b (band) on val window by lag regression (slope)
def slope(a, b):
    o = np.isfinite(a)&np.isfinite(b)
    return np.polyfit(a[o], b[o], 1)[0]
x_all = Xres[t_fit_end:]
rho_scalar = {}
for h in range(1,9):
    a = x_all[:-h]; b = x_all[h:]
    o = np.isfinite(a)&np.isfinite(b)
    rho_scalar[h] = np.corrcoef(a[o], b[o])[0,1]
rho_band = {}
for k in range(5):
    for h in range(1,9):
        a = Bval[:-h,k,:]; b = Bval[h:,k,:]
        o = np.isfinite(a)&np.isfinite(b)
        rho_band[(k,h)] = slope(a[o], b[o])

errs = {'persist1': [], 'scalar': [], 'band': []}
for t in range(t_fit_end, T):
    la = last_anchor(t)
    if la is None or t == la: continue
    h = t - la
    tgt = Xres[t]
    o = np.isfinite(tgt)
    e = (1.0*Xres[la] - tgt)[o]
    errs['persist1'].append((h, (e**2).mean()))
    e = (pred_scalar(t, la, lambda hh: rho_scalar.get(hh, 0.0)) - tgt)[o]
    errs['scalar'].append((h, (e**2).mean()))
    e = (pred_band(t, la, lambda k, hh: rho_band.get((k,hh), 0.0)) - tgt)[o]
    errs['band'].append((h, (e**2).mean()))

print("\nmasked-row RMSE by gap (val window; n months per h shown):")
all_h = sorted(set(h for h,_ in errs['scalar']))
hdr = "  method    " + " ".join(f"h={h}(n={sum(1 for hh,_ in errs['scalar'] if hh==h)})" for h in all_h)
print(hdr)
for m in errs:
    row = f"  {m:9s} "
    for h in all_h:
        v = np.sqrt(np.mean([e for hh,e in errs[m] if hh==h]))
        row += f"{v:.3f}      "
    print(row)
# calendar-weighted (test h distribution)
wts = {1:4,2:3,3:2,4:1,5:1,6:1}
for m in errs:
    num = 0; den = 0
    for h in all_h:
        w = wts.get(h, 0)
        v = np.mean([e for hh,e in errs[m] if hh==h])
        num += w*v; den += w
    if den: print(f"  {m:9s} cal-wtd RMSE = {np.sqrt(num/den):.4f}")
print(f"\nval Xres std: {np.nanstd(Xres[t_fit_end:]):.4f}")
