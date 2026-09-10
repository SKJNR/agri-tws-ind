"""2-b step 12: BANDED KALMAN PROTOTYPE on 2013-15 val window (test-calendar-like anchors).
Measures achievable masked-row RMSE with: per-band phi evolution + per-band cov observations +
exact anchor observations. Compares vs total-persistence and scalar-phi baselines."""
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
rng = np.random.default_rng(0)
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, yms, t_abs_tr, lat_c, lon_c = d['F'].astype(np.float64), d['yms'], d['t_abs_tr'], d['lat_c'], d['lon_c']
SPEI1, SPEI3, SPEI6, SPEI12, SM = [d[k].astype(np.float64) for k in ['SPEI1','SPEI3','SPEI6','SPEI12','SM']]
T, n_cells = F.shape
COVS = {'SPEI_01':SPEI1,'SPEI_03':SPEI3,'SPEI_06':SPEI6,'SPEI_12':SPEI12,'SM':SM}

# ---------- split: fit 2002-2012, val 2013-2015 ----------
t_fit_end = int(np.searchsorted(yms, 201301))
t_val0 = t_fit_end
t_tr = t_abs_tr[:t_fit_end]
mu = np.nanmean(F[:t_fit_end], axis=0)
td = t_tr - t_tr.mean()
beta = (td[:,None]*(F[:t_fit_end]-mu)).sum(0)/np.sum(td**2)
tbar = t_tr.mean()
# residual anomaly (the state): F - mu - trend
Xres = F - mu[None,:] - np.outer(t_abs_tr - tbar, beta)

# ---------- spatial bands (fixed Gaussian basis) ----------
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
    return np.array(b)          # (5, n_cells)
BAND_NAMES = ['<2deg','2-4','4-9','9-24','>24']

# ---------- fit per-band phi and observation model on 2002-2012 ----------
# composite cov W: ridge regression of cov devs on Xres (per cell too noisy; global weights)
cov_devs = {nm: M - np.nanmean(M[:t_fit_end], axis=0, keepdims=True) for nm, M in COVS.items()}
# global composite weights via pooled regression
Y = Xres[:t_fit_end]
A_ = np.stack([cov_devs[nm][:t_fit_end] for nm in COVS], axis=0)  # (5, T_fit, n)
ok = np.isfinite(Y) & np.all(np.isfinite(A_), axis=0)
# solve min || Y - sum w_k A_k ||^2 pooled: 5x5 normal equations
Av = A_[:, ok]; Yv = Y[ok]
G_ = Av@Av.T; g_ = Av@Yv
w_cov = np.linalg.solve(G_ + 1e-6*np.eye(5), g_)
Wcomp = sum(w_cov[k]*cov_devs[nm] for k,nm in enumerate(COVS))
print("composite cov weights:", {nm: round(float(w_cov[k]),3) for k,nm in enumerate(COVS)})

# band statistics on fit era
Bfit = np.stack([bands_of(Xres[t]) for t in range(t_fit_end)])   # (T_fit, 5, n)
Bfit_var = np.nanvar(Bfit, axis=(0,2))
phi_b = np.zeros(5); r_obs = np.zeros(5)
Wfit = Wcomp[:t_fit_end]
Bw = np.stack([bands_of(Wfit[t]) for t in range(t_fit_end)])
for k in range(5):
    x = Bfit[:,k,:]; w = Bw[:,k,:]
    o = np.isfinite(x)&np.isfinite(w)
    phi_b[k] = np.corrcoef(x[:-1][o[:,:-1] & o[1:]]*0+ (x[:-1]*(o[:-1]&o[1:])) , 0)[0,1] if False else None
# simpler: pooled lag-1 corr per band (mask-aware)
for k in range(5):
    x = Bfit[:,k,:]; o = np.isfinite(x)
    a = x[:-1][o[:-1]&o[1:]]; b = x[1:][o[:-1]&o[1:]]
    phi_b[k] = np.corrcoef(a, b)[0,1]
    w = Bw[:,k,:]; ow = np.isfinite(w)&o
    r_obs[k] = np.corrcoef(w[ow], x[ow])[0,1]
print("\nband | var share | phi(lag1, fit era) | corr(W_band, X_band)")
for k in range(5):
    print(f"  {BAND_NAMES[k]:6s} | {Bfit_var[k]/Bfit_var.sum()*100:4.1f}%    | {phi_b[k]:.3f}              | {r_obs[k]:.3f}")
print(f"total W-vs-X corr: {np.corrcoef(Wfit[np.isfinite(Wfit)&np.isfinite(Y)], Y[np.isfinite(Wfit)&np.isfinite(Y)])[0,1]:.3f}")

# ---------- val protocol ----------
anchor_yms_val = [201309, 201401, 201406, 201412, 201505]
val_range = range(t_val0, T)
ym_arr = yms.astype(int)
anchor_t = [int(np.searchsorted(ym_arr, y)) for y in anchor_yms_val]
print("\nval anchors at t:", anchor_t, "yms:", [ym_arr[t] for t in anchor_t])

def run_kalman(phi_use, use_cov_obs=True, obs_at_target=False, R_scale=1.0):
    """band-space Kalman with exact anchor updates; returns pred for every val month"""
    nb = 5
    # state mean and var per band per cell
    mu_s = np.zeros((nb, n_cells)); P_s = np.full((nb, n_cells), Bfit_var[:, None])  # start uncertain
    preds = {}
    last_anchor = None
    # walk through time from t_val0-12 to end
    t_start = t_val0 - 12
    # init at first anchor before val
    for t in range(t_start, T):
        is_anchor = t in anchor_t
        # observation update
        if is_anchor:
            x = Xres[t]
            B = bands_of(x)
            mu_s = np.where(np.isfinite(B), B, mu_s); P_s[:] = 0.0
            last_anchor = t
        elif use_cov_obs and np.isfinite(Wcomp[t]).any():
            w = Wcomp[t]
            Bw_t = bands_of(w)
            for k in range(nb):
                o = np.isfinite(Bw_t[k]) & np.isfinite(mu_s[k])
                H = r_obs[k]
                sW = np.nanstd(Bw[:,k,:]); sX = np.sqrt(Bfit_var[k])
                Hk = H*sW/sX
                Rk = max(sW**2*(1-r_obs[k]**2), 1e-9)*R_scale
                K = P_s[k]*Hk/(Hk*Hk*P_s[k]+Rk)
                mu_s[k] = np.where(o, mu_s[k]+K*(Bw_t[k]-Hk*mu_s[k]), mu_s[k])
                P_s[k] = np.where(o, (1-K*Hk)*P_s[k], P_s[k])
        # predict (store forecast for NEXT month before evolving)
        if t+1 < T and t+1 >= t_val0 and (t+1) not in anchor_t:
            preds[t+1] = (mu_s.copy(), last_anchor)
        # evolve
        mu_s = phi_use[:,None]*mu_s
        P_s = phi_use[:,None]**2 * P_s + (1-phi_use[:,None]**2)*Bfit_var[:,None]
        if t+1 < T and (t+1) in anchor_t:
            pass  # will be reset exactly at anchor
    return preds

# evaluate: RMSE of predicting Xres[t] for each val month t (given last anchor & cov obs)
def evaluate(phi_use, **kw):
    preds = run_kalman(phi_use, **kw)
    errs_h = {}
    for t, (mu_s, la) in preds.items():
        if la is None: continue
        h = t - la
        xhat = mu_s.sum(0)
        x = Xres[t]
        o = np.isfinite(x) & np.isfinite(xhat)
        e = xhat[o]-x[o]
        errs_h.setdefault(h, []).append((e**2).mean())
    hs = sorted(errs_h)
    return {h: np.sqrt(np.mean(np.concatenate([np.sqrt(v)]*0+[v]))) if False else np.sqrt(np.mean(v)) for h,v in errs_h.items()}

phi_detrended = phi_b.copy()
phi_slowboost = phi_b.copy(); phi_slowboost[3:] = np.minimum(phi_b[3:]+0.15, 0.99); phi_slowboost[4]=0.99
print("\nval masked-row RMSE by gap h (Xres units; std of val Xres = %.3f):" % np.nanstd(Xres[t_val0:]))
for name, ph, kw in [
    ("persist-total (phi=1 all bands)", np.ones(5), dict(use_cov_obs=False)),
    ("scalar phi=0.74 all bands", np.full(5,0.74), dict(use_cov_obs=False)),
    ("banded phi (fit)", phi_detrended, dict(use_cov_obs=False)),
    ("banded phi + cov obs", phi_detrended, dict(use_cov_obs=True)),
    ("banded phi(slowboost) + cov obs", phi_slowboost, dict(use_cov_obs=True)),
    ("banded phi(slowboost) + cov obs, R x0.5", phi_slowboost, dict(use_cov_obs=True, R_scale=0.5)),
]:
    out = evaluate(ph, **kw)
    # test-calendar h distribution: h1:4, h2:3, h3:2, h4:1, h5:1, h6:1 months of rows
    wts = {1:4,2:3,3:2,4:1,5:1,6:1}
    tot = np.sqrt(sum(wts[h]*out[h]**2 for h in wts if h in out)/12)
    print(f"  {name:38s}: h1={out.get(1,0):.3f} h2={out.get(2,0):.3f} h3={out.get(3,0):.3f} h4={out.get(4,0):.3f} h5={out.get(5,0):.3f} h6={out.get(6,0):.3f} h7={out.get(7,0):.3f} | cal-wtd={tot:.4f}")
