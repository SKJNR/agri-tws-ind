"""2-b step 8: REALISTIC multi-scale oracle with trend extrapolation (no in-sample detrending).
Setup mirrors competition: fit mu_c, beta_c, PC basis, phi on 2002-2010; forecast 2011-2015 at h=1..7."""
import numpy as np
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, yms, t_abs_tr = d['F'].astype(np.float64), d['yms'], d['t_abs_tr']
full = ~np.isnan(F).any(axis=0)
Ff = F[:, full]
T, n_full = Ff.shape
t_split = int(np.searchsorted(yms, 201101))
t_tr = t_abs_tr[:t_split]

# ---- fit on train era ----
mu = Ff[:t_split].mean(0)
td = t_tr - t_tr.mean()
beta = (td[:,None]*(Ff[:t_split]-mu)).sum(0)/(td**2).sum()
R_tr = Ff[:t_split] - mu[None,:] - np.outer(t_tr-t_tr.mean(), beta)   # train residuals (mu+beta+t_resid)
U,S2,Vt = np.linalg.svd(R_tr - R_tr.mean(0), full_matrices=False)
# note: R_tr.mean(0)~0 by construction
scores_tr = U[:,:99]*S2[:99]

# residual-band phi, honest: split train into 2 halves, basis from half1, phi_res from half2 residuals
h1 = t_split//2
Rh1 = R_tr[:h1]; Rh2 = R_tr[h1:]
Uh,Sh,Vh = np.linalg.svd(Rh1 - Rh1.mean(0), full_matrices=False)
rec_h2 = (Rh2 - Rh1.mean(0)) @ Vh[:40].T @ Vh[:40]
res_h2 = (Rh2 - Rh1.mean(0)) - rec_h2
phi_res = np.corrcoef(res_h2[:-1].ravel(), res_h2[1:].ravel())[0,1]
print(f"honest residual-band phi (basis half1, residual half2, K=40): {phi_res:.3f}")

def band_phis(K, nband=6):
    P2 = scores_tr[:, :K]
    phi_raw = np.array([np.corrcoef(P2[:-1,k],P2[1:,k])[0,1] for k in range(K)])
    v = S2[:K]**2; w = v/v.sum()
    # bands: [0:5],[5:10],[10:20],[20:40],[40:60],[60:K]
    edges = [0,5,10,20,40,60,K]
    edges = sorted(set([e for e in edges if e<=K]))
    phi_band = np.zeros(K); band_id = np.zeros(K,int)
    for b in range(len(edges)-1):
        sl = slice(edges[b], edges[b+1])
        ww = w[sl]/w[sl].sum() if w[sl].sum()>0 else None
        pb = float(np.sum(ww*phi_raw[sl])) if ww is not None else phi_pool
        phi_band[sl] = pb; band_id[sl] = b
    phi_pool = float(np.sum(w*phi_raw))
    return phi_band, phi_raw, phi_pool, band_id

K = 60
phi_band, phi_raw, phi_pool, band_id = band_phis(K)
edges_all = [0,5,10,20,40,60,K]
edges_all = sorted(set([e for e in edges_all if e<=K]))
print("band phis:", [f"PC{edges_all[b]+1}-{edges_all[b+1]}:{phi_band[band_id==b][0]:.3f}" for b in range(len(edges_all)-1) if (band_id==b).any()])
print(f"pooled phi: {phi_pool:.3f}")

Lk = Vt[:K].T   # (n_full, K) loadings from FULL train era
Pin = np.linalg.pinv(Lk)

# per-cell AR(1) on train residuals
phi_cell = np.array([np.corrcoef(R_tr[:-1,j],R_tr[1:,j])[0,1] for j in range(n_full)])
print(f"per-cell AR1 phi: mean={phi_cell.mean():.3f}")

# ---- eval 2011-2015 ----
t_te = t_abs_tr[t_split:]
Fte = Ff[t_split:]
n_te = Fte.shape[0]
tbar = t_tr.mean()
res = {name: {h:[] for h in range(1,8)} for name in ['cellAR1','singlePhi','banded','oraclePC']}

for i in range(1, n_te-8):
    x = Fte[i-1]; t_now = t_te[i-1]
    mu_hat = mu + (t_now - tbar)*beta          # current baseline (trend-extrapolated mean)
    xres = x - mu_hat                           # current state estimate (residual space)
    s = Pin @ xres
    r = xres - Lk @ s
    # per-PC phis fit IN-SAMPLE on eval window (upper bound) - computed below via regression
    for h in range(1,8):
        tgt = Fte[i-1+h]
        t_tgt = t_te[i-1+h]
        base_tgt = mu + (t_tgt - tbar)*beta
        tgt_res = tgt - base_tgt
        # (a) per-cell AR1
        p = mu_hat + (phi_cell**h)*xres
        res['cellAR1'][h].append(p - tgt)
        # (b) single-phi denoised (pipeline proxy): denoise with K=60, single pooled phi
        p = mu_hat + Lk @ ((phi_pool**h)*s)
        res['singlePhi'][h].append(p - tgt)
        # (c) banded phi
        p = mu_hat + Lk @ (phi_band**h * s) + (phi_res**h)*r
        res['banded'][h].append(p - tgt)

# (d) in-sample per-PC oracle: regress tgt_res on s(t) per h (fit on eval window itself)
print("\nbuilding in-sample per-PC oracle regressions...")
S_eval = np.zeros((n_te, K))
for i in range(n_te):
    x = Fte[i]; t_now = t_te[i]
    mu_hat = mu + (t_now - tbar)*beta
    S_eval[i] = Pin @ (x - mu_hat)
for h in range(1,8):
    X0 = S_eval[:-h]; Y0 = []
    for i in range(n_te-h):
        tgt = Fte[i+h]; t_tgt = t_te[i+h]
        Y0.append(tgt - (mu + (t_tgt-tbar)*beta))
    Y0 = np.array(Y0)
    A_ = np.column_stack([np.ones(len(X0)), X0])
    c,*_ = np.linalg.lstsq(A_, Y0, rcond=None)
    pred = A_@c
    res['oraclePC'][h] = (pred - Y0)  # in-sample upper bound

print("\nRMSE by method (anomaly incl. trend-extrapolation error), eval 2011-2015:")
print("std of target residual:", np.sqrt(np.mean([(Fte[i]-(mu+(t_te[i]-tbar)*beta))**2 for i in range(n_te)])))
for name in ['cellAR1','singlePhi','banded','oraclePC']:
    out = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res[name].items()}
    print(f"  {name:10s}: " + " ".join(f"h{h}={out[h]:.4f}" for h in range(1,8)))
# gains
o_b = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res['banded'].items()}
o_s = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res['singlePhi'].items()}
o_c = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res['cellAR1'].items()}
o_o = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in res['oraclePC'].items()}
print(f"  banded vs singlePhi gain: " + " ".join(f"h{h}={o_s[h]-o_b[h]:+.4f}" for h in range(1,8)))
print(f"  banded vs cellAR1  gain: " + " ".join(f"h{h}={o_c[h]-o_b[h]:+.4f}" for h in range(1,8)))
print(f"  oracle upper bound gap: " + " ".join(f"h{h}={o_b[h]-o_o[h]:+.4f}" for h in range(1,8)))
