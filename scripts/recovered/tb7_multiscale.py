"""2-b step 7: MULTI-SCALE oracle. Truth: field = sum of spatial modes with persistence
spectrum (phi ~0.97 large scale -> ~0 small scale), essentially NO white noise.
Test: banded per-PC AR forecast vs per-cell AR(1) (our pipeline proxy), honest 2002-10 -> 2011-15."""
import numpy as np
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
A_dt, yms = d['A_dt'], d['yms']
full = ~np.isnan(A_dt).any(axis=0)
A_f = A_dt[:, full] - np.nanmean(A_dt[:, full], axis=0)
t_split = int(np.searchsorted(yms, 201101))

Atr = A_f[:t_split] - A_f[:t_split].mean(0)
Ate = A_f[t_split:]
mu = Atr.mean(0)  # ~0
U2,S2,V2 = np.linalg.svd(Atr, full_matrices=False)

# per-cell AR(1) baseline
rng = np.random.default_rng(0)
pos = rng.choice(A_f.shape[1], 2000, replace=False)
Xs = A_f[:, pos]
Xtr, Xte = Xs[:t_split], Xs[t_split:]
phi_cell = np.array([np.corrcoef(Xtr[:-1,j],Xtr[1:,j])[0,1] for j in range(2000)])
mu_cell = Xtr.mean(0)

def eval_h(errors): return {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in errors.items()}

# ---------- banded per-PC forecast ----------
def banded_forecast(K, phi_shrink=0.0, resid_phi_mode='fit'):
    Lk = V2[:K].T                      # (n_full, K) loadings (train basis)
    P2 = (U2[:,:K]*S2[:K])             # train scores
    # phi per PC via lag-1 corr of scores, shrunk toward pooled by variance weight
    phi_raw = np.array([np.corrcoef(P2[:-1,k],P2[1:,k])[0,1] for k in range(K)])
    v = S2[:K]**2; w = v/v.sum()
    phi_pool = float(np.sum(w*phi_raw))
    phi = (1-phi_shrink)*phi_raw + phi_shrink*phi_pool
    # residual persistence: pooled lag-1 corr of residual field (train era)
    rec_tr = P2 @ V2[:K]
    resid_tr = Atr - rec_tr
    if resid_phi_mode=='fit':
        r1_res = np.corrcoef(resid_tr[:-1].ravel(), resid_tr[1:].ravel())[0,1]
    else:
        r1_res = 0.0
    return Lk, phi, r1_res, phi_raw, phi_pool

results = {}
for K in [20, 40, 60, 80, 99]:
    Lk, phi, phi_res, phi_raw, phi_pool = banded_forecast(K)
    Lsub = Lk[pos]; Pin = np.linalg.pinv(Lsub)
    mu_sub = mu[pos]
    errs = {h:[] for h in range(1,8)}
    errs_s = {h:[] for h in range(1,8)}
    n_te = Xte.shape[0]
    for i in range(1, n_te-8):
        x_last = Xs[t_split+i-1]
        s = Pin @ (x_last - mu_sub)
        r = (x_last - mu_sub) - Lsub @ s
        for h in range(1,8):
            tgt = Xs[t_split+i-1+h]
            pred_b = mu_sub + Lsub @ (phi**h * s) + (phi_res**h)*r
            errs[h].append(pred_b - tgt)
            pred_s = mu_cell + (phi_cell**h)*(x_last-mu_cell)
            errs_s[h].append(pred_s - tgt)
    out_b = eval_h(errs); out_s = eval_h(errs_s)
    results[K] = (out_s, out_b, phi_raw, phi_pool, phi_res)
    print(f"K={K:3d} resid_phi={phi_res:.3f} | per-cell AR1: " + " ".join(f"h{h}={out_s[h]:.3f}" for h in range(1,8)))
    print(f"          banded  PC : " + " ".join(f"h{h}={out_b[h]:.3f}" for h in range(1,8)))
    print(f"          gain       : " + " ".join(f"h{h}={out_s[h]-out_b[h]:+.3f}" for h in range(1,8)))

# phi spectrum summary for K=60
Lk, phi, phi_res, phi_raw, phi_pool = banded_forecast(60)
v = S2[:60]**2; w = v/v.sum()
print(f"\nphi spectrum (K=60, honest train-fit): pooled={phi_pool:.3f}")
print("  PC1-5 :", np.round(phi_raw[:5],3), " w=", np.round(w[:5],3))
print("  PC6-10:", np.round(phi_raw[5:10],3), " w=", np.round(w[5:10],3))
print("  PC11-20:", np.round(phi_raw[10:20],2))
print("  PC21-40:", np.round(phi_raw[20:40],2))
print("  PC41-60:", np.round(phi_raw[40:60],2))
print(f"  variance-weighted phi^h: " + " ".join(f"h{h}={np.sum(w*phi_raw**h):.3f}" for h in range(1,8)))

# theoretical: variance-weighted (phi^h) vs (pooled phi)^h
print(f"  (pooled phi)^h       : " + " ".join(f"h{h}={phi_pool**h:.3f}" for h in range(1,8)))

# ---------- ALSO: single-scale denoised AR(1) (proxy for our pipeline) ----------
K=60
Lk = V2[:K].T; Lsub = Lk[pos]; Pin = np.linalg.pinv(Lsub); mu_sub = mu[pos]
P2 = (U2[:,:K]*S2[:K])
phi_pool2 = np.corrcoef(P2.ravel(), np.roll(P2,1,axis=0).ravel())[0,1]  # rough pooled
errs_k = {h:[] for h in range(1,8)}
for i in range(1, Xte.shape[0]-8):
    x_last = Xs[t_split+i-1]
    s = Pin @ (x_last - mu_sub)
    for h in range(1,8):
        tgt = Xs[t_split+i-1+h]
        pred = mu_sub + Lsub @ ((phi_pool**h)*s)   # single phi on denoised state
        errs_k[h].append(pred - tgt)
out_k = eval_h(errs_k)
print(f"\nsingle-phi denoised AR(1) (pipeline proxy): " + " ".join(f"h{h}={out_k[h]:.3f}" for h in range(1,8)))
print(f"banded gain vs pipeline proxy: " + " ".join(f"h{h}={out_k[h]-results[60][1][h]:+.3f}" for h in range(1,8)))
