"""2-b step 4: PCA of detrended field + per-PC dynamics; test anchors; D decomposition."""
import numpy as np, pandas as pd
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
A_dt, yms, mu_c, beta_c, tbar_c, t_abs_tr = d['A_dt'], d['yms'], d['mu_c'], d['beta_c'], d['tbar_c'], d['t_abs_tr']
lat_c, lon_c = d['lat_c'], d['lon_c']
T, n_cells = A_dt.shape

full = ~np.isnan(A_dt).any(axis=0)
A_f = A_dt[:, full] - np.nanmean(A_dt[:, full], axis=0)
U, S_, Vt = np.linalg.svd(A_f, full_matrices=False)
v_exp = (S_**2)/(S_**2).sum()
print(f"(D) detrended field spectrum: PC1={v_exp[0]*100:.1f}% PC1-5={v_exp[:5].sum()*100:.1f}% PC1-10={v_exp[:10].sum()*100:.1f}% PC1-50={v_exp[:50].sum()*100:.1f}% PC1-100={v_exp[:100].sum()*100:.1f}%")
P = (U[:,:100]*S_[:100]).astype(np.float64)   # PC scores (T,100)
L = Vt[:100].T                                 # loadings (n_full,100)

# spatial structure of loadings: corr with lat (hemispheric?), with lon, and roughness
lat_f = lat_c[full]; lon_f = lon_c[full]
print("\nPC loading structure (corr loading vs |lat|, lat, lon; loading spatial smoothness = corr with 1-deg neighbor mean):")
# neighbor map
key = {(int(round(a*10)), int(round(b*10))): i for i,(a,b) in enumerate(zip(lat_c, lon_c))}
key_f = {(int(round(a*10)), int(round(b*10))): i for i,(a,b) in enumerate(zip(lat_f, lon_f))}
nb_counts = np.zeros(len(lat_f))
for (a,b), i in key_f.items():
    for da,db in [(10,0),(-10,0),(0,10),(0,-10)]:
        if (a+da, b+db) in key_f: nb_counts[i]+=1
print(f"  cells with 4 neighbors: {(nb_counts==4).mean()*100:.1f}%")
# neighbor mean via adjacency (vectorized-ish)
import collections
adj = collections.defaultdict(list)
for (a,b), i in key_f.items():
    for da,db in [(10,0),(-10,0),(0,10),(0,-10)]:
        j = key_f.get((a+da, b+db))
        if j is not None: adj[i].append(j)
for k in [0,1,2,4,9,49,99]:
    l = L[:,k]
    r_lat = np.corrcoef(l, lat_f)[0,1]
    r_alat = np.corrcoef(l, np.abs(lat_f))[0,1]
    r_lon = np.corrcoef(l, lon_f)[0,1]
    # neighbor smoothness: corr(l_c, mean nbrs)
    sm_i = []
    own_i = []
    for i, js in adj.items():
        if len(js)>=2: sm_i.append(np.mean(l[js])); own_i.append(l[i])
    r_sm = np.corrcoef(np.array(own_i), np.array(sm_i))[0,1]
    print(f"  PC{k+1:3d} (v={v_exp[k]*100:5.2f}%): corr|lat|={r_alat:+.3f} corr_lat={r_lat:+.3f} corr_lon={r_lon:+.3f} nbr-smooth={r_sm:.4f}")

# per-PC ACF and AR fits
print("\nPC score dynamics (AR(1) phi, AR(2) roots, 1-step R2 of AR1 vs AR2):")
def ar_fit(x, order):
    X0 = x[order:]
    A_ = np.column_stack([np.ones(len(X0))]+[x[order-1-k: len(x)-1-k] for k in range(order)])
    c, res, *_ = np.linalg.lstsq(A_, X0, rcond=None)
    pred = A_@c
    return c, 1 - ((X0-pred)**2).mean()/X0.var()
rows=[]
for k in range(100):
    s = P[:,k]
    r1 = np.corrcoef(s[:-1], s[1:])[0,1]
    c2, r2in = ar_fit(s, 2)
    disc = c2[1]**2+4*c2[2]
    rows.append((k+1, v_exp[k], r1, c2[1], c2[2], disc, r2in))
rows=np.array(rows)
print("  PC | varexp | AR1_phi | AR2_phi1 | AR2_phi2 | disc | AR2_R2_in")
for k in list(range(12))+[19,29,49,99]:
    r = rows[k-1]
    print(f"  PC{k:3d} {r[1]*100:6.2f}%  {r[2]:+.3f}  {r[3]:+.3f}  {r[4]:+.3f}  {r[5]:+.3f}  {r[6]:.3f}")
phis = rows[:,2]
w = rows[:,1]/rows[:,1].sum()
print(f"  AR1 phi: weighted-mean={np.sum(w*phis):.3f}, range=[{phis.min():.3f},{phis.max():.3f}], frac phi>0.9: {(phis>0.9).mean():.2f}, frac phi<0.4: {(phis<0.4).mean():.2f}")
print(f"  variance-weighted phi^h for h=1..7: " + " ".join(f"h{h}={np.sum(w*(phis**h)):.3f}" for h in range(1,8)))
n_complex = (rows[:,5]<0).sum()
print(f"  PCs with complex AR(2) roots: {n_complex}/100 (in top-20: {(rows[:20,5]<0).sum()})")
# oscillation period for complex ones in top 20
for k in range(20):
    if rows[k,5]<0:
        R = np.sqrt(-rows[k,4]); th = np.arccos(np.clip(rows[k,3]/(2*R),-1,1))
        print(f"    PC{k+1}: R={R:.3f} period={2*np.pi/th:.1f} mo")

# ---------- ORACLE: per-PC AR forecast vs scalar-phi, in CELL space ----------
print("\nORACLE per-PC AR(1) vs scalar-phi AR(1), cell-space multi-step RMSE (fit 2002-10, eval 2011-15, 2000 cells):")
rng = np.random.default_rng(0)
pos = rng.choice(A_f.shape[1], 2000, replace=False)
Xs = A_f[:, pos]
t_split = int(np.searchsorted(yms, 201101))
# per-cell scalar phi (fit on train part)
Xtr = Xs[:t_split]; Xte = Xs[t_split:]
phi_cell = np.array([np.corrcoef(Xtr[:-1,j], Xtr[1:,j])[0,1] for j in range(Xs.shape[1])])
mu_cell = Xtr.mean(0)
# per-PC phi from full-basis PCA (fit on train era only, honest)
Atr = A_f[:t_split]; Atr = Atr - Atr.mean(0)
U2,S2,V2 = np.linalg.svd(Atr, full_matrices=False)
K=99
P2 = (U2[:,:K]*S2[:K])
phi_pc = np.array([np.corrcoef(P2[:-1,k],P2[1:,k])[0,1] for k in range(K)])
mean_pc = P2.mean(0)
mu_full = A_f[:t_split].mean(0)  # cell mean
Lk = V2[:K].T  # loadings (n_full, K)
# express test cell values in PC space using train loadings
def pc_forecast(hist_fields, h):
    # hist_fields: (T_h, n_cells_sub) -> project each month? we only need last state
    # state = last month's field; project onto PCs using pseudo-inverse loadings on SUBSET cells
    Lsub = Lk[pos]  # (2000, K)
    Pin = np.linalg.pinv(Lsub)  # (K, 2000)
    x_last = hist_fields[-1] - mu_full[pos]
    s = Pin @ x_last
    s_pred = mean_pc + (phi_pc**h)*(s-mean_pc)
    return mu_full[pos] + Lsub @ s_pred
n_te = Xte.shape[0]
err_scalar = {h:[] for h in range(1,8)}
err_pc = {h:[] for h in range(1,8)}
for i in range(1, n_te-8):
    x_last = Xs[t_split+i-1]  # last observed month
    for h in range(1,8):
        tgt = Xs[t_split+i-1+h]
        pred_s = mu_cell + (phi_cell**h)*(x_last-mu_cell)
        err_scalar[h].append(pred_s - tgt)
        pred_p = pc_forecast(Xs[:t_split+i], h)
        err_pc[h].append(pred_p - tgt)
out_s = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in err_scalar.items()}
out_p = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in err_pc.items()}
print("  per-cell AR1: " + " ".join(f"h{h}={v:.4f}" for h,v in out_s.items()))
print("  per-PC  AR1: " + " ".join(f"h{h}={v:.4f}" for h,v in out_p.items()))
print("  gain:        " + " ".join(f"h{h}={out_s[h]-out_p[h]:+.4f}" for h in out_s))
np.savez('/home/z/my-project/scripts/tb_pc.npz', v_exp=v_exp, P=P, L=L, phis=phis, phi_pc=phi_pc)
