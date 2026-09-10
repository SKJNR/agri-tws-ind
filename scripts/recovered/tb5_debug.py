"""Debug per-PC oracle: why does projection-based forecast look worse at h1?"""
import numpy as np
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
A_dt, yms = d['A_dt'], d['yms']
full = ~np.isnan(A_dt).any(axis=0)
A_f = A_dt[:, full] - np.nanmean(A_dt[:, full], axis=0)
rng = np.random.default_rng(0)
pos = rng.choice(A_f.shape[1], 2000, replace=False)
Xs = A_f[:, pos]
t_split = int(np.searchsorted(yms, 201101))

Atr = A_f[:t_split]; Atr = Atr - Atr.mean(0)
U2,S2,V2 = np.linalg.svd(Atr, full_matrices=False)
K=99
Lk = V2[:K].T
Lsub = Lk[pos]
Pin = np.linalg.pinv(Lsub)
mu_full = A_f[:t_split].mean(0)
P2 = (U2[:,:K]*S2[:K])
phi_pc = np.array([np.corrcoef(P2[:-1,k],P2[1:,k])[0,1] for k in range(K)])

# 1) projection quality on eval window
Xte = Xs[t_split:]
mu_sub = mu_full[pos]
proj = mu_sub + Lsub @ (Pin @ (Xte - mu_sub).T).T if False else None
Z = (Xte - mu_sub) @ Lsub  # scores via normal equations? no...
# correct: scores = pinv(Lsub) @ x  -> x_hat = Lsub @ pinv(Lsub) @ x
P99 = Lsub @ Pin  # (2000,2000) projection matrix
Xhat = (Xte - mu_sub) @ P99.T + mu_sub
resid = Xte - Xhat
print(f"eval anomaly var: {Xte.var():.4f}")
print(f"projection residual var: {resid.var():.4f}  (fraction dropped: {resid.var()/Xte.var():.3f})")
# how much of the residual is noise? compare with train-era
Xtr = Xs[:t_split]
Xhat_tr = (Xtr - mu_sub) @ P99.T + mu_sub
print(f"train-era projection residual var: {(Xtr-Xhat_tr).var():.4f}")

# 2) correlation of per-PC pred with target at h1
phi_cell = np.array([np.corrcoef(Xtr[:-1,j],Xtr[1:,j])[0,1] for j in range(2000)])
n_te = Xte.shape[0]
i = 5
x_last = Xs[t_split+i-1]; tgt = Xs[t_split+i]
pred_cell = mu_sub + phi_cell*(x_last-mu_sub)
s = Pin @ (x_last - mu_sub)
pred_pc = mu_sub + Lsub @ (phi_pc*s)
print(f"\nh1 example: corr(pred_cell, tgt)={np.corrcoef(pred_cell,tgt)[0,1]:.4f}, corr(pred_pc,tgt)={np.corrcoef(pred_pc,tgt)[0,1]:.4f}")
print(f"rmse cell={np.sqrt(np.mean((pred_cell-tgt)**2)):.4f}, rmse pc={np.sqrt(np.mean((pred_pc-tgt)**2)):.4f}")
print(f"var pred_cell={pred_cell.var():.4f}, var pred_pc={pred_pc.var():.4f}, var tgt={tgt.var():.4f}")
print(f"var scores: {s.var():.4f}; phi_pc top10: {np.round(phi_pc[:10],3)}")
# check: are scores from x_last similar to train-era scores? (drift in PC space?)
s_tr = Pin @ (Xtr[-1] - mu_sub)
print(f"score rms train-last vs eval: {np.sqrt(np.mean(s_tr**2)):.3f} vs {np.sqrt(np.mean(s**2)):.3f}")
# score variance per PC in train era
S_tr_all = (Xtr - mu_sub) @ np.linalg.pinv(Lsub).T if False else (Pin @ (Xtr - mu_sub).T).T
print(f"train score var (first 10): {np.round(S_tr_all.var(0)[:10],2)}")
print(f"eval  score var (first 10): {np.round(((Pin @ (Xte - mu_sub).T).T).var(0)[:10],2)}")
