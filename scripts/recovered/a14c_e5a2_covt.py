"""
Agent 14-c E5a-2: does the LOCAL cov field at month t (time-varying D tracker) beat
static S in the D-tilde? Feature families with cov_field(t) and drift, protocol 1.
"""
import numpy as np

DATA = '/home/z/my-project/data'
cache = np.load(f'{DATA}/a14c_cache.npz')
cache3 = np.load(f'{DATA}/a14c_cache3.npz')
mu_c = cache['mu_c']; beta_c = cache['beta_c']; tbar_c = cache['tbar_c']
A = cache['A']; anchors = [int(a) for a in cache['anchors']]; t_anc = cache['t_anc'].astype(float)
t_mid = float(cache['t_mid']); n_cells = len(mu_c)
V_an = cache['V_an']; full = cache['full']; full_idx = np.where(full)[0]
S = cache3['S']; S_early = cache3['S_early']; S_late = cache3['S_late']
Sdrift = S_late - S_early
cfs = cache3['cov_field_stack']; all_m = [int(m) for m in cache3['all_m']]
cov_field = {m: cfs[i] for i, m in enumerate(all_m)}
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float64)

def cov_local(t, half=6):
    """mean of cov fields over test months within +/-half months of t"""
    ms = [m for m in all_m if abs(m - t) <= half]
    if not ms: ms = [min(all_m, key=lambda m: abs(m - t))]
    return np.nanmean(np.array([cov_field[m] for m in ms]), axis=0), len(ms)

t_e = t_anc[[0,1,2,3]].mean(); t_l = t_anc[[4,5]].mean()
def u_era(t): return np.clip((np.float64(t) - t_e)/(t_l - t_e), 0.0, 1.0)

def linfit_loo(j, Kb=50):
    idx = [k for k in range(6) if k != j]
    P = A[idx]; tp = t_anc[idx]; tm = tp.mean()
    X = np.column_stack([np.ones(len(idx)), tp - tm])
    okc = np.isfinite(P).all(axis=0)
    a_ = np.full(n_cells, np.nan); b_ = np.zeros(n_cells)
    co = np.linalg.solve(X.T@X, X.T@P[:, okc])
    a_[okc] = co[0] + co[1]*(tm - t_mid); b_[okc] = co[1]
    xg = b_[full_idx]; okg = np.isfinite(xg)
    VK = V_an[:, :Kb]
    b_[full_idx] = np.where(okg, VK @ (VK.T @ np.where(okg, xg, 0.0)), xg)
    return a_, b_

def dhat_loo(j):
    return np.nanmean(A[[k for k in range(6) if k != j]], axis=0)

def build_feats(j, t, Kb=50):
    f = {'Dhat': dhat_loo(j), 'S': S, 'trendex': trendex(t),
         'covt': cov_local(t)[0], 'covt3': cov_local(t, 3)[0],
         'covdrift': Sdrift*u_era(t)}
    a_, b_ = linfit_loo(j, Kb=Kb)
    f['drift'] = b_*(np.float64(t)-t_mid)
    return f

def loo_eval(featnames, ridge=1e-2, intercept=False):
    rms, per, ws = [], [], []
    for j in range(6):
        Xs, ys = [], []
        for a in range(6):
            if a == j: continue
            f = build_feats(a, t_anc[a])
            cols = [f[nm] for nm in featnames] + ([np.ones(n_cells)] if intercept else [])
            X = np.column_stack(cols)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(A[a])
            Xs.append(X[ok]); ys.append(A[a][ok])
        Xf = np.vstack(Xs); yf = np.concatenate(ys)
        w = np.linalg.solve(Xf.T@Xf + ridge*np.eye(Xf.shape[1]), Xf.T@yf)
        f = build_feats(j, t_anc[j])
        cols = [f[nm] for nm in featnames] + ([np.ones(n_cells)] if intercept else [])
        X = np.column_stack(cols)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(A[j])
        r = float(np.sqrt(np.mean((X[ok]@w - A[j][ok])**2)))
        rms.append(r); per.append(r); ws.append(w)
    return float(np.mean(rms)), per, np.mean(ws, axis=0)

print("=== E5a-2: local cov field as time-varying D tracker (protocol 1) ===")
for nm, feats in [
    ("v4 base [Dhat,S,trendex] (fixed)", None),
    ("[Dhat,S,trendex] refit", ['Dhat','S','trendex']),
    ("[Dhat,covt,trendex] (swap S->covt)", ['Dhat','covt','trendex']),
    ("[Dhat,S,covt,trendex]", ['Dhat','S','covt','trendex']),
    ("[Dhat,S,covt3,trendex] (3mo local)", ['Dhat','S','covt3','trendex']),
    ("[Dhat,covt] (drop trendex)", ['Dhat','covt']),
    ("[Dhat,S,covt,trendex,drift]", ['Dhat','S','covt','trendex','drift']),
    ("[Dhat,S,covt,trendex,covdrift]", ['Dhat','S','covt','trendex','covdrift']),
]:
    if feats is None:
        rms = 0.8189; per = None; w = np.array([0.70, 0.45, 0.073])
    else:
        rms, per, w = loo_eval(feats)
    print(f"  {nm:<42}: {rms:.4f}  w={np.round(w,3)}")

# per-anchor for the swap candidate
_, per_b, _ = loo_eval(['Dhat','S','trendex'])
_, per_c, _ = loo_eval(['Dhat','S','covt','trendex'])
print("\nper-anchor: base-refit vs +covt")
for a, rb, rc in zip(anchors, per_b, per_c):
    print(f"  {a}: {rb:.4f} -> {rc:.4f} ({rc-rb:+.4f})")

# how much does covt correlate with A at anchors (vs S)?
print("\ncorr(cov_field[a], A_a) vs corr(S, A_a):")
for j, a in enumerate(anchors):
    m = np.isfinite(cov_field[a]) & np.isfinite(A[j])
    print(f"  {a}: covt {np.corrcoef(cov_field[a][m], A[j][m])[0,1]:.3f}   S {np.corrcoef(S[m], A[j][m])[0,1]:.3f}")
