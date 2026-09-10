"""
Agent 14-c E5a: D-tilde anchor-LOO with feature-based candidates (protocol 1).

LOO protocol: for each target anchor j, all anchor-derived features (D-hat, linear fit)
are built from anchors != j; weights are ridge-fit on the 5 non-target anchors
(target = their own fields A_a); RMSE evaluated on A_j (includes target fast+noise,
floor sqrt(0.377)=0.614). Baselines: D-hat only (known ~0.842), v2 combo (~0.820),
v4 combo (0.70/0.45/0.073).

Families:
  base    : [Dhat, S, trendex(t)]                     (current)
  drift   : base + b_hat*(t-t_mid)   (PC-smoothed per-cell slope from LOO anchors)
  covdrift: base + (S_late-S_early)*u(t)              (era ramp)
  full    : base + drift + covdrift
  era     : era-interpolated D-hat + S + trendex
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
cache = np.load(f'{DATA}/a14c_cache.npz')
cache3 = np.load(f'{DATA}/a14c_cache3.npz')
mu_c = cache['mu_c']; beta_c = cache['beta_c']; tbar_c = cache['tbar_c']
A = cache['A']; anchors = [int(a) for a in cache['anchors']]; t_anc = cache['t_anc'].astype(float)
t_mid = float(cache['t_mid']); n_cells = len(mu_c)
V_an = cache['V_an']; full = cache['full']; full_idx = np.where(full)[0]
S = cache3['S']; S_early = cache3['S_early']; S_late = cache3['S_late']
Sdrift = S_late - S_early
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float64)
def corr2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0,1])

# era ramps
t_e = t_anc[[0,1,2,3]].mean(); t_l = t_anc[[4,5]].mean()
def u_era(t): return np.clip((np.float64(t) - t_e)/(t_l - t_e), 0.0, 1.0)

def linfit_loo(j, Kb=20, Ka=None):
    """per-cell (a, b) from anchors != j, PC-smoothed. a at t_mid."""
    idx = [k for k in range(6) if k != j]
    P = A[idx]; tp = t_anc[idx]
    tm = tp.mean()
    X = np.column_stack([np.ones(len(idx)), tp - tm])
    okc = np.isfinite(P).all(axis=0)
    a_ = np.full(n_cells, np.nan); b_ = np.full(n_cells, np.nan)
    if okc.sum() > 0:
        co = np.linalg.solve(X.T@X, X.T@P[:, okc])
        # translate intercept to t_mid reference
        a_[okc] = co[0] + co[1]*(tm - t_mid); b_[okc] = co[1]
    # fill b with 0 (no drift info) where missing
    b_ = np.where(np.isfinite(b_), b_, 0.0)
    # PC smoothing on full-history cells
    if Kb is not None:
        xg = b_[full_idx]; okg = np.isfinite(xg)
        VK = V_an[:, :Kb]
        b_[full_idx] = np.where(okg, VK @ (VK.T @ np.where(okg, xg, 0.0)), xg)
        # renormalize variance? no - keep raw projection
    if Ka is not None:
        xg = a_[full_idx]; okg = np.isfinite(xg)
        VK = V_an[:, :Ka]
        a_[full_idx] = np.where(okg, VK @ (VK.T @ np.where(okg, xg, 0.0)), xg)
    return a_, b_

def dhat_loo(j):
    idx = [k for k in range(6) if k != j]
    return np.nanmean(A[idx], axis=0)

def d_era_loo(j):
    early = [k for k in [0,1,2,3] if k != j]; late = [k for k in [4,5] if k != j]
    De = np.nanmean(A[early], axis=0) if early else np.full(n_cells, np.nan)
    Dl = np.nanmean(A[late], axis=0) if late else np.full(n_cells, np.nan)
    return De, Dl

def build_feats(j, t, Kb=20, with_drift=True, with_covdrift=True, use_era=False):
    dh = dhat_loo(j)
    f = {'Dhat': dh, 'S': S, 'trendex': trendex(t)}
    if with_drift:
        a_, b_ = linfit_loo(j, Kb=Kb)
        f['drift'] = b_ * (np.float64(t) - t_mid)
    if with_covdrift:
        f['covdrift'] = Sdrift * u_era(t)
    if use_era:
        De, Dl = d_era_loo(j)
        u = u_era(t)
        f['Dhat_era'] = (1-u)*np.nan_to_num(De) + u*np.nan_to_num(Dl)
    return f

def loo_eval(featnames, Kb=20, ridge=1e-2, use_era=False, fixed_w=None):
    """returns mean RMSE over 6 folds (+ per-fold)"""
    rms, per = [], []
    for j in range(6):
        # fit weights on anchors != j
        Xs, ys = [], []
        for a in range(6):
            if a == j: continue
            f = build_feats(a, t_anc[a], Kb=Kb, use_era=use_era)
            cols = [f[nm] for nm in featnames]
            X = np.column_stack(cols + [np.ones(n_cells)])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(A[a])
            Xs.append(X[ok]); ys.append(A[a][ok])
        Xf = np.vstack(Xs); yf = np.concatenate(ys)
        if fixed_w is not None:
            w = fixed_w
        else:
            w = np.linalg.solve(Xf.T@Xf + ridge*np.eye(Xf.shape[1]), Xf.T@yf)
        # evaluate on j
        f = build_feats(j, t_anc[j], Kb=Kb, use_era=use_era)
        X = np.column_stack([f[nm] for nm in featnames] + [np.ones(n_cells)])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(A[j])
        pred = X[ok] @ w
        r = float(np.sqrt(np.mean((pred - A[j][ok])**2)))
        rms.append(r); per.append((anchors[j], r))
    return float(np.mean(rms)), per, w

print("=========== E5a: D-tilde anchor-LOO (protocol 1) ===========")
print(f"floor (fast+noise at target anchor): {np.sqrt(0.377):.3f}\n")

# sanity baselines
r, per, _ = loo_eval(['Dhat'], fixed_w=np.array([1.0, 0.0]))
print(f"D-hat only                      : {r:.4f}   per-anchor {np.round([p[1] for p in per],3)}")
r, per, w = loo_eval(['Dhat','S','trendex'], fixed_w=np.array([0.650, 0.456, 0.073, 0.0]))
print(f"v2 combo (0.650/0.456/0.073)    : {r:.4f}")
r, per, w = loo_eval(['Dhat','S','trendex'], fixed_w=np.array([0.70, 0.45, 0.073, 0.0]))
print(f"v4 combo (0.70/0.45/0.073)      : {r:.4f}   <- BASELINE")
r, per, w = loo_eval(['Dhat','S','trendex'])
print(f"base LOO-refit weights          : {r:.4f}   w={np.round(w,3)}")

print()
for Kb in [10, 20, 50]:
    r, per, w = loo_eval(['Dhat','S','trendex','drift'], Kb=Kb)
    print(f"+drift(Kb={Kb:>2}) refit            : {r:.4f}   w={np.round(w,3)}")
r, per, w = loo_eval(['Dhat','S','trendex','covdrift'])
print(f"+covdrift refit                 : {r:.4f}   w={np.round(w,3)}")
r, per, w = loo_eval(['Dhat','S','trendex','drift','covdrift'], Kb=20)
print(f"+drift(Kb=20)+covdrift refit    : {r:.4f}   w={np.round(w,3)}")
r, per, w = loo_eval(['Dhat_era','S','trendex'], use_era=True)
print(f"era-interp D-hat + S + trendex  : {r:.4f}   w={np.round(w,3)}")
r, per, w = loo_eval(['Dhat','S','drift','covdrift'], Kb=20)
print(f"no-trendex variant              : {r:.4f}   w={np.round(w,3)}")

# per-anchor detail for the best candidates
print("\nper-anchor RMSE detail (v4 baseline vs +drift+covdrift):")
_, per_b, _ = loo_eval(['Dhat','S','trendex'], fixed_w=np.array([0.70, 0.45, 0.073, 0.0]))
_, per_n, _ = loo_eval(['Dhat','S','trendex','drift','covdrift'], Kb=20)
for (a1, r1), (a2, r2) in zip(per_b, per_n):
    print(f"  anchor {a1}: base {r1:.4f} -> new {r2:.4f}  ({r2-r1:+.4f})")

# what does drift alone look like (replace trendex with drift entirely)?
r, per, w = loo_eval(['Dhat','S','drift'], Kb=20)
print(f"\nDhat+S+drift (no trendex)       : {r:.4f}   w={np.round(w,3)}")
# drift with stronger smoothing and era-interp
r, per, w = loo_eval(['Dhat_era','S','trendex','drift','covdrift'], Kb=20, use_era=True)
print(f"era+drift+covdrift              : {r:.4f}   w={np.round(w,3)}")
