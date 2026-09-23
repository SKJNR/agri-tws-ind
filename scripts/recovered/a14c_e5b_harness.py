"""
Agent 14-c E5b: FULL anchor-LOO harness (protocol 2, the trusted one) with
time-varying cov-based D-tilde candidates.

Port of v3_dtilde.py protocol: predict anchor field j from previous anchor i via
forward Kalman (+ backward from next anchor), D-tilde LOO (excludes target j),
H/R recalibrated per fold. 4 consecutive pairs (gaps<=8), phi=0.74, K=200 denoise.
Baseline to beat: 0.7185-0.7200.

New candidates: D-tilde includes covt3(t) = mean cov-regression field over test
months within +/-3 of t (time-varying slow tracker), and obs variants
W'_m = cov_field_m - covt3(m) (local-mean removed from fast obs).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
cache = np.load(f'{DATA}/a14c_cache.npz')
mu_c = cache['mu_c']; beta_c = cache['beta_c']; tbar_c = cache['tbar_c']
V = cache['V_dt']; full = cache['full']; full_idx = np.where(full)[0]
A = cache['A']; anchors = [int(a) for a in cache['anchors']]; t_anc = cache['t_anc'].astype(float)
t_mid = float(cache['t_mid']); n_cells = len(mu_c)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)
def dn(field):
    out = field.copy()
    x = field[full_idx]
    okx = np.isfinite(x)
    out[full_idx] = np.where(okx, V @ (V.T @ np.where(okx, x, 0.0)), x)
    return out

# ---- rebuild cov fields / W ----
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['t_abs'] = test['time'].dt.year*12 + test['time'].dt.month - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values
Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[int(m)] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
AF = {a: A[j] for j, a in enumerate(anchors)}

def cov_local(t, half=3, exclude_self=False):
    ms = [m for m in all_m if abs(m - t) <= half and not (exclude_self and m == t)]
    if not ms:
        ms = [min(all_m, key=lambda m: abs(m - t))]
        if exclude_self and len(ms) == 1 and ms[0] == t:
            ms = sorted(all_m, key=lambda m: abs(m - t))[:2]
            ms = [m for m in ms if m != t] or [min(all_m, key=lambda m: abs(m-t))]
    return np.nanmean(np.array([cov_field[m] for m in ms]), axis=0), ms

def dhat_loo(jj):
    """jj = target anchor INDEX (0..5)"""
    return np.nanmean(A[[k for k in range(6) if k != jj]], axis=0)

def linfit_loo(jj, Kb=50):
    idx = [k for k in range(6) if k != jj]
    P = A[idx]; tp = t_anc[idx]; tm = tp.mean()
    X = np.column_stack([np.ones(len(idx)), tp - tm])
    okc = np.isfinite(P).all(axis=0)
    b_ = np.zeros(n_cells)
    co = np.linalg.solve(X.T@X, X.T@P[:, okc])
    b_[okc] = co[1]
    xg = b_[full_idx]; okg = np.isfinite(xg)
    VK = V[:, :Kb]  # use detrended basis for slope smoothing
    b_[full_idx] = np.where(okg, VK @ (VK.T @ np.where(okg, xg, 0.0)), xg)
    return b_

# ---- per-fold weight fitting (protocol 1 machinery, on anchors != target) ----
def fit_weights(jm, featnames, ridge=1e-2):
    """jm = target anchor MONTH"""
    jj = anchors.index(jm)
    Xs, ys = [], []
    for k in range(6):
        if anchors[k] == jm: continue
        t = t_anc[k]
        f = {'Dhat': dhat_loo(jj), 'S': S, 'trendex': trendex(t),
             'covt3': cov_local(t)[0], 'covexcl': cov_local(t, exclude_self=True)[0],
             'covt6': cov_local(t, 6)[0]}
        f['drift'] = linfit_loo(jj)*(t - t_mid)
        X = np.column_stack([f[nm] for nm in featnames])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(A[k])
        Xs.append(X[ok]); ys.append(A[k][ok])
    Xf = np.vstack(Xs); yf = np.concatenate(ys)
    return np.linalg.solve(Xf.T@Xf + ridge*np.eye(len(featnames)), Xf.T@yf)

def Dtil(jm, t, family, w=None):
    jj = anchors.index(jm)
    dh = dhat_loo(jj)
    f = {'Dhat': dh, 'S': S, 'trendex': trendex(t),
         'covt3': cov_local(t)[0], 'covexcl': cov_local(t, exclude_self=True)[0],
         'covt6': cov_local(t, 6)[0]}
    f['drift'] = linfit_loo(jj)*(np.float64(t) - t_mid)
    if family == 'base_fixed':
        return 0.70*dh + 0.45*S + 0.073*trendex(t)
    if family == 'v2_fixed':
        return 0.650*dh + 0.456*S + 0.073*trendex(t)
    return np.column_stack([f[nm] for nm in family]) @ w

def calib_HR(jm, phi, family, w, obs_mode):
    cs, zs, vfs = [], [], []
    for k in range(6):
        a = anchors[k]
        if a == jm: continue
        dj = Dtil(jm, a, family, w)
        obs = (cov_field[a] - S) if obs_mode == 'W' else (cov_field[a] - cov_local(a)[0])
        obs = dn(obs)
        ok = np.isfinite(obs) & np.isfinite(A[k]) & np.isfinite(dj)
        cs.append(np.cov(obs[ok], (A[k]-dj)[ok])[0,1]); zs.append(np.var(obs[ok]))
        vfs.append(np.nanvar(A[k]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

def obs_at(m, obs_mode):
    if m not in cov_field: return None
    if obs_mode == 'W': return cov_field[m] - S
    return cov_field[m] - cov_local(m)[0]

def fwd_pass(i, j, phi, family, w, obs_mode):
    H, R, var_f = calib_HR(j, phi, family, w, obs_mode)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = dn(AF[i] - Dtil(j, i, family, w))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    for m in range(i+1, j+1):
        x = phi*x; P = phi**2*P + q
        o = obs_at(m, obs_mode)
        if o is not None:
            o_ = dn(o); okw = np.isfinite(o_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(o_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

def bwd_pass(k, j, phi, family, w, obs_mode):
    H, R, var_f = calib_HR(j, phi, family, w, obs_mode)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = dn(AF[k] - Dtil(j, k, family, w))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[k]), P0, var_f).astype(np.float32)
    for m in range(k-1, j-1, -1):
        x = phi*x; P = phi**2*P + q
        o = obs_at(m, obs_mode)
        if o is not None:
            o_ = dn(o); okw = np.isfinite(o_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(o_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

pairs = []
for idx in range(len(anchors)-1):
    i, j = anchors[idx], anchors[idx+1]
    if j - i <= 8: pairs.append((i, j))

def run(family, obs_mode='W', phi=0.74, pred_offset=1, wfix=None):
    errs, details = [], []
    for (i, j) in pairs:
        if isinstance(family, list):
            w = wfix if wfix is not None else fit_weights(j, family)
        else:
            w = None
        xf, Pf = fwd_pass(i, j, phi, family, w, obs_mode)
        later = [a for a in anchors if a > j]
        if later:
            xb, Pb = bwd_pass(later[0], j, phi, family, w, obs_mode)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        pred = Dtil(j, j+pred_offset, family, w) + x
        tgt = AF[j]
        ok = np.isfinite(pred) & np.isfinite(tgt)
        e = float(np.sqrt(np.mean((pred[ok]-tgt[ok])**2)))
        errs.append(e); details.append((j, e))
    return float(np.mean(errs)), details

print("=== E5b: FULL anchor-LOO harness (protocol 2) ===")
print(f"pairs: {[(i,j,j-i) for i,j in pairs]}\n")
r, d = run('base_fixed'); print(f"baseline v4 (0.70/0.45/0.073), obs W  : {r:.4f}  {np.round([x[1] for x in d],4)}")
r, d = run('v2_fixed');  print(f"baseline v2 (0.650/0.456/0.073), obs W : {r:.4f}")
r, d = run('base_fixed', pred_offset=0); print(f"baseline v4, Dtil at t=j (not j+1)      : {r:.4f}")
print()
FAMS = [
    (['Dhat','S','covt3','trendex'], 'covt3 family'),
    (['Dhat','covt3','trendex'], 'covt3 no-S'),
    (['Dhat','S','covt6','trendex'], 'covt6 family'),
    (['Dhat','S','covexcl','trendex'], 'covexcl (no concurrent month)'),
    (['Dhat','S','covt3','trendex','drift'], 'covt3 + drift'),
]
for fam, nm in FAMS:
    w_all = [fit_weights(j, fam) for (i, j) in pairs]
    wm = np.mean(w_all, axis=0)
    r, d = run(fam, wfix=None)
    # note: run() refits weights per fold internally when wfix None
    print(f"{nm:<38}: {r:.4f}  mean w={np.round(wm,3)}")
    r2, d2 = run(fam, obs_mode='Wloc')
    print(f"{'':<38}  obs=W' (local-mean removed): {r2:.4f}")
