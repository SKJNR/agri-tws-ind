"""
V3 experiments: spatial denoising + backward anchor pass. No LB cost — anchor-LOO validated.

E1: baseline (v2b config, phi=0.74) on 4 consecutive-anchor pairs  [expect ~0.74]
E2: + PC-denoise anchor init (K = 50/100/200/400)
E3: + PC-denoise cov observations
E4: + backward pass from next anchor (precision-weighted blend)
E5: best combo summary
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

# ---------------- load train ----------------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_tr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)

# per-cell trend (calendar OLS)
F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
beta_c = np.where(np.nansum(td*td, axis=0) > 100, np.nansum(td*F64, axis=0)/np.where(np.nansum(td*td,axis=0)>0, np.nansum(td*td,axis=0), 1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# detrended field + PC basis (full cells)
full = ~np.isnan(F).any(axis=0)
n_full = int(full.sum())
TD = t_abs_tr[:, None] - tbar_c[None, :]                    # (T, n_cells)
A_dt = F64 - mu_c[None, :] - TD * beta_c[None, :]           # detrended anomaly
A_dtf = A_dt[:, full]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
U, S_, Vt = np.linalg.svd(A_dtf, full_matrices=False)
v_exp = S_**2/(S_**2).sum()
print(f"detrended field spectrum: PC1={v_exp[0]*100:.1f}% PC1-10={v_exp[:10].sum()*100:.1f}% PC1-50={v_exp[:50].sum()*100:.1f}% PC1-100={v_exp[:100].sum()*100:.1f}% PC1-200={v_exp[:200].sum()*100:.1f}% PC1-400={v_exp[:400].sum()*100:.1f}%")
VK = Vt[:400].T.astype(np.float32)   # (n_full, 400) basis
cells_full_idx = np.where(full)[0]
# map full-index -> all-cells index helpers
def denoise(field):
    """field: (n_cells,) anomaly; project full-cells part onto top-400 PCs, keep rest"""
    out = field.copy()
    x = field[full]
    okx = np.isfinite(x)
    xz = np.where(okx, x, 0.0)
    proj = VK @ (VK.T @ xz)
    out[full] = np.where(okx, proj, x)
    return out

# ---------------- load test ----------------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values
cc_t = test['cc'].values
msk = test['masked'].values

Z = train[COVS].values.astype('float32')
yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

Zt_raw = test[COVS].values.astype('float32')
okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

# D-tilde (LOO-safe) with trendex, weights from v2 build
w1, w2, w3 = 0.650, 0.456, 0.073
def Dtil_loo(j, t):
    dloo = np.nanmean(np.array([AF[a] for a in anchors if a != j]), axis=0)
    return (w1*dloo + w2*S + w3*trendex(t)).astype(np.float32)

# W quality in PC space (diagnostic)
print("\nW vs (F - D-tilde) correlation, raw vs PC-denoised (at anchors, LOO D-tilde):")
for a in anchors:
    dj = Dtil_loo(a, a)
    raw_r = np.corrcoef(np.nan_to_num(W[a]), np.nan_to_num(AF[a]-dj))[0,1]
    wd = denoise(W[a]); fd = denoise(AF[a]-dj)
    dn_r = np.corrcoef(np.nan_to_num(wd), np.nan_to_num(fd))[0,1]
    print(f"  {a%10000}: raw {raw_r:.3f} -> denoised {dn_r:.3f}")

# ---------------- Kalman passes ----------------
def calib_HR(j, phi, dn_obs):
    cs, zs, vfs = [], [], []
    for a in anchors:
        if a == j: continue
        dj = Dtil_loo(a, a)
        w_ = denoise(W[a]) if dn_obs else W[a]
        ok = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

def fwd_pass(i, j, phi, dn_init, dn_obs):
    """forward Kalman from anchor i to target month j; returns (x at j after obs_j, P at j)"""
    H, R, var_f = calib_HR(j, phi, dn_obs)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    dj = Dtil_loo(j, i)
    f0 = AF[i] - dj
    f0 = denoise(f0) if dn_init else f0
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    for m in range(i+1, j+1):
        x = phi*x; P = phi**2*P + q
        w_ = W.get(m)
        if w_ is not None:
            w_ = denoise(w_) if dn_obs else w_
            okw = np.isfinite(w_)
            K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + K*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-K*H)*P, P)
    return x, P

def bwd_pass(k, j, phi, dn_init, dn_obs):
    """backward Kalman from later anchor k down to target month j; returns (x at j, P at j)"""
    H, R, var_f = calib_HR(j, phi, dn_obs)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    dj = Dtil_loo(j, k)
    f0 = AF[k] - dj
    f0 = denoise(f0) if dn_init else f0
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[k]), P0, var_f).astype(np.float32)
    for m in range(k-1, j-1, -1):
        x = phi*x; P = phi**2*P + q
        w_ = W.get(m)
        if w_ is not None:
            w_ = denoise(w_) if dn_obs else w_
            okw = np.isfinite(w_)
            K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + K*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-K*H)*P, P)
    return x, P

# ---------------- harness ----------------
pairs = []
for idx in range(len(anchors)-1):
    i, j = anchors[idx], anchors[idx+1]
    if j - i <= 8:
        pairs.append((i, j, j-i))

def run_config(phi, dn_init, dn_obs, use_bwd, K=None):
    global VK
    if K is not None:
        VK = Vt[:K].T.astype(np.float32)
    errs = []
    for (i, j, g) in pairs:
        xf, Pf = fwd_pass(i, j, phi, dn_init, dn_obs)
        later = [a for a in anchors if a > j]
        if use_bwd and later:
            k = later[0]
            xb, Pb = bwd_pass(k, j, phi, dn_init, dn_obs)
            wgt = (1.0/np.maximum(Pf,1e-6)) / (1.0/np.maximum(Pf,1e-6) + 1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        dj = Dtil_loo(j, j+1)
        pred = dj + x
        tgt = AF[j]
        ok = np.isfinite(pred) & np.isfinite(tgt)
        errs.append(float(np.sqrt(np.mean((pred[ok]-tgt[ok])**2))))
    return np.mean(errs), errs

print("\n=== E1: baselines (phi=0.74) ===")
base, pe = run_config(0.74, False, False, False)
print(f"  v2b-equivalent (no denoise, fwd only): {base:.4f}  per-pair {np.round(pe,3)}")

print("\n=== E2: + PC-denoise anchor init ===")
for K in [50, 100, 200, 400]:
    r, pe = run_config(0.74, True, False, False, K=K)
    print(f"  init-denoise K={K:3d}: {r:.4f}  per-pair {np.round(pe,3)}")

print("\n=== E3: + PC-denoise observations (init raw) ===")
for K in [50, 100, 200]:
    r, pe = run_config(0.74, False, True, False, K=K)
    print(f"  obs-denoise K={K:3d}: {r:.4f}  per-pair {np.round(pe,3)}")

print("\n=== E3b: both denoised ===")
for K in [100, 200]:
    r, pe = run_config(0.74, True, True, False, K=K)
    print(f"  both K={K:3d}: {r:.4f}  per-pair {np.round(pe,3)}")

print("\n=== E4: + backward pass ===")
for K in [100, 200]:
    r, pe = run_config(0.74, True, True, True, K=K)
    print(f"  both+bwd K={K:3d}: {r:.4f}  per-pair {np.round(pe,3)}")
r, pe = run_config(0.74, False, False, True)
print(f"  no-denoise + bwd:    {r:.4f}  per-pair {np.round(pe,3)}")

print("\n=== E5: phi sweep on best config ===")
for phi in [0.70, 0.74, 0.78, 0.82]:
    r, pe = run_config(phi, True, True, True, K=200)
    print(f"  phi={phi}: {r:.4f}  per-pair {np.round(pe,3)}")
