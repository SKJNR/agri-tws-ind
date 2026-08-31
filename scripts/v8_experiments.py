"""
V8 EXPERIMENTS — test-era harness (anchor-LOO) + lambda_test estimation.

E0a: spatial-roughness noise estimate: anchors (test era) vs train months -> lambda_test
E0b: LAM x PHI sweep on the 4 consecutive-anchor LOO pairs (v2b-style core)
E0c: sanity — TWS_t NaN among masked rows; cov availability
E2 : drift-weighted D-hat (tau sweep) on LOO
E3 : era-split S on LOO
E4 : 5-dim per-covariate Kalman obs vs scalar regression obs on LOO

All LOO numbers are comparable to history: v2b-core LOO ~0.7414 (train-reg W),
harness floor ~0.7185-0.7200 (with denoise+bwd).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
rng = np.random.default_rng(0)

# ================= load =================
print("loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique()); T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_tr = np.array([(int(v)//100)*12+(int(v)%100)-1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

# per-cell OLS trend
F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:,None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None,:]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where(sxx>100, np.nansum(td*F64,axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t)-tbar_c)*beta_c).astype(np.float32)

# denoiser basis (top-200 PCs of detrended train field, full-history cells)
full = ~np.isnan(F).any(axis=0)
TDm = t_abs_tr[:,None] - tbar_c[None,:]
A_dt = F64 - mu_c[None,:] - TDm*beta_c[None,:]
A_dtf = A_dt[:, full]; A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
_,_,Vt = np.linalg.svd(A_dtf, full_matrices=False)
V = Vt[:200].T.astype(np.float32)
def dn(field):
    out = field.copy(); x = field[full]; okx = np.isfinite(x)
    out[full] = np.where(okx, V@(V.T@np.where(okx,x,0.0)), x)
    return out

# global cov regression (TWS ~ covs) on train
Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# ================= test =================
print("loading test...", flush=True)
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'],test['lon'])]
assert (test['cc']>=0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values

# E0c sanity
print(f"\n[E0c] masked rows with non-NaN TWS_t: {int(np.isfinite(test['TWS_t'].values[msk]).sum())} (expect 0)")
covfin = {c: float(np.isfinite(test[c].values).mean()) for c in COVS}
print(f"[E0c] test cov finite fractions: {covfin}")

# cov fields per test month
Zt_raw = test[COVS].values.astype('float32')
okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta==m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W = {int(m): cov_field[m]-S for m in all_m}

# raw per-cov fields (for E4)
covF = {}
for j,c in enumerate(COVS):
    for m in np.sort(test['t_abs'].unique()):
        selm = (ta==m)
        fm = np.full(n_cells, np.nan, dtype=np.float32)
        v = test[c].values[selm]
        fm[cc_t[selm][np.isfinite(v)]] = v[np.isfinite(v)]
        covF[(j,int(m))] = fm
# per-cov temporal mean over test months, and deviation obs
muCov = {}
for j in range(5):
    M = np.array([covF[(j,int(m))] for m in all_m])
    muCov[j] = np.nanmean(M, axis=0)
Wj = {(j,int(m)): covF[(j,int(m))]-muCov[j] for j in range(5) for m in all_m}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac<0.01].index)
AF = {}
for a in anchors:
    sel = (ta==a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
print(f"anchors: {anchors}")

# ================= E0a: spatial roughness -> lambda =================
print("\n[E0a] spatial-roughness noise estimation", flush=True)
from scipy.spatial import cKDTree
cl = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
latv = cl['lat'].values.astype(np.float64); lonv = cl['lon'].values.astype(np.float64)
tree = cKDTree(np.column_stack([latv, lonv]))
d_self, nb = tree.query(np.column_stack([latv, lonv]), k=6)   # self + 5 nearest
NB = nb[:,1:].astype(np.int32)
dnn = np.median(d_self[:,1])
keep_nb = d_self[:,1:] <= max(1.6*dnn, dnn+0.05)
print(f"  median NN dist={dnn:.3f} deg; nbrs kept per cell: {keep_nb.sum(axis=1).mean():.2f}")

def noise_var(A):
    """A: (n_cells,) field. returns sigma^2 of spatially-white component."""
    idx = np.where(np.isfinite(A))[0]
    nbv = A[NB]                       # (n_cells,5)
    okn = np.isfinite(nbv) & keep_nb
    m = okn.sum(axis=1)
    valid = np.isfinite(A) & (m>=3)
    if valid.sum() < 100: return np.nan, 0
    nbm = np.where(okn, nbv, 0).sum(axis=1)/np.maximum(m,1)
    r = A - nbm
    sig2 = (r[valid]**2 * m[valid]/(m[valid]+1)).mean()
    return float(sig2), int(valid.sum())

# train months (anomaly field, full-history cells)
tr_s2 = []
Anom = F - mu_c[None,:]
for ti in range(T):
    s2, n = noise_var(Anom[ti])
    if n > 5000: tr_s2.append(s2)
tr_s2 = np.array(tr_s2)
tr_var = float(np.nanvar(Anom[:, full], axis=1).mean())
print(f"  train months: noise var mean={tr_s2.mean():.4f} (std {tr_s2.std():.4f}), "
      f"anomaly var={tr_var:.4f} -> implied lambda_train={1-tr_s2.mean()/tr_var:.3f}")

an_s2, an_var = [], []
for a in anchors:
    s2, n = noise_var(AF[a])
    an_s2.append(s2)
    an_var.append(float(np.nanvar(AF[a])))
    print(f"  anchor {a}: noise var={s2:.4f} n={n} anomaly var={an_var[-1]:.4f} -> lam_impl={1-s2/an_var[-1]:.3f}")
an_s2 = np.array(an_s2); an_var = np.array(an_var)
print(f"  >>> anchors: mean noise var={an_s2.mean():.4f}, mean anomaly var={an_var.mean():.4f}, "
      f"implied lambda_test={1-an_s2.mean()/an_var.mean():.3f}  (train reference: 0.84)")
# PC-residual cross-check
res_anchor = [float(np.nanvar(AF[a]-dn(AF[a]))) for a in anchors]
res_train  = [float(np.nanvar(Anom[ti]-dn(Anom[ti]))) for ti in range(T)]
print(f"  PC-residual: anchors mean={np.mean(res_anchor):.4f}  train mean={np.mean(res_train):.4f} "
      f"ratio={np.mean(res_anchor)/np.mean(res_train):.3f}")

# ================= LOO harness =================
PAIRS = [(anchors[i], anchors[i+1]) for i in range(len(anchors)-1)]
PAIRS = [(i,k) for i,k in PAIRS if k-i <= 8]
print(f"\nLOO pairs: {PAIRS}")

def fit_weights(a_list, target_a):
    """LOO-fit D-tilde weights using anchors a_list (excluding target), returning (w1,w2,w3)."""
    Xs, ys = [], []
    for a in a_list:
        dloo = np.nanmean(np.array([AF[b] for b in a_list if b!=a]), axis=0)
        tx = trendex(a)
        ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])&np.isfinite(tx)
        Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(3), Xw.T@yw)
    return float(w_[0]), float(w_[1]), float(w_[2])

def make_Dtil(anchors_avail, lam_split=None):
    """Return Dtil(t) fn using anchors_avail (all except LOO target)."""
    w1,w2,w3 = fit_weights(anchors_avail, None)
    Dhat = np.nanmean(np.array([AF[a] for a in anchors_avail]), axis=0)
    def Dtil(t, dh=None):
        return (w1*(dh if dh is not None else Dhat) + w2*S + w3*trendex(t)).astype(np.float32)
    return Dtil, (w1,w2,w3), Dhat

def calibrate(Dtil, anchors_avail, lam, per_cov=False):
    cs, zs, vfs = [], [], []
    for a in anchors_avail:
        dj = Dtil(a); w_ = dn(W[a])
        ok = np.isfinite(w_)&np.isfinite(AF[a])&np.isfinite(dj)
        cs.append(np.cov(w_[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(lam*var_f); R = max(varz - c_*c_/(lam*var_f), 1e-4)
    return H, R, var_f

def fwd_to(i, tm, phi, lam, H, R, var_f, use_dn, obs='scalar'):
    q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f
    x = np.where(np.isfinite(AF[i]), lam*np.nan_to_num(AF[i]), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    for m in range(i+1, tm+1):
        x = phi*x; P = phi**2*P + q
        if obs=='scalar' and m in W:
            w_ = dn(W[m]) if use_dn else W[m]
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x+Kg*(np.nan_to_num(w_)-H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
        elif obs=='multi' and m in W:
            for j in range(5):
                w_ = Wj[(j,m)]
                okw = np.isfinite(w_)
                if okw.sum()==0: continue
                Hj_, Rj_ = H[j], R[j]
                Kg = np.where(okw, P*Hj_/(Hj_*Hj_*P+Rj_), 0).astype(np.float32)
                x = np.where(okw, x+Kg*(np.nan_to_num(w_)-Hj_*x), x)
                P = np.where(okw, (1-Kg*Hj_)*P, P)
    return x

def loo_run(phi, lam, use_dn=False, obs='scalar', dhat_mode='static', tau=np.inf, S_mode='global'):
    """Returns mean RMSE over LOO pairs (predict anchor k field from anchor i)."""
    errs, deterrs = [], []
    for i,k in PAIRS:
        avail = [a for a in anchors if a != k]
        Dtil, wts, Dhat = make_Dtil(avail)
        if dhat_mode=='drift':
            t_k = k
            wts_a = np.array([np.exp(-abs(a-t_k)/tau) for a in avail]); wts_a/=wts_a.sum()
            Dhat_d = sum(w*AF[a] for w,a in zip(wts_a, avail))
            w1,w2,w3 = wts
            def Dtil(t, _D=Dhat_d, _w=(w1,w2,w3)):
                return (_w[0]*_D + _w[1]*S + _w[2]*trendex(t)).astype(np.float32)
        if S_mode=='era':
            early_m = [m for m in all_m if m < 2017*12]; late_m = [m for m in all_m if m >= 2017*12]
            S_e = np.nanmean(np.array([cov_field[m] for m in early_m]), axis=0)
            S_l = np.nanmean(np.array([cov_field[m] for m in late_m]), axis=0)
            w1,w2,w3 = fit_weights(avail, None)
            def Dtil(t, _w=(w1,w2,w3)):
                return (_w[0]*Dhat + _w[1]*(S_l if t>=2017*12 else S_e) + _w[2]*trendex(t)).astype(np.float32)
        if obs=='multi':
            # per-cov calibration at avail anchors
            Hj_, Rj_, vfs = np.zeros(5), np.zeros(5), []
            states = {a: AF[a]-Dtil(a) for a in avail}
            vfs = float(np.mean([np.nanvar(s) for s in states.values()]))
            for j in range(5):
                cj, zj = [], []
                for a in avail:
                    w_ = Wj[(j,a)]
                    ok = np.isfinite(w_)&np.isfinite(states[a])
                    cj.append(np.cov(w_[ok], states[a][ok])[0,1]); zj.append(np.var(w_[ok]))
                c_ = float(np.mean(cj)); z_ = float(np.mean(zj))
                Hj_[j] = c_/(lam*vfs); Rj_[j] = max(z_ - c_*c_/(lam*vfs), 1e-4)
            H, R = Hj_, Rj_
        else:
            H, R, var_f = calibrate(Dtil, avail, lam)
            vfs = var_f
        var_f = vfs
        x = fwd_to(i, k, phi, lam, H, R, var_f, use_dn, obs)
        pred = Dtil(k) + x
        ok = np.isfinite(pred)&np.isfinite(AF[k])
        errs.append(float(np.sqrt(np.mean((pred[ok]-AF[k][ok])**2))))
        dj = Dtil(k)
        okd = np.isfinite(dj)&np.isfinite(AF[k])
        deterrs.append(float(np.sqrt(np.mean((dj[okd]-AF[k][okd])**2))))
    return float(np.mean(errs)), float(np.mean(deterrs)), errs

# ---- E0b: lam x phi sweep (v2b-style core: no dn) ----
print("\n[E0b] lam x phi sweep (scalar cov obs, no denoise):")
print(f"{'lam':>5} | " + " | ".join(f"phi={p}" for p in [0.70,0.74,0.80]))
for lam in [0.84, 0.90, 0.95, 1.00]:
    row = []
    for phi in [0.70,0.74,0.80]:
        e,_,_ = loo_run(phi, lam)
        row.append(f"{e:.4f}")
    print(f"{lam:5.2f} | " + " | ".join(row))

# baseline reference: lam=0.84, phi=0.74 with denoise (v4 core)
e_dn,_,_ = loo_run(0.74, 0.84, use_dn=True)
print(f"ref: lam=0.84 phi=0.74 WITH denoise: {e_dn:.4f}  (v3_experiments got ~0.7424 no-dn / 0.7237 dn)")

# ---- E2: drift-weighted D-hat ----
print("\n[E2] drift-weighted D-hat (tau months):")
for tau in [np.inf, 36, 24, 12, 8]:
    e, de, _ = loo_run(0.74, 0.84, dhat_mode='drift', tau=tau)
    print(f"  tau={tau if np.isfinite(tau) else 'inf':>4}: full RMSE={e:.4f}  D-tilde err={de:.4f}")

# ---- E3: era-split S ----
print("\n[E3] era-split S:")
e, de, _ = loo_run(0.74, 0.84, S_mode='era')
print(f"  era-split S: full RMSE={e:.4f}  D-tilde err={de:.4f}")
e, de, _ = loo_run(0.74, 0.84)
print(f"  global S   : full RMSE={e:.4f}  D-tilde err={de:.4f}")

# ---- E4: 5-dim cov obs ----
print("\n[E4] multi-cov Kalman obs (5 sequential updates) vs scalar:")
e_m, _, _ = loo_run(0.74, 0.84, obs='multi')
e_s, _, _ = loo_run(0.74, 0.84, obs='scalar')
print(f"  multi: {e_m:.4f}   scalar: {e_s:.4f}")

print("\nDONE experiments.")
