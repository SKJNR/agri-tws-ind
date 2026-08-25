"""E9: D-tilde via random-walk-with-drift smoothing between anchors (replaces static anchor-mean).
Slow state v(t) = slow(t) - trendex(t), v ~ RW(sigma_s); anchors observe v + fast + noise.
Fwd/bwd precision blend. Fast Kalman = scalar-H best config (denoise K=200 + bwd)."""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

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
F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where(sxx > 100, np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

full = ~np.isnan(F).any(axis=0)
full_idx = np.where(full)[0]
TD = t_abs_tr[:, None] - tbar_c[None, :]
A_dt = F64 - mu_c[None, :] - TD * beta_c[None, :]
A_dtf = A_dt[:, full]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
U, S_, Vt = np.linalg.svd(A_dtf, full_matrices=False)
K = 200
V = Vt[:K].T.astype(np.float32)
def dn(field):
    out = field.copy()
    x = field[full]
    okx = np.isfinite(x)
    out[full] = np.where(okx, V @ (V.T @ np.where(okx, x, 0.0)), x)
    return out

Z = train[COVS].values.astype('float32')
yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

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

# observation noise for slow state: fast var + obs noise var
var_fast_noise = float(np.mean([np.nanvar(AF[a] - dn(AF[a])) for a in anchors]))  # crude: high-freq energy
var_fn = float(np.mean([np.nanvar(AF[a]) for a in anchors])) * (1-LAM_F) + 0.5   # fallback
OBS_VAR = max(var_fast_noise, 0.5)
print(f"slow obs var (fast+noise at anchors): {OBS_VAR:.3f}")

def Dtil_rw(t, j_excl, sigma_s, wS):
    """RW-with-drift smoothed slow estimate at month t, excluding anchor j_excl (LOO) or None."""
    anch = [a for a in anchors if a != j_excl]
    # observation of v(a) = slow - trendex at each anchor
    obs_v = {a: AF[a] - wS*S - trendex(a) for a in anch}
    # forward pass
    def fwd(tgt):
        u = None; P = None
        for a in anch:
            if a > tgt: break
            if u is None:
                u = LAM_S*obs_v[a]; P = OBS_VAR + SIG_S2*(0)  # first obs
                P = np.full(n_cells, OBS_VAR, dtype=np.float32)
            else:
                dt = a - a_prev
                u = u.copy(); P = P + SIG_S2*dt
                Kg = P/(P+OBS_VAR)
                u = u + Kg*(obs_v[a]-u); P = (1-Kg)*P
            a_prev = a
        if u is None:
            return None, None
        dt = tgt - a_prev
        return u, P + SIG_S2*max(dt,0)
    def bwd(tgt):
        u = None; P = None
        for a in reversed(anch):
            if a < tgt: break
            if u is None:
                u = obs_v[a].copy(); P = np.full(n_cells, OBS_VAR, dtype=np.float32)
            else:
                dt = a_prev - a
                u = u.copy(); P = P + SIG_S2*dt
                Kg = P/(P+OBS_VAR)
                u = u + Kg*(obs_v[a]-u); P = (1-Kg)*P
            a_prev = a
        if u is None:
            return None, None
        dt = a_prev - tgt
        return u, P + SIG_S2*max(dt,0)
    uf, Pf = fwd(t)
    ub, Pb = bwd(t)
    if uf is None: v = ub
    elif ub is None: v = uf
    else:
        wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
        v = wgt*uf + (1-wgt)*ub
    return (wS*S + trendex(t) + v).astype(np.float32)

# fast Kalman (scalar-H, denoise, bwd) with Dtil function injected
def calib_HR(j, phi, Dfun):
    cs, zs, vfs = [], [], []
    for a in anchors:
        if a == j: continue
        dj = Dfun(a, j)
        w_ = dn(W[a])
        ok = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    return c_/(LAM_F*var_f), max(varz - c_*c_/(LAM_F*var_f), 1e-4), var_f

def fwd_pass(i, j, phi, Dfun):
    H, R, var_f = calib_HR(j, phi, Dfun)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = dn(AF[i] - Dfun(i, j))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    for m in range(i+1, j+1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = dn(W[m])
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

def bwd_pass(k, j, phi, Dfun):
    H, R, var_f = calib_HR(j, phi, Dfun)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = dn(AF[k] - Dfun(k, j))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[k]), P0, var_f).astype(np.float32)
    for m in range(k-1, j-1, -1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = dn(W[m])
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

pairs = []
for idx in range(len(anchors)-1):
    i, j = anchors[idx], anchors[idx+1]
    if j - i <= 8:
        pairs.append((i, j, j-i))

# reference: static D-tilde (E7 best: w1=0.7, w2=0.45, w3=0.073)
w1_, w2_, w3_ = 0.70, 0.45, 0.073
def Dtil_static(t, j):
    dloo = np.nanmean(np.array([AF[a] for a in anchors if a != j]), axis=0)
    return (w1_*dloo + w2_*S + w3_*trendex(t)).astype(np.float32)

def run(phi, Dfun, use_bwd=True):
    errs = []
    for (i, j, g) in pairs:
        xf, Pf = fwd_pass(i, j, phi, Dfun)
        later = [a for a in anchors if a > j]
        if use_bwd and later:
            xb, Pb = bwd_pass(later[0], j, phi, Dfun)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        pred = Dfun(j+1, j) + x
        tgt = AF[j]
        ok = np.isfinite(pred) & np.isfinite(tgt)
        errs.append(float(np.sqrt(np.mean((pred[ok]-tgt[ok])**2))))
    return np.mean(errs), errs

print(f"\nreference static D-tilde: {run(0.74, Dtil_static)[0]:.4f}")

print("\n=== E9: RW-smoothed D-tilde ===")
for sigma_s in [0.05, 0.08, 0.12, 0.18]:
    for wS in [0.3, 0.45]:
        globals()['SIG_S2'] = sigma_s**2
        globals()['LAM_S'] = 0.8
        Dfun = lambda t, j: Dtil_rw(t, j, sigma_s, wS)
        r, pe = run(0.74, Dfun)
        print(f"  sigma_s={sigma_s:.2f} wS={wS:.2f}: {r:.4f}  per-pair {np.round(pe,3)}")
