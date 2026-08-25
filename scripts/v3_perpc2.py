"""E8: per-PC Kalman with TRAIN-calibrated H_k/R_k (138 months), scaled to test coupling.
Decisive test of whether per-direction observation quality (PC2-5 r=0.85-0.95) is exploitable."""
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

# ---- train cov fields (fit <2010 to avoid leakage in H estimation) ----
Z = train[COVS].values.astype('float32')
yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
early = (train['time'] < '2010-01-01').values
fitm = okz & early
coef_tr = np.linalg.solve(Z1[fitm].T@Z1[fitm] + 1e-2*np.eye(6), Z1[fitm].T@yv[fitm])
cov_est_tr = np.full(len(train), np.nan, dtype=np.float32)
okz_all = np.isfinite(Z1).all(axis=1)
cov_est_tr[okz_all] = np.column_stack([Z[okz_all], np.ones(okz_all.sum())]) @ coef_tr
CF_tr = np.full((T, n_cells), np.nan, dtype=np.float32)
CF_tr[train['ym'].map(ym_to_i).values[okz_all], train['cc'].values[okz_all]] = cov_est_tr[okz_all]
CF_tr = CF_tr - mu_c
W_tr = CF_tr - np.nanmean(CF_tr, axis=0, keepdims=True)

# per-PC train stats
X_fast = A_dtf - 0.0                       # (T, n_full) detrended anomaly (fast+noise)
X_w = W_tr[:, full]
okm = np.isfinite(X_w).all(axis=1) & np.isfinite(X_fast).all(axis=1)
Fk = (X_fast[okm] @ V)                     # (T_ok, K) fast scores
Wk = (X_w[okm] @ V)                        # (T_ok, K) obs scores
cov_wf = (Wk*Fk).mean(axis=0)
var_w = (Wk*Wk).mean(axis=0)
var_f = (Fk*Fk).mean(axis=0)
var_x = LAM_F*var_f
H_tr_pc = cov_wf/var_x
R_tr_pc = np.maximum(var_w - cov_wf**2/var_x, 1e-6)
# pooled scalar versions
H_tr_scalar = cov_wf.sum()/var_x.sum()
print(f"train per-PC H (first 10): {np.round(H_tr_pc[:10],3)}")
print(f"train pooled scalar H: {H_tr_scalar:.3f}")
q_pc = LAM_F*var_f*(1-0.74**2)

# ---------------- test ----------------
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
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef_tr
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
w1, w2, w3 = 0.70, 0.45, 0.073
def Dtil_loo(j, t):
    dloo = np.nanmean(np.array([AF[a] for a in anchors if a != j]), axis=0)
    return (w1*dloo + w2*S + w3*trendex(t)).astype(np.float32)

# test pooled scalar H (for scale factor)
def test_scalar_H(j):
    cs, zs, vfs = [], [], []
    for a in anchors:
        if a == j: continue
        dj = Dtil_loo(a, a)
        w_ = W[a]
        ok = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f_ = float(np.mean(vfs))
    return c_/(LAM_F*var_f_)

scale = test_scalar_H(4192)/H_tr_scalar
print(f"test/train coupling scale: {scale:.3f}")
H_pc = H_tr_pc*scale
R_pc = R_tr_pc*(np.nanvar(np.array([W[m][full] for m in all_m]))/var_w[0])  # crude global var scaling
R_pc = np.maximum(R_pc, 1e-6)

def pc_scores(field):
    xf = np.where(np.isfinite(field[full]), field[full], 0.0)
    return V.T @ xf
def unpc(s):
    out = np.zeros(n_cells, dtype=np.float32)
    out[full] = V @ s
    return out
def dn(field):   # scalar-H denoise helper (PC projection on full cells)
    out = field.copy()
    x = field[full]
    okx = np.isfinite(x)
    out[full] = np.where(okx, V @ (V.T @ np.where(okx, x, 0.0)), x)
    return out

P0_pc = LAM_F*(1-LAM_F)*var_f

def fwd_pc(i, j, phi):
    q = LAM_F*var_f*(1-phi**2)
    dj = Dtil_loo(j, i)
    x = LAM_F*pc_scores(AF[i]-dj)
    P = P0_pc.copy()
    for m in range(i+1, j+1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = pc_scores(W[m])
            Kg = P*H_pc/(H_pc*H_pc*P + R_pc)
            x = x + Kg*(w_ - H_pc*x)
            P = (1-Kg*H_pc)*P
    return x, P
def bwd_pc(k, j, phi):
    q = LAM_F*var_f*(1-phi**2)
    dj = Dtil_loo(j, k)
    x = LAM_F*pc_scores(AF[k]-dj)
    P = P0_pc.copy()
    for m in range(k-1, j-1, -1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = pc_scores(W[m])
            Kg = P*H_pc/(H_pc*H_pc*P + R_pc)
            x = x + Kg*(w_ - H_pc*x)
            P = (1-Kg*H_pc)*P
    return x, P

pairs = []
for idx in range(len(anchors)-1):
    i, j = anchors[idx], anchors[idx+1]
    if j - i <= 8:
        pairs.append((i, j, j-i))

def run_pc(phi, use_bwd):
    errs = []
    for (i, j, g) in pairs:
        xf, Pf = fwd_pc(i, j, phi)
        later = [a for a in anchors if a > j]
        if use_bwd and later:
            xb, Pb = bwd_pc(later[0], j, phi)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        pred = Dtil_loo(j, j+1) + unpc(x)
        tgt = AF[j]
        ok = np.isfinite(pred) & np.isfinite(tgt)
        errs.append(float(np.sqrt(np.mean((pred[ok]-tgt[ok])**2))))
    return np.mean(errs), errs

print("\n=== E8: train-calibrated per-PC Kalman ===")
for phi in [0.70, 0.74, 0.78]:
    r, pe = run_pc(phi, False)
    print(f"  perPC-train phi={phi} fwd-only: {r:.4f}  per-pair {np.round(pe,3)}")
r, pe = run_pc(0.74, True)
print(f"  perPC-train phi=0.74 +bwd: {r:.4f}  per-pair {np.round(pe,3)}")
r, pe = run_pc(0.70, True)
print(f"  perPC-train phi=0.70 +bwd: {r:.4f}  per-pair {np.round(pe,3)}")

# reference: best scalar-H config from E7 weight grid (0.7185)
print("\nreference: scalar-H best (E7 grid): 0.7185")
