"""E6: per-PC Kalman (each PC gets its own H_k, R_k) vs scalar-H. Anchor-LOO harness."""
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
TD = t_abs_tr[:, None] - tbar_c[None, :]
A_dt = F64 - mu_c[None, :] - TD * beta_c[None, :]
A_dtf = A_dt[:, full]
A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
U, S_, Vt = np.linalg.svd(A_dtf, full_matrices=False)

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
w1, w2, w3 = 0.650, 0.456, 0.073
def Dtil_loo(j, t):
    dloo = np.nanmean(np.array([AF[a] for a in anchors if a != j]), axis=0)
    return (w1*dloo + w2*S + w3*trendex(t)).astype(np.float32)

# ---------- per-PC machinery ----------
K = 200
V = Vt[:K].T.astype(np.float32)          # (n_full, K)
def pc(x):
    """(n_cells,) -> PC scores (K,) using full cells; nan->0"""
    xf = np.where(np.isfinite(x[full]), x[full], 0.0)
    return V.T @ xf
def unpc(s):
    out = np.zeros(n_cells, dtype=np.float32)
    out[full] = V @ s
    return out

def calib_pc(j):
    """per-PC H_k, R_k, var_f_k from anchors != j"""
    Hs, Rs, vfs = [], [], []
    for a in anchors:
        if a == j: continue
        dj = Dtil_loo(a, a)
        w_ = pc(W[a]); f_ = pc(AF[a]-dj)
        Hs.append(w_*f_); Rs.append(w_*w_); vfs.append(f_*f_)
    cov_wf = np.mean(Hs, axis=0)          # E[w*f] per PC
    var_w = np.mean(Rs, axis=0)           # E[w^2]
    var_ff = np.mean(vfs, axis=0)         # E[f^2] ~ var_f + var_x-ish (sample)
    var_x = LAM_F*var_ff
    H = cov_wf/var_x
    R = np.maximum(var_w - cov_wf**2/var_x, 1e-5)
    return H, R, var_ff

def fwd_pc(i, j, phi, shrink=0.0):
    H, R, var_ff = calib_pc(j)
    # shrink per-PC H toward pooled mean H (stability)
    Hs = (1-shrink)*H + shrink*np.mean(H)
    q = LAM_F*var_ff*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_ff
    dj = Dtil_loo(j, i)
    x = LAM_F*pc(AF[i]-dj)
    has = np.isfinite(AF[i][full]).mean()
    P = np.where(has > 0.99, P0, var_ff).astype(np.float32)
    for m in range(i+1, j+1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = pc(W[m])
            Kg = P*Hs/(Hs*Hs*P + R)
            x = x + Kg*(w_ - Hs*x)
            P = (1-Kg*Hs)*P
    return unpc(x), P

def bwd_pc(k, j, phi, shrink=0.0):
    H, R, var_ff = calib_pc(j)
    Hs = (1-shrink)*H + shrink*np.mean(H)
    q = LAM_F*var_ff*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_ff
    dj = Dtil_loo(j, k)
    x = LAM_F*pc(AF[k]-dj)
    P = P0.astype(np.float32)
    for m in range(k-1, j-1, -1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = pc(W[m])
            Kg = P*Hs/(Hs*Hs*P + R)
            x = x + Kg*(w_ - Hs*x)
            P = (1-Kg*Hs)*P
    return unpc(x), P

pairs = []
for idx in range(len(anchors)-1):
    i, j = anchors[idx], anchors[idx+1]
    if j - i <= 8:
        pairs.append((i, j, j-i))

def run_pc(phi, use_bwd, shrink):
    errs = []
    for (i, j, g) in pairs:
        xf, Pf = fwd_pc(i, j, phi, shrink)
        later = [a for a in anchors if a > j]
        if use_bwd and later:
            xb, Pb = bwd_pc(later[0], j, phi, shrink)
            # combine in PC space would be better; approximate per-cell precision blend
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            # Pf/Pb are (K,) in PC space; broadcast to cells via V weighting: use mean weight
            wf = float(np.mean(wgt))
            x = wf*xf + (1-wf)*xb
        else:
            x = xf
        pred = Dtil_loo(j, j+1) + x
        tgt = AF[j]
        ok = np.isfinite(pred) & np.isfinite(tgt)
        errs.append(float(np.sqrt(np.mean((pred[ok]-tgt[ok])**2))))
    return np.mean(errs), errs

print("per-PC H (first 10 PCs, LOO target 4192):")
H, R, vf = calib_pc(4192)
print("  H:", np.round(H[:10],3))
print("  R:", np.round(R[:10],4))
print("  var_f:", np.round(vf[:10],3))

print("\n=== E6: per-PC Kalman ===")
for shrink in [0.0, 0.5]:
    r, pe = run_pc(0.74, False, shrink)
    print(f"  perPC fwd-only shrink={shrink}: {r:.4f}  per-pair {np.round(pe,3)}")
r, pe = run_pc(0.74, True, 0.0)
print(f"  perPC +bwd shrink=0.0: {r:.4f}  per-pair {np.round(pe,3)}")
r, pe = run_pc(0.74, True, 0.5)
print(f"  perPC +bwd shrink=0.5: {r:.4f}  per-pair {np.round(pe,3)}")
for phi in [0.70, 0.78]:
    r, pe = run_pc(phi, True, 0.0)
    print(f"  perPC +bwd phi={phi}: {r:.4f}  per-pair {np.round(pe,3)}")
