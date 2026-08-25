"""E7: denoise D-tilde via anchor-field PC projection. The biggest remaining error component."""
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
def denoise(field):
    out = field.copy()
    x = field[full]
    okx = np.isfinite(x)
    xz = np.where(okx, x, 0.0)
    out[full] = np.where(okx, V @ (V.T @ xz), x)
    return out

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

# D-tilde with optional anchor-PC projection of the D-hat part
def Dhat_loo_proj(j, J):
    """mean of other anchors' fields, projected onto top-J PCs of those fields"""
    flds = np.array([AF[a] for a in anchors if a != j])      # (5, n_cells)
    dhat = np.nanmean(flds, axis=0)
    if J is None:
        return dhat
    flds_f = flds[:, full]
    ok_cols = ~np.isnan(flds_f).any(axis=0)                  # over full cells
    X = flds_f[:, ok_cols]                                   # RAW fields (NO mean-removal: PC1 = common D pattern)
    _, _, VtA = np.linalg.svd(X, full_matrices=False)
    VJ = VtA[:J].T                                           # (n_ok, J)
    out = dhat.copy()
    xg = full_idx[ok_cols]                                   # global indices of ok cols
    out[xg] = VJ @ (VJ.T @ dhat[xg])
    return out

def Dtil_loo(j, t, J=None):
    return (w1*Dhat_loo_proj(j, J) + w2*S + w3*trendex(t)).astype(np.float32)

def calib_HR(j, phi, DJ):
    cs, zs, vfs = [], [], []
    for a in anchors:
        if a == j: continue
        dj = Dtil_loo(a, a, DJ)
        w_ = denoise(W[a])
        ok = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[ok], (AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

def fwd_pass(i, j, phi, DJ):
    H, R, var_f = calib_HR(j, phi, DJ)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = denoise(AF[i] - Dtil_loo(j, i, DJ))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    for m in range(i+1, j+1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = denoise(W[m])
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
            P = np.where(okw, (1-Kg*H)*P, P)
    return x, P

def bwd_pass(k, j, phi, DJ):
    H, R, var_f = calib_HR(j, phi, DJ)
    q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f
    f0 = denoise(AF[k] - Dtil_loo(j, k, DJ))
    x = np.where(np.isfinite(f0), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
    P = np.where(np.isfinite(AF[k]), P0, var_f).astype(np.float32)
    for m in range(k-1, j-1, -1):
        x = phi*x; P = phi**2*P + q
        if m in W:
            w_ = denoise(W[m])
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

def run(phi, DJ, use_bwd):
    errs = []
    for (i, j, g) in pairs:
        xf, Pf = fwd_pass(i, j, phi, DJ)
        later = [a for a in anchors if a > j]
        if use_bwd and later:
            xb, Pb = bwd_pass(later[0], j, phi, DJ)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        pred = Dtil_loo(j, j+1, DJ) + x
        tgt = AF[j]
        ok = np.isfinite(pred) & np.isfinite(tgt)
        errs.append(float(np.sqrt(np.mean((pred[ok]-tgt[ok])**2))))
    return np.mean(errs), errs

print("=== E7: D-tilde denoising (anchor-PC projection) ===")
print(f"  J=None (v3 baseline): {run(0.74, None, True)[0]:.4f}")
for J in [1, 2, 3, 4, 5]:
    r, pe = run(0.74, J, True)
    print(f"  J={J}: {r:.4f}  per-pair {np.round(pe,3)}")

# what does projection do to D error directly? (LOO D-hat vs target anchor field)
print("\nD-hat LOO error vs target anchor field (includes fast+noise of target):")
for J in [None, 1, 2, 3]:
    es = []
    for j in anchors:
        d = Dhat_loo_proj(j, J)
        ok = np.isfinite(d) & np.isfinite(AF[j])
        es.append(float(np.sqrt(np.mean((d[ok]-AF[j][ok])**2))))
    print(f"  J={J}: mean RMSE {np.mean(es):.4f}")

# re-optimize weights with projected D-hat? quick grid on J=2
best = None
for J in [2, 3]:
    for w1_ in [0.5, 0.6, 0.7]:
        for w2_ in [0.3, 0.45, 0.6]:
            globals()['w1'], globals()['w2'] = w1_, w2_
            r, _ = run(0.74, J, True)
            if best is None or r < best[0]: best = (r, J, w1_, w2_)
    globals()['w1'], globals()['w2'] = 0.650, 0.456
print(f"\nbest (RMSE, J, w1, w2) after weight grid: {best}")
