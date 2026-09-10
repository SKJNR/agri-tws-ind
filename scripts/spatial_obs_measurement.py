"""DECISIVE MEASUREMENT: spatial structure of the cov->TWS observation noise.

If cov obs noise is spatially WHITE while the TWS fast field is SMOOTH (rank~100),
then spatially averaged covariates should recover the fast field much better than
the per-cell r~0.43 that the current scalar-Kalman pipeline exploits.

Measurements (all on TRAIN, fit <=2012, eval 2013-15):
  M1. per-cell corr(w, f)  vs  corr(smoothed w, smoothed f) at sigma = 1,2,3,5 deg
  M2. neighbor ACF of obs residual e = w - beta*f  (white vs smooth noise)
  M3. direct target test (k0 style):
        y ~ [TWS_t anomaly, raw cov anomalies(t+1)]           (current k0)
        y ~ [TWS_t anomaly, SMOOTHED cov-composite(t+1)]      (new)
        y ~ [TWS_t anomaly, raw covs(t+1) + smoothed composite]  (both)
  M4. innovation visibility: corr(smoothed cov-innovation(t+1), eta(t+1))
      where eta = f(t+1) - phi*f(t)  -- the unpredictable part!
"""
import numpy as np, pandas as pd, time
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)
log(f"train {len(train):,} rows, {n_cells} cells")

# smoother (Gaussian on sphere)
_SMOOTH = {}
def get_smoother(sigma_deg):
    key = float(sigma_deg)
    if key in _SMOOTH: return _SMOOTH[key]
    la = np.deg2rad(cell_xy[:,0]); lo = np.deg2rad(cell_xy[:,1])
    pts = np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
    tree = cKDTree(pts)
    rad = np.deg2rad(3.0*sigma_deg)
    rows, cols, vals = [], [], []
    for i in range(len(pts)):
        nb = tree.query_ball_point(pts[i], rad)
        if len(nb) == 0: nb = [i]
        d2 = ((pts[nb]-pts[i])**2).sum(axis=1)
        w = np.exp(-d2/(2*np.deg2rad(sigma_deg)**2))
        w = w/np.maximum(w.sum(), 1e-12)
        rows.extend([i]*len(nb)); cols.extend(nb); vals.extend(w.tolist())
    K = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    _SMOOTH[key] = K
    return K

# ---- infra on fit<=2012 ----
fit = train[train['time'].dt.year <= 2012]
yms = np.sort(fit['ym'].unique()); T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
coef = np.linalg.solve(Z1.T@Z1 + 1e-2*np.eye(6), Z1.T@yv)
log("infra done")

# ---- month -> field matrices for eval period 2013-15 ----
ev = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]
ev_months = np.sort(ev['ym'].unique())
fields = {}   # ym -> dict(f=fast field, w=cov composite anomaly, cov anomalies per cov)
for ym in ev_months:
    sub = ev[ev['ym']==ym]
    f = np.full(n_cells, np.nan, dtype=np.float32); f[sub['cc'].values] = sub['TWS_t'].values
    w = np.full(n_cells, np.nan, dtype=np.float32)
    Zm = np.column_stack([sub[c].values for c in COVS] + [np.ones(len(sub))]).astype('float32')
    w[sub['cc'].values] = Zm @ coef
    fields[int(ym)] = (f - mu_c, w - mu_c)
log(f"built {len(fields)} eval month fields")

# ================= M1: per-cell vs smoothed corr =================
log("=== M1: corr(w, f) per-cell vs spatially smoothed ===")
sigmas = [0.0, 1.0, 2.0, 3.0, 5.0]
acc = {s: [] for s in sigmas}
for ym, (f, w) in fields.items():
    ok = np.isfinite(f) & np.isfinite(w)
    for s in sigmas:
        if s == 0.0:
            ws = w
        else:
            ws = get_smoother(s) @ np.where(np.isfinite(w), w, 0.0)
        c = np.corrcoef(f[ok], ws[ok])[0,1]
        acc[s].append(c)
for s in sigmas:
    print(f"  sigma={s:3.1f}deg: mean corr = {np.mean(acc[s]):.4f}")

# ================= M2: neighbor ACF of obs residual =================
log("=== M2: spatial structure of obs residual e = w - beta*f ===")
# per-month beta = cov(w,f)/var(f) over cells
res_acf = {1: [], 2: [], 3: []}
for ym, (f, w) in list(fields.items())[::3]:
    ok = np.isfinite(f) & np.isfinite(w)
    beta = np.cov(w[ok], f[ok])[0,1]/np.var(f[ok])
    e = np.where(ok, w - beta*f, np.nan)
    ef = np.where(np.isfinite(e), e, 0.0)
    for ddeg in res_acf:
        K = get_smoother(0.4)  # ~nearest-neighbor kernel
        # build distance-d neighbor pairs via KDTree
        la = np.deg2rad(cell_xy[:,0]); lo = np.deg2rad(cell_xy[:,1])
        pts = np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
        tree = cKDTree(pts)
        pairs = tree.query_pairs(np.deg2rad(ddeg+0.25))
        if len(pairs):
            i, j = np.array(list(pairs)).T
            m = ok[i] & ok[j]
            if m.sum() > 100:
                res_acf[ddeg].append(np.corrcoef(ef[i[m]], ef[j[m]])[0,1])
for d in res_acf:
    print(f"  neighbor ACF at ~{d}deg: {np.mean(res_acf[d]):.4f}")

# ================= M3: k0-style direct target test =================
log("=== M3: k0 target prediction: raw covs(t+1) vs smoothed composite(t+1) ===")
# next-month cov composite per (cc, t_abs+1)
ev2 = ev.copy()
nxt = {}
for ym in ev_months:
    sub = ev[ev['ym']==ym]
    Zm = np.column_stack([sub[c].values for c in COVS] + [np.ones(len(sub))]).astype('float32')
    wv = (Zm @ coef) - mu_c[sub['cc'].values]
    nxt[ym] = dict(zip(sub['cc'].values.astype(int), wv))
# smoothed composite per month
nxt_sm = {}
for ym, d_ in nxt.items():
    fld = np.full(n_cells, np.nan, dtype=np.float32)
    ccv = np.array(list(d_.keys())); wv = np.array(list(d_.values()))
    fld[ccv] = wv
    K2 = get_smoother(2.0); K3 = get_smoother(3.0)
    sm2 = (K2 @ np.where(np.isfinite(fld), fld, 0.0))
    sm3 = (K3 @ np.where(np.isfinite(fld), fld, 0.0))
    nxt_sm[ym] = (fld, sm2, sm3)

rows = []
for ym in ev_months[:-1]:
    sub = ev[ev['ym']==ym].copy()
    y, m = divmod(int(ym), 100)
    ym_n = (y+1)*100+1 if m==12 else ym+1
    if ym_n not in nxt: continue
    f_raw, sm2, sm3 = nxt_sm[ym_n]
    sub['w_raw'] = sub['cc'].map(lambda c: f_raw[c]).values.astype('float32')
    sub['w_sm2'] = sm2[sub['cc'].values]
    sub['w_sm3'] = sm3[sub['cc'].values]
    rows.append(sub)
R = pd.concat(rows)
yv_ = (R['target'].values - mu_c[R['cc'].values]).astype('float64')
x_tw = (R['TWS_t'].values - mu_c[R['cc'].values]).astype('float64')
ok = np.isfinite(yv_) & np.isfinite(x_tw) & np.isfinite(R['w_raw'].values)

def fit_rmse(cols):
    Xf = np.column_stack(cols + [np.ones(len(R))])[ok]
    cf = np.linalg.solve(Xf.T@Xf + 1e-3*np.eye(Xf.shape[1]), Xf.T@yv_[ok])
    return float(np.sqrt(np.mean((Xf@cf - yv_[ok])**2)))

print(f"  persistence:                  {fit_rmse([x_tw]):.4f}")
print(f"  + raw composite(t+1):         {fit_rmse([x_tw, R['w_raw'].values]):.4f}")
print(f"  + smooth2 composite(t+1):     {fit_rmse([x_tw, R['w_sm2'].values]):.4f}")
print(f"  + smooth3 composite(t+1):     {fit_rmse([x_tw, R['w_sm3'].values]):.4f}")
print(f"  + raw AND smooth2:            {fit_rmse([x_tw, R['w_raw'].values, R['w_sm2'].values]):.4f}")

# ================= M4: innovation visibility =================
log("=== M4: eta(t+1) visibility from smoothed cov composite(t+1) ===")
# eta = f(t+1) - phi*f(t) with phi=0.80; how much does w(t+1) see it?
phi = 0.80
etas, wraws, wsm2s, wsm3s = [], [], [], []
yms_list = sorted(fields.keys())
for ym in yms_list[:-1]:
    y, m = divmod(int(ym), 100)
    ym_n = (y+1)*100+1 if m==12 else ym+1
    if ym_n not in fields: continue
    f1, w1 = fields[ym]; f2, w2 = fields[ym_n]
    ok2 = np.isfinite(f1) & np.isfinite(f2) & np.isfinite(w2)
    eta = f2 - phi*f1
    K2 = get_smoother(2.0); K3 = get_smoother(3.0)
    wsm2 = K2 @ np.where(np.isfinite(w2), w2, 0.0)
    wsm3 = K3 @ np.where(np.isfinite(w2), w2, 0.0)
    etas.append(eta[ok2]); wraws.append(w2[ok2]); wsm2s.append(wsm2[ok2]); wsm3s.append(wsm3[ok2])
eta = np.concatenate(etas); wr = np.concatenate(wraws); w2s = np.concatenate(wsm2s); w3s = np.concatenate(wsm3s)
print(f"  std(eta) = {eta.std():.4f}")
print(f"  corr(eta, raw w(t+1))     = {np.corrcoef(eta, wr)[0,1]:.4f}   R2={np.corrcoef(eta,wr)[0,1]**2:.3f}")
print(f"  corr(eta, smooth2 w(t+1)) = {np.corrcoef(eta, w2s)[0,1]:.4f}   R2={np.corrcoef(eta,w2s)[0,1]**2:.3f}")
print(f"  corr(eta, smooth3 w(t+1)) = {np.corrcoef(eta, w3s)[0,1]:.4f}   R2={np.corrcoef(eta,w3s)[0,1]**2:.3f}")
print(f"  => residual std after smooth2: {eta.std()*np.sqrt(1-np.corrcoef(eta,w2s)[0,1]**2):.4f}")

# ================= M5: PER-PC correlation (spectral separation) =================
log("=== M5: per-PC corr of cov composite w with FAST field ===")
# fast field on train: detrend per cell using fit<=2012 slope
F64 = F.astype(np.float64)
t_abs_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_arr[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0)
A_dt = F64 - mu_c[None,:] - (t_abs_arr[:,None]-tbar_c[None,:])*beta_c[None,:]
full = ~np.isnan(F).any(axis=0)
A = A_dt[:, full]; A = A - A.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(A, full_matrices=False)
V = Vt[:100].T  # top-100 PCs of fast field (full cells only)

# for eval months: fast field (detrended) and w projected onto V
r_per_pc = np.zeros((len(fields), 100))
for i, (ym, (f, w)) in enumerate(sorted(fields.items())):
    f_fast = f - (np.mean(t_abs_arr)-tbar_c)*0.0  # trend removal for eval month:
    # eval month trend component: (t_abs(ym) - tbar_c)*beta_c
    t_abs_ym = (int(ym)//100)*12 + (int(ym)%100) - 1
    f_fast = f - (t_abs_ym - tbar_c)*beta_c
    okc = np.isfinite(f_fast) & np.isfinite(w) & full
    for k in range(100):
        vk = V[:, k]
        a_ = vk[okc] @ f_fast[okc]; b_ = vk[okc] @ w[okc]
        na = np.sqrt(vk[okc] @ vk[okc] * f_fast[okc] @ f_fast[okc])
        nb = np.sqrt(vk[okc] @ vk[okc] * w[okc] @ w[okc])
        r_per_pc[i, k] = (a_*b_)/(na*nb)
r_mean = r_per_pc.mean(axis=0)
print("  per-PC corr (avg over months), by PC block:")
for blk in range(10):
    lo_, hi_ = blk*10, blk*10+10
    print(f"    PC{lo_+1:3d}-{hi_:3d}: {r_mean[lo_:hi_].mean():+.4f}   " +
          " ".join(f"{v:+.2f}" for v in r_mean[lo_:hi_]))
print("  => spectral separation exists if top-PC corr >> tail-PC corr")

