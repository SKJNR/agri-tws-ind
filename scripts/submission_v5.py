"""
SUBMISSION V5: v2b architecture + SPATIAL post-processing (validated on trusted sparse CV).

CV evidence (spatial_prize_test2.py, sparse-anchor protocol that correctly ranked v2b>v4a/v4c):
  - v2b baseline CV 0.7023
  - Gaussian sigma=2.0deg post-hoc prediction smoothing: -0.0193 (gain in EVERY month, every k)
  - + W-pool(box r=2) inside Kalman: -0.0198 (partially redundant, small extra)
  Expected LB: 0.7137 - ~0.019 => ~0.695

Variants:
  v5a: phi=0.74, v2b stack unchanged + gau2.0 smoothing of MASKED-row predictions
  v5b: phi=0.74, W-pooled (box r=2) cov obs + gau2.0 smoothing of masked rows
  v5c: v5b + smoothing applied to k=0 (unmasked) rows as well (experiment: tests the
       k=0 smoothing channel; truth field is rank~50 smooth so should not hurt)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------------- load train ----------------
print("Loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_train = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

F64 = F.astype(np.float64)
ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_train[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0)
td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100) & (ok_t.sum(axis=0) >= 24),
                  np.nansum(td*F64, axis=0)/np.where(np.nansum(td*td, axis=0) > 0, np.nansum(td*td, axis=0), 1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# ---------------- grid mapping ----------------
lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)
for (la, lo), grp in train.groupby(['lat','lon']):
    cc_grid[lat_i[la], lon_i[lo]] = int(grp['cc'].iloc[0])
mask_g = cc_grid >= 0

def to_grid(v_cell):
    g = np.full((NI, NJ), np.nan, dtype=np.float32); g[mask_g] = v_cell[cc_grid[mask_g]]; return g
def from_grid(g):
    out = np.full(n_cells, np.nan, dtype=np.float32); out[cc_grid[mask_g]] = g[mask_g]; return out
def shift(g, di, dj):
    gg = np.roll(g, dj, axis=1)
    if di > 0: gg = np.vstack([np.full((di, NJ), np.nan, np.float32), gg[:-di]])
    if di < 0: gg = np.vstack([gg[-di:], np.full((-di, NJ), np.nan, np.float32)])
    return gg
GAU2 = {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*2.0*2.0)))
        for di in range(-4,5) for dj in range(-4,5)
        if np.exp(-(di*di+dj*dj)/(2*2.0*2.0)) > 0.01}
BOX2 = {(di,dj): 1.0 for di in range(-2,3) for dj in range(-2,3)}
def kernel_pool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

# ---------------- load test ----------------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
assert (test['cc'] >= 0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values
cc_t = test['cc'].values
msk = test['masked'].values

# ---------------- cov fields ----------------
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
W_raw = {int(m): cov_field[m] - S for m in all_m}
W_pool = {m: from_grid(kernel_pool(to_grid(v), BOX2)) for m, v in W_raw.items()}

# ---------------- anchors ----------------
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"test anchors: {anchors}")
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)

Xs, ys = [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]]))
    ys.append(AF[a][ok])
Xw = np.vstack(Xs); yw = np.concatenate(ys)
w_ = np.linalg.solve(Xw.T@Xw + np.array([1e-3,1e-3,1e-3]), Xw.T@yw)
w1, w2, w3 = float(w_[0]), float(w_[1]), float(w_[2])
print(f"D-tilde weights (LOO): D-hat={w1:.3f}, S={w2:.3f}, trendex={w3:.3f}")
def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

# ---------------- Kalman ----------------
LAM_F = 0.84
def calibrate(Wd):
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a)
        ok = np.isfinite(Wd[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(Wd[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(Wd[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    var_x = LAM_F*var_f
    return c_/var_x, max(varz - c_*c_/var_x, 1e-4), var_f

def kalman_predict(phi_f, Wd):
    H, R, var_f = calibrate(Wd)
    q = LAM_F*var_f*(1-phi_f**2)
    P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for a in anchors:
        fa = AF[a] - Dtil(a)
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        sel0 = np.where((ta == a) & msk)[0]
        if len(sel0):
            pred[sel0] = mu_c[cc_t[sel0]] + Dtil(a+1)[cc_t[sel0]] + phi_f*x[cc_t[sel0]]
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in Wd:
                wv = Wd[m]; okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((ta == m) & msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in Wd:
                wv = Wd[tm]; okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[cc_t[sel]] + Dtil(tm)[cc_t[sel]] + x2[cc_t[sel]]
    return pred

# ---------------- k=0 model (unchanged from V1/V2) ----------------
print("Fitting k=0 model...", flush=True)
Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy()
Lk['t_next'] = Lk['t_abs'] + 1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values
yr = mgk['t_abs'].values // 12
Xf = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm],
    *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))
]).astype(np.float32)
yA = (mgk['target'].values - mu_c[ccm]).astype(np.float32)
Xf = np.nan_to_num(Xf, nan=0.0)
w_rec = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
sw = np.sqrt(w_rec)
selF = has_nxt_tr
A_f = Xf[selF]*sw[selF,None]
coefF = np.linalg.solve(A_f.T@A_f + 1e-3*np.eye(12), A_f.T@(yA[selF]*sw[selF]))
colsR = [0,1,2,3,4,5,11]
A_r = Xf[:, colsR]*sw[:,None]
coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(7), A_r.T@(yA*sw))

Lt = test[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = test[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
Xt = np.column_stack([
    mgt['TWS_t'].values - mu_c[cct],
    *[mgt[c].values - clim[cct, j] for j, c in enumerate(COVS)],
    *[mgt[c+'_nxt'].values - clim[cct, j] for j, c in enumerate(COVS)],
    np.ones(len(mgt))
]).astype(np.float32)
Xt = np.nan_to_num(Xt, nan=0.0)
k0 = np.full(len(test), np.nan, dtype=np.float64)
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
k0[use_full] = mu_c[cct[use_full]] + Xt[use_full] @ coefF
k0[use_red]  = mu_c[cct[use_red]]  + Xt[use_red][:, colsR] @ coefR
bad = (~msk) & np.isnan(k0)
if bad.any(): k0[bad] = mu_c[cct][bad]

# ---------------- post-hoc spatial smoothing ----------------
def smooth_rows(pred, rows):
    """rows: boolean over test rows; smooth those rows' predictions per month on the grid."""
    out = pred.copy()
    for m in np.unique(ta[rows & np.isfinite(pred)]):
        selm = np.where((ta == m) & rows & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[cc_t[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kernel_pool(to_grid(cm), GAU2))
        out[selm] = sm[cc_t[selm]]
    return out

# ---------------- assemble ----------------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm = ~msk

def assemble(masked_pred, tag, smooth_masked=True, smooth_k0=False):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0[unm]
    pred[msk] = masked_pred[msk]
    pred = np.where(np.isnan(pred), mu_c[cc_t], pred)
    if smooth_masked: pred = smooth_rows(pred, msk)
    if smooth_k0:     pred = smooth_rows(pred, unm)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    n_nan = int(out['Target'].isna().sum())
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} "
          f"(unm std={pred[unm].std():.3f}, masked std={pred[msk].std():.3f}) nan={n_nan}", flush=True)

print("\n--- v5a: v2b + gau2.0 smoothing (masked rows) ---")
p_raw = kalman_predict(0.74, W_raw)
assemble(p_raw, 'v5a', smooth_masked=True, smooth_k0=False)
print("--- v5b: W-pooled Kalman + gau2.0 smoothing (masked rows) ---")
p_wp = kalman_predict(0.74, W_pool)
assemble(p_wp, 'v5b', smooth_masked=True, smooth_k0=False)
print("--- v5c: v5b + smoothing k=0 rows too ---")
assemble(p_wp, 'v5c', smooth_masked=True, smooth_k0=True)

# sanity vs v2b submission: correlation of change
old = pd.read_csv(f'{DL}/submission_v2b.csv')
new = pd.read_csv(f'{DL}/submission_v5a.csv')
m = old.merge(new, on='ID', suffixes=('_old','_new'))
d = (m['Target_new']-m['Target_old'])
print(f"\nv5a vs v2b: mean diff {d.mean():+.5f}, std {d.std():.5f}, corr {np.corrcoef(m['Target_old'],m['Target_new'])[0,1]:.5f}")
print("Done.")
