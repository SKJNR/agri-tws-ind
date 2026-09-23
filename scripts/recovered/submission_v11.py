"""SUBMISSION V11 — the D-lane: era-weighted Dhat + cov-slow student.

Base: v10b (LB 0.699997215, NEW BEST) = v6c masked stack (W-pool Kalman
phi-ens{.74,.80} + gau2.0) + v8 two-comp D-aware k0 blend + gau1.0.

What's new (validated on the 6 real test anchors, leave-one-anchor-out):
  1. Era-weighted Dhat (exp decay tau=12) instead of static 6-anchor mean:
     LOO 0.8224 -> 0.8034 (all), 0.7864 -> 0.7639 (late/private era).
     Static Dhat is stale at both era ends; std(Dhat_late-Dhat_early)=0.759.
  2. Cov-slow student (v11b only): ridge on hw=4 time-averaged raw covariate
     deviations predicting the teacher's (D-tilde) LOO residuals:
     additional 0.8034 -> 0.7874.  ("student learning from teacher failures")
  3. Honest anchor handling: Kalman init / H-R calibration / k0-B D-features
     at anchor months use leave-one-out Dtil (excludes the anchor's own field,
     no self-shrinkage); prediction months use full-weight Dtil.

Split-decode context: public = first 7 test months (2015-09..2016-08);
private (61% of rows) contains the 2017 h=1..6 block with the largest D.
Era weighting should help public AND private (more on private).

Variants:
  v11a: era-weighted Dhat only
  v11b: era-weighted Dhat + cov-slow student
"""
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
TAU = 12.0
HW = 4       # student averaging half-window (months)
LAM_STUD = 1000.0

# ---------------- load train (v6 verbatim) ----------------
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
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_train = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((len(yms), n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')

F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_train[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# ---------------- grid utils (v6 verbatim) ----------------
lats = np.sort(train['lat'].unique()); lons = np.sort(train['lon'].unique())
lat_i = {v:i for i,v in enumerate(lats)}; lon_i = {v:i for i,v in enumerate(lons)}
NI, NJ = len(lats), len(lons)
cc_grid = np.full((NI, NJ), -1, dtype=np.int32)
for (la, lo), grp in train.groupby(['lat','lon']):
    cc_grid[lat_i[la], lon_i[lo]] = int(grp['cc'].iloc[0])
mask_g = cc_grid >= 0
def to_grid(v):
    g = np.full((NI, NJ), np.nan, dtype=np.float32); g[mask_g] = v[cc_grid[mask_g]]; return g
def from_grid(g):
    out = np.full(n_cells, np.nan, dtype=np.float32); out[cc_grid[mask_g]] = g[mask_g]; return out
def shift(g, di, dj):
    gg = np.roll(g, dj, axis=1)
    if di > 0: gg = np.vstack([np.full((di, NJ), np.nan, np.float32), gg[:-di]])
    if di < 0: gg = np.vstack([gg[-di:], np.full((-di, NJ), np.nan, np.float32)])
    return gg
def gaussW(sig, r=4):
    return {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*sig*sig)))
            for di in range(-r,r+1) for dj in range(-r,r+1)
            if np.exp(-(di*di+dj*dj)/(2*sig*sig)) > 0.01}
BOX2 = {(di,dj): 1.0 for di in range(-2,3) for dj in range(-2,3)}
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)
GAU2 = gaussW(2.0); GAU10 = gaussW(1.0)

# ---------------- load test (v6 verbatim) ----------------
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
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values
test_months = np.array(sorted(test['t_abs'].unique()))

# ---------------- cov fields + anchors ----------------
Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in test_months:
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
S = np.nanmean(np.array([cov_field[m] for m in test_months]), axis=0)
W_raw = {int(m): cov_field[m] - S for m in test_months}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W_raw.items()}

# raw covariate deviation fields (for the student)
Zdev = {}
for m in test_months:
    selm = ta == m
    f = np.full((n_cells, len(COVS)), np.nan, dtype=np.float32)
    for j, c in enumerate(COVS):
        v = np.full(n_cells, np.nan, dtype=np.float32); v[cc_t[selm]] = test[c].values[selm]
        f[:, j] = v - clim[:, j]
    Zdev[int(m)] = f
Zdev_slow = {int(m): np.nanmean(np.array([Zdev[x] for x in test_months if abs(x-m) <= HW]), axis=0)
             for m in test_months}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"anchors: {anchors}")
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

# ---------------- era-weighted Dhat + LOO weight fit ----------------
def dhat_era(t, excl=None):
    others = [b for b in anchors if b != excl]
    ws = np.array([np.exp(-abs(t-b)/TAU) for b in others], dtype=np.float64)
    stack = np.array([AF[b] for b in others], dtype=np.float64)
    ok = np.isfinite(stack)
    wmat = np.where(ok, ws[:, None], 0.0)
    num = np.nansum(np.where(ok, stack, 0.0)*wmat, axis=0)
    den = wmat.sum(axis=0)
    d = np.where(den > 1e-9, num/np.maximum(den, 1e-9), np.nan)
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)

Xs, ys = [], []
for a in anchors:
    d_ = dhat_era(a, excl=a)
    tx = trendex(a)
    ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([d_[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]),
                     np.vstack(Xs).T@np.concatenate(ys))
w1, w2, w3 = map(float, w_)
print(f"era D-tilde weights (tau={TAU:.0f}): {w1:.3f}/{w2:.3f}/{w3:.3f}")

# baseline LOO RMSE for the era teacher (report)
res = []
for a in anchors:
    d_ = dhat_era(a, excl=a); tx = trendex(a)
    ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    dt = w1*d_ + w2*S + w3*tx
    res.append(np.sqrt(np.nanmean((dt[ok]-AF[a][ok])**2)))
print(f"era D-tilde LOO RMSE: all={np.sqrt(np.mean(np.array(res)**2)):.4f}")

# ---------------- cov-slow student (v11b) ----------------
def dtil_era(t, excl=None):
    return (w1*dhat_era(t, excl=excl) + w2*S + w3*trendex(t)).astype(np.float32)

Xtr, ytr = [], []
for a in anchors:
    dtr = dtil_era(a, excl=a)
    fs = Zdev_slow[a]
    ok = np.isfinite(dtr) & np.isfinite(AF[a]) & np.isfinite(fs).all(axis=1)
    Xtr.append(np.column_stack([fs[ok], np.ones(ok.sum())]))
    ytr.append((AF[a]-dtr)[ok])
XA, yA = np.vstack(Xtr), np.concatenate(ytr)
beta = np.linalg.solve(XA.T@XA + LAM_STUD*np.eye(6), XA.T@yA)
print(f"student beta: {np.round(beta,4)}")
def student_corr(t):
    fs = Zdev_slow[int(t)] if int(t) in Zdev_slow else None
    if fs is None: return np.zeros(n_cells, dtype=np.float32)
    c = np.column_stack([np.nan_to_num(fs), np.ones(n_cells)]) @ beta
    return c.astype(np.float32)
def dtil_stud(t, excl=None):
    return (dtil_era(t, excl=excl) + student_corr(t)).astype(np.float32)

# honest LOO check of the student-corrected D-tilde
res_s = []
for a in anchors:
    # refit beta without anchor a (honest)
    Xb, yb = [], []
    for b in anchors:
        if b == a: continue
        dtr = dtil_era(b, excl=b); fs = Zdev_slow[b]
        ok = np.isfinite(dtr) & np.isfinite(AF[b]) & np.isfinite(fs).all(axis=1)
        Xb.append(np.column_stack([fs[ok], np.ones(ok.sum())])); yb.append((AF[b]-dtr)[ok])
    XBB, yBB = np.vstack(Xb), np.concatenate(yb)
    bb = np.linalg.solve(XBB.T@XBB + LAM_STUD*np.eye(6), XBB.T@yBB)
    fs = Zdev_slow[a]
    ok = np.isfinite(AF[a]) & np.isfinite(fs).all(axis=1)
    c = np.column_stack([np.nan_to_num(fs[ok]), np.ones(ok.sum())]) @ bb
    dt = dtil_era(a, excl=a)[ok]
    res_s.append(np.sqrt(np.mean((dt+c-AF[a][ok])**2)))
print(f"student-corrected D-tilde LOO RMSE: all={np.sqrt(np.mean(np.array(res_s)**2)):.4f}")

# ---------------- Kalman (parameterized by Dtil function) ----------------
LAM_F = 0.84
def calibrate(Wd, dtil):
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = dtil(a, excl=a)
        ok = np.isfinite(Wd[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(Wd[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(Wd[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    return c_/(LAM_F*var_f), max(varz - c_*c_/(LAM_F*var_f), 1e-4), var_f

def kalman_predict(phi_f, Wd, dtil):
    H, R, var_f = calibrate(Wd, dtil)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    print(f"    kalman phi={phi_f}: H={H:.4f} R={R:.4f} var_f={var_f:.4f}", flush=True)
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for a in anchors:
        fa = AF[a] - dtil(a, excl=a)
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
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
            pred[sel] = mu_c[cc_t[sel]] + dtil(tm)[cc_t[sel]] + x2[cc_t[sel]]
    return pred

def smooth_rows(pred, rows, Wt):
    out = pred.copy()
    for m in np.unique(ta[rows & np.isfinite(pred)]):
        selm = np.where((ta == m) & rows & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[cc_t[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), Wt))
        out[selm] = sm[cc_t[selm]]
    return out

# ---------------- k=0 model B: v8 two-comp D-aware blend (train once) ----------------
print("k0-B: training two-comp + LGBM...", flush=True)
Lk = train[['cc','t_abs','TWS_t','target']+COVS].copy(); Lk['t_next'] = Lk['t_abs']+1
Rk = train[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgk = Lk.merge(Rk, on=['cc','t_next'], how='left')
mgk = mgk[mgk['target'].notna() & mgk['TWS_t'].notna()]
has_nxt_tr = mgk[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
ccm = mgk['cc'].values; yr = mgk['t_abs'].values // 12
colsR = [0,1,2,3,4,5,11]

def trendex_vec(t_arr, cc_arr):
    return ((np.asarray(t_arr, dtype=np.float64)-tbar_c[cc_arr])*beta_c[cc_arr]).astype(np.float32)
slow0 = trendex_vec(mgk['t_abs'].values, ccm); slow1 = trendex_vec(mgk['t_next'].values, ccm)
Xlin = np.column_stack([
    mgk['TWS_t'].values - mu_c[ccm] - slow0,
    *[mgk[c].values - clim[ccm, j] for j, c in enumerate(COVS)],
    *[mgk[c+'_nxt'].values - clim[ccm, j] for j, c in enumerate(COVS)],
    np.ones(len(mgk))]).astype(np.float32)
yB = (mgk['target'].values - mu_c[ccm] - slow1).astype(np.float32)
Xlin = np.nan_to_num(Xlin, nan=0.0)
w_recB = np.where(yr <= 2006, 1.0, np.where(yr <= 2009, 1.5, 2.0)).astype(np.float32)
swB = np.sqrt(w_recB); selF = has_nxt_tr
A_fB = Xlin[selF]*swB[selF,None]
coefF_B = np.linalg.solve(A_fB.T@A_fB + 1e-3*np.eye(12), A_fB.T@(yB[selF]*swB[selF]))
A_rB = Xlin[:, colsR]*swB[:,None]
coefR_B = np.linalg.solve(A_rB.T@A_rB + 1e-3*np.eye(7), A_rB.T@(yB*swB))

lat_cc = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
lat_arr = lat_cc['lat'].values.astype(np.float32); lon_arr = lon_arr = lat_cc['lon'].values.astype(np.float32)
mon_tr = (mgk['t_abs'].values % 12) + 1
XLB = np.column_stack([Xlin[:,0], slow1, Xlin[:,1:11],
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm], lat_arr[ccm], lon_arr[ccm]]).astype(np.float32)
okY = np.isfinite(yB)
params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=8)
ds  = lgb.Dataset(XLB[selF], label=yB[selF], weight=w_recB[selF])
dsr = lgb.Dataset(XLB[~selF & okY], label=yB[~selF & okY], weight=w_recB[~selF & okY])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, dsr, num_boost_round=400)
print("k0-B LGBM trained.")

Lt = test[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = test[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
mon_te = (mgt['t_abs'].values % 12) + 1

def build_k0B(dtil):
    uniq_m = np.unique(mgt['t_abs'].values)
    Dcache = {}
    for m in uniq_m:
        d0 = dtil(int(m), excl=int(m))      # honest at own anchor month
        d1 = dtil(int(m)+1)
        Dcache[int(m)] = (d0, d1)
    Dt0 = np.zeros(len(mgt), dtype=np.float32); Dt1 = np.zeros(len(mgt), dtype=np.float32)
    for i, (m, tn) in enumerate(zip(mgt['t_abs'].values, mgt['t_next'].values)):
        d0, d1 = Dcache[int(m)]
        Dt0[i] = d0[cct[i]]; Dt1[i] = d1[cct[i]]
    Xv = np.column_stack([
        mgt['TWS_t'].values - mu_c[cct] - Dt0,
        *[mgt[c].values - clim[cct, j] for j, c in enumerate(COVS)],
        *[mgt[c+'_nxt'].values - clim[cct, j] for j, c in enumerate(COVS)],
        np.ones(len(mgt))]).astype(np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0)
    XLv = np.column_stack([Xv[:,0], Dt1, Xv[:,1:11],
        has_nxt_te.astype(np.float32),
        np.sin(2*np.pi*mon_te/12), np.cos(2*np.pi*mon_te/12),
        beta_c[cct], mu_c[cct], lat_arr[cct], lon_arr[cct]]).astype(np.float32)
    k0_B_lin = np.full(len(test), np.nan, dtype=np.float64)
    k0_B_lin[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + Xv[use_full] @ coefF_B
    k0_B_lin[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + Xv[use_red][:, colsR] @ coefR_B
    k0_B_lgb = np.full(len(test), np.nan, dtype=np.float64)
    k0_B_lgb[use_full] = mu_c[cct[use_full]] + Dt1[use_full] + bst.predict(XLv[use_full])
    k0_B_lgb[use_red]  = mu_c[cct[use_red]]  + Dt1[use_red]  + bst_r.predict(XLv[use_red])
    k0_blendB = 0.5*np.nan_to_num(k0_B_lin) + 0.5*np.nan_to_num(k0_B_lgb)
    badB = (~msk) & np.isnan(k0_blendB)
    if badB.any(): k0_blendB[badB] = mu_c[cc_t][badB]
    k0_B = smooth_rows(np.where(np.isfinite(k0_blendB), k0_blendB, mu_c[cc_t]), ~msk & tw_ok, GAU10)
    return np.where(np.isfinite(k0_B), k0_B, k0_blendB)

# ---------------- assemble ----------------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm = ~msk

def assemble(masked_pred, k0_pred, tag):
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm] = k0_pred[unm]
    pred[msk] = masked_pred[msk]
    pred = np.where(np.isnan(pred), mu_c[cc_t], pred)
    pred = smooth_rows(pred, msk, GAU2)
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    print(f"saved {tag}: mean={pred.mean():.4f} std={pred.std():.4f} nan={int(out['Target'].isna().sum())}", flush=True)
    return pred

print("\nmasked Kalman runs (era Dtil)...", flush=True)
p74_a = kalman_predict(0.74, W_pool, dtil_era)
p80_a = kalman_predict(0.80, W_pool, dtil_era)
p_ens_a = 0.5*p74_a + 0.5*p80_a
print("k0-B (era Dtil)...", flush=True)
k0B_a = build_k0B(dtil_era)
print("\n--- v11a: era-weighted Dhat (tau=12) ---", flush=True)
pred_a = assemble(p_ens_a, k0B_a, 'v11a')

print("\nmasked Kalman runs (era + student Dtil)...", flush=True)
p74_b = kalman_predict(0.74, W_pool, dtil_stud)
p80_b = kalman_predict(0.80, W_pool, dtil_stud)
p_ens_b = 0.5*p74_b + 0.5*p80_b
print("k0-B (era + student Dtil)...", flush=True)
k0B_b = build_k0B(dtil_stud)
print("\n--- v11b: era-weighted Dhat + cov-slow student ---", flush=True)
pred_b = assemble(p_ens_b, k0B_b, 'v11b')

# ---------------- verification ----------------
print("\n=== verification ===", flush=True)
base = pd.read_csv(f'{DL}/submission_v10b.csv')['Target'].values
for tag, pp in [('v11a', pred_a), ('v11b', pred_b)]:
    o = pd.read_csv(f'{DL}/submission_{tag}.csv')
    assert len(o) == 280961, tag
    assert (o['ID'].values == sub['ID'].values).all(), tag
    assert o['Target'].notna().all() and np.isfinite(o['Target']).all(), tag
    d = o['Target'].values - base
    dm = d[msk]; dk = d[unm]
    # public-window diff (first 7 test months)
    pub = ta < (int(test_months.min()) + 12)
    print(f"{tag}: corr vs v10b={np.corrcoef(o['Target'].values, base)[0,1]:.5f} "
          f"mean|d|={np.abs(d).mean():.4f} | masked|d|={np.abs(dm).mean():.4f} k0|d|={np.abs(dk).mean():.4f} "
          f"| public-rows mean|d|={np.abs(d[pub]).mean():.4f} | range [{o['Target'].min():.3f}, {o['Target'].max():.3f}]")
print("\nDONE.")
