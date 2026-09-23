"""V12 ATTRIBUTION STUDY — decompose the v11b LB regression (0.7000 -> 0.7106) on the
honest 2013-15 val window (real targets, Spearman 1.0 vs LB across 5 shipped variants).

v11 bundled 3 changes on top of v10b. This study isolates each:
  V0  v10b-analog : static Dhat, FULL-at-anchor calib/init (the proven recipe)
  V1  honest-calib: V0 but LOO-at-anchor calib/init only        <- isolates change (3)
  V2  era-surgical: V0 exactly, but PRED line uses era Dtil(tm)  <- isolates change (1)
  V2d era-dedup   : V2 minus adjacent-anchor fast contamination
  V3  v11a-analog : era Dtil everywhere + LOO calib/init         <- changes (1)+(3)
  V4  stud-surgical: V2 + student in PRED line only              <- adds change (2)
  V5  v11b-analog : era+student everywhere + LOO calib           <- all three (the shipped v11b)
  + tau sweep {6,12,24} on V2.

All variants: W-pool box2 obs, phi-ens {0.74,0.80}, gau2.0 smoothing = the v10b masked
pipeline. Eval: masked val rows, RMSE overall and by horizon h (target-anchor distance).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
HW = 4; LAM_STUD = 1000.0

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
yms_all = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms_all)}

fit = train[train['time'].dt.year <= 2012].copy()
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
print(f"fit {len(fit):,} | val {len(val):,}", flush=True)

# ---------------- masking pattern from test ----------------
test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
test_raw['masked'] = test_raw['TWS_t_masked'].astype(bool)
test_raw['cal_mon'] = test_raw['time'].dt.month
cal_mask_frac = test_raw.groupby('cal_mon')['masked'].mean()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values; val_ta = val['t_abs'].values; val_msk = val['masked'].values

# ---------------- infra on fit only ----------------
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
sxy = np.nansum(td*F64, axis=0); sxx = np.nansum(td*td, axis=0)
beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), sxy/np.where(sxx > 0, sxx, 1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

# grid utils
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
BOX2 = {(di,dj): 1.0 for di in range(-2,3) for dj in range(-2,3)}
def gaussW(sig, r=4):
    return {(di,dj): float(np.exp(-(di*di+dj*dj)/(2*sig*sig)))
            for di in range(-r,r+1) for dj in range(-r,r+1)
            if np.exp(-(di*di+dj*dj)/(2*sig*sig)) > 0.01}
GAU2 = gaussW(2.0)
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

# ---------------- val cov fields, S, W_pool ----------------
Zv = val[COVS].values.astype('float32')
okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W_raw = {int(m): cov_field[m] - S for m in all_m_v}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W_raw.items()}

# ---------------- val anchors + AF ----------------
mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = sorted(int(v) for v in mfrac_v[mfrac_v < 0.5].index)
print(f"val anchors ({len(anchors_v)}): {anchors_v}")
AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c

Dhat_raw = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)
Dhat = np.where(np.isfinite(Dhat_raw), Dhat_raw, 0.0).astype(np.float32)

def dhat_era(t, excl=None, tau=12.0):
    others = [b for b in anchors_v if b != excl]
    ws = np.array([np.exp(-abs(t-b)/tau) for b in others], dtype=np.float64)
    stack = np.array([AF[b] for b in others], dtype=np.float64)
    ok = np.isfinite(stack)
    wmat = np.where(ok, ws[:, None], 0.0)
    num = np.nansum(np.where(ok, stack, 0.0)*wmat, axis=0)
    den = wmat.sum(axis=0)
    d = np.where(den > 1e-9, num/np.maximum(den, 1e-9), np.nan)
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)

def dhat_static(excl=None):
    others = [b for b in anchors_v if b != excl]
    d = np.nanmean(np.array([AF[b] for b in others]), axis=0)
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)

# LOO weight fits (static + era)
def fit_weights(dhatfun):
    Xs, ys = [], []
    for a in anchors_v:
        d_ = dhatfun(a, excl=a); tx = trendex(a)
        ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
        Xs.append(np.column_stack([d_[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
    w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
    return tuple(map(float, w_))
w1s, w2s, w3s = fit_weights(lambda t, excl: dhat_static(excl))
w1e, w2e, w3e = fit_weights(lambda t, excl: dhat_era(t, excl=excl))
print(f"static weights: {w1s:.3f}/{w2s:.3f}/{w3s:.3f} | era weights: {w1e:.3f}/{w2e:.3f}/{w3e:.3f}")

def Dtil_static(t, excl=None): return (w1s*dhat_static(excl) + w2s*S + w3s*trendex(t)).astype(np.float32)
def Dtil_era(t, excl=None, tau=12.0): return (w1e*dhat_era(t, excl=excl, tau=tau) + w2e*S + w3e*trendex(t)).astype(np.float32)

# dedup: remove adjacent-anchor FAST contamination from era Dhat at pred months
# fast_est(b) = AF[b] - static-LOO Dhat at b;  era weight w_b on AF[b] carries w_b*fast_est(b)
FAST = {b: (AF[b] - (w1s*dhat_static(excl=b) + w2s*S + w3s*trendex(b))).astype(np.float32) for b in anchors_v}
def Dtil_era_dedup(t, tau=12.0):
    base = Dtil_era(t, tau=tau).copy()
    others = [b for b in anchors_v if b != t]
    ws = np.array([np.exp(-abs(t+0.5-b)/tau) for b in others]); ws = ws/ws.sum()
    corr = np.zeros(n_cells, dtype=np.float32)
    for b, w in zip(others, ws):
        corr += w*w1e*np.nan_to_num(FAST[b])
    return (base - corr).astype(np.float32)

# student (v11 recipe on val)
Zdev = {}
for m in all_m_v:
    selm = val_ta == m
    f = np.full((n_cells, len(COVS)), np.nan, dtype=np.float32)
    for j, c in enumerate(COVS):
        v = np.full(n_cells, np.nan, dtype=np.float32); v[val_cc[selm]] = val[c].values[selm]
        f[:, j] = v - clim[:, j]
    Zdev[int(m)] = f
Zdev_slow = {int(m): np.nanmean(np.array([Zdev[x] for x in all_m_v if abs(x-m) <= HW]), axis=0)
             for m in all_m_v}
Xtr, ytr = [], []
for a in anchors_v:
    dtr = Dtil_era(a, excl=a); fs = Zdev_slow[a]
    ok = np.isfinite(dtr) & np.isfinite(AF[a]) & np.isfinite(fs).all(axis=1)
    Xtr.append(np.column_stack([fs[ok], np.ones(ok.sum())])); ytr.append((AF[a]-dtr)[ok])
XA, yA = np.vstack(Xtr), np.concatenate(ytr)
beta_stud = np.linalg.solve(XA.T@XA + LAM_STUD*np.eye(6), XA.T@yA)
def student_corr(t):
    fs = Zdev_slow[int(t)] if int(t) in Zdev_slow else None
    if fs is None: return np.zeros(n_cells, dtype=np.float32)
    return (np.column_stack([np.nan_to_num(fs), np.ones(n_cells)]) @ beta_stud).astype(np.float32)
def Dtil_stud(t, excl=None): return (Dtil_era(t, excl=excl) + student_corr(t)).astype(np.float32)
print(f"student std on pred months: {np.mean([np.nanstd(student_corr(int(m))) for m in all_m_v]):.4f}")

# ---------------- parameterized Kalman ----------------
def calibrate(calib_dtil):
    cs, zs, vfs = [], [], []
    for a in anchors_v:
        dj = calib_dtil(a)
        ok = np.isfinite(W_pool[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W_pool[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W_pool[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    return c_/(LAM_F*var_f), max(varz - c_*c_/(LAM_F*var_f), 1e-4), var_f

def kalman_predict(phi_f, calib_dtil, init_dtil, pred_dtil, h_track=None):
    """calib_dtil(a): Dtil for H/R calibration at anchor a (v10b: FULL static; v11: LOO).
       init_dtil(a):  Dtil for the Kalman init residual at anchor a.
       pred_dtil(tm): Dtil added at the PREDICTION line."""
    H, R, var_f = calibrate(calib_dtil)
    q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
    pred = np.full(len(val), np.nan, dtype=np.float64)
    if h_track is None: h_track = np.full(len(val), -1, dtype=np.int32)
    for a in anchors_v:
        fa = AF[a] - init_dtil(a)
        x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
        for k in range(1, 9):
            m = a + k
            x = phi_f*x; P = phi_f**2*P + q
            if m in W_pool:
                wv = W_pool[m]; okw = np.isfinite(wv)
                K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + K*(wv - H*x), x)
                P = np.where(okw, (1-K*H)*P, P)
            sel = np.where((val_ta == m) & val_msk)[0]
            if len(sel) == 0: continue
            tm = m + 1
            x2 = phi_f*x
            if tm in W_pool:
                wv = W_pool[tm]; okw = np.isfinite(wv)
                P2 = phi_f**2*P + q
                K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
            pred[sel] = mu_c[val_cc[sel]] + pred_dtil(tm)[val_cc[sel]] + x2[val_cc[sel]]
            h_track[sel] = k + 1
    return pred, h_track

def smooth_rows(pred, rows, Wt):
    out = pred.copy()
    for m in np.unique(val_ta[rows & np.isfinite(pred)]):
        selm = np.where((val_ta == m) & rows & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[val_cc[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), Wt))
        out[selm] = sm[val_cc[selm]]
    return out

def run_variant(name, calib_dtil, init_dtil, pred_dtil, era_tau=None):
    p1, h1 = kalman_predict(0.74, calib_dtil, init_dtil, pred_dtil)
    p2, _ = kalman_predict(0.80, calib_dtil, init_dtil, pred_dtil)
    p = 0.5*p1 + 0.5*p2
    p = smooth_rows(p, val_msk, GAU2)
    ok = val_msk & np.isfinite(p) & np.isfinite(val_target)
    rmse = float(np.sqrt(np.mean((p[ok]-val_target[ok])**2)))
    # per-horizon
    hs = {}; msg = []
    for hh in [2,3,4,5,6,7,8,9]:
        mm = ok & (h1 == hh)
        if mm.sum() > 500:
            hs[hh] = float(np.sqrt(np.mean((p[mm]-val_target[mm])**2)))
            msg.append(f"h{hh}:{hs[hh]:.4f}({int(mm.sum())})")
    print(f"  {name:<28} RMSE={rmse:.4f}  rows={int(ok.sum()):,}  {' '.join(msg)}", flush=True)
    return rmse

print("\n=== V12 ATTRIBUTION STUDY (val 2013-15 masked rows, v10b pipeline) ===")
results = {}
results['V0  v10b-analog (static/full)'] = run_variant('V0  v10b-analog',
    lambda a: Dtil_static(a), lambda a: Dtil_static(a), lambda t: Dtil_static(t))
results['V1  honest-calib only (LOO)'] = run_variant('V1  honest-calib',
    lambda a: Dtil_static(a, excl=a), lambda a: Dtil_static(a, excl=a), lambda t: Dtil_static(t))
results['V2  era-surgical (pred only)'] = run_variant('V2  era-surgical',
    lambda a: Dtil_static(a), lambda a: Dtil_static(a), lambda t: Dtil_era(t))
results['V2d era-surgical + dedup'] = run_variant('V2d era-dedup',
    lambda a: Dtil_static(a), lambda a: Dtil_static(a), lambda t: Dtil_era_dedup(t))
results['V3  v11a-analog (era+LOO)'] = run_variant('V3  v11a-analog',
    lambda a: Dtil_era(a, excl=a), lambda a: Dtil_era(a, excl=a), lambda t: Dtil_era(t))
results['V4  V2 + student (pred only)'] = run_variant('V4  era+stud-surgical',
    lambda a: Dtil_static(a), lambda a: Dtil_static(a), lambda t: Dtil_stud(t))
results['V5  v11b-analog (all three)'] = run_variant('V5  v11b-analog',
    lambda a: Dtil_stud(a, excl=a), lambda a: Dtil_stud(a, excl=a), lambda t: Dtil_stud(t))

print("\n--- tau sweep on V2 (era-surgical) ---")
for tau in [6.0, 18.0, 24.0]:
    results[f'V2_tau{int(tau)}'] = run_variant(f'V2_tau{int(tau)}',
        lambda a: Dtil_static(a), lambda a: Dtil_static(a), lambda t: Dtil_era(t, tau=tau))

print("\n--- calibration H/R per mode (context) ---")
for lbl, f in [('static-full (v10b)', lambda a: Dtil_static(a)),
               ('static-LOO', lambda a: Dtil_static(a, excl=a)),
               ('era-LOO (v11a)', lambda a: Dtil_era(a, excl=a)),
               ('era+stud-LOO (v11b)', lambda a: Dtil_stud(a, excl=a))]:
    H, R, var_f = calibrate(f)
    print(f"  {lbl:<24} H={H:.4f} R={R:.4f} var_f={var_f:.4f} K0={LAM_F*(1-LAM_F)*var_f*H/(H*H*LAM_F*(1-LAM_F)*var_f+R):.4f}")

print("\n=== SUMMARY (sorted) ===")
for k, v in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"  {v:.4f}  {k}")
print("\nDONE.")
