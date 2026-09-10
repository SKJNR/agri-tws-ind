"""
15-b LOO AUDIT — Mission A (masked-row constants sensitivity) + Mission C (D-error attack)
Harness: test-era anchor-LOO at TRUE test distances (generalized from v8_loo2.py).
Baseline to reproduce: obs-dn + per-cell H/R (v8a config) = 0.7369.

Sections:
  0. baseline reproduction + per-pair errors
  1. MISSION A: PHI / LAM_F / K / percell-blend / clip / D-tilde-weight sensitivities
  2. MISSION C-a: D-residual spatial smoothness + D-hat smoothing / PC-projection experiments
  3. MISSION C-b: anchor-pair drift variants (global / per-cell / smoothed slope, shrunk alphas)
  4. MISSION C-c: 4th D-tilde regressor (S_late / SPEI_12 static / SPEI_12 deviation(t))
"""
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ================= infra (from v8_loo2.py) =================
print("loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
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
F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:,None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None,:]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where(sxx>100, np.nansum(td*F64,axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t)-tbar_c)*beta_c).astype(np.float32)

full = ~np.isnan(F).any(axis=0)
TDm = t_abs_tr[:,None]-tbar_c[None,:]
A_dt = F64 - mu_c[None,:] - TDm*beta_c[None,:]
A_dtf = A_dt[:,full]; A_dtf = A_dtf - A_dtf.mean(axis=0, keepdims=True)
print("SVD...", flush=True)
_,_,Vt = np.linalg.svd(A_dtf, full_matrices=False)
Vt = Vt[:400].astype(np.float32)   # keep top 400 for K-sweeps
def dn(field, K):
    V = Vt[:K].T
    out = field.copy(); x = field[full]; okx = np.isfinite(x)
    out[full] = np.where(okx, V@(V.T@np.where(okx,x,0.0)), x)
    return out

Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1)&np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# per-cell H/R sufficient stats (LAM-independent; Hc_raw(lam) computed on demand)
cov_est_tr = np.full(len(train), np.nan, dtype=np.float32)
Zok = np.isfinite(Z).all(axis=1)
cov_est_tr[Zok] = np.column_stack([Z[Zok], np.ones(Zok.sum())]) @ coef
tr_ta = train['t_abs'].values; tr_cc = train['cc'].values
A_tr = yv - mu_c[tr_cc] - ((tr_ta - tbar_c[tr_cc])*beta_c[tr_cc]).astype(np.float32)
gm = pd.Series(cov_est_tr).groupby(tr_cc).mean().reindex(range(n_cells)).fillna(0).values
Wd_tr = np.where(np.isfinite(cov_est_tr), cov_est_tr - gm[tr_cc], np.nan)
tok = np.isfinite(Wd_tr)&np.isfinite(A_tr)
ci = tr_cc[tok].astype(np.int64)
Wv = Wd_tr[tok].astype(np.float64); Av = A_tr[tok].astype(np.float64)
n_c = np.bincount(ci, minlength=n_cells)
sW = np.bincount(ci, weights=Wv, minlength=n_cells); sA = np.bincount(ci, weights=Av, minlength=n_cells)
sWW = np.bincount(ci, weights=Wv*Wv, minlength=n_cells); sAA = np.bincount(ci, weights=Av*Av, minlength=n_cells)
sWA = np.bincount(ci, weights=Wv*Av, minlength=n_cells)
with np.errstate(invalid='ignore', divide='ignore'):
    varW_c = sWW/n_c-(sW/n_c)**2; varA_c = sAA/n_c-(sA/n_c)**2; covWA_c = sWA/n_c-(sW/n_c)*(sA/n_c)
def percell_raw(lam):
    Hr = covWA_c/(lam*varA_c); Rr = varW_c - Hr*Hr*(lam*varA_c)
    val = (n_c>=100)&(varA_c>1e-6)&(varW_c>1e-6)&(Rr>0)
    Hr = np.where(val, Hr, np.nan); Rr = np.where(val, Rr, np.nan)
    return Hr, Rr, val

# S_late: mean cov-regression field over late-train window (2013-01..2015-08), minus mu_c
late = (tr_ta//12)>=2013
S_late = pd.Series(np.where(late, cov_est_tr, np.nan)).groupby(tr_cc).mean().reindex(range(n_cells)).values - mu_c
S_late = S_late.astype(np.float32)

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

Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}; spei12_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta==m)&okt
    fm = np.full(n_cells, np.nan, dtype=np.float32); f12 = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]; f12[cc_t[selm]] = Zt_raw[selm,3]
    cov_field[m] = fm - mu_c; spei12_field[m] = f12
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W = {int(m): cov_field[m]-S for m in all_m}
M12 = np.nanmean(np.array([spei12_field[m] for m in all_m]), axis=0)
dev12 = {int(m): spei12_field[m]-M12 for m in all_m}
WdnCache = {}
def Wdn(m, K):
    key = (m, K)
    if key not in WdnCache: WdnCache[key] = dn(W[m], K)
    return WdnCache[key]

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac<0.01].index)
AF = {}
for a in anchors:
    sel = (ta==a)&(~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
print(f"anchors: {anchors}")

PAIRS = [(anchors[i],anchors[i+1]) for i in range(len(anchors)-1)]
PAIRS = [(i,k) for i,k in PAIRS if k-i<=8]
print(f"PAIRS: {PAIRS}")

# ---- spatial smoothing helper (1-degree grid, normalized Gaussian convolution) ----
lat_g = codes['lat'].values.astype(np.float64); lon_g = codes['lon'].values.astype(np.float64)
ulat = np.sort(np.unique(lat_g)); ulon = np.sort(np.unique(lon_g))
iy = np.searchsorted(ulat, lat_g); ix = np.searchsorted(ulon, lon_g)
NY, NX = len(ulat), len(ulon)
def smooth_field(field, sigma):
    G = np.full((NY,NX), np.nan, dtype=np.float64); G[iy,ix] = field
    M = np.isfinite(G); Gf = np.where(M, G, 0.0)
    num = gaussian_filter(Gf, sigma, mode='nearest'); den = gaussian_filter(M.astype(np.float64), sigma, mode='nearest')
    out = np.where(den>1e-8, num/np.maximum(den,1e-8), np.nan)
    return out[iy,ix].astype(np.float32)

# ---- extra regressors ----
XREG_STATIC = {'S_late': S_late, 'SPEI12': M12.astype(np.float32)}
def xreg_field(xreg, t):
    if xreg=='S_late': return S_late
    if xreg=='SPEI12': return M12.astype(np.float32)
    if xreg=='s12dev': return dev12[int(t)]
    raise ValueError(xreg)

# ================= generalized run =================
def fit_weights(avail, ridge=1e-3, xreg=None, transform=None):
    Xs, ys = [], []
    for a in avail:
        dloo = np.nanmean(np.array([AF[b] for b in avail if b!=a]), axis=0)
        if transform is not None: dloo = transform(dloo)
        tx = trendex(a)
        ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])&np.isfinite(tx)
        if xreg is not None:
            ex = xreg_field(xreg, a); ok = ok&np.isfinite(ex)
            Xs.append(np.column_stack([dloo[ok],S[ok],tx[ok],ex[ok]]))
        else:
            Xs.append(np.column_stack([dloo[ok],S[ok],tx[ok]]))
        ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    p = Xw.shape[1]
    w_ = np.linalg.solve(Xw.T@Xw + ridge*np.eye(p), Xw.T@yw)
    return np.asarray(w_, dtype=np.float64)

def run(phi=0.74, lam=0.84, K=200, percell=True, pblend=0.5, clip=(0.3,3.0),
        wmode='refit', ridge=1e-3, drift=None, dhat_smooth=0.0, dhat_proj=0,
        dtil_smooth=0.0, xreg=None, obs_dn=True):
    """drift: None | (mode, alpha) with mode in {'global','cell','smooth3'}"""
    Hr_l, Rr_l, val_l = percell_raw(lam)
    Hg_tr = float(np.nanmean(Hr_l)); Rg_tr = float(np.nanmean(Rr_l))
    def transform(Dh):
        out = Dh.copy()
        if dhat_proj>0:
            Vp = Vt[:dhat_proj].T
            xf = np.where(np.isfinite(Dh[full]), Dh[full], 0.0).astype(np.float32)
            out[full] = (Vp@(Vp.T@xf)).astype(np.float32)
        if dhat_smooth>0: out = smooth_field(out, dhat_smooth)
        return out
    errs = []
    for (i,k) in PAIRS:
        avail = [a for a in anchors if a!=k]
        t_av = np.array(avail, dtype=np.float64); tbar = t_av.mean()
        Dh = np.nanmean(np.array([AF[a] for a in avail]), axis=0)
        DhT = transform(Dh)
        # weights
        if wmode=='refit': w_ = fit_weights(avail, ridge, xreg, transform)
        elif wmode=='fixed': w_ = np.array([0.650,0.456,0.073])
        elif wmode=='noTrend': w_ = np.append(fit_weights2(avail, ridge, transform), 0.0)
        elif wmode.startswith('pert'):
            w_ = fit_weights(avail, ridge, None, transform)
            _idx, _fac = wmode[4:].split('_')
            w_ = w_.copy(); w_[int(_idx)-1] *= float(_fac)
        elif wmode=='ridge1e-1': w_ = fit_weights(avail, 0.1, xreg, transform)
        else: raise ValueError(wmode)
        # drift field
        driftf = None
        if drift is not None:
            mode, alpha = drift
            stack = np.array([AF[a] for a in avail])
            Mf = np.nanmean(stack, axis=0)
            sc = np.zeros(n_cells); stt = ((t_av-tbar)**2).sum()
            for a in avail: sc += (a-tbar)*np.nan_to_num(AF[a]-Mf)
            slope = sc/max(stt,1e-9)
            if mode=='global': slope = np.full(n_cells, np.nanmean(slope), dtype=np.float64)
            elif mode=='smooth3': slope = smooth_field(slope.astype(np.float32), 3.0)
            driftf = (alpha*slope).astype(np.float32)
        def Dtil(t):
            d = w_[0]*DhT + w_[1]*S + w_[2]*trendex(t)
            if xreg is not None: d = d + w_[3]*xreg_field(xreg, t)
            if driftf is not None: d = d + driftf*(np.float64(t)-tbar)
            d = d.astype(np.float32)
            if dtil_smooth>0: d = smooth_field(d, dtil_smooth)
            return d
        # global H/R calibration
        cs, zs, vfs = [], [], []
        for a in avail:
            dj = Dtil(a); wv = W[a]
            ok = np.isfinite(wv)&np.isfinite(AF[a])&np.isfinite(dj)
            cs.append(np.cov(wv[ok],(AF[a]-dj)[ok])[0,1]); zs.append(np.var(wv[ok])); vfs.append(np.nanvar(AF[a]-dj))
        c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
        H = c_/(lam*var_f); R = max(varz-c_*c_/(lam*var_f),1e-4)
        if percell:
            lo, hi = clip
            rH = np.clip(Hr_l/Hg_tr, lo, hi); rR = np.clip(Rr_l/Rg_tr, lo, hi)
            Hc = np.where(val_l, H*(pblend+(1-pblend)*rH), H).astype(np.float32)
            Rc = np.where(val_l, np.maximum(R*(pblend+(1-pblend)*rR),1e-4), R).astype(np.float32)
        else:
            Hc = np.full(n_cells, H, np.float32); Rc = np.full(n_cells, R, np.float32)
        q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f
        x = np.where(np.isfinite(AF[i]), lam*np.nan_to_num(AF[i]-Dtil(i)), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        for m in range(i+1, k+1):
            x = phi*x; P = phi**2*P + q
            if m in W:
                wv = Wdn(m,K) if obs_dn else W[m]
                okw = np.isfinite(wv)
                Kg = np.where(okw, P*Hc/(Hc*Hc*P+Rc), 0).astype(np.float32)
                x = np.where(okw, x+Kg*(np.nan_to_num(wv)-Hc*x), x)
                P = np.where(okw, (1-Kg*Hc)*P, P)
        pred = Dtil(k) + x
        ok = np.isfinite(pred)&np.isfinite(AF[k])
        errs.append(float(np.sqrt(np.mean((pred[ok]-AF[k][ok])**2))))
    return float(np.mean(errs)), errs

def fit_weights2(avail, ridge, transform):
    """2-regressor (Dhat, S) fit"""
    Xs, ys = [], []
    for a in avail:
        dloo = np.nanmean(np.array([AF[b] for b in avail if b!=a]), axis=0)
        if transform is not None: dloo = transform(dloo)
        ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])
        Xs.append(np.column_stack([dloo[ok],S[ok]])); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    return np.linalg.solve(Xw.T@Xw + ridge*np.eye(2), Xw.T@yw)

# ================= SECTION 0: baseline =================
print("\n"+"="*70+"\nSECTION 0: baseline reproduction (expect 0.7369)\n"+"="*70)
b, e = run(); print(f"baseline obs-dn+percell: {b:.4f}  per-pair {['%.4f'%v for v in e]}")
b2, e2 = run(percell=False); print(f"obs-dn only (no percell): {b2:.4f}  per-pair {['%.4f'%v for v in e2]}")

# ================= SECTION 1: MISSION A — constants sensitivity =================
import os
if not os.environ.get('SKIP1'):
    print("\n"+"="*70+"\nSECTION 1: MISSION A — masked-row constants (deltas vs baseline)\n"+"="*70)
    print("--- PHI sweep ---")
    for phi in [0.66, 0.70, 0.74, 0.78, 0.82, 0.86]:
        r, e = run(phi=phi); print(f"phi={phi:.2f}: {r:.4f} ({r-b:+.4f})  {['%.4f'%v for v in e]}")
    print("--- LAM_F sweep ---")
    for lam in [0.78, 0.81, 0.84, 0.87, 0.90]:
        r, e = run(lam=lam); print(f"lam={lam:.2f}: {r:.4f} ({r-b:+.4f})")
    print("--- K (PC-denoise basis) sweep ---")
    for K in [50, 100, 150, 200, 300, 400]:
        r, e = run(K=K); print(f"K={K}: {r:.4f} ({r-b:+.4f})")
    print("--- per-cell blend factor sweep (0=raw train ratios, 1=global only) ---")
    for pb in [0.0, 0.3, 0.5, 0.7, 1.0]:
        r, e = run(pblend=pb); print(f"pblend={pb:.1f}: {r:.4f} ({r-b:+.4f})")
    print("--- clip bounds ---")
    for clip in [(0.2,5.0), (0.3,3.0), (0.5,2.0), (0.15,7.0)]:
        r, e = run(clip=clip); print(f"clip={clip}: {r:.4f} ({r-b:+.4f})")
    print("--- D-tilde weight modes ---")
    r,_ = run(wmode='fixed');    print(f"weights fixed 0.650/0.456/0.073: {r:.4f} ({r-b:+.4f})  [IN-SAMPLE: fixed wts were fit on these very anchors]")
    r,_ = run(wmode='noTrend');  print(f"weights 2-regressor (no trendex): {r:.4f} ({r-b:+.4f})")
    r,_ = run(wmode='ridge1e-1');print(f"ridge 1e-1 (vs 1e-3): {r:.4f} ({r-b:+.4f})")
    r,_ = run(wmode='pert1_1.15'); print(f"W1*1.15: {r:.4f} ({r-b:+.4f})")
    r,_ = run(wmode='pert1_0.85'); print(f"W1*0.85: {r:.4f} ({r-b:+.4f})")
    r,_ = run(wmode='pert2_1.15'); print(f"W2*1.15: {r:.4f} ({r-b:+.4f})")
    r,_ = run(wmode='pert2_0.85'); print(f"W2*0.85: {r:.4f} ({r-b:+.4f})")

# ================= SECTION 2: MISSION C-a — residual smoothness + D-hat smoothing =================
print("\n"+"="*70+"\nSECTION 2: MISSION C-a — D-residual spatial structure\n"+"="*70)
def grid_lag_corr(r, lags):
    G = np.full((NY,NX), np.nan, dtype=np.float64); G[iy,ix] = r
    out = []
    for (dy,dx) in lags:
        if dy>=0: tr = slice(dy,None); sr = slice(0,NY-dy)
        else:     tr = slice(0,dy);    sr = slice(-dy,None)
        if dx>=0: tc = slice(dx,None); sc = slice(0,NX-dx)
        else:     tc = slice(0,dx);    sc = slice(-dx,None)
        B = np.full_like(G, np.nan)
        B[tr,tc] = G[sr,sc]
        m = np.isfinite(G)&np.isfinite(B)
        if m.sum()<100: out.append(np.nan); continue
        a = G[m]-G[m].mean(); bb = B[m]-B[m].mean()
        out.append(float((a*bb).sum()/np.sqrt((a*a).sum()*(bb*bb).sum())))
    return float(np.nanmean(out)), out
LAGS1 = [(1,0),(0,1),(-1,0),(0,-1)]; LAGS2 = [(2,0),(0,2),(-2,0),(0,-2)]; LAGSD = [(1,1),(-1,-1),(1,-1),(-1,1)]
for (i,k) in PAIRS:
    avail = [a for a in anchors if a!=k]
    Dh = np.nanmean(np.array([AF[a] for a in avail]), axis=0)
    w_ = fit_weights(avail)
    Dtk = (w_[0]*Dh + w_[1]*S + w_[2]*trendex(k)).astype(np.float32)
    rD = AF[k]-Dtk
    c1,_ = grid_lag_corr(np.nan_to_num(rD), LAGS1); c2,_ = grid_lag_corr(np.nan_to_num(rD), LAGS2); cd,_ = grid_lag_corr(np.nan_to_num(rD), LAGSD)
    rng = np.random.default_rng(0); rsh = rng.permutation(np.nan_to_num(rD))
    cn,_ = grid_lag_corr(rsh, LAGS1)
    print(f"pair {i}->{k}: D-resid lag1={c1:.3f} lag2={c2:.3f} diag1.4={cd:.3f} | shuffled-null lag1={cn:.3f}")
print("--- D-hat smoothing / projection experiments ---")
for s in [1.0, 2.0, 3.0]:
    r, e = run(dhat_smooth=s); print(f"Dhat gaussian smooth sigma={s}deg: {r:.4f} ({r-b:+.4f})  {['%.4f'%v for v in e]}")
for s in [1.0, 2.0]:
    r, e = run(dtil_smooth=s); print(f"Dtil(t) gaussian smooth sigma={s}deg: {r:.4f} ({r-b:+.4f})")
for pr in [50, 100, 200]:
    r, e = run(dhat_proj=pr); print(f"Dhat PC-projection r={pr}: {r:.4f} ({r-b:+.4f})")

# ================= SECTION 3: MISSION C-b — drift =================
print("\n"+"="*70+"\nSECTION 3: MISSION C-b — anchor-drift variants\n"+"="*70)
for mode in ['global','cell','smooth3']:
    for alpha in [0.25, 0.5, 1.0]:
        r, e = run(drift=(mode, alpha))
        print(f"drift mode={mode} alpha={alpha:.2f}: {r:.4f} ({r-b:+.4f})  {['%.4f'%v for v in e]}")

# ================= SECTION 4: MISSION C-c — 4th regressor =================
print("\n"+"="*70+"\nSECTION 4: MISSION C-c — 4th D-tilde regressor\n"+"="*70)
for xr in ['S_late','SPEI12','s12dev']:
    r, e = run(xreg=xr)
    print(f"xreg={xr}: {r:.4f} ({r-b:+.4f})  {['%.4f'%v for v in e]}")
# combined: s12dev + best drift if any
r, e = run(xreg='s12dev', drift=('smooth3',0.5))
print(f"xreg=s12dev + drift smooth3/0.5: {r:.4f} ({r-b:+.4f})  {['%.4f'%v for v in e]}")

print("\nDONE audit_15b_loo.")
