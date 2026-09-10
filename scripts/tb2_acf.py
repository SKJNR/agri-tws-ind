"""2-b step 2: seasonality check (TWS + covariates) and detrended ACF to lag 60 + AR(2) roots."""
import numpy as np
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
F, A_dt, yms = d['F'], d['A_dt'], d['yms']
lat_c, lon_c = d['lat_c'], d['lon_c']
SPEI1, SPEI3, SPEI6, SPEI12, SM = d['SPEI1'], d['SPEI3'], d['SPEI6'], d['SPEI12'], d['SM']
T, n_cells = A_dt.shape

months = (yms % 100).astype(int)

# ---------- (A) Seasonality ----------
print("="*70)
print("(A) SEASONALITY CHECK")
mm = np.array([f"{m:02d}" for m in range(1,13)])
for name, M in [('TWS detrended anomaly', A_dt), ('TWS raw-minus-mean', F.astype(np.float64)-np.nanmean(F,axis=0)),
                ('SPEI_01', SPEI1.astype(np.float64)), ('SPEI_12', SPEI12.astype(np.float64)), ('SOIL_MOISTURE', SM.astype(np.float64))]:
    # remove per-cell time mean of the cov itself, then monthly means
    Mc = M - np.nanmean(M, axis=0, keepdims=True)
    clim = np.array([np.nanmean(Mc[months==m]) for m in range(1,13)])
    print(f"  {name:24s}: monthly clim amplitude (max-min)={clim.max()-clim.min():.4f}, pooled std={np.nanstd(Mc):.4f}, ratio={(clim.max()-clim.min())/np.nanstd(Mc):.4f}")

# per-cell seasonal amplitude of TWS anomaly (fit sin/cos)
t = np.arange(T)
cosv = np.cos(2*np.pi*t/12); sinv = np.sin(2*np.pi*t/12)
cosv = cosv - cosv.mean(); sinv = sinv - sinv.mean()
amp = np.zeros(n_cells)
for lim in [slice(None)]:
    Ac = A_dt - np.nanmean(A_dt, axis=0, keepdims=True)
    a1 = np.nansum(Ac*cosv[:,None],axis=0)/np.sum(cosv**2)
    a2 = np.nansum(Ac*sinv[:,None],axis=0)/np.sum(sinv**2)
    amp = np.sqrt(a1**2+a2**2)
print(f"  per-cell annual amplitude (sin/cos fit): mean={amp.mean():.4f}, p90={np.percentile(amp,90):.4f} vs per-cell std={np.nanstd(A_dt,axis=0).mean():.4f}")

# ---------- (B) Detrended ACF to lag 60 ----------
print("="*70)
print("(B) DETRENDED ANOMALY ACF (per-cell, averaged; calendar-correct lags)")
Ac = A_dt - np.nanmean(A_dt, axis=0, keepdims=True)
Ac = np.where(np.isfinite(Ac), Ac, np.nan)
# use only cells with full obs
full = ~np.isnan(Ac).any(axis=0)
print(f"  full-obs cells: {full.sum()}")
X = Ac[:, full]
Xc = X - X.mean(axis=0)
var = (Xc**2).mean(axis=0)
acf = np.zeros(61)
for k in range(61):
    if k==0: acf[k]=1.0
    else:
        num = (Xc[:-k]*Xc[k:]).mean(axis=0)
        acf[k] = np.mean(num/var)
print("  pooled per-cell ACF k=0..60:")
for i in range(0, 61, 4):
    print("   k=%2d r=%.4f" % (i, acf[i]))
neg = np.where(acf[1:]<0)[0]+1
print(f"  first negative lag: {neg[0] if len(neg) else None}; min acf={acf[1:].min():.4f} at k={acf[1:].argmin()+1}")

# fit AR(2) to ACF via Yule-Walker on r1,r2
r1, r2 = acf[1], acf[2]
phi2 = (r2 - r1**2)/(1 - r1**2)
phi1 = r1*(1-phi2)
print(f"  Yule-Walker AR(2) from pooled ACF: phi1={phi1:.4f} phi2={phi2:.4f}")
disc = phi1**2 + 4*phi2
if disc < 0:
    R = np.sqrt(-phi2); theta = np.arccos(phi1/(2*np.sqrt(-phi2)))
    print(f"  COMPLEX ROOTS: R={R:.4f}, theta={theta:.4f} rad -> period = {2*np.pi/theta:.1f} months ({2*np.pi/theta/12:.2f} yr)")
else:
    print(f"  real roots, disc={disc:.4f}")

# per-cell AR(2) fits on a sample (least squares on 3 consecutive)
print("\n  per-cell AR(2) OLS on detrended (sample 3000 cells):")
rng = np.random.default_rng(0)
idx = rng.choice(np.where(full)[0], 3000, replace=False)
Xs = Ac[:, idx]
Xs = Xs - np.nanmean(Xs, axis=0, keepdims=True)
# build lag matrices
rows=[]
for j in range(Xs.shape[1]):
    x = Xs[:, j]
    ok = np.isfinite(x)
    xx = x[ok]
    if len(xx) < 60: continue
    X0 = xx[2:]; X1 = xx[1:-1]; X2 = xx[:-2]
    A_ = np.column_stack([np.ones(len(X0)), X1, X2])
    coef, *_ = np.linalg.lstsq(A_, X0, rcond=None)
    rows.append(coef)
rows=np.array(rows)
print(f"    mean intercept={rows[:,0].mean():.4f}, phi1={rows[:,1].mean():.4f} (std {rows[:,1].std():.3f}), phi2={rows[:,2].mean():.4f} (std {rows[:,2].std():.3f})")
disc_c = rows[:,1]**2 + 4*rows[:,2]
frac_c = (disc_c<0).mean()
print(f"    fraction of cells with complex roots: {frac_c:.3f}")
cc = rows[disc_c<0]
if len(cc):
    R = np.sqrt(-cc[:,2]); th = np.arccos(np.clip(cc[:,1]/(2*R), -1, 1))
    per = 2*np.pi/th
    print(f"    among complex: median R={np.median(R):.3f}, median period={np.median(per):.1f} months, IQR period=({np.percentile(per,25):.1f},{np.percentile(per,75):.1f})")

# ---------- AR(1) vs AR(2) multi-step forecast comparison (oracle on train) ----------
print("\n  ORACLE multi-step forecast RMSE, AR(1) vs AR(2), detrended (train, pooled sample):")
# fit per-cell AR(1)/AR(2) on 2002-2010, evaluate multi-step forecasts on 2011-2015
t_split = np.searchsorted(yms, 201101)
tr_x = Xs[:t_split]; te_x = Xs[t_split:]
res = {}
for name in ['AR1','AR2']:
    errs = {h: [] for h in range(1,8)}
    for j in range(Xs.shape[1]):
        xtr = tr_x[:, j]; ok = np.isfinite(xtr); xtr = xtr[ok]
        xte = te_x[:, j]; okt = np.isfinite(xte)
        if len(xtr)<80 or okt.sum()<40: continue
        xte = xte[okt]
        if name=='AR1':
            p = np.array([np.corrcoef(xtr[:-1], xtr[1:])[0,1]]) if np.std(xtr)>1e-9 else np.array([0.9])
            mu = xtr.mean()
            def fc(hist, h): return mu + (p[0]**h)*(hist[-1]-mu)
        else:
            X0=xtr[2:]; X1=xtr[1:-1]; X2=xtr[:-2]
            A_=np.column_stack([np.ones(len(X0)),X1,X2])
            c,*_=np.linalg.lstsq(A_,X0,rcond=None)
            mu=c[0]; a1,a2=c[1],c[2]
            def fc(hist,h):
                x1,x2=hist[-1],hist[-2]; m=mu
                for _ in range(h):
                    x2,x1 = x1, mu+a1*(x1-m)+a2*(x2-m)
                return x1
        # evaluate: from each origin i in test, forecast h=1..7 (need 7 steps ahead available)
        n_te = len(xte)
        for i in range(0, n_te-8):
            for h in range(1,8):
                errs[h].append(fc(xte[:i+1], h) - xte[i+h])
    out = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in errs.items()}
    res[name]=out
    print(f"    {name}: " + " ".join(f"h{h}={v:.4f}" for h,v in out.items()))
gains = {h: res['AR1'][h]-res['AR2'][h] for h in res['AR1']}
print("    AR(2) gain:   " + " ".join(f"h{h}={v:+.4f}" for h,v in gains.items()))
np.save('/home/z/my-project/scripts/tb_acf.npy', acf)
