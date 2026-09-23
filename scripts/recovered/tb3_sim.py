"""2-b step 3: (i) is the negative long-lag ACF a detrend artifact? simulate generator.
(ii) oracle AR(1) vs AR(2) multi-step on detrended (fixed threshold bug)."""
import numpy as np
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
A_dt, yms = d['A_dt'], d['yms']
T, n_cells = A_dt.shape
months = (yms % 100).astype(int)

full = ~np.isnan(A_dt).any(axis=0)
X = A_dt[:, full]
Xc = X - X.mean(axis=0)

def pooled_acf(Xc, kmax=60):
    var = (Xc**2).mean(axis=0)
    return np.array([1.0]+[np.mean((Xc[:-k]*Xc[k:]).mean(axis=0)/var) for k in range(1,kmax+1)])

acf_emp = pooled_acf(Xc)
print("empirical detrended ACF r1..r12:", np.round(acf_emp[:13],4))

# ---------- (i) SIMULATION: mu + RW slow + AR(1) fast + white noise ----------
# params: fast phi=0.80, lam=0.84 (signal fraction), slow as RW with step std set so
# linear detrending leaves similar long-lag negative plateau.
rng = np.random.default_rng(42)
T, N = Xc.shape
def simulate(phi, lam, rw_step, ncell=3000, T=138):
    sig_var_total = 1.0
    # decompose: fast var vf, slow var vs, noise vn, with vf+vs+vn = 1, lam=(vf+vs)/1
    # choose slow to explain ~25% of anomaly variance (measured trend share)
    vs = 0.25*lam; vf = lam - vs; vn = 1-lam
    phi_v = 0.5  # placeholder, set outside
    S = np.zeros((T, ncell)); s = rng.normal(0, np.sqrt(vs), ncell)
    for t in range(T):
        s = s + rng.normal(0, rw_step, ncell)
        S[t] = s
    # rescale slow to have var vs
    S = S/ S.std()*np.sqrt(vs)
    F = np.zeros((T, ncell)); f = rng.normal(0, np.sqrt(vf), ncell)
    for t in range(T):
        f = phi*f + np.sqrt(vf*(1-phi**2))*rng.normal(size=ncell)
        F[t] = f
    E = rng.normal(0, np.sqrt(vn), (T, ncell))
    return S + F + E

# linear-detrend each simulated series, pooled ACF
def detrend_acf(Y):
    t = np.arange(Y.shape[0])
    tb = t.mean(); td = t - tb
    beta = (td[:,None]*(Y-Y.mean(0))).sum(0)/(td**2).sum()
    R = Y - Y.mean(0) - td[:,None]*beta[None,:]
    return pooled_acf(R - R.mean(0))

print("\nSIMULATION A: RW slow (step so detrended slow leaves plateau) + AR(1) fast + noise")
for rw_mult, phi in [(0.05, 0.80), (0.10, 0.80), (0.05, 0.70), (0.08, 0.65)]:
    Y = simulate(phi, 0.84, rw_mult, ncell=3000)
    a = detrend_acf(Y)
    print(f"  rw={rw_mult} phi={phi}: r1={a[1]:.3f} r4={a[4]:.3f} r8={a[8]:.3f} r12={a[12]:.3f} r24={a[24]:.3f} r34={a[34]:.3f} r48={a[48]:.3f}")

print("\nSIMULATION B: pure AR(2) complex roots (damped oscillation), no slow")
for phi1, phi2 in [(1.0, -0.35), (0.9, -0.3), (1.1, -0.4)]:
    Y = np.zeros((138, 3000))
    f1 = rng.normal(0,1,3000); f2 = rng.normal(0,1,3000)
    v = 1.0
    for t in range(138):
        f0 = phi1*f1 + phi2*f2 + np.sqrt(max(v*(1-0),1e-9))*0 + rng.normal(0,0.5,3000)
        # keep var ~1
        f0 = f0/f0.std()
        Y[t] = f0; f2, f1 = f1, f0
    Y = Y*np.sqrt(0.84) + rng.normal(0, np.sqrt(0.16), Y.shape)
    a = detrend_acf(Y)
    print(f"  phi1={phi1} phi2={phi2}: r1={a[1]:.3f} r4={a[4]:.3f} r8={a[8]:.3f} r12={a[12]:.3f} r24={a[24]:.3f} r34={a[34]:.3f} r48={a[48]:.3f}")

# ---------- (ii) oracle AR1 vs AR2 multi-step (fixed) ----------
print("\nORACLE multi-step forecast on detrended train (fit 2002-2010, eval 2011-2015):")
rng = np.random.default_rng(0)
pos = rng.choice(X.shape[1], 2000, replace=False)
Xs = X[:, pos] - X[:, pos].mean(0)
t_split = int(np.searchsorted(yms, 201101))
res = {}
for name in ['AR1','AR2']:
    errs = {h: [] for h in range(1,8)}
    for j in range(Xs.shape[1]):
        xtr = Xs[:t_split, j]; xte = Xs[t_split:, j]
        if np.std(xtr)<1e-9: continue
        if name=='AR1':
            p = np.corrcoef(xtr[:-1], xtr[1:])[0,1]
            mu = xtr.mean()
            def fc(hist, h, p=p, mu=mu): return mu + (p**h)*(hist[-1]-mu)
        else:
            X0=xtr[2:]; X1=xtr[1:-1]; X2=xtr[:-2]
            A_=np.column_stack([np.ones(len(X0)),X1,X2])
            c,*_=np.linalg.lstsq(A_,X0,rcond=None)
            mu=c[0]; a1,a2=c[1],c[2]
            def fc(hist,h,mu=mu,a1=a1,a2=a2):
                x1,x2=hist[-1],hist[-2]
                for _ in range(h):
                    x2,x1 = x1, mu+a1*(x1-mu)+a2*(x2-mu)
                return x1
        n_te = len(xte)
        for i in range(2, n_te-8):
            for h in range(1,8):
                errs[h].append(fc(xte[:i+1], h) - xte[i+h])
    out = {h: np.sqrt(np.mean(np.array(e)**2)) for h,e in errs.items()}
    res[name]=out
    print(f"  {name}: " + " ".join(f"h{h}={v:.4f}" for h,v in out.items()))
print("  AR(2) gain: " + " ".join(f"h{h}={res['AR1'][h]-res['AR2'][h]:+.4f}" for h in res['AR1']))

# persistence baseline (phi from full train) and climatology
print(f"  detrended anomaly std (eval window): {np.nanstd(Xs[t_split:]):.4f}")
