"""
V8 LOO-2 — test-era arbitration of masked-row components at TRUE anchor distances.
Decomposes what v4a got wrong on the LB: denoise(init vs obs) vs backward pass.

LOO pairs (fwd-only):   (i,k) consecutive anchors, predict field at k from i
LOO triples (fwd+bwd):  (i,k,l) — predict field at k from i (fwd) + l (bwd)
Fold-safe: D-hat/D-tilde exclude target anchor k only.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
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
_,_,Vt = np.linalg.svd(A_dtf, full_matrices=False)
V = Vt[:200].T.astype(np.float32)
def dn(field):
    out = field.copy(); x = field[full]; okx = np.isfinite(x)
    out[full] = np.where(okx, V@(V.T@np.where(okx,x,0.0)), x)
    return out

Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1)&np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# per-cell H/R on full train (era-scaled source)
clim = train.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
cov_est_tr = np.full(len(train), np.nan, dtype=np.float32)
Zok = np.isfinite(Z).all(axis=1)
cov_est_tr[Zok] = np.column_stack([Z[Zok], np.ones(Zok.sum())]) @ coef
tmp = pd.DataFrame({'cc':train['cc'].values,'t_abs':train['t_abs'].values,'cov':cov_est_tr,'tws':yv})
tmp['A'] = tmp['tws'] - mu_c[tmp['cc'].values] - ((tmp['t_abs'].values - tbar_c[tmp['cc'].values])*beta_c[tmp['cc'].values])
gmean = tmp.groupby('cc')['cov'].mean().reindex(range(n_cells)).fillna(0).values
tmp['Wd'] = tmp['cov'] - gmean[tmp['cc'].values]
tok = tmp.dropna()
ci = tok['cc'].values.astype(np.int64)
Wv = tok['Wd'].values.astype(np.float64); Av = tok['A'].values.astype(np.float64)
n_c = np.bincount(ci, minlength=n_cells)
sW = np.bincount(ci, weights=Wv, minlength=n_cells); sA = np.bincount(ci, weights=Av, minlength=n_cells)
sWW = np.bincount(ci, weights=Wv*Wv, minlength=n_cells); sAA = np.bincount(ci, weights=Av*Av, minlength=n_cells)
sWA = np.bincount(ci, weights=Wv*Av, minlength=n_cells)
with np.errstate(invalid='ignore', divide='ignore'):
    varW_c = sWW/n_c-(sW/n_c)**2; varA_c = sAA/n_c-(sA/n_c)**2; covWA_c = sWA/n_c-(sW/n_c)*(sA/n_c)
    Hc_raw = covWA_c/(LAM_F*varA_c); Rc_raw = varW_c - Hc_raw**2*(LAM_F*varA_c)
valid_c = (n_c>=100)&(varA_c>1e-6)&(varW_c>1e-6)&(Rc_raw>0)
Hg_tr = float(np.nanmean(np.where(valid_c,Hc_raw,np.nan))); Rg_tr = float(np.nanmean(np.where(valid_c,Rc_raw,np.nan)))

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'],test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values

Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta==m)&okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
W = {int(m): cov_field[m]-S for m in all_m}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac<0.01].index)
AF = {}
for a in anchors:
    sel = (ta==a)&(~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

PAIRS = [(anchors[i],anchors[i+1]) for i in range(len(anchors)-1)]
PAIRS = [(i,k) for i,k in PAIRS if k-i<=8]
TRIPLES = [(anchors[i],anchors[i+1],anchors[i+2]) for i in range(len(anchors)-2)]
TRIPLES = [(i,k,l) for i,k,l in TRIPLES if k-i<=8 and l-k<=20]
print(f"PAIRS: {PAIRS}\nTRIPLES: {TRIPLES}")

def fit_weights(a_list):
    Xs, ys = [], []
    for a in a_list:
        dloo = np.nanmean(np.array([AF[b] for b in a_list if b!=a]), axis=0)
        tx = trendex(a)
        ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a])&np.isfinite(tx)
        Xs.append(np.column_stack([dloo[ok],S[ok],tx[ok]])); ys.append(AF[a][ok])
    Xw = np.vstack(Xs); yw = np.concatenate(ys)
    w_ = np.linalg.solve(Xw.T@Xw + 1e-3*np.eye(3), Xw.T@yw)
    return map(float, w_)

def run(phi=0.74, shrink=1.0, dn_init=False, dn_obs=False, use_bwd=False, percell=False):
    errs = []
    for triple in (TRIPLES if use_bwd else PAIRS):
        if use_bwd: i,k,l = triple
        else: i,k = triple
        avail = [a for a in anchors if a!=k]
        w1,w2,w3 = fit_weights(avail)
        Dhat = np.nanmean(np.array([AF[a] for a in avail]), axis=0)
        def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)
        # calibrate global H/R (per-cell scaled if requested)
        cs, zs, vfs = [], [], []
        for a in avail:
            dj = Dtil(a); w_ = W[a]
            ok = np.isfinite(w_)&np.isfinite(AF[a])&np.isfinite(dj)
            cs.append(np.cov(w_[ok],(AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok])); vfs.append(np.nanvar(AF[a]-dj))
        c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
        H = c_/(LAM_F*var_f); R = max(varz-c_*c_/(LAM_F*var_f),1e-4)
        if percell:
            rH = np.clip(Hc_raw/Hg_tr, 0.3, 3.0); rR = np.clip(Rc_raw/Rg_tr, 0.3, 3.0)
            Hc = np.where(valid_c, H*(0.5+0.5*rH), H).astype(np.float32)
            Rc = np.where(valid_c, np.maximum(R*(0.5+0.5*rR),1e-4), R).astype(np.float32)
        else:
            Hc, Rc = np.full(n_cells,H,np.float32), np.full(n_cells,R,np.float32)
        q = LAM_F*var_f*(1-phi**2); P0 = LAM_F*(1-LAM_F)*var_f

        def obs_update(x, P, m):
            if m not in W: return x, P
            w_ = dn(W[m]) if dn_obs else W[m]
            okw = np.isfinite(w_)
            Kg = np.where(okw, P*Hc/(Hc*Hc*P+Rc), 0).astype(np.float32)
            x = np.where(okw, x+Kg*(np.nan_to_num(w_)-Hc*x), x)
            P = np.where(okw, (1-Kg*Hc)*P, P)
            return x, P

        f0 = AF[i]-Dtil(i)
        f0 = dn(f0) if dn_init else f0
        x = np.where(np.isfinite(AF[i]), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        for m in range(i+1, k+1):
            x = phi*x; P = phi**2*P + q
            x, P = obs_update(x, P, m)
        xf, Pf = x, P
        if use_bwd:
            b0 = AF[l]-Dtil(l)
            b0 = dn(b0) if dn_init else b0
            xb = np.where(np.isfinite(AF[l]), LAM_F*np.nan_to_num(b0), 0.0).astype(np.float32)
            Pb = np.where(np.isfinite(AF[l]), P0, var_f).astype(np.float32)
            for m in range(l-1, k-1, -1):
                xb = phi*xb; Pb = phi**2*Pb + q
                xb, Pb = obs_update(xb, Pb, m)
            wf = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wf*xf + (1-wf)*xb
        else:
            x = xf
        pred = Dtil(k) + shrink*x
        ok = np.isfinite(pred)&np.isfinite(AF[k])
        errs.append(float(np.sqrt(np.mean((pred[ok]-AF[k][ok])**2))))
    return float(np.mean(errs)), errs

print("\n--- decomposition at true test distances (phi=0.74) ---")
base, e = run();                     print(f"1. v2b core (no dn, no bwd, shrink=1):        {base:.4f}  {['%.4f'%v for v in e]}")
r_, e = run(shrink=0.85);            print(f"2. + shrink 0.85:                              {r_:.4f}")
r_, e = run(dn_init=True);           print(f"3. + init-denoise only:                        {r_:.4f}")
r_, e = run(dn_obs=True);            print(f"4. + obs-denoise only:                         {r_:.4f}")
r_, e = run(dn_init=True, dn_obs=True); print(f"5. + both denoise (=v4 core):                  {r_:.4f}")
r_, e = run(use_bwd=True);           print(f"6. + bwd (triples, no dn):                     {r_:.4f}")
r_, e = run(use_bwd=True, dn_init=True, dn_obs=True); print(f"7. + bwd + both dn (v4a-style, triples):       {r_:.4f}")
r_, e = run(use_bwd=True, dn_init=True); print(f"8. + bwd + init-dn only:                       {r_:.4f}")

print("\n--- shrink x phi on core ---")
for phi in [0.70, 0.74, 0.80]:
    row = [f"shrink {s}: {run(phi=phi, shrink=s)[0]:.4f}" for s in [1.0, 0.85]]
    print(f"phi={phi}: " + " | ".join(row))

print("\n--- best combos ---")
r_, e = run(shrink=0.85, dn_init=True, use_bwd=True);  print(f"shrink.85 + init-dn + bwd:    {r_:.4f}")
r_, e = run(shrink=0.85, dn_init=True, dn_obs=True, use_bwd=True); print(f"shrink.85 + both-dn + bwd:    {r_:.4f}")
r_, e = run(shrink=0.85, percell=True);               print(f"shrink.85 + percell H/R:      {r_:.4f}")
r_, e = run(shrink=0.85, percell=True, dn_init=True, use_bwd=True)
print(f"shrink.85 + percell + init-dn + bwd: {r_:.4f}")

print("\n--- round 2: finalize masked config ---")
r_, e = run(dn_obs=True); print(f"obs-dn:                    {r_:.4f}  {['%.4f'%v for v in e]}")
r_, e = run(dn_obs=True, percell=True); print(f"obs-dn + percell:          {r_:.4f}  {['%.4f'%v for v in e]}")
r_, e = run(dn_obs=True, percell=True, dn_init=True); print(f"obs-dn + percell + init-dn:{r_:.4f}  {['%.4f'%v for v in e]}")
for s in [0.90, 0.95]:
    r_, e = run(dn_obs=True, shrink=s); print(f"obs-dn + shrink {s}:       {r_:.4f}")
r_, e = run(dn_obs=True, percell=True, shrink=0.90); print(f"obs-dn + percell + shrink .90: {r_:.4f}")
for phi in [0.70, 0.78]:
    r_, e = run(dn_obs=True, phi=phi); print(f"obs-dn phi={phi}:          {r_:.4f}")
print("\nDONE loo2.")
