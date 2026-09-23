"""
AUDIT 15-c part 7: LOW-COV-QUALITY GATING A/B on the true-distance anchor-LOO harness.
Baseline = v8a masked config (obs-dn + per-cell H/R) = 0.7369 per v8_loo2.
Variants: widen rH clip floor, hard-gate Hc to 0 for low |r_cell| cells.
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
    r_cell = covWA_c/np.sqrt(varW_c*varA_c)
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
print(f"PAIRS: {PAIRS}")

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

def run(phi=0.74, shrink=1.0, dn_obs=True, percell=True, rH_lo=0.3, rH_hi=3.0, gate=None, mix=0.5):
    """gate: None or threshold t -> Hc=0 where |r_cell|<t"""
    errs = []
    for (i,k) in PAIRS:
        avail = [a for a in anchors if a!=k]
        w1,w2,w3 = fit_weights(avail)
        Dhat = np.nanmean(np.array([AF[a] for a in avail]), axis=0)
        def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)
        cs, zs, vfs = [], [], []
        for a in avail:
            dj = Dtil(a); w_ = W[a]
            ok = np.isfinite(w_)&np.isfinite(AF[a])&np.isfinite(dj)
            cs.append(np.cov(w_[ok],(AF[a]-dj)[ok])[0,1]); zs.append(np.var(w_[ok])); vfs.append(np.nanvar(AF[a]-dj))
        c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
        H = c_/(LAM_F*var_f); R = max(varz-c_*c_/(LAM_F*var_f),1e-4)
        if percell:
            rH = np.clip(Hc_raw/Hg_tr, rH_lo, rH_hi); rR = np.clip(Rc_raw/Rg_tr, 0.3, 3.0)
            Hc = np.where(valid_c, H*(mix+(1-mix)*rH), H).astype(np.float32)
            Rc = np.where(valid_c, np.maximum(R*(mix+(1-mix)*rR),1e-4), R).astype(np.float32)
            if gate is not None:
                Hc = np.where(np.abs(r_cell) < gate, 0.0, Hc).astype(np.float32)
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
        x = np.where(np.isfinite(AF[i]), LAM_F*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        for m in range(i+1, k+1):
            x = phi*x; P = phi**2*P + q
            x, P = obs_update(x, P, m)
        pred = Dtil(k) + shrink*x
        ok = np.isfinite(pred)&np.isfinite(AF[k])
        errs.append(float(np.sqrt(np.mean((pred[ok]-AF[k][ok])**2))))
    return float(np.mean(errs)), errs

print("\n--- gating A/B on true-distance LOO (v8a masked config family) ---")
r, e = run();                                print(f"baseline obs-dn + percell (v8a cfg):  {r:.4f}  {['%.4f'%v for v in e]}")
r, e = run(percell=False);                   print(f"obs-dn only (v8c cfg):                 {r:.4f}  {['%.4f'%v for v in e]}")
r, e = run(rH_lo=0.1);                       print(f"rH clip floor 0.3->0.1:                {r:.4f}  {['%.4f'%v for v in e]}")
r, e = run(rH_lo=0.0);                       print(f"rH clip floor 0.3->0.0:                {r:.4f}  {['%.4f'%v for v in e]}")
r, e = run(gate=0.2);                        print(f"hard gate |r|<0.2 -> H=0 (n={int((np.abs(r_cell)<0.2).sum()):,} cells): {r:.4f}")
r, e = run(gate=0.3);                        print(f"hard gate |r|<0.3 -> H=0 (n={int((np.abs(r_cell)<0.3).sum()):,} cells): {r:.4f}")
r, e = run(rH_hi=5.0);                       print(f"rH clip ceiling 3.0->5.0:              {r:.4f}")
r, e = run(mix=0.25);                        print(f"mix 50/50 -> 25/75 (more per-cell):    {r:.4f}")
r, e = run(rH_lo=0.1, mix=0.25);             print(f"floor 0.1 + mix 25/75:                 {r:.4f}")
print("\nDONE part 7")
