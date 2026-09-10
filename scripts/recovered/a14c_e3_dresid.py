"""
Agent 14-c E3: D_resid structure + PSEUDO-ERA CALIBRATION (the decisive null test).

E3a: D_resid(c) = A_bar(c) - mean_j trendex_j(c)  (deviation from pure trend continuation)
     - spatial smoothness (neighbor corr), rank (PC projection onto train bases),
       correlates (mu_c, |lat|, |beta|, train anomaly var, S), stability across anchor subsets.
E3b: PSEUDO-ERA calibration inside train: era A (2002-2012) = pseudo-train,
     era B (2013-2015) = pseudo-test with 6 pseudo-anchors mimicking the real anchor
     spacing (gaps 4,5,6,19,4 compressed by 31/38). Truth: NO regime change inside train
     (or a natural RW decorrelation - that's what we measure). Gives the NULL distribution
     for r_hat, alpha_hat, independent-slope variance, off-field variance, PC-proj var.
     => tells us whether the test-era numbers are genuine regime change or estimator artifacts.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
cache = np.load(f'{DATA}/a14c_cache.npz')
mu_c = cache['mu_c']; beta_c = cache['beta_c']; tbar_c = cache['tbar_c']
V_dt = cache['V_dt']; V_an = cache['V_an']; full = cache['full']
A = cache['A']; anchors = list(cache['anchors']); t_anc = cache['t_anc']
b_sl = cache['b_sl']; a_int = cache['a_int']; slope_noise2 = float(cache['slope_noise2'])
s_h = float(cache['s_h']); attn = float(cache['attn']); alpha_off = float(cache['alpha_off'])
n_cells = len(mu_c); full_idx = np.where(full)[0]; n_full = int(full.sum())
t_mid = float(cache['t_mid'])
def corr2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100: return np.nan
    return float(np.corrcoef(x[m], y[m])[0,1])
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c)

# rebuild S (static cov field) for correlates
print("Rebuilding S from test covariates...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['t_abs'] = (test['time'].dt.year*100 + test['time'].dt.month)*0 + (test['time'].dt.year*12 + test['time'].dt.month - 1)
Zt_raw = test[COVS].values.astype('float32'); okt = np.isfinite(Zt_raw).all(axis=1)
cov_est = np.full(len(test), np.nan, dtype=np.float32)
cov_est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef
cc_t = test['cc'].values; ta = test['t_abs'].values
cov_field = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est[selm]
    cov_field[int(m)] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)

# neighbor map for smoothness
grid = {}
for cc, la, lo in zip(codes['cc'], codes['lat'], codes['lon']):
    grid[(int(round(float(la)*10)), int(round(float(lo)*10)))] = int(cc)
def neighbor_corr(fld):
    xs, ys, xn, yn = [], [], [], []
    la_arr = codes['lat'].values; lo_arr = codes['lon'].values; cc_arr = codes['cc'].values
    for la, lo, cc in zip(la_arr, lo_arr, cc_arr):
        v = fld[cc]
        if not np.isfinite(v): continue
        e = grid.get((int(round(la*10)), int(round((lo+1)*10))))
        n_ = grid.get((int(round((la+1)*10)), int(round(lo*10))))
        if e is not None and np.isfinite(fld[e]): xs.append(v); ys.append(fld[e])
        if n_ is not None and np.isfinite(fld[n_]): xn.append(v); yn.append(fld[n_])
    return corr2(np.array(xs), np.array(ys)), corr2(np.array(xn), np.array(yn))

def proj_var(fld, Vb, K):
    x = fld[full_idx]; ok = np.isfinite(x)
    VK = Vb[:, :K]
    proj = VK @ (VK.T @ np.where(ok, x, 0.0))
    return float(np.nanvar(proj))

# ================= E3a: D_resid structure =================
print("\n================ E3a: D_resid STRUCTURE ================")
Abar = np.nanmean(A, axis=0)
tx_mean = np.nanmean(np.array([trendex(a) for a in anchors]), axis=0)
D_resid = Abar - tx_mean
lat_c = codes['lat'].values; lon_c = codes['lon'].values
train_anom_var = np.full(n_cells, np.nan)
# train anomaly variance (detrended) from cache? recompute quickly from beta_noise2? use F
yms = np.sort(train['ym'].unique()); ym_to_i = {int(v):i for i,v in enumerate(yms)}
T = len(yms)
F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
F64 = F.astype(np.float64)
TD = (np.array([(int(v)//100)*12+(int(v)%100)-1 for v in yms], dtype=np.float64)[:,None] - tbar_c[None,:])
train_dt_var = np.nanvar(F64 - mu_c[None,:] - TD*beta_c[None,:], axis=0)

print(f"D_resid: std={np.nanstd(D_resid):.3f}")
ew, ns = neighbor_corr(D_resid)
print(f"neighbor corr: E-W {ew:.3f}  N-S {ns:.3f}  (for reference D-hat ~0.99/0.97 style fields)")
for K in [5, 10, 20, 50]:
    print(f"  PC-proj var (anomaly basis) K={K}: {proj_var(D_resid, V_an, K):.4f} ({proj_var(D_resid,V_an,K)/np.nanvar(D_resid)*100:.0f}% of var)")
print("\ncorrelates of D_resid:")
for nm, fld in [("mu_c", mu_c), ("|lat|", np.abs(lat_c)), ("|beta_c|", np.abs(beta_c)),
                ("train anomaly var", train_dt_var), ("S", S), ("trendex(t_mid)", trendex(t_mid)),
                ("beta_c", beta_c)]:
    print(f"  corr(D_resid, {nm:>16}) = {corr2(D_resid, fld):+.3f}")
# stability across anchor subsets
D_res_13 = np.nanmean(A[[0,1,2]], axis=0) - np.nanmean([trendex(anchors[i]) for i in [0,1,2]], axis=0)
D_res_46 = np.nanmean(A[[3,4,5]], axis=0) - np.nanmean([trendex(anchors[i]) for i in [3,4,5]], axis=0)
print(f"\nstability: corr(D_resid[anchors 1-3], D_resid[anchors 4-6]) = {corr2(D_res_13, D_res_46):.3f}")
print(f"           std(diff) = {np.nanstd(D_res_13 - D_res_46):.3f}")

# raw D-hat correlates for reference
print(f"\nreference: corr(Abar, mu_c) = {corr2(Abar, mu_c):+.3f}, corr(Abar, S) = {corr2(Abar, S):+.3f}")
# regression of D_resid on mu_c
ok = np.isfinite(D_resid) & np.isfinite(mu_c)
g = np.polyfit(mu_c[ok], D_resid[ok], 1)
r2 = corr2(D_resid, mu_c)**2
print(f"D_resid = {g[0]:.3f}*mu_c + {g[1]:.3f}  (R^2={r2:.3f}); residual-after-mu std = {np.std(D_resid[ok]-g[0]*mu_c[ok]-g[1]):.3f}")

# ================= E3b: PSEUDO-ERA CALIBRATION =================
print("\n================ E3b: PSEUDO-ERA CALIBRATION (null test) ================")
t_abs_yms = np.array([(int(v)//100)*12+(int(v)%100)-1 for v in yms], dtype=np.float64)
have = set(t_abs_yms.astype(int).tolist())

def pseudo_experiment(eraA_end_year, tB_start, offsets, n_draw=4):
    """eraA: 2002..eraA_end_year; pseudo-anchors near tB_start+offsets (snapped to available months)."""
    ymA = (train['t_abs'] <= eraA_end_year*12+11)
    F_A = np.full((T, n_cells), np.nan, dtype=np.float32)
    F_A[train['ym'].map(ym_to_i).values, train['cc'].values] = np.where(ymA, train['TWS_t'].values, np.nan)
    muA = np.nanmean(F_A, axis=0)
    F64A = F_A.astype(np.float64); okA = np.isfinite(F64A)
    tmat = np.where(okA, t_abs_yms[:,None], np.nan)
    tbarA = np.nanmean(tmat, axis=0); tdA = tmat - tbarA[None,:]
    sxxA = np.nansum(tdA*tdA, axis=0)
    betaA = np.where(sxxA>100, np.nansum(tdA*F64A,axis=0)/np.where(sxxA>0,sxxA,1), 0.0).astype(np.float32)
    resA = F64A - muA[None,:] - tdA*betaA[None,:]
    dofA = np.maximum(okA.sum(axis=0)-2, 1)
    sig2resA = np.nansum(np.where(okA, resA**2, np.nan), axis=0)/dofA
    betaA_noise2 = np.where(sxxA>100, sig2resA/np.where(sxxA>0,sxxA,1), np.nan)
    out = []
    for d in range(n_draw):
        rng = np.random.default_rng(1000+d)
        tgt = [tB_start + int(o) + int(rng.integers(-1, 2)) for o in offsets]
        ps = []
        for p in tgt:
            best, bd = None, 99
            for q in range(p-2, p+3):
                if q in have and abs(q-p) < bd:
                    best, bd = q, abs(q-p)
            if best is not None and best not in ps: ps.append(best)
        if len(ps) < 6: continue
        P = np.full((6, n_cells), np.nan, dtype=np.float32)
        for i, p in enumerate(ps[:6]):
            ii = int(np.where(t_abs_yms == p)[0][0])
            P[i] = F[ii] - muA
        tp = np.array(ps, dtype=np.float64); tm = tp.mean()
        X = np.column_stack([np.ones(6), tp - tm]); XtXi = np.linalg.inv(X.T@X)
        okc = np.isfinite(P).all(axis=0)
        co = np.linalg.solve(X.T@X, X.T@P[:, okc])
        aP = np.full(n_cells, np.nan); bP = np.full(n_cells, np.nan)
        aP[okc] = co[0]; bP[okc] = co[1]
        res = P[:, okc] - (X@co)
        s2c = (res**2).sum(axis=0)/4
        s2anc = float(np.median(s2c)); Sxxp = float(((tp-tm)**2).sum())
        slnoise2 = s2anc/Sxxp
        okm = okc & np.isfinite(betaA) & (betaA_noise2>0)
        x = betaA[okm].astype(np.float64); y = bP[okm].astype(np.float64)
        # huber
        s_, c_ = np.polyfit(x, y, 1)
        for _ in range(8):
            e = y - (s_*x+c_); sc = 1.4826*np.median(np.abs(e-np.median(e)))+1e-12
            u = np.abs(e)/(2.0*sc); w = np.where(u<=1, 1.0, 1.0/np.maximum(u,1e-12))
            Xw = np.column_stack([x, np.ones(len(x))]); sw = np.sqrt(w)
            sol = np.linalg.solve((Xw*sw[:,None]).T@(Xw*sw[:,None]), (Xw*sw[:,None]).T@(y*sw))
            s_, c_ = sol
        attnA = 1.0/(1.0+float(np.nanmean(betaA_noise2[okm]))/np.var(x))
        var_b = float(np.nanvar(bP)); var_btrue = max(var_b - slnoise2, 0)
        cov_bb = float(np.cov(bP[okm], betaA[okm])[0,1])
        indep_var = max(var_b - slnoise2 - cov_bb**2/np.var(betaA[okm]), 0)
        # intercept analysis
        txm = ((tm - tbarA)*betaA)
        ok2 = okc & np.isfinite(txm)
        al_, ac_ = np.polyfit(txm[ok2].astype(np.float64), aP[ok2].astype(np.float64), 1)
        intnoise2 = s2anc*float(XtXi[0,0])
        var_a = float(np.nanvar(aP)); var_atrue = max(var_a-intnoise2, 0)
        offP = aP - al_*txm
        offP_var_true = max(var_atrue - (al_**2)*np.nanvar(txm), 0)
        # PC projections of residual slope + off fields
        sres = bP - s_*betaA
        pj_slope = proj_var(sres, V_an, 10); pj_slope50 = proj_var(sres, V_an, 50)
        pj_off = proj_var(np.where(np.isfinite(offP), offP, np.nan), V_an, 10)
        out.append(dict(r=s_/attnA, r_raw=s_, alpha=al_, var_btrue=var_btrue, indep_var=indep_var,
                        slnoise2=slnoise2, pj_slope10=pj_slope, pj_slope50=pj_slope50,
                        pj_off10=pj_off, offP_var_true=offP_var_true, var_atrue=var_atrue,
                        cov_bb=cov_bb, var_beta=float(np.var(betaA[okm]))))
    return out

# real anchor offsets: 0,4,9,15,34,38 (span 38); compressed x0.82 for 31-month windows
offs_real = [0, 4, 9, 15, 34, 38]
offs_comp = [0, 3, 7, 12, 28, 31]
res_all = []
for eraA_end, tB_start, offs, tag in [
        (2006, 24084, offs_real, "A=02-06 B=07-10 (real spacing)"),
        (2009, 24120, offs_real, "A=02-09 B=10-13 (real spacing)"),
        (2012, 24156, offs_comp, "A=02-12 B=13-15 (compressed)")]:
    out = pseudo_experiment(eraA_end, tB_start, offs, n_draw=6)
    if not out:
        print(f"  {tag}: skipped (missing months)"); continue
    r = np.array([o['r'] for o in out]); al = np.array([o['alpha'] for o in out])
    iv = np.array([o['indep_var'] for o in out]); pj10 = np.array([o['pj_slope10'] for o in out])
    pj50 = np.array([o['pj_slope50'] for o in out]); pjo = np.array([o['pj_off10'] for o in out])
    ofv = np.array([o['offP_var_true'] for o in out]); vat = np.array([o['var_atrue'] for o in out])
    print(f"  {tag}: r_hat={r.mean():.3f}±{r.std():.3f}  alpha_hat={al.mean():.3f}±{al.std():.3f}  "
          f"indep_slope_var={iv.mean():.2e}  projSlope(K10)={pj10.mean():.2e} (K50={pj50.mean():.2e})  "
          f"off_var_true={ofv.mean():.3f} projOff(K10)={pjo.mean():.3f} var_a_true={vat.mean():.3f}")
    res_all += out

print("\n--- TEST-ERA measurements (from E2) for comparison ---")
cov_bb_test = float(np.cov(b_sl[np.isfinite(b_sl)&np.isfinite(beta_c)], beta_c[np.isfinite(b_sl)&np.isfinite(beta_c)])[0,1])
indep_test = max(float(np.nanvar(b_sl)) - slope_noise2 - cov_bb_test**2/np.nanvar(beta_c), 0)
sres_test = b_sl - s_h*beta_c
print(f"  r_hat={s_h/attn:.3f}  alpha_hat={alpha_off:.3f}  indep_slope_var={indep_test:.2e}")
print(f"  projSlope(K10)={proj_var(sres_test, V_an, 10):.2e}  projSlope(K50)={proj_var(sres_test, V_an, 50):.2e}")
offT = a_int - alpha_off*trendex(t_mid)
print(f"  off_var_true={max(float(np.nanvar(a_int))-float(cache['sig2_anc']*cache['XtX_inv'][0,0])-(alpha_off**2)*np.nanvar(trendex(t_mid)),0):.3f}  projOff(K10)={proj_var(offT, V_an, 10):.3f}")
print(f"  var_a_true={max(float(np.nanvar(a_int))-float(cache['sig2_anc']*cache['XtX_inv'][0,0]),0):.3f}")

np.savez_compressed(f'{DATA}/a14c_cache2.npz', S=S, D_resid=D_resid, Abar=Abar,
                    train_dt_var=train_dt_var, cov_field_mean=S)
print("\ncache2 saved.")
