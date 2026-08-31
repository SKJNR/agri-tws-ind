"""V11 EXPERIMENT — era-weighted Dhat + cov-slow student (the D-lane).

Context (from split_decode.py):
  Test = 18 sparse months; 6 fully-unmasked anchors (201509, 201601, 201606,
  201612, 201807, 201811); 12 fully-masked months. Public split = first 7
  months (201509..201608, 38.9% of rows). Private = the rest incl. the 2017
  long-horizon block (h=1..6) with the largest D offsets.
  Current D-tilde uses a STATIC Dhat (mean of all 6 anchors) for every month
  -> stale at both era ends; std(Dhat_late - Dhat_early) = 0.759.

Measurements (leave-one-anchor-out on the 6 test anchors):
  A0. static baseline (expect D-tilde LOO ~0.818)
  A.  era-weighted Dhat: exp-decay tau sweep, recent-K, linear interpolation
      -> LOO RMSE overall AND on the 2 late anchors (private-era proxy)
  B.  cov-slow student: ridge predicting (AF[a] - Dtil(a)) from time-averaged
      raw covariate deviations around month a (D-residual lane)

Ship rule: any scheme beating static D-tilde LOO by >= 0.010 goes into v11.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

# ---------------- load train ----------------
print("loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
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

# ---------------- load test ----------------
print("loading test...", flush=True)
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked']+COVS)
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
print(f"n_cells={n_cells}  test months={len(test_months)}  k0 rows={int((~msk).sum())}")

# raw covariate deviation fields per test month (n_cells, 5)
Zdev = {}
for m in test_months:
    selm = ta == m
    f = np.full((n_cells, len(COVS)), np.nan, dtype=np.float32)
    for j, c in enumerate(COVS):
        v = np.full(n_cells, np.nan, dtype=np.float32); v[cc_t[selm]] = test[c].values[selm]
        f[:, j] = v - clim[:, j]
    Zdev[int(m)] = f

# anchors
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
late_anchors = [a for a in anchors if a >= 24210]   # 201807, 201811
print(f"anchors: {anchors}  late(private-era): {late_anchors}")

# combined cov field for S (v6-verbatim definition)
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

# ---------------- LOO machinery ----------------
def wmean_stack(stack, ws):
    """weighted mean over axis 0 with NaN handling; stack (K, n_cells)"""
    ok = np.isfinite(stack)
    wmat = np.where(ok, ws[:, None], 0.0)
    num = np.nansum(np.where(ok, stack, 0.0)*wmat, axis=0)
    den = wmat.sum(axis=0)
    return np.where(den > 1e-9, num/np.maximum(den, 1e-9), np.nan)

def loo_eval(dhat_fn, name, w=None):
    """LOO over anchors. Returns (all_rmse, late_rmse, rows, w)."""
    preds = {}
    for a in anchors:
        preds[a] = dhat_fn(a, excl=a)
    if w is None:
        Xs, ys = [], []
        for a in anchors:
            d_, tx = preds[a], trendex(a)
            ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
            Xs.append(np.column_stack([d_[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
        XA, yA = np.vstack(Xs), np.concatenate(ys)
        w = np.linalg.solve(XA.T@XA + np.array([1e-3,1e-3,1e-3]), XA.T@yA)
    rows = {}
    for a in anchors:
        d_, tx = preds[a], trendex(a)
        ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
        dt = w[0]*d_ + w[1]*S + w[2]*tx
        rows[a] = (np.sqrt(np.nanmean((dt[ok]-AF[a][ok])**2)), ok, dt)
    allr = float(np.sqrt(np.mean([rows[a][0]**2 for a in anchors])))
    later = float(np.sqrt(np.mean([rows[a][0]**2 for a in late_anchors])))
    print(f"  {name:<34} LOO all={allr:.4f}  late={later:.4f}  w=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f})")
    return allr, later, rows, w

print("\n=== A0. static baseline (current pipeline) ===")
def static(t, excl):
    others = [b for b in anchors if b != excl]
    d = np.nanmean(np.array([AF[b] for b in others], dtype=np.float64), axis=0)
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)
base_all, base_late, base_rows, base_w = loo_eval(static, "static Dhat mean")
# Dhat-only baseline for reference
res = []
for a in anchors:
    others = [b for b in anchors if b != excl] if False else [b for b in anchors if b != a]
    d = np.nanmean(np.array([AF[b] for b in others], dtype=np.float64), axis=0)
    ok = np.isfinite(d) & np.isfinite(AF[a])
    res.append(np.sqrt(np.nanmean((d[ok]-AF[a][ok])**2)))
print(f"  (Dhat-only LOO = {np.sqrt(np.mean(np.array(res)**2)):.4f})")

print("\n=== A. era-weighted schemes ===")
def exp_tau(tau):
    def fn(t, excl):
        others = [b for b in anchors if b != excl]
        ws = np.array([np.exp(-abs(t-b)/tau) for b in others], dtype=np.float64)
        d = wmean_stack(np.array([AF[b] for b in others], dtype=np.float64), ws)
        return np.where(np.isfinite(d), d, 0.0).astype(np.float32)
    return fn

schemes = {}
for tau in [4, 6, 9, 12, 18, 24, 36]:
    a_, l_, _, _ = loo_eval(exp_tau(tau), f"exp-decay tau={tau}")
    schemes[f'exp tau={tau}'] = (a_, l_, exp_tau(tau))

def recent_k(K):
    def fn(t, excl):
        others = sorted([b for b in anchors if b != excl], key=lambda b: abs(t-b))[:K]
        d = np.nanmean(np.array([AF[b] for b in others], dtype=np.float64), axis=0)
        return np.where(np.isfinite(d), d, 0.0).astype(np.float32)
    return fn
for K in [2, 3, 4]:
    a_, l_, _, _ = loo_eval(recent_k(K), f"recent-{K} anchors")
    schemes[f'recent-{K}'] = (a_, l_, recent_k(K))

def lininterp(t, excl):
    others = sorted([b for b in anchors if b != excl])
    if t <= others[0]:
        d = AF[others[0]].astype(np.float64)
    elif t >= others[-1]:
        d = AF[others[-1]].astype(np.float64)
    else:
        d = np.full(n_cells, np.nan)
        for i in range(len(others)-1):
            if others[i] <= t <= others[i+1]:
                lam = (others[i+1]-t)/(others[i+1]-others[i])
                a1 = np.nan_to_num(AF[others[i]]); a2 = np.nan_to_num(AF[others[i+1]])
                ok = np.isfinite(AF[others[i]]) & np.isfinite(AF[others[i+1]])
                d = np.where(ok, lam*a1+(1-lam)*a2, np.nan)
                break
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)
a_, l_, _, _ = loo_eval(lininterp, "linear time interpolation")
schemes['lininterp'] = (a_, l_, lininterp)

best_name = min(schemes, key=lambda k: schemes[k][0])
ba, bl, bfn = schemes[best_name]
print(f"\nBEST: {best_name}  all={ba:.4f} (static {base_all:.4f}, gain {base_all-ba:+.4f})  "
      f"late={bl:.4f} (static {base_late:.4f}, gain {base_late-bl:+.4f})")

# ---------------- B. cov-slow student ----------------
print("\n=== B. cov-slow student on best teacher ===")
def slow_feats(t, hw):
    ms = [int(m) for m in test_months if abs(m-t) <= hw]
    return np.nanmean(np.array([Zdev[m] for m in ms]), axis=0)  # (n_cells, 5)

_, _, rows_t, w_t = loo_eval(bfn, f"teacher = {best_name}")
for hw in [0, 2, 4, 6, 9]:
    feats = {a: slow_feats(a, hw) for a in anchors}
    for lam in [10.0, 100.0, 1000.0]:
        errs_t, errs_s = [], []
        for a in anchors:
            others = [b for b in anchors if b != a]
            Xtr, ytr = [], []
            for b in others:
                fb = feats[b]; okb = np.isfinite(fb).all(axis=1) & np.isfinite(AF[b]) & rows_t[b][1]
                Xtr.append(np.column_stack([fb[okb], np.ones(okb.sum())]))
                ytr.append((AF[b]-rows_t[b][2])[okb])   # residual after teacher D-tilde
            XA, yA = np.vstack(Xtr), np.concatenate(ytr)
            try:
                cf = np.linalg.solve(XA.T@XA + lam*np.eye(6), XA.T@yA)
            except np.linalg.LinAlgError:
                continue
            fa = feats[a]; oka = np.isfinite(fa).all(axis=1) & rows_t[a][1]
            corr = np.column_stack([fa[oka], np.ones(oka.sum())]) @ cf
            base = rows_t[a][2][oka]; tgt = AF[a][oka]
            errs_t.append(np.sqrt(np.mean((base-tgt)**2)))
            errs_s.append(np.sqrt(np.mean((base+corr-tgt)**2)))
        rt = np.sqrt(np.mean(np.array(errs_t)**2)); rs = np.sqrt(np.mean(np.array(errs_s)**2))
        if rs < rt - 0.002:
            print(f"  hw={hw} lam={lam:.0f}: teacher={rt:.4f} +student={rs:.4f}  gain={rt-rs:+.4f}  coefs={np.round(cf[:5],3)}")

print("\nDONE.")
