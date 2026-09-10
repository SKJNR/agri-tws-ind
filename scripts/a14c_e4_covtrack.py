"""
Agent 14-c E4: covariate-D tracking (dynamic vs static).

Questions:
 (a) S_early (mean cov-regression field over 2015-16 test months) vs S_late (2017-18):
     do they track the anchor anomalies of their OWN era better than the other era?
 (b) corr(S_late - S_early, D_late - D_early): does the cov field's drift track D's drift?
 (c) per-anchor: cov_field[a] vs A_a, and cov deviation W_a vs anchor deviation from
     the fitted linear D trajectory (slow deviation tracking).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
cache = np.load(f'{DATA}/a14c_cache.npz')
mu_c = cache['mu_c']; beta_c = cache['beta_c']; tbar_c = cache['tbar_c']
A = cache['A']; anchors = [int(a) for a in cache['anchors']]; t_anc = cache['t_anc']
t_mid = float(cache['t_mid']); n_cells = len(mu_c)
def corr2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100: return np.nan
    return float(np.corrcoef(x[m], y[m])[0,1])

# rebuild cov fields (need per-month)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS: train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
Z = train[COVS].values.astype('float32'); yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['t_abs'] = test['time'].dt.year*12 + test['time'].dt.month - 1
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

# eras: early = months <= 2016-12 (24203), late = >= 2017-01 (24204)
early_m = [m for m in all_m if m <= 24203]
late_m  = [m for m in all_m if m >= 24204]
S_early = np.nanmean(np.array([cov_field[m] for m in early_m]), axis=0)
S_late  = np.nanmean(np.array([cov_field[m] for m in late_m]), axis=0)
print(f"test months: early={len(early_m)} late={len(late_m)}")
print(f"std: S={np.nanstd(S):.3f} S_early={np.nanstd(S_early):.3f} S_late={np.nanstd(S_late):.3f} "
      f"drift S_late-S_early={np.nanstd(S_late-S_early):.3f}  corr(S_early,S_late)={corr2(S_early,S_late):.3f}")

D_early = np.nanmean(A[[0,1,2,3]], axis=0)
D_late  = np.nanmean(A[[4,5]], axis=0)
print(f"\n--- (a) era-specific tracking ---")
print(f"corr(S_early, D_early) = {corr2(S_early, D_early):.3f}   corr(S_early, D_late) = {corr2(S_early, D_late):.3f}")
print(f"corr(S_late,  D_early) = {corr2(S_late, D_early):.3f}   corr(S_late,  D_late) = {corr2(S_late, D_late):.3f}")
print(f"corr(S, D_early) = {corr2(S, D_early):.3f}  corr(S, D_late) = {corr2(S, D_late):.3f}")
print(f"\n--- (b) drift tracking ---")
print(f"corr(S_late - S_early, D_late - D_early) = {corr2(S_late - S_early, D_late - D_early):.3f}")
print(f"std(D_late - D_early) = {np.nanstd(D_late - D_early):.3f}")
# regression: does S-drift improve D-drift prediction beyond slope-model drift?
b_sl = cache['b_sl']; s_h = float(cache['s_h'])
slope_drift = s_h*beta_c*(t_anc[[4,5]].mean() - t_anc[[0,1,2,3]].mean())
dd = D_late - D_early
X = np.column_stack([np.ones(n_cells), np.nan_to_num(slope_drift), np.nan_to_num(S_late - S_early)])
ok = np.isfinite(dd) & np.isfinite(S_late) & np.isfinite(S_early) & np.isfinite(slope_drift)
w = np.linalg.solve(X[ok].T@X[ok] + 1e-3*np.eye(3), X[ok].T@dd[ok])
r2a = 1 - np.var(dd[ok] - X[ok]@w)/np.var(dd[ok])
X2 = np.column_stack([np.ones(n_cells), np.nan_to_num(slope_drift)])[ok]
w2 = np.linalg.solve(X2.T@X2 + 1e-3*np.eye(2), X2.T@dd[ok])
r2b = 1 - np.var(dd[ok] - X2@w2)/np.var(dd[ok])
print(f"D_drift ~ slope-model: R^2={r2b:.3f};  + S-drift: R^2={r2a:.3f}  coefs={np.round(w,3)}")

print(f"\n--- (c) per-anchor cov tracking ---")
# per-cell linear trajectory from all 6 anchors (diagnostic only)
X6 = np.column_stack([np.ones(6), t_anc - t_mid])
okc = np.isfinite(A).all(axis=0)
co = np.linalg.solve(X6.T@X6, X6.T@A[:, okc])
afit = np.full(n_cells, np.nan); bfit = np.full(n_cells, np.nan)
afit[okc] = co[0]; bfit[okc] = co[1]
print(f"{'anchor':>8} {'corr(cov_a,A_a)':>16} {'corr(W_a, dev_a)':>17} {'corr(cov_a,dev)':>16}")
for j, a in enumerate(anchors):
    Wa = cov_field[a] - S
    dev = A[j] - (afit + bfit*(t_anc[j]-t_mid))     # anchor deviation from linear fit (fast+slow-dev)
    dev_slow = afit + bfit*(t_anc[j]-t_mid) - np.nanmean(A, axis=0)  # slow trajectory dev from mean
    print(f"{a:>8} {corr2(cov_field[a], A[j]):>16.3f} {corr2(Wa, dev):>17.3f} {corr2(cov_field[a], dev_slow):>16.3f}")
# also: does W_a track the slow DEVIATION of anchor a from D-hat (i.e., D(c,t_a) - Dbar)?
print(f"\ncorr(W_a, A_a - Dhat) per anchor (fast tracking, for reference):")
Dhat = np.nanmean(A, axis=0)
for j, a in enumerate(anchors):
    print(f"  {a}: {corr2(cov_field[a]-S, A[j]-Dhat):.3f}")
np.savez_compressed(f'{DATA}/a14c_cache3.npz', S_early=S_early, S_late=S_late, S=S,
                    cov_field_stack=np.array([cov_field[m] for m in all_m]), all_m=all_m)
print("\ncache3 saved.")
