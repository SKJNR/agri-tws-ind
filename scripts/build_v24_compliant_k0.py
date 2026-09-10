"""SUBMISSION V24 — the COMPLIANT rescue of the Dcache-k0 advantage (Task 21).

Facts that motivate this build (all measured, Sep 5):
  - v22_splice 0.699118155 and v23_splice 0.683548406 landed exactly on the
    pre-registered diagonal: v23^2 - v21a^2 = -(v22^2 - v12b^2) (5-decimal match).
  - => K_a15^2 - K_dc^2 = +0.0122: v12b's Dcache k0 beats v21a's a15 k0 by
    ~0.010 RMSE on the k0 block (public), worth ~0.0038 public / ~0.0024 private.
  - v12b's k0 is NON-SELECTABLE only because its k0-B LGBM (XLB) includes raw
    lat_arr/lon_arr columns — prohibited by the organizer ruling of 19 Aug
    (forum thread 34450). The masked block, the linear k0 part, the Dcache
    (era-inclusive Dtil), and the GAU10 neighbourhood smoothing are all clean.

v24 = v21a's masked block (bit-copied from the submitted CSV) + a COMPLIANT
rebuild of v12b's k0: identical code path to submission_v12.py build_k0B
(Dtil_era) with the ONLY change = XLB/Xv drop the two raw coordinate columns.
mu_c / beta_c (per-cell train-TWS statistics, not coordinate encodings) stay.

Pre-registered band (before submission): S24 in [0.683548, 0.687374]
  (lower = full Dcache capture = v23's score; upper = a15-equivalence).
Adopt rule (pre-registered): adopt v24 as SLOT-1 iff S24 < 0.687374.
"""
import numpy as np, pandas as pd
import lightgbm as lgb

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
TAU = 12.0

# ---------------- load train (v12 verbatim) ----------------
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

# ---------------- grid utils (v12 verbatim) ----------------
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
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)
GAU10 = gaussW(1.0)

# ---------------- load test (v12 verbatim) ----------------
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

# ---------------- cov fields + anchors (v12 verbatim) ----------------
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

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"anchors: {anchors}")
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

# ---------------- Dhat era (v12 verbatim) ----------------
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

def fit_weights(get_dhat):
    Xs, ys = [], []
    for a in anchors:
        d_ = get_dhat(a, a); tx = trendex(a)
        ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
        Xs.append(np.column_stack([d_[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
    w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
    return tuple(map(float, w_))

w1e, w2e, w3e = fit_weights(lambda t, e: dhat_era(t, excl=e))
print(f"era weights: {w1e:.3f}/{w2e:.3f}/{w3e:.3f}")
def Dtil_era(t):    return (w1e*dhat_era(t) + w2e*S + w3e*trendex(t)).astype(np.float32)

# ---------------- k0 model B: COMPLIANT LGBM (lat/lon columns REMOVED) ----------------
print("k0-B: training two-comp + COMPLIANT LGBM (no lat/lon)...", flush=True)
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
print(f"k0-B linear coef[FAST_est]={coefF_B[0]:.4f}")

# >>> THE COMPLIANCE FIX: XLB without lat_arr/lon_arr raw coordinate columns <<<
mon_tr = (mgk['t_abs'].values % 12) + 1
XLB = np.column_stack([Xlin[:,0], slow1, Xlin[:,1:11],
    has_nxt_tr.astype(np.float32),
    np.sin(2*np.pi*mon_tr/12), np.cos(2*np.pi*mon_tr/12),
    beta_c[ccm], mu_c[ccm]]).astype(np.float32)
okY = np.isfinite(yB)
params = dict(objective='regression', learning_rate=0.05, num_leaves=63,
              min_child_samples=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbosity=-1, seed=0, num_threads=8)
ds  = lgb.Dataset(XLB[selF], label=yB[selF], weight=w_recB[selF])
dsr = lgb.Dataset(XLB[~selF & okY], label=yB[~selF & okY], weight=w_recB[~selF & okY])
bst = lgb.train(params, ds, num_boost_round=500)
bst_r = lgb.train(params, dsr, num_boost_round=400)
print("k0-B COMPLIANT LGBM trained (17 columns, no lat/lon).")

Lt = test[['cc','t_abs','TWS_t']+COVS].copy(); Lt['t_next'] = Lt['t_abs']+1
Rt = test[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
mgt = Lt.merge(Rt, on=['cc','t_next'], how='left')
assert len(mgt) == len(test), "merge must preserve test row order 1:1"
has_nxt_te = mgt[[c+'_nxt' for c in COVS]].notna().all(axis=1).values
cct = mgt['cc'].values
tw_ok = np.isfinite(mgt['TWS_t'].values)
use_full = (~msk) & has_nxt_te & tw_ok
use_red  = (~msk) & (~has_nxt_te) & tw_ok
mon_te = (mgt['t_abs'].values % 12) + 1

def smooth_rows(pred, rows, Wt):
    out = pred.copy()
    for m in np.unique(ta[rows & np.isfinite(pred)]):
        selm = np.where((ta == m) & rows & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[cc_t[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), Wt))
        out[selm] = sm[cc_t[selm]]
    return out

def build_k0B_compliant(dtil):
    uniq_m = np.unique(mgt['t_abs'].values)
    Dcache = {int(m): (dtil(int(m)), dtil(int(m)+1)) for m in uniq_m}
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
    # >>> XLv likewise WITHOUT lat/lon columns (must mirror XLB) <<<
    XLv = np.column_stack([Xv[:,0], Dt1, Xv[:,1:11],
        has_nxt_te.astype(np.float32),
        np.sin(2*np.pi*mon_te/12), np.cos(2*np.pi*mon_te/12),
        beta_c[cct], mu_c[cct]]).astype(np.float32)
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

# ---------------- assemble: v21a masked (bit-copy) + compliant k0 ----------------
print("\nk0-B compliant (era-inclusive Dcache)...", flush=True)
k0_compliant = build_k0B_compliant(Dtil_era)

sub_ids = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')[['ID']]
v21a = pd.read_csv(f'{DL}/submission_v21a.csv')
v21a.columns = ['ID', 'Target']
assert (v21a['ID'].values == sub_ids['ID'].values).all()
masked_vals = v21a['Target'].values.astype(np.float64)

pred = np.empty(len(test), dtype=np.float64)
pred[~msk] = k0_compliant[~msk]
pred[msk]  = masked_vals[msk]          # bit-copy of the submitted v21a masked block
pred = np.where(np.isnan(pred), mu_c[cc_t], pred)

out = sub_ids.copy()
out['Target'] = pred.astype(np.float32)
out.to_csv(f'{DL}/submission_v24.csv', index=False)
print(f"saved v24: mean={pred.mean():.4f} std={pred.std():.4f} nan={int(out['Target'].isna().sum())}", flush=True)

# ---------------- verification ----------------
print("\n=== verification ===", flush=True)
v24 = pd.read_csv(f'{DL}/submission_v24.csv'); v24.columns = ['ID', 'Target']
v12b = pd.read_csv(f'{DL}/submission_v12b.csv'); v12b.columns = ['ID', 'Target']
pub = ta < (int(test_months.min()) + 12)
checks = [
    ('rows 280,961', len(v24) == 280961),
    ('ID order == SampleSubmission', (v24['ID'].values == sub_ids['ID'].values).all()),
    ('all finite', bool(np.isfinite(v24['Target'].values).all())),
    ('masked bit-exact vs v21a', float(np.abs(v24['Target'].values[msk] - masked_vals[msk]).max()) == 0.0),
]
for name, okk in checks:
    print(f"  [{'PASS' if okk else 'FAIL'}] {name}")

d24_12b = v24['Target'].values - v12b['Target'].values
d24_21a = v24['Target'].values - v21a['Target'].values
for wname, wmask in [('public', pub), ('private', ~pub)]:
    for cname, cmask in [('k0', ~msk), ('msk', msk)]:
        mm = wmask & cmask
        if mm.sum():
            print(f"  v24-vs-v12b {wname} {cname}: n={int(mm.sum()):6d} rms={np.sqrt((d24_12b[mm]**2).mean()):.4f} "
                  f"corr={np.corrcoef(v24['Target'].values[mm], v12b['Target'].values[mm])[0,1]:.5f}")
print(f"  v24-vs-v21a k0 all: rms={np.sqrt((d24_21a[~msk]**2).mean()):.4f}")

# pre-registered band arithmetic
S21 = 0.687374005; S23 = 0.683548406
print(f"\n  pre-registered band: [{S23:.6f}, {S21:.6f}]  (v23 floor .. v21a ceiling)")
print(f"  adopt rule: v24 -> SLOT-1 iff S24 < {S21:.6f}; else keep v21a.")
print("DONE.")
