"""V11 FAILURE AUDIT — decompose the v11b LB regression (0.6999972 -> 0.7105711).

v11 bundled THREE changes on top of v10b:
  (1) era-weighted Dhat (tau=12)  [validated: LOO 0.8224 -> 0.8034]
  (2) cov-slow student             [validated: LOO 0.8034 -> 0.7867]
  (3) "honest" LOO anchor handling in calibrate()/Kalman-init/k0-B Dt0
      [NOT validated anywhere - made on principle]

This audit measures, offline:
  A. CSV deltas v11a/v11b vs v10b by window (public/private) x class (k0/masked)
  B. Kalman H/R/var_f recomputed for all three calibration modes
     (static-full = v10b recipe, era-LOO = v11a, era+stud-LOO = v11b)
  C. Dtil(tm) deltas per test month: static vs era vs era+student
  D. h=1 double-count sizing: era weight on the ADJACENT anchor's field
     (whose fast state is ALSO carried by the Kalman x0) for h=1 pred months
  E. sel0 check: masked rows at anchor months (v10b had an h=0 block; v11 dropped it)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
TAU = 12.0; HW = 4; LAM_STUD = 1000.0; LAM_F = 0.84

# ---------------- load train (v10/v11 verbatim) ----------------
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
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

# ---------------- load test ----------------
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

# ---------------- cov fields + anchors ----------------
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
W_raw = {int(m): cov_field[m] - S for m in test_months}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W_raw.items()}

mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

# ---------------- Dhat variants ----------------
Dhat_raw = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
Dhat = np.where(np.isfinite(Dhat_raw), Dhat_raw, 0.0).astype(np.float32)

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

# static LOO weights (v10 recipe)
Xs, ys = [], []
for a in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b != a]), axis=0)
    tx = trendex(a)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
w1s, w2s, w3s = map(float, w_)
print(f"static D-tilde weights: {w1s:.3f}/{w2s:.3f}/{w3s:.3f}")

# era LOO weights
Xs, ys = [], []
for a in anchors:
    d_ = dhat_era(a, excl=a); tx = trendex(a)
    ok = np.isfinite(d_) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([d_[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
w1e, w2e, w3e = map(float, w_)
print(f"era D-tilde weights:    {w1e:.3f}/{w2e:.3f}/{w3e:.3f}")

def Dtil_static(t): return (w1s*Dhat + w2s*S + w3s*trendex(t)).astype(np.float32)
def Dtil_era(t, excl=None): return (w1e*dhat_era(t, excl=excl) + w2e*S + w3e*trendex(t)).astype(np.float32)

# student (v11 recipe)
Zdev = {}
for m in test_months:
    selm = ta == m
    f = np.full((n_cells, len(COVS)), np.nan, dtype=np.float32)
    for j, c in enumerate(COVS):
        v = np.full(n_cells, np.nan, dtype=np.float32); v[cc_t[selm]] = test[c].values[selm]
        f[:, j] = v - clim[:, j]
    Zdev[int(m)] = f
Zdev_slow = {int(m): np.nanmean(np.array([Zdev[x] for x in test_months if abs(x-m) <= HW]), axis=0)
             for m in test_months}
Xtr, ytr = [], []
for a in anchors:
    dtr = Dtil_era(a, excl=a); fs = Zdev_slow[a]
    ok = np.isfinite(dtr) & np.isfinite(AF[a]) & np.isfinite(fs).all(axis=1)
    Xtr.append(np.column_stack([fs[ok], np.ones(ok.sum())])); ytr.append((AF[a]-dtr)[ok])
XA, yA = np.vstack(Xtr), np.concatenate(ytr)
beta = np.linalg.solve(XA.T@XA + LAM_STUD*np.eye(6), XA.T@yA)
def student_corr(t):
    fs = Zdev_slow[int(t)] if int(t) in Zdev_slow else None
    if fs is None: return np.zeros(n_cells, dtype=np.float32)
    c = np.column_stack([np.nan_to_num(fs), np.ones(n_cells)]) @ beta
    return c.astype(np.float32)
def Dtil_stud(t, excl=None): return (Dtil_era(t, excl=excl) + student_corr(t)).astype(np.float32)

print(f"student component std on pred months: {np.mean([np.nanstd(student_corr(int(m))) for m in test_months]):.4f}")

# ---------------- B. Kalman calibration: three modes ----------------
def calibrate(dtil_full, dtil_loo, label):
    """dtil_full(a): Dtil at anchor a INCLUDING own field (v10b recipe).
       dtil_loo(a):  Dtil at anchor a EXCLUDING own field (v11 recipe)."""
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = dtil_loo(a)
        ok = np.isfinite(W_pool[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W_pool[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W_pool[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(LAM_F*var_f); R = max(varz - c_*c_/(LAM_F*var_f), 1e-4)
    print(f"  {label}: H={H:.4f} R={R:.4f} var_f={var_f:.4f}  (K0 gain={LAM_F*(1-LAM_F)*var_f*H/(H*H*LAM_F*(1-LAM_F)*var_f+R):.4f})")
    return H, R, var_f

print("\n[B] Kalman calibration comparison:")
calibrate(Dtil_static, Dtil_static, "v10b  static + FULL-at-anchor (shipped, LB 0.7000)")
calibrate(Dtil_era,    Dtil_era,    "v11a  era + LOO-at-anchor               ")
calibrate(Dtil_stud,   Dtil_stud,   "v11b  era+stud + LOO-at-anchor (LB 0.7106)")

# what v10b's H/R would be with era Dtil but FULL-at-anchor:
# (Dtil_era with excl=None at anchor a = includes own anchor with era weight ~1.0 -> near self)
calibrate(lambda a: Dtil_era(a), lambda a: Dtil_era(a, excl=a), "era   full-at-anchor calib    ")

# ---------------- E. sel0 check ----------------
print("\n[E] masked rows AT anchor months (v11 dropped the h=0 block):")
for a in anchors:
    n_masked_at_anchor = int(((ta == a) & msk).sum())
    print(f"  anchor {a}: masked rows = {n_masked_at_anchor}")

# ---------------- C. Dtil(tm) deltas per month ----------------
print("\n[C] Dtil(tm) deltas at PREDICTION months (masked rows), static -> era -> era+stud:")
public_months = test_months[:7]
for m in test_months:
    tm = int(m) + 1
    if tm not in [int(x)+1 for x in test_months if msk[ta == x].all()]: pass
    selm = (ta == m) & msk
    if selm.sum() == 0: continue
    ds = Dtil_static(tm)[cc_t[selm]]; de = Dtil_era(tm)[cc_t[selm]]; dst = Dtil_stud(tm)[cc_t[selm]]
    w = 'PUBLIC ' if m in public_months else 'private'
    # h from nearest preceding anchor
    h = min([m - a for a in anchors if m > a] + [999])
    print(f"  {w} month {int(m)} (h={int(h)}): mean|era-static|={np.abs(de-ds).mean():.4f} "
          f"mean|stud|={np.abs(dst-de).mean():.4f} std(static)={ds.std():.3f} std(era)={de.std():.3f} std(stud)={dst.std():.3f}")

# ---------------- D. h=1 double-count ----------------
print("\n[D] era weight on ADJACENT anchor's field in Dhat_era(m+1) for h=1 months (double-count with Kalman x0):")
for a in anchors:
    m1 = a + 1
    if not ((ta == m1) & msk).any(): continue
    others = [b for b in anchors]
    ws = np.array([np.exp(-abs(m1-b)/TAU) for b in others]); ws = ws/ws.sum()
    wself = ws[others.index(a)] if a in others else 0.0
    print(f"  pred month {m1} (h=1 after anchor {a}): own-anchor weight in era-Dhat = {wself:.3f} (static: 0.167)")

# ---------------- A. CSV deltas ----------------
print("\n[A] CSV deltas vs v10b by window x class:")
base = pd.read_csv(f'{DL}/submission_v10b.csv')['Target'].values
pub = ta < (int(test_months.min()) + 12)
for tag in ['v11a', 'v11b']:
    o = pd.read_csv(f'{DL}/submission_{tag}.csv')['Target'].values
    d = o - base
    for wname, wmask in [('public', pub), ('private', ~pub)]:
        for cname, cmask in [('k0  ', ~msk), ('mask', msk)]:
            mm = wmask & cmask
            if mm.sum():
                print(f"  {tag} {wname} {cname}: n={int(mm.sum()):6d} mean|d|={np.abs(d[mm]).mean():.4f} rms(d)={np.sqrt((d[mm]**2).mean()):.4f} mean(d)={d[mm].mean():+.4f}")
print("\nDONE.")
