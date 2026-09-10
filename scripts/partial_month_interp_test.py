"""
partial_month_interp_test.py — THE OVERLOOKED CHANNEL:
use VISIBLE cells' TWS_t at partial months (~50% visible) to interpolate the
fast state of MASKED cells at the same month (field is rank~50, neighbor corr 0.97).

Currently masked rows at partial months rely on Kalman propagated from anchors
4-8 months stale. A spatial interpolation from ~7,800 visible same-month cells
should be far fresher.

Tests (sparse-anchor honest CV):
  P1: accuracy of interpolated anomaly at masked cells (corr, RMSE vs true)
  P2: masked-row RMSE at partial months: current stack vs +interp observation
  P3: full-stack impact (all masked rows)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

print("Loading...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms_all = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms_all)}

fit = train[train['time'].dt.year <= 2012].copy()
val  = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
cal_mask_frac = test_raw.assign(m=test_raw['TWS_t_masked'].astype(bool), cm=test_raw['time'].dt.month).groupby('cm')['m'].mean()
print("calendar-month mask fractions:", {int(k): round(v,3) for k,v in cal_mask_frac.items()})
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values; val_ta = val['t_abs'].values; val_msk = val['masked'].values

# how many masked rows sit at partial months (vs full months)?
val['mfrac'] = val['cal_mon'].map(cal_mask_frac).values
partial_m = [m for m, f in cal_mask_frac.items() if 0.05 < f < 0.95]
print(f"partial calendar months: {partial_m}")
n_masked_partial = int((val_msk & np.isin(val['cal_mon'].values, partial_m)).sum())
n_masked_full    = int((val_msk & ~np.isin(val['cal_mon'].values, partial_m)).sum())
print(f"masked rows at partial months: {n_masked_partial:,} | at full months: {n_masked_full:,}")

# grid
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
BOX2 = {(di,dj): 1.0 for di in range(-2,3) for dj in range(-2,3)}
def kpool(g, Wt):
    num = np.zeros_like(g, dtype=np.float64); den = np.zeros_like(g, dtype=np.float64)
    for (di, dj), w in Wt.items():
        gg = shift(g, di, dj); ok = np.isfinite(gg)
        num[ok] += w*gg[ok]; den[ok] += w
    return np.where(den > 1e-8, num/np.maximum(den, 1e-8), np.nan).astype(np.float32)

mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

Zv = val[COVS].values.astype('float32'); okv = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_c
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m_v}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W.items()}

cand = [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6]
anchors_v = [m for m in cand if m in set(val_ta.tolist())]
AF = {}
for a in anchors_v:
    sel = (val_ta == a) & (~val_msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[val_cc[sel]] = val_tws_visible.values[sel]
    AF[a] = fa - mu_c
Dhat = np.nanmean(np.array([AF[a] for a in anchors_v]), axis=0)
Xs, ys = [], []
for a in anchors_v:
    dloo = np.nanmean(np.array([AF[b] for b in anchors_v if b != a]), axis=0)
    tx = trendex(a).astype(np.float32)
    ok = np.isfinite(dloo) & np.isfinite(S) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S[ok], tx[ok]])); ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + np.array([1e-3,1e-3,1e-3]), np.vstack(Xs).T@np.concatenate(ys))
w1, w2, w3 = map(float, w_)
def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

cs, zs, vfs = [], [], []
for a in anchors_v:
    dj = Dtil(a)
    ok = np.isfinite(W_pool[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
    cs.append(np.cov(W_pool[a][ok], (AF[a]-dj)[ok])[0,1])
    zs.append(np.var(W_pool[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
H = c_/(LAM_F*var_f); R = max(varz - c_*c_/(LAM_F*var_f), 1e-4)

# ---------------- P1: interpolation quality at masked cells ----------------
print("\n=== P1: interp quality (visible->masked, partial months) ===")
val_mon = val['cal_mon'].values
for sig in [1.0, 1.5, 2.0]:
    errs, corrs, ns = [], [], []
    for m in np.unique(val_ta):
        if m in anchors_v: continue
        rows_m = (val_ta == m)
        vis = rows_m & (~val_msk)
        hid = rows_m & val_msk
        if vis.sum() < 1000 or hid.sum() < 500: continue
        # visible anomaly field
        fv = np.full(n_cells, np.nan, dtype=np.float32); fv[val_cc[vis.values]] = val_tws_visible.values[vis.values]
        g = to_grid(fv)
        gi = kpool(g, gaussW(sig))   # kernel mean of VISIBLE neighbors only
        interp = from_grid(gi)
        true_f = np.full(n_cells, np.nan, dtype=np.float32); true_f[val_cc[hid.values]] = val_tws_visible.values[hid.values]  # hidden but we know it (val)
        okc = np.isfinite(interp) & np.isfinite(true_f)
        if okc.sum() < 100: continue
        errs.append(np.sqrt(np.mean((interp[okc]-true_f[okc])**2)))
        corrs.append(np.corrcoef(interp[okc], true_f[okc])[0,1])
        ns.append(int(okc.sum()))
    print(f"  sig={sig}: interp RMSE at masked cells {np.mean(errs):.4f}, corr {np.mean(corrs):.4f} (n~{np.mean(ns):,.0f}/month)")

# ---------------- P2/P3: inject interp as Kalman observation ----------------
print("\n=== P2/P3: stack with interp observations at partial months ===")

def run_stack(use_interp, sig_interp=1.5, R_int=0.09):
    pred = np.full(len(val), np.nan, dtype=np.float64)
    for phi_f in [0.74, 0.80]:
        pphi = np.full(len(val), np.nan, dtype=np.float64)
        q = LAM_F*var_f*(1-phi_f**2); P0 = LAM_F*(1-LAM_F)*var_f
        for a in anchors_v:
            fa = AF[a] - Dtil(a)
            x = np.where(np.isfinite(fa), LAM_F*np.nan_to_num(fa), 0.0).astype(np.float32)
            P = np.where(np.isfinite(AF[a]), P0, var_f).astype(np.float32)
            sel0 = np.where((val_ta == a) & val_msk)[0]
            if len(sel0):
                pphi[sel0] = mu_c[val_cc[sel0]] + Dtil(a+1)[val_cc[sel0]] + phi_f*x[val_cc[sel0]]
            for k in range(1, 9):
                m = a + k
                x = phi_f*x; P = phi_f**2*P + q
                # cov observation (all months)
                if m in W_pool:
                    wv = W_pool[m]; okw = np.isfinite(wv)
                    K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                    x = np.where(okw, x + K*(wv - H*x), x)
                    P = np.where(okw, (1-K*H)*P, P)
                # NEW: interp observation at partial months
                if use_interp and m in val['t_abs'].values:
                    rows_m = (val_ta == m)
                    vis = rows_m & (~val_msk)
                    hid = rows_m & val_msk
                    if vis.sum() > 1000 and hid.sum() > 500:
                        fv = np.full(n_cells, np.nan, dtype=np.float32); fv[val_cc[vis.values]] = val_tws_visible.values[vis.values]
                        gi = from_grid(kpool(to_grid(fv), gaussW(sig_interp)))
                        # observation: fast state ~= interp anomaly (minus D)
                        z_int = gi - Dtil(m)
                        okw = np.isfinite(gi) & (P > 0)
                        K_int = np.where(okw, P/(P+R_int), 0).astype(np.float32)
                        x = np.where(okw, x + K_int*(z_int - x), x)
                        P = np.where(okw, (1-K_int)*P, P)
                sel = np.where((val_ta == m) & val_msk)[0]
                if len(sel) == 0: continue
                tm = m + 1
                x2 = phi_f*x
                if tm in W_pool:
                    wv = W_pool[tm]; okw = np.isfinite(wv)
                    P2_ = phi_f**2*P + q
                    K2 = np.where(okw, P2_*H/(H*H*P2_+R), 0).astype(np.float32)
                    x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
                pphi[sel] = mu_c[val_cc[sel]] + Dtil(tm)[val_cc[sel]] + x2[val_cc[sel]]
        contrib = np.where(np.isfinite(pphi), 0.5*pphi, 0.0)
        pred = np.where(np.isfinite(pphi), np.nan_to_num(pred) + contrib, pred)
    # fixed sigma 2.0 smoothing
    for m in np.unique(val_ta[val_msk & np.isfinite(pred)]):
        selm = np.where((val_ta == m) & val_msk & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[val_cc[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), gaussW(2.0)))
        pred[selm] = sm[val_cc[selm]]
    ok = np.isfinite(pred) & val_msk
    r_all = float(np.sqrt(np.mean((pred[ok]-val_target[ok])**2)))
    okp = ok & np.isin(val_mon, partial_m)
    r_part = float(np.sqrt(np.mean((pred[okp]-val_target[okp])**2)))
    return r_all, r_part

r0_all, r0_part = run_stack(False)
print(f"baseline (no interp):  all {r0_all:.4f} | partial-month masked {r0_part:.4f}")
for sig, rint in [(1.0, 0.06), (1.5, 0.09), (1.5, 0.15), (2.0, 0.15), (1.0, 0.03)]:
    r_all, r_part = run_stack(True, sig, rint)
    print(f"interp sig={sig} R={rint}:  all {r_all:.4f} ({r_all-r0_all:+.4f}) | partial {r_part:.4f} ({r_part-r0_part:+.4f})")

print("\nDone.", flush=True)
