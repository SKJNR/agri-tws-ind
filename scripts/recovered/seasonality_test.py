"""
seasonality_test.py — the never-tested channel: per-cell CALENDAR-MONTH climatology.

All 16 rounds used mu_c = per-cell all-month mean. If TWS anomalies have a seasonal
cycle (month-of-year structure), mu_{c,m} is free signal.

Tests:
  S1: variance decomposition — how much of per-cell TWS variance is month-of-year?
      (fit months clim on 2002-2012, measure on 2013-15)
  S2: masked-row prediction with mu_{c,month} replacing mu_c in the v5b/v6 stack
      (Kalman init, D-tilde, final prediction) — sparse-anchor honest CV.
  S3: k=0 model with month-of-year features (harmonic encodings) — quick LGBM check.
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
train['cal_mon'] = train['time'].dt.month
n_cells = int(train['cc'].max())+1

fit = train[train['time'].dt.year <= 2012].copy()
val  = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()

test_raw = pd.read_csv(f'{DATA}/Test (2).csv')
test_raw['time'] = pd.to_datetime(test_raw['time'])
cal_mask_frac = test_raw.assign(m=test_raw['TWS_t_masked'].astype(bool), cm=test_raw['time'].dt.month).groupby('cm')['m'].mean()
val['cal_mon'] = val['time'].dt.month
val['masked'] = val['cal_mon'].map(cal_mask_frac).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_cc = val['cc'].values
val_ta = val['t_abs'].values
val_msk = val['masked'].values
val_mon = val['cal_mon'].values

# ---------------- S1: variance decomposition ----------------
print("\n=== S1: month-of-year variance (fit 2002-2012) ===")
mu_c = fit.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
mucm = np.zeros((12, n_cells), dtype=np.float32)
cnt_m = np.zeros((12, n_cells), dtype=np.int32)
acc_m = np.zeros((12, n_cells), dtype=np.float64)
np.add.at(acc_m, (fit['cal_mon'].values-1, fit['cc'].values), fit['TWS_t'].values)
np.add.at(cnt_m, (fit['cal_mon'].values-1, fit['cc'].values), 1)
mucm[cnt_m > 0] = (acc_m/cnt_m)[cnt_m > 0]
# fill missing month-cell combos with annual mean
for m in range(12):
    no = cnt_m[m] == 0
    mucm[m, no] = mu_c[no]
# shrink seasonal toward annual mean: mucm_s = mu + s*(mucm - mu)
# evaluate on val: total anomaly var vs seasonal-explained var
okv = np.isfinite(val_target)
anom_annual = val_tws_visible.values - mu_c[val_cc]     # uses TWS_t (all val rows have it before masking? use raw)
raw_tws = train.set_index(['cc','t_abs'])['TWS_t']
# simpler: load raw val TWS from original train (unmasked copy exists as val_tws_visible BEFORE we nan'd? we nan'd in place)
valTWS = val_tws_visible.fillna(0).values  # masked set to NaN; those cells irrelevant for var decomp
okfin = np.isfinite(val_tws_visible.values)
anom_seas = val_tws_visible.values[okfin] - mucm[val_mon[okfin]-1, val_cc[okfin]]
anom_ann  = val_tws_visible.values[okfin] - mu_c[val_cc[okfin]]
print(f"val raw TWS anomaly var (annual mu):    {np.nanvar(anom_ann):.4f}")
print(f"val raw TWS anomaly var (monthly mu):   {np.nanvar(anom_seas):.4f}")
print(f"seasonal share = {1 - np.nanvar(anom_seas)/np.nanvar(anom_ann):.4f}")

# also: does target anomaly have seasonal structure? (target = TWS at t+1)
tgt_anom_ann = val_target[okv] - mu_c[val_cc[okv]]
tgt_month = val['cal_mon'].values
tgt_anom_seas = val_target[okv] - mucm[(tgt_month[okv])%12, val_cc[okv]]  # target month = current+1
print(f"target anomaly var (annual mu):  {np.nanvar(tgt_anom_ann):.4f}")
print(f"target anomaly var (monthly mu): {np.nanvar(tgt_anom_seas):.4f}")
print(f"seasonal share (target) = {1 - np.nanvar(tgt_anom_seas)/np.nanvar(tgt_anom_ann):.4f}")

# ---------------- S2: full stack with monthly climatology ----------------
print("\n=== S2: masked-row stack with mu_{c,month} (sparse-anchor CV) ===")
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
GAU2 = gaussW(2.0)

# infra on fit
clim = fit.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
Z = fit[COVS].values.astype('float32'); yv = fit['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
yms_all = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms_all)}
Fmat = np.full((len(yms_all), n_cells), np.nan, dtype=np.float32)
Fmat[fit['ym'].map(ym_to_i).values, fit['cc'].values] = fit['TWS_t'].values
t_abs_yms = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms_all], dtype=np.float64)
F64 = Fmat.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_yms[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0).astype(np.float32)
def trendex(t): return ((np.float64(t) - tbar_c) * beta_c).astype(np.float32)

def mu_of(t_abs):
    """cell mean for calendar month of t_abs (monthly climatology)."""
    mon = int(t_abs) % 12
    return mucm[mon]

# cov fields on val
Zv = val[COVS].values.astype('float32'); okv2 = np.isfinite(Zv).all(axis=1)
cov_est_v = np.full(len(val), np.nan, dtype=np.float32)
cov_est_v[okv2] = np.column_stack([Zv[okv2], np.ones(okv2.sum())]) @ coef
cov_field = {}
for m in np.sort(val['t_abs'].unique()):
    selm = (val_ta == m) & okv2
    fm = np.full(n_cells, np.nan, dtype=np.float32); fm[val_cc[selm]] = cov_est_v[selm]
    cov_field[m] = fm - mu_of(m)
all_m_v = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m_v]), axis=0)
W = {int(m): cov_field[m] - S for m in all_m_v}
W_pool = {m: from_grid(kpool(to_grid(v), BOX2)) for m, v in W.items()}

cand = [2013*12+0, 2013*12+6, 2014*12+0, 2014*12+8, 2015*12+0, 2015*12+6]
anchors_v = [m for m in cand if m in set(val_ta.tolist())]

def run_stack(use_monthly_mu):
    def MU(t_abs):
        return mu_of(t_abs) if use_monthly_mu else mu_c
    AF = {}
    for a in anchors_v:
        sel = (val_ta == a) & (~val_msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32); fa[val_cc[sel]] = val_tws_visible.values[sel]
        AF[a] = fa - MU(a)
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
    # calibrate + predict (phi-ensemble like v6c)
    cs, zs, vfs = [], [], []
    for a in anchors_v:
        dj = Dtil(a)
        ok = np.isfinite(W_pool[a]) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(W_pool[a][ok], (AF[a]-dj)[ok])[0,1])
        zs.append(np.var(W_pool[a][ok])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(LAM_F*var_f); R = max(varz - c_*c_/(LAM_F*var_f), 1e-4)
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
                pphi[sel0] = MU(a+1)[val_cc[sel0]] + Dtil(a+1)[val_cc[sel0]] + phi_f*x[val_cc[sel0]]
            for k in range(1, 9):
                m = a + k
                x = phi_f*x; P = phi_f**2*P + q
                if m in W_pool:
                    wv = W_pool[m]; okw = np.isfinite(wv)
                    K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                    x = np.where(okw, x + K*(wv - H*x), x)
                    P = np.where(okw, (1-K*H)*P, P)
                sel = np.where((val_ta == m) & val_msk)[0]
                if len(sel) == 0: continue
                tm = m + 1
                x2 = phi_f*x
                if tm in W_pool:
                    wv = W_pool[tm]; okw = np.isfinite(wv)
                    P2 = phi_f**2*P + q
                    K2 = np.where(okw, P2*H/(H*H*P2+R), 0).astype(np.float32)
                    x2 = np.where(okw, x2 + K2*(wv - H*x2), x2)
                pphi[sel] = MU(tm)[val_cc[sel]] + Dtil(tm)[val_cc[sel]] + x2[val_cc[sel]]
        pred = np.where(np.isfinite(pphi), pred + 0.5*pphi, pred)
    # smooth masked rows
    for m in np.unique(val_ta[val_msk & np.isfinite(pred)]):
        selm = np.where((val_ta == m) & val_msk & np.isfinite(pred))[0]
        cm = np.full(n_cells, np.nan, dtype=np.float32); cm[val_cc[selm]] = pred[selm].astype(np.float32)
        sm = from_grid(kpool(to_grid(cm), GAU2))
        pred[selm] = sm[val_cc[selm]]
    ok = np.isfinite(pred) & val_msk
    return float(np.sqrt(np.mean((pred[ok]-val_target[ok])**2)))

r_annual = run_stack(False)
print(f"v6c-equivalent CV (annual mu):    {r_annual:.4f}")
r_monthly = run_stack(True)
print(f"same stack with monthly mu_{c,m}: {r_monthly:.4f}   ({r_monthly-r_annual:+.4f})")

print("\nDone.", flush=True)
