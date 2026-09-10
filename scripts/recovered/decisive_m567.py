"""M5-M7: THE decisive tests.
M5: per-PC corr of cov composite w with fast field (spectral separation).
M6: on the 6 REAL test anchors: corr over cells between cov-field and TWS anchor field.
M7: on M2 mirror: does w(tm) add information BEYOND the v18 pipeline prediction?
    (a) pipeline pred (v18 cfg0)
    (b) direct regression pred from w(tm) alone
    (c) optimal linear blend of (a) and (b)  <-- if this beats (a) by >0.01, BREAKTHROUGH
"""
import numpy as np, pandas as pd, time
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)

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
log("data loaded")

def build_infra(fit_df):
    d = {}
    yms = np.sort(fit_df['ym'].unique()); T = len(yms)
    ym_to_i = {int(v):i for i,v in enumerate(yms)}
    t_abs_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)
    F = np.full((T, n_cells), np.nan, dtype=np.float32)
    F[fit_df['ym'].map(ym_to_i).values, fit_df['cc'].values] = fit_df['TWS_t'].values
    d['mu_c'] = np.nanmean(F, axis=0)
    d['clim'] = fit_df.groupby('cc')[COVS].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
    Z = fit_df[COVS].values.astype('float32'); yv = fit_df['TWS_t'].values.astype('float32')
    Z1 = np.column_stack([Z, np.ones(len(Z))])
    okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
    d['coef'] = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
    F64 = F.astype(np.float64)
    ok_t = np.isfinite(F64)
    t_mat = np.where(ok_t, t_abs_arr[:, None], np.nan)
    d['tbar_c'] = np.nanmean(t_mat, axis=0)
    td = t_mat - d['tbar_c'][None, :]
    sxx = np.nansum(td*td, axis=0)
    d['beta_c'] = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), np.nansum(td*F64, axis=0)/np.where(sxx>0,sxx,1), 0.0)
    A_dt = F64 - d['mu_c'][None,:] - (t_abs_arr[:,None]-d['tbar_c'][None,:])*d['beta_c'][None,:]
    full = ~np.isnan(F).any(axis=0)
    A = A_dt[:, full]; A = A - A.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    d['V'] = Vt[:100].T
    d['full'] = full
    d['full_idx'] = np.where(full)[0]
    return d

_SMOOTH = {}
def get_smoother(sigma_deg):
    key = float(sigma_deg)
    if key in _SMOOTH: return _SMOOTH[key]
    la = np.deg2rad(cell_xy[:,0]); lo = np.deg2rad(cell_xy[:,1])
    pts = np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
    tree = cKDTree(pts)
    rad = np.deg2rad(3.0*sigma_deg)
    rows, cols, vals = [], [], []
    for i in range(len(pts)):
        nb = tree.query_ball_point(pts[i], rad)
        if len(nb) == 0: nb = [i]
        d2 = ((pts[nb]-pts[i])**2).sum(axis=1)
        w = np.exp(-d2/(2*np.deg2rad(sigma_deg)**2))
        w = w/np.maximum(w.sum(), 1e-12)
        rows.extend([i]*len(nb)); cols.extend(nb); vals.extend(w.tolist())
    K = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    _SMOOTH[key] = K
    return K

infra = build_infra(train)
log("infra built")

def cov_field_of(df):
    """ym -> (cov composite field, smoothed)"""
    Zm = np.column_stack([df[c].values for c in COVS] + [np.ones(len(df))]).astype('float32')
    est = Zm @ infra['coef']
    out = {}
    for ym in np.sort(df['ym'].unique()):
        sel = df['ym'].values == ym
        fld = np.full(n_cells, np.nan, dtype=np.float32)
        fld[df['cc'].values[sel]] = est[sel]
        out[int(ym)] = fld
    return out

# ================= M6: real test anchors coupling =================
log("=== M6: cov composite vs TWS anchor field at the 6 REAL test anchors ===")
tcv = cov_field_of(test)
mfrac = test.groupby('ym')['masked'].mean()
anchors_te = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
print(f"test anchors: {anchors_te}")
S_te = np.nanmean(np.array([tcv[a] for a in anchors_te]), axis=0)
for a in anchors_te:
    sel = (test['ym'].values == a) & (~test['masked'].values)
    af = np.full(n_cells, np.nan, dtype=np.float32)
    af[test['cc'].values[sel]] = test['TWS_t'].values[sel]
    af = af - infra['mu_c']
    ok = np.isfinite(af) & np.isfinite(tcv[a])
    r_raw = np.corrcoef(af[ok], tcv[a][ok])[0,1]
    # smoothed versions
    K2 = get_smoother(2.0)
    af_s = K2 @ np.where(np.isfinite(af), af, 0.0)
    w_s = K2 @ np.where(np.isfinite(tcv[a]), tcv[a], 0.0)
    r_sm = np.corrcoef(af_s[ok], w_s[ok])[0,1]
    # regression RMSE of anchor field from cov composite
    b = np.polyfit(tcv[a][ok], af[ok], 1)
    rm = np.sqrt(np.mean((af[ok] - np.polyval(b, tcv[a][ok]))**2))
    print(f"  {a}: corr(raw)={r_raw:.4f} corr(sm2)={r_sm:.4f} rmse(reg)={rm:.4f} anchor_std={np.nanstd(af):.4f}")

# ================= M5: per-PC on train eval months =================
log("=== M5: per-PC corr of w with fast field (2013-15, train-fit infra) ===")
infra_cv = build_infra(train[train['time'].dt.year <= 2012])
ev = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]
Zm = np.column_stack([ev[c].values for c in COVS] + [np.ones(len(ev))]).astype('float32')
est = Zm @ infra_cv['coef']
ev2 = ev.assign(cov_est=est)
fidx = infra_cv['full_idx']
V = infra_cv['V']
# proper per-PC corr: corr of PC projections across MONTHS (time series corr per PC).
proj_f, proj_w = [], []
for ym in np.sort(ev2['ym'].unique()):
    sel = ev2['ym'].values == ym
    f = np.full(n_cells, np.nan, dtype=np.float32); f[ev2['cc'].values[sel]] = ev2['TWS_t'].values[sel]
    w = np.full(n_cells, np.nan, dtype=np.float32); w[ev2['cc'].values[sel]] = ev2['cov_est'].values[sel]
    t_abs_ym = (int(ym)//100)*12 + (int(ym)%100) - 1
    f_fast = (f - infra_cv['mu_c']) - (t_abs_ym - infra_cv['tbar_c'])*infra_cv['beta_c']
    w_dev = w - infra_cv['mu_c']
    ff = f_fast[fidx]; ww = w_dev[fidx]
    m = np.isfinite(ff) & np.isfinite(ww)
    proj_f.append(V[m].T @ ff[m])
    proj_w.append(V[m].T @ ww[m])
proj_f = np.array(proj_f); proj_w = np.array(proj_w)   # (months, 100)
print("  per-PC corr (across months, 2013-15):")
for blk in range(10):
    lo_, hi_ = blk*10, blk*10+10
    rs = [np.corrcoef(proj_f[:,k], proj_w[:,k])[0,1] for k in range(lo_,hi_)]
    print(f"    PC{lo_+1:3d}-{hi_:3d}: mean r = {np.mean(rs):+.4f}   " + " ".join(f"{v:+.2f}" for v in rs))

# ================= M7: M2 mirror — does w(tm) add info beyond pipeline? =================
log("=== M7: M2 mirror blend test ===")
val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
M2_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2, 24173:3, 24176:3, 24177:3,
           24178:3, 24179:3, 24180:3, 24181:3, 24182:3, 24183:3, 24187:4}
months = set(M2_ANCHORS) | set(M2_RUNS)
mir = val_all[val_all['t_abs'].isin(months)].copy()
mir['masked'] = mir['t_abs'].isin(M2_RUNS)
mir.loc[mir['masked'], 'TWS_t'] = np.nan

# --- pipeline pred (v18 cfg0, copy of run_masked_v18) ---
def run_masked(infra, ev_df, ev_tws_visible, phi, use_bwd=True, dtil_w=(0.70,0.45,0.073),
               lam=0.80, bwd_max_gap=8, dhat_tau=24.0, dhat_mode='full', smooth_sigma=2.0):
    mu_c = infra['mu_c']; n = len(ev_df)
    ta = ev_df['t_abs'].values; cc = ev_df['cc'].values; msk = ev_df['masked'].values
    Zv = ev_df[COVS].values.astype('float32')
    ok = np.isfinite(Zv).all(axis=1)
    cov_est = np.full(n, np.nan, dtype=np.float32)
    cov_est[ok] = np.column_stack([Zv[ok], np.ones(ok.sum())]) @ infra['coef']
    cov_field = {}
    for m in np.sort(ev_df['t_abs'].unique()):
        selm = (ta == m) & ok
        fm = np.full(n_cells, np.nan, dtype=np.float32)
        fm[cc[selm]] = cov_est[selm]
        cov_field[int(m)] = fm - mu_c
    all_m = np.array(sorted(cov_field.keys()))
    S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
    W = {int(m): cov_field[m] - S for m in all_m}
    mfrac = ev_df.groupby('t_abs')['masked'].mean()
    anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
    tws_vis = ev_tws_visible.loc[ev_df.index].values
    AF = {}
    for a in anchors:
        sel = (ta == a) & (~msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    Dhat_s = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    w1,w2,w3 = dtil_w
    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil_s(t): return (w1*Dhat_s + w2*S + w3*trendex(t)).astype(np.float32)
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil_s(a); w_ = W[a]
        okc = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[okc], (AF[a]-dj)[okc])[0,1]); zs.append(np.var(w_[okc])); vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(lam*var_f); R = max(varz - c_*c_/(lam*var_f), 1e-4)
    q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f
    A_arr = np.array(sorted(AF.keys()), dtype=np.float64)
    A_stack = np.array([AF[a] for a in A_arr]); A_fin = np.isfinite(A_stack)
    def Dhat_era(t):
        wts = np.exp(-np.abs(A_arr - float(t))/dhat_tau)[:, None] * A_fin
        num = (np.where(A_fin, A_stack, 0.0) * wts).sum(axis=0)
        den = wts.sum(axis=0)
        return np.where(den > 1e-6, num/np.maximum(den,1e-6), Dhat_s).astype(np.float32)
    def Dtil_b(t, era):
        if not era: return Dtil_s(t)
        return (w1*Dhat_era(t) + w2*S + w3*trendex(t)).astype(np.float32)
    Sm = get_smoother(smooth_sigma) if smooth_sigma else None
    def pass_kalman(i, tm, direction):
        f0 = AF[i] - Dtil_b(i, era=True)
        x = np.where(np.isfinite(f0), lam*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        rng = range(i+1, tm+1) if direction>0 else range(i-1, tm-1, -1)
        for m in rng:
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = W[m]; okw = np.isfinite(w_)
                Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
                P = np.where(okw, (1-Kg*H)*P, P)
        return x, P
    pred = np.full(n, np.nan, dtype=np.float64)
    anchors_arr = np.array(anchors)
    masked_months = sorted(set(ta[msk].tolist()))
    for m in masked_months:
        tm = m + 1
        sel = np.where((ta == m) & msk)[0]
        if len(sel) == 0: continue
        i = int(anchors_arr[np.searchsorted(anchors_arr, m, side='right')-1])
        xf, Pf = pass_kalman(i, tm, +1)
        later = anchors_arr[anchors_arr > m]
        use_b = use_bwd and len(later) and (bwd_max_gap is None or (int(later[0]) - tm) <= bwd_max_gap)
        if use_b:
            k = int(later[0])
            xb, Pb = pass_kalman(k, tm, -1)
            wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            x = xf
        if Sm is not None: x = Sm @ x
        dT = Dtil_b(tm, era=True)
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
    return pred, W, cov_field, S

pipe_pred, W, cov_field, S = run_masked(infra_cv, mir, val_tws_visible, 0.80)
mm = mir['masked'].values & np.isfinite(pipe_pred) & np.isfinite(mir['target'].values)
y_true = mir['target'].values[mm]
p_pipe = pipe_pred[mm]
rmse_pipe = np.sqrt(np.mean((p_pipe - y_true)**2))
log(f"  pipeline M2 masked RMSE: {rmse_pipe:.4f}  (expect ~0.6688)")

# w(tm): cov composite (raw and smoothed) at target month
ta_m = mir['t_abs'].values
K2 = get_smoother(2.0)
w_raw = np.full(len(mir), np.nan); w_sm = np.full(len(mir), np.nan)
for m in np.unique(ta_m):
    fld = cov_field.get(int(m)+1)
    if fld is None: continue
    smf = K2 @ np.where(np.isfinite(fld), fld, 0.0)
    sel = ta_m == m
    w_raw[sel] = fld[mir['cc'].values[sel]]
    w_sm[sel] = smf[mir['cc'].values[sel]]
okw = np.isfinite(w_raw)
m1 = mm & okw
print(f"  rows with w(tm): {m1.sum():,} / {mm.sum():,}")

y_true = mir['target'].values  # full length
def blend_rmse(cols):
    X = np.column_stack(cols)
    X = X[m1]
    X = np.column_stack([X, np.ones(len(X))])
    c = np.linalg.solve(X.T@X + 1e-6*np.eye(X.shape[1]), X.T@y_true[m1])
    return float(np.sqrt(np.mean((X@c - y_true[m1])**2)))

p_pipe_full = pipe_pred
print(f"  (a) pipeline alone:        {np.sqrt(np.mean((p_pipe_full[m1]-y_true[m1])**2)):.4f}")
wr = w_raw[m1]
b0 = np.polyfit(wr, y_true[m1], 1)
print(f"  (b) w(tm) alone:           {np.sqrt(np.mean((np.polyval(b0,wr)-y_true[m1])**2)):.4f}")
print(f"  (c) pipeline + w_raw(tm):  {blend_rmse([p_pipe_full, w_raw]):.4f}")
print(f"  (d) pipeline + w_sm(tm):   {blend_rmse([p_pipe_full, w_sm]):.4f}")
print(f"  (e) pipe + w_raw + w_sm:   {blend_rmse([p_pipe_full, w_raw, w_sm]):.4f}")
print(f"  (f) w_raw + w_sm alone:    {blend_rmse([w_raw, w_sm]):.4f}")
