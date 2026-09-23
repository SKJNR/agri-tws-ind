"""BLEND TEST: does adding real-GRACE estimates (GravIS/COST-G/CSR) to our pipeline
improve masked-row RMSE on the M2 mirror?

External estimate construction:
  - per-cell affine map: comp_TWS ~ a_c * ext + b_c, fit on train <=2012 months.
  - ext value at TARGET month tm -> estimate of comp TWS at tm.
Test on M2 mirror masked rows (target = comp TWS at tm):
  (a) pipeline alone
  (b) ext alone (per product)
  (c) ext ensemble (mean of 3 mapped products)
  (d) pipeline + ext blend (OLS)
  (e) pipeline + ext ensemble blend
"""
import numpy as np, pandas as pd, xarray as xr, time
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ---------- load external products at 1deg, month -> field ----------
def load_product(kind):
    if kind == 'gravis':
        ds = xr.open_dataset(f'{S}/gravis_tws_grid.nc')
        f = ds['tws'].values.astype(np.float64)
        tv = pd.to_datetime(ds['time'].values)
        latg = ds['lat'].values  # desc 89.5..-89.5
        # store as (ym, lat_idx_desc) ; use lat position map by value
        return f, tv, {round(float(v),1): i for i, v in enumerate(latg)}, 'desc'
    if kind == 'costg':
        ds = xr.open_dataset(f'{S}/gravis_costg_tws.nc')
        f = ds['tws'].values.astype(np.float64)
        tv = pd.to_datetime(ds['time'].values)
        latg = ds['lat'].values
        return f, tv, {round(float(v),1): i for i, v in enumerate(latg)}, 'desc'
    if kind == 'csr':
        ds = xr.open_dataset(f'{S}/csr_mascons_all.nc')
        tv = pd.to_datetime(ds['time'].values, unit='D', origin=pd.Timestamp('2002-01-01'))
        lwe = ds['lwe_thickness'].values.astype(np.float32)
        T = lwe.shape[0]; out = np.full((T, 180, 360), np.nan, dtype=np.float32)
        for i in range(180):
            for j in range(360):
                blk = lwe[:, 4*i:4*i+4, 4*j:4*j+4]
                n = np.isfinite(blk).sum(axis=(1,2)); s = np.where(np.isfinite(blk), blk, 0).sum(axis=(1,2))
                out[:, i, j] = np.where(n > 0, s/np.maximum(n,1), np.nan)
        lat1 = np.array([-89.875 + 0.25*(4*i+1.5) for i in range(180)])
        # flip to descending 89.5..-89.5
        out = out[:, ::-1, :]
        lat1_desc = lat1[::-1]
        return out.astype(np.float64), tv, {round(float(v),1): i for i, v in enumerate(lat1_desc)}, 'desc'

log("loading external products ...")
PROD = {}
for kind in ['gravis', 'costg', 'csr']:
    f, tv, latpos, order = load_product(kind)
    ym_idx = {t.year*100+t.month: i for i, t in enumerate(tv)}
    PROD[kind] = (f, ym_idx, latpos)
    log(f"  {kind}: {f.shape[0]} months")

def gval(kind, la, lo, ym):
    f, ym_idx, latpos = PROD[kind]
    if ym not in ym_idx: return np.nan
    glo = lo if lo >= 0 else lo + 360
    li = latpos.get(round(float(la), 1))
    if li is None: li = int(np.argmin(np.abs(np.array(sorted(latpos.keys())) - la)))
    lo_i = int(round((glo - 0.5))) % 360
    return f[ym_idx[ym], li, lo_i]

# ---------- competition data ----------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)

# ---------- per-cell affine fit on <=2012 ----------
log("fitting per-cell affine maps (train <=2012) ...")
fitdf = train[train['time'].dt.year <= 2012]
cell_lat = cell_xy[:,0]; cell_lon = cell_xy[:,1]
A_MAP = {}
for kind in ['gravis', 'costg', 'csr']:
    a_c = np.zeros(n_cells); b_c = np.zeros(n_cells); n_c = np.zeros(n_cells, dtype=int)
    # group by cell: use pivot approach for speed
    gv = np.full(len(fitdf), np.nan)
    las = fitdf['lat'].values; los = fitdf['lon'].values; yms = fitdf['ym'].values
    f, ym_idx, latpos = PROD[kind]
    # vectorize: build 1deg field per needed ym
    needed = np.unique(yms)
    for ym in needed:
        if ym not in ym_idx: continue
        fi = f[ym_idx[ym]]
        m = yms == ym
        glo = np.where(los[m] >= 0, los[m], los[m]+360)
        li = np.array([latpos.get(round(float(x),1), -1) for x in las[m]])
        okm = li >= 0
        lo_i = np.round(glo - 0.5).astype(int) % 360
        gv[np.where(m)[0][okm]] = fi[li[okm], lo_i[okm]]
    df = pd.DataFrame({'cc': fitdf['cc'].values, 'g': gv, 'y': fitdf['TWS_t'].values})
    df = df[np.isfinite(df['g']) & np.isfinite(df['y'])]
    g_ = df.groupby('cc')
    n_ = g_.size().reindex(range(n_cells), fill_value=0)
    mx = g_['g'].mean().reindex(range(n_cells), fill_value=0)
    my = g_['y'].mean().reindex(range(n_cells), fill_value=0)
    df['gx'] = df['g'] - df['cc'].map(mx)
    df['gy'] = df['y'] - df['cc'].map(my)
    sxy = (df['gx']*df['gy']).groupby(df['cc']).sum().reindex(range(n_cells), fill_value=0)
    sxx = (df['gx']**2).groupby(df['cc']).sum().reindex(range(n_cells), fill_value=0)
    a_c = np.where((n_ >= 40) & (sxx > 1e-6), sxy/np.maximum(sxx, 1e-6), 0.0)
    b_c = np.where(n_ >= 40, my - a_c*mx, 0.0)
    A_MAP[kind] = (a_c, b_c)
    log(f"  {kind}: cells fitted = {(n_>=40).sum():,}")

def ext_estimate(kind, cc_arr, la_arr, lo_arr, ym_arr):
    a_c, b_c = A_MAP[kind]
    v = np.array([gval(kind, la, lo, ym) for la, lo, ym in zip(la_arr, lo_arr, ym_arr)])
    return a_c[cc_arr]*v + b_c[cc_arr]

# ---------- pipeline (v18 cfg0) on M2 mirror ----------
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
    return d

infra_cv = build_infra(train[train['time'].dt.year <= 2012])

def run_masked(ev_df, ev_tws_visible, phi=0.80, lam=0.80, dtil_w=(0.70,0.45,0.073),
               use_bwd=True, bwd_max_gap=8, dhat_tau=24.0, smooth_sigma=2.0):
    mu_c = infra_cv['mu_c']; n = len(ev_df)
    ta = ev_df['t_abs'].values; cc = ev_df['cc'].values; msk = ev_df['masked'].values
    Zv = ev_df[COVS].values.astype('float32')
    ok = np.isfinite(Zv).all(axis=1)
    cov_est = np.full(n, np.nan, dtype=np.float32)
    cov_est[ok] = np.column_stack([Zv[ok], np.ones(ok.sum())]) @ infra_cv['coef']
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
    def trendex(t): return ((np.float64(t) - infra_cv['tbar_c']) * infra_cv['beta_c']).astype(np.float32)
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
    def Dtil_b(t, era=True):
        return (w1*Dhat_era(t) + w2*S + w3*trendex(t)).astype(np.float32)
    Sm = get_smoother(smooth_sigma)
    def pass_kalman(i, tm, direction):
        f0 = AF[i] - Dtil_b(i)
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
        x = Sm @ x
        dT = Dtil_b(tm)
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
    return pred

# ---------- M2 mirror ----------
val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
M2_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2, 24173:3, 24176:3, 24177:3,
           24178:3, 24179:3, 24180:3, 24181:3, 24182:3, 24183:3, 24187:4}
months = set(M2_ANCHORS) | set(M2_RUNS)
mir = val_all[val_all['t_abs'].isin(months)].copy()
mir['masked'] = mir['t_abs'].isin(M2_RUNS)
mir.loc[mir['masked'], 'TWS_t'] = np.nan

log("running pipeline on M2 mirror ...")
pipe_pred = run_masked(mir, val_tws_visible)
mm = mir['masked'].values & np.isfinite(pipe_pred) & np.isfinite(mir['target'].values)
y = mir['target'].values
log(f"  pipeline M2: {np.sqrt(np.mean((pipe_pred[mm]-y[mm])**2)):.4f}  (ref 0.6688)")

# ---------- external estimates at target months ----------
log("computing external estimates ...")
la_m = mir['lat'].values; lo_m = mir['lon'].values; cc_m = mir['cc'].values
ta_m = mir['t_abs'].values
ym_of_t = ((ta_m + 1)//12)*100 + ((ta_m+1) % 12 + 1)
ext = {}
for kind in ['gravis', 'costg', 'csr']:
    ext[kind] = ext_estimate(kind, cc_m, la_m, lo_m, ym_of_t)
    m2 = mm & np.isfinite(ext[kind])
    if m2.sum():
        rm = np.sqrt(np.mean((ext[kind][m2]-y[m2])**2))
        r = np.corrcoef(ext[kind][m2], y[m2])[0,1]
        log(f"  {kind} alone: RMSE={rm:.4f} corr={r:+.4f} n={m2.sum():,}")
ext_ens = np.nanmean(np.column_stack([ext[k] for k in ext]), axis=1)

def blend(cols, mask):
    X = np.column_stack(cols)[mask]
    X = np.column_stack([X, np.ones(len(X))])
    c = np.linalg.solve(X.T@X + 1e-6*np.eye(X.shape[1]), X.T@y[mask])
    return float(np.sqrt(np.mean((X@c - y[mask])**2)))

m_all = mm & np.isfinite(ext_ens)
print(f"\nrows with ext available: {m_all.sum():,} / {mm.sum():,}")
print(f"(a) pipeline alone:            {np.sqrt(np.mean((pipe_pred[m_all]-y[m_all])**2)):.4f}")
for kind in ['gravis','costg','csr']:
    mk = mm & np.isfinite(ext[kind])
    print(f"(b) {kind} alone:              {np.sqrt(np.mean((ext[kind][mk]-y[mk])**2)):.4f}")
mk = mm & np.isfinite(ext_ens)
print(f"(c) ext ensemble alone:        {np.sqrt(np.mean((ext_ens[mk]-y[mk])**2)):.4f}")
print(f"(d) pipeline + gravis:         {blend([pipe_pred, ext['gravis']], m_all):.4f}")
print(f"(e) pipeline + costg:          {blend([pipe_pred, ext['costg']], m_all):.4f}")
print(f"(f) pipeline + csr:            {blend([pipe_pred, ext['csr']], m_all):.4f}")
print(f"(g) pipeline + ext_ens:        {blend([pipe_pred, ext_ens], m_all):.4f}")
print(f"(h) pipeline + all 3 ext:      {blend([pipe_pred, ext['gravis'], ext['costg'], ext['csr']], m_all):.4f}")

# error correlation diagnostic
mk = mm & np.isfinite(ext_ens)
e1 = pipe_pred[mk]-y[mk]; e2 = ext_ens[mk]-y[mk]
print(f"\nerror corr pipeline vs ext_ens: {np.corrcoef(e1, e2)[0,1]:+.4f}")
