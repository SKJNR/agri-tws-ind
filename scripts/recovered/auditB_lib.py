"""
AUDIT B library — exact replication of rebuild_v17b.py pipeline with hooks.
No original files modified.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84
W1, W2, W3 = 0.70, 0.45, 0.073


def load_cached():
    tr = np.load('/home/z/my-project/scripts/auditB_train.npz')
    te = np.load('/home/z/my-project/scripts/auditB_test.npz')
    return tr, te


def build_train_df(tr):
    df = pd.DataFrame({'cc': tr['cc'], 't_abs': tr['t_abs'], 'TWS_t': tr['TWS'],
                       'target': tr['target']})
    for j, c in enumerate(COVS):
        df[c] = tr['covs'][:, j]
    df['ym'] = (df['t_abs']//12)*100 + (df['t_abs']%12) + 1
    return df


def build_infra(fit_df, n_cells, label='', keep_PCs=True):
    """Verbatim copy of rebuild_v17b.build_infra (plus keeping full Vt for K experiments)."""
    d = {}
    yms = np.sort(fit_df['ym'].unique()); T = len(yms)
    ym_to_i = {int(v): i for i, v in enumerate(yms)}
    t_abs_arr = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)
    F = np.full((T, n_cells), np.nan, dtype=np.float32)
    F[fit_df['ym'].map(ym_to_i).values, fit_df['cc'].values] = fit_df['TWS_t'].values
    with np.errstate(all='ignore'):
        d['mu_c'] = np.where(np.isfinite(np.nanmean(F, axis=0)), np.nanmean(F, axis=0), 0.0).astype(np.float32)
    d['clim'] = fit_df.groupby('cc')[COVS].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
    Z = fit_df[COVS].values.astype('float32'); yv = fit_df['TWS_t'].values.astype('float32')
    Z1 = np.column_stack([Z, np.ones(len(Z))])
    okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
    d['coef'] = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])
    F64 = F.astype(np.float64)
    ok_t = np.isfinite(F64)
    t_mat = np.where(ok_t, t_abs_arr[:, None], np.nan)
    tbar_c = np.nanmean(t_mat, axis=0)
    td = t_mat - tbar_c[None, :]
    sxx = np.nansum(td*td, axis=0)
    beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24), np.nansum(td*F64, axis=0)/np.where(sxx>0, sxx, 1), 0.0).astype(np.float32)
    d['tbar_c'], d['beta_c'] = tbar_c, beta_c
    full = ~np.isnan(F).any(axis=0)
    A_dt = F64 - d['mu_c'][None,:] - (t_abs_arr[:,None]-tbar_c[None,:])*beta_c[None,:]
    A = A_dt[:, full]; A = A - A.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    d['V'] = Vt[:200].T.astype(np.float32)
    d['Vt_full'] = Vt.astype(np.float32) if keep_PCs else None
    d['full'] = full
    d['T_fit'] = T
    d['A_dt_full_cells'] = A_dt[:, full]
    d['t_abs_fit'] = t_abs_arr
    return d


def dn(infra, field, use_denoise=True, K=None, mode='proj'):
    """Generalized denoiser. mode: 'proj' (v17b), 'shrink' (50/50 raw+proj)."""
    if not use_denoise or K == 0:
        return field
    out = field.copy()
    x = field[infra['full']]
    okx = np.isfinite(x)
    V = infra['V'] if K is None else infra['Vt_full'][:K].T.astype(np.float32)
    xp = V @ (V.T @ np.where(okx, x, 0.0))
    if mode == 'shrink':
        xp = 0.5*(np.where(okx, x, 0.0) + xp)
    out[infra['full']] = np.where(okx, xp, x)
    return out


def run_masked2(infra, ev_df, ev_tws_visible, phi, use_bwd=True, use_denoise=True,
                dhat_denoise=False, blend='precision', bwd_max_gap=None, dtil_w=(W1,W2,W3),
                lam=LAM_F, dn_K=None, dn_mode='proj', dn_where='both', raw_HR=False,
                diag=False):
    """Verbatim run_masked from rebuild_v17b + hooks:
      dn_K/dn_mode: denoiser rank/mode; dn_where: 'both'|'W'|'init'; raw_HR: estimate H,R from raw fields.
      diag: return per-row diagnostics dict."""
    mu_c = infra['mu_c']; n = len(ev_df)
    ta = ev_df['t_abs'].values; cc = ev_df['cc'].values; msk = ev_df['masked'].values
    Zv = ev_df[COVS].values.astype('float32')
    ok = np.isfinite(Zv).all(axis=1)
    cov_est = np.full(n, np.nan, dtype=np.float32)
    cov_est[ok] = np.column_stack([Zv[ok], np.ones(ok.sum())]) @ infra['coef']
    cov_field = {}
    for m in np.sort(ev_df['t_abs'].unique()):
        selm = (ta == m) & ok
        fm = np.full(infra['mu_c'].shape[0], np.nan, dtype=np.float32)
        fm[cc[selm]] = cov_est[selm]
        cov_field[int(m)] = fm - mu_c
    all_m = np.array(sorted(cov_field.keys()))
    S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
    W = {int(m): cov_field[m] - S for m in all_m}
    mfrac = ev_df.groupby('t_abs')['masked'].mean()
    anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
    tws_vis = ev_tws_visible.values
    AF = {}
    for a in anchors:
        sel = (ta == a) & (~msk)
        fa = np.full(infra['mu_c'].shape[0], np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    if dhat_denoise:
        Dhat = np.nanmean(np.array([dn(infra, AF[a], use_denoise, dn_K, dn_mode) for a in anchors]), axis=0)
    else:
        Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    w1, w2, w3 = dtil_w
    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil(t): return (w1*Dhat + w2*S + w3*trendex(t)).astype(np.float32)

    def dnW(m):
        if not use_denoise or dn_where in ('init',):
            return W[m]
        return dn(infra, W[m], use_denoise, dn_K, dn_mode)
    def dnF(f0):
        if not use_denoise or dn_where in ('W',):
            return f0
        return dn(infra, f0, use_denoise, dn_K, dn_mode)

    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a)
        w_ = dnW(a) if raw_HR is False else W[a]
        okc = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[okc], (AF[a]-dj)[okc])[0,1]); zs.append(np.var(w_[okc]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(lam*var_f); R = max(varz - c_*c_/(lam*var_f), 1e-4)
    q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f

    def pass_kalman(i, tm, direction):
        f0 = dnF(AF[i] - Dtil(i))
        x = np.where(np.isfinite(f0), lam*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        rng = range(i+1, tm+1) if direction > 0 else range(i-1, tm-1, -1)
        for m in rng:
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = dnW(m); okw = np.isfinite(w_)
                Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
                P = np.where(okw, (1-Kg*H)*P, P)
        return x, P

    pred = np.full(n, np.nan, dtype=np.float64)
    DG = dict(rows=[], m=[], gap_back=[], gap_fwd=[], xf=[], xb=[], wgt=[], Dtil=[], dhat_cells=[])
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
            if blend == 'precision':
                wgt = (1.0/np.maximum(Pf,1e-6))/(1.0/np.maximum(Pf,1e-6)+1.0/np.maximum(Pb,1e-6))
            else:
                wgt = np.float32(blend)
            x = wgt*xf + (1-wgt)*xb
        else:
            xb = np.full_like(xf, np.nan); wgt = np.ones_like(Pf).astype(np.float64)
            x = xf
        dT = Dtil(tm)
        pred[sel] = mu_c[cc[sel]] + dT[cc[sel]] + x[cc[sel]]
        if diag:
            DG['rows'].append(sel); DG['m'].append(m)
            DG['gap_back'].append(tm - i)
            DG['gap_fwd'].append(int(later[0]) - tm if len(later) else -1)
            DG['xf'].append(xf[cc[sel]]); DG['xb'].append(xb[cc[sel]])
            DG['wgt'].append(np.asarray(wgt)[cc[sel]])
            DG['Dtil'].append(dT[cc[sel]])
    diag_out = None
    if diag:
        diag_out = dict(meta=pd.DataFrame({'m': DG['m'], 'gap_back': DG['gap_back'], 'gap_fwd': DG['gap_fwd']}),
                        xf=np.concatenate(DG['xf']), xb=np.concatenate(DG['xb']),
                        wgt=np.concatenate(DG['wgt']), Dtil=np.concatenate(DG['Dtil']),
                        rows=np.concatenate(DG['rows']), rows_list=DG['rows'],
                        H=H, R=R, var_f=var_f, varz=varz, c_=c_, q=q, P0=P0, Dhat=Dhat, S=S,
                        anchors=anchors, W=W, AF=AF, Dtil_fn=Dtil)
    return (pred, diag_out) if diag else pred


# ---------- k0 (verbatim v17b) ----------
def build_k0_features(df, infra, n_cells=None):
    has_target = 'target' in df.columns
    cols = ['cc','t_abs','TWS_t']+(['target'] if has_target else [])+COVS
    L = df[cols].copy(); L['t_next'] = L['t_abs']+1
    R = df[['cc','t_abs']+COVS].rename(columns={'t_abs':'t_next', **{c: c+'_nxt' for c in COVS}})
    mg = L.merge(R, on=['cc','t_next'], how='left')
    ccm = mg['cc'].values
    Xt = np.column_stack([
        mg['TWS_t'].values - infra['mu_c'][ccm],
        *[mg[c].values - infra['clim'][ccm, j] for j, c in enumerate(COVS)],
        *[mg[c+'_nxt'].values - infra['clim'][ccm, j] for j, c in enumerate(COVS)],
    ]).astype(np.float32)
    if has_target:
        y = (mg['target'].values - infra['mu_c'][ccm]).astype(np.float32)
    else:
        y = np.full(len(mg), np.nan, dtype=np.float32)
    yr = mg['t_abs'].values // 12
    return Xt, y, yr, mg['TWS_t'].notna().values, mg


def k0_linear(Xtr, ytr, wtr, Xev):
    has_nxt_tr = np.isfinite(Xtr[:, 6:11]).all(axis=1)
    sw = np.sqrt(wtr)
    Xtr0 = np.nan_to_num(Xtr, nan=0.0)
    A = Xtr0[has_nxt_tr]*sw[has_nxt_tr,None]
    coefF = np.linalg.solve(A.T@A + 1e-3*np.eye(11), A.T@(ytr[has_nxt_tr]*sw[has_nxt_tr]))
    colsR = np.array([0,1,2,3,4,5])
    A_r = Xtr0[:, colsR]*sw[:,None]
    coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(6), A_r.T@(ytr*sw))
    has_nxt_ev = np.isfinite(Xev[:, 6:11]).all(axis=1)
    Xe = np.nan_to_num(Xev, nan=0.0)
    pred = np.empty(len(Xev), dtype=np.float32)
    pred[has_nxt_ev] = Xe[has_nxt_ev] @ coefF
    pred[~has_nxt_ev] = Xe[~has_nxt_ev][:, colsR] @ coefR
    return pred, coefF, coefR


def k0_lgb(Xtr, ytr, wtr, Xev, rounds=400):
    import lightgbm as lgb
    params = dict(objective='regression', metric='rmse', num_leaves=63,
                  learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, num_threads=4,
                  seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    bst = lgb.train(params, dtr, num_boost_round=rounds)
    return bst.predict(Xev), bst
