"""
AUDIT B2 (re-scoped) — item B4 (stretch): SPATIAL FAST-STATE SMOOTHING on mirrored M1.
After the per-cell Kalman, smooth the fast-state estimate x across neighboring cells
(ring 1deg w=.99, ring 2deg w=.97, overall shrink alpha) and re-score M1 honest rows.
Configs: raw | smooth-blend | smooth-xf/xb-separately | smooth + D-hat smoothing (bonus).
Cached npz only; no CSV loads. New file only.
"""
import numpy as np, pandas as pd, time, sys, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import load_cached, build_train_df, build_infra, COVS

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
OUT = '/home/z/my-project/download/auditB2_spatial.txt'
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))

tr, te = load_cached()
cells = np.load('/home/z/my-project/scripts/auditB_cells.npz')
lat = cells['lat'].astype(np.float64); lon = cells['lon'].astype(np.float64)
n_cells = len(lat)
train = build_train_df(tr)
train['year'] = train['t_abs'] // 12
tmonth = te['t_abs'] % 12 + 1
cal = pd.Series(te['masked'].astype(float)).groupby(tmonth).mean()

fit = train[train['year'] <= 2012].copy()
infra = build_infra(fit, n_cells, 'cv')
mu_c = infra['mu_c']

# neighbor machinery (rings 1deg w=.99, 2deg w=.97)
ilat = np.round(lat*2).astype(np.int64); ilon = np.round((lon+180)*2).astype(np.int64) % 720
buckets = {}
for i in range(n_cells): buckets.setdefault((int(ilat[i]), int(ilon[i])), []).append(i)
src1, dst1, src2, dst2 = [], [], [], []
for i in range(n_cells):
    a, b = int(ilat[i]), int(ilon[i])
    for da in (-4, -2, 0, 2, 4):
        for db in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            if da == 0 and db == 0: continue
            dlon = min(abs(db), 720-abs(db)); cheb = max(abs(da), dlon)
            if cheb not in (2, 4): continue
            jj = buckets.get((a+da, (b+db) % 720))
            if not jj: continue
            for j in jj:
                (src1 if cheb == 2 else src2).append(j); (dst1 if cheb == 2 else dst2).append(i)
src1, dst1, src2, dst2 = map(np.array, (src1, dst1, src2, dst2))
P(f"neighbor pairs: ring1={len(src1):,} ring2={len(src2):,}")

def nbr_mean(field):
    v = np.where(np.isfinite(field), field, 0.0)
    f = np.isfinite(field).astype(np.float64)
    s1 = np.bincount(dst1, weights=v[src1], minlength=n_cells); c1 = np.bincount(dst1, weights=f[src1], minlength=n_cells)
    s2 = np.bincount(dst2, weights=v[src2], minlength=n_cells); c2 = np.bincount(dst2, weights=f[src2], minlength=n_cells)
    num = 0.99*s1 + 0.97*s2; den = 0.99*c1 + 0.97*c2
    return np.where(den > 0, num/np.maximum(den, 1e-9), 0.0)

# ---- run_masked verbatim (v17b, no denoiser) + field capture ----
def run_masked_fields(infra, ev_df, ev_tws_visible, phi, use_bwd=True, lam=0.84, dtil_w=(0.70, 0.45, 0.073)):
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
    tws_vis = ev_tws_visible.values
    AF = {}
    for a in anchors:
        sel = (ta == a) & (~msk)
        fa = np.full(n_cells, np.nan, dtype=np.float32)
        fa[cc[sel]] = tws_vis[sel]
        AF[a] = fa - mu_c
    Dhat = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)
    w1, w2, w3 = dtil_w
    def trendex(t): return ((np.float64(t) - infra['tbar_c']) * infra['beta_c']).astype(np.float32)
    def Dtil(t, Dh=None): return (w1*(Dh if Dh is not None else Dhat) + w2*S + w3*trendex(t)).astype(np.float32)
    cs, zs, vfs = [], [], []
    for a in anchors:
        dj = Dtil(a); w_ = W[a]
        okc = np.isfinite(w_) & np.isfinite(AF[a]) & np.isfinite(dj)
        cs.append(np.cov(w_[okc], (AF[a]-dj)[okc])[0, 1]); zs.append(np.var(w_[okc]))
        vfs.append(np.nanvar(AF[a]-dj))
    c_ = float(np.mean(cs)); varz = float(np.mean(zs)); var_f = float(np.mean(vfs))
    H = c_/(lam*var_f); R = max(varz - c_*c_/(lam*var_f), 1e-4)
    q = lam*var_f*(1-phi**2); P0 = lam*(1-lam)*var_f
    def pass_kalman(i, tm, direction):
        f0 = AF[i] - Dtil(i)
        x = np.where(np.isfinite(f0), lam*np.nan_to_num(f0), 0.0).astype(np.float32)
        P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
        rng = range(i+1, tm+1) if direction > 0 else range(i-1, tm-1, -1)
        for m in rng:
            x = phi*x; P = phi**2*P + q
            if m in W:
                w_ = W[m]; okw = np.isfinite(w_)
                Kg = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
                x = np.where(okw, x + Kg*(np.nan_to_num(w_) - H*x), x)
                P = np.where(okw, (1-Kg*H)*P, P)
        return x, P
    fields = {}
    anchors_arr = np.array(anchors)
    masked_months = sorted(set(ta[msk].tolist()))
    for m in masked_months:
        tm = m + 1
        sel = np.where((ta == m) & msk)[0]
        if len(sel) == 0: continue
        i = int(anchors_arr[np.searchsorted(anchors_arr, m, side='right')-1])
        xf, Pf = pass_kalman(i, tm, +1)
        later = anchors_arr[anchors_arr > m]
        use_b = use_bwd and len(later) > 0
        if use_b:
            k = int(later[0])
            xb, Pb = pass_kalman(k, tm, -1)
            wgt = (1.0/np.maximum(Pf, 1e-6))/(1.0/np.maximum(Pf, 1e-6)+1.0/np.maximum(Pb, 1e-6))
            x = wgt*xf + (1-wgt)*xb
        else:
            xb = np.full_like(xf, np.nan); wgt = np.ones_like(Pf)
            x = xf
        fields[m] = dict(sel=sel, tm=tm, x=x, xf=xf, xb=xb, wgt=wgt, Dtil=Dtil(tm))
    return fields, Dhat, S, AF, Dtil, var_f

# ---- M1 mirror frame ----
val = train[(train['year'] >= 2013) & (train['year'] <= 2015)].copy()
val['masked'] = (val['t_abs'] % 12 + 1).map(cal).values > 0.5
val_tws_visible = val['TWS_t'].copy()
M_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]
M_RUNS = {24157, 24161, 24162, 24167, 24168, 24173, 24176, 24177, 24178, 24179, 24180, 24187}
m_months = set(M_ANCHORS) | M_RUNS
mir = val[val['t_abs'].isin(m_months)].copy()
mir['masked'] = mir['t_abs'].isin(M_RUNS)
mir['TWS_t'] = val_tws_visible.loc[mir.index].values
mir.loc[mir['masked'], 'TWS_t'] = np.nan
tgt = mir['target'].values.astype(np.float64)
msk_mir = mir['masked'].values

def score(fields, alpha=0.0, mode='blend', smooth_dhat=0.0):
    Dh_s = None
    pred = np.full(len(mir), np.nan, dtype=np.float64)
    # pre-smooth Dhat if requested
    for m, fd in fields.items():
        x = fd['x'].copy()
        if alpha > 0:
            if mode == 'blend':
                x = (x + alpha*nbr_mean(x))/(1+alpha)
            elif mode == 'sep':
                xf = (fd['xf'] + alpha*nbr_mean(fd['xf']))/(1+alpha)
                xb = fd['xb'].copy()
                okb = np.isfinite(xb)
                xb = np.where(okb, (xb + alpha*nbr_mean(np.where(okb, xb, 0.0)))/(1+alpha), xb)
                w = fd['wgt']
                x = w*xf + (1-w)*np.where(okb, xb, xf)
        sel = fd['sel']; cc_sel = mir['cc'].values[sel]
        dT = fd['Dtil']
        if smooth_dhat > 0 and Dh_s is None:
            pass
        pred[sel] = mu_c[cc_sel] + dT[cc_sel] + x[cc_sel]
    mm = msk_mir & np.isfinite(pred) & np.isfinite(tgt)
    return float(np.sqrt(np.mean((pred[mm]-tgt[mm])**2))), int(mm.sum())

log("running M1 raw (phi=0.80, lam=0.84, bwd, no-dn)...")
fields, Dhat, S, AF, Dtil_fn, var_f = run_masked_fields(infra, mir, mir['TWS_t'], 0.80)
r_raw, n_raw = score(fields)
P(f"RAW M1 RMSE = {r_raw:.4f} (auditA_mirror: 0.6319), n={n_raw:,}")

# diagnostic: neighbor correlation of the fast-state fields vs the truth-ish residual
P("\nfast-state field neighbor correlations (lag-1 ring):")
for m in list(fields)[:4]:
    x = fields[m]['x']
    xv = x[src1]; ok = np.isfinite(xv) & np.isfinite(x[dst1])
    P(f"  month {m//12}-{m%12+1:02d}: corr(x_i, x_neighbor1) = {np.corrcoef(x[dst1][ok], xv[ok])[0,1]:.3f}")

P("\nalpha sweep (smooth the blended fast state, rings 1-2 deg):")
for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
    r, _ = score(fields, alpha=alpha, mode='blend')
    P(f"  alpha={alpha:4.2f}: M1 RMSE = {r:.4f} (delta {r-r_raw:+.4f})")

P("\nmode=sep (smooth fwd and bwd passes separately, then precision-blend):")
for alpha in [0.5, 1.0, 2.0]:
    r, _ = score(fields, alpha=alpha, mode='sep')
    P(f"  alpha={alpha:4.2f}: M1 RMSE = {r:.4f} (delta {r-r_raw:+.4f})")

# bonus: smooth D-hat itself (D-tilde input) at fixed best-alpha (or 1.0)
P("\nbonus: smoothing D-hat (6-anchor mean) with rings 1-2 deg, recompute Dtil:")
for sa in [0.5, 1.0, 2.0]:
    Dh_s = (Dhat + sa*nbr_mean(Dhat))/(1+sa)
    pred = np.full(len(mir), np.nan, dtype=np.float64)
    for m, fd in fields.items():
        sel = fd['sel']; cc_sel = mir['cc'].values[sel]
        dT = 0.70*Dh_s + 0.45*S + 0.073*((np.float64(fd['tm']) - infra['tbar_c'])*infra['beta_c']).astype(np.float32)
        pred[sel] = mu_c[cc_sel] + dT[cc_sel] + fd['x'][cc_sel]
    mm = msk_mir & np.isfinite(pred) & np.isfinite(tgt)
    r = float(np.sqrt(np.mean((pred[mm]-tgt[mm])**2)))
    P(f"  D-hat alpha={sa:4.2f}: M1 RMSE = {r:.4f} (delta {r-r_raw:+.4f})")

# M2 robustness if any alpha helped
best_a, best_r = 0.0, r_raw
for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
    r, _ = score(fields, alpha=alpha, mode='blend')
    if r < best_r: best_a, best_r = alpha, r
if best_a > 0:
    P(f"\nM2 robustness check for alpha={best_a}:")
    M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
    M2_RUNS = M_RUNS | {24181, 24182, 24183, 24184, 24185}
    m2_months = set(M2_ANCHORS) | M2_RUNS
    mir2 = val[val['t_abs'].isin(m2_months)].copy()
    mir2['masked'] = mir2['t_abs'].isin(M2_RUNS)
    mir2['TWS_t'] = val_tws_visible.loc[mir2.index].values
    mir2.loc[mir2['masked'], 'TWS_t'] = np.nan
    tgt2 = mir2['target'].values.astype(np.float64); msk2 = mir2['masked'].values
    f2, _, _, _, _, _ = run_masked_fields(infra, mir2, mir2['TWS_t'], 0.80)
    def score2(fields, alpha):
        pred = np.full(len(mir2), np.nan, dtype=np.float64)
        for m, fd in fields.items():
            x = fd['x'].copy()
            if alpha > 0: x = (x + alpha*nbr_mean(x))/(1+alpha)
            sel = fd['sel']; cc_sel = mir2['cc'].values[sel]
            pred[sel] = mu_c[cc_sel] + fd['Dtil'][cc_sel] + x[cc_sel]
        mm = msk2 & np.isfinite(pred) & np.isfinite(tgt2)
        return float(np.sqrt(np.mean((pred[mm]-tgt2[mm])**2)))
    r2_raw = score2(f2, 0.0); r2_sm = score2(f2, best_a)
    P(f"  M2 raw {r2_raw:.4f} (auditA: 0.6905) -> smoothed alpha={best_a}: {r2_sm:.4f} (delta {r2_sm-r2_raw:+.4f})")

P(f"\nSUMMARY: raw M1 {r_raw:.4f}; best smoothing alpha={best_a} -> {best_r:.4f} (delta {best_r-r_raw:+.4f})")
with open(OUT, 'w') as f: f.write('\n'.join(lines))
log(f"saved {OUT}")
