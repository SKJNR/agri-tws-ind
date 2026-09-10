"""
a14b_common.py — shared infrastructure for the latent-factor Kalman filter (Agent 14-b).

Replaces the per-cell scalar Kalman (submission_v4.py) with a K-dim factor Kalman on the
detrended anomaly field's PCA scores. Conventions (cell grid cc, t_abs, cov regression,
D-tilde) follow submission_v4.py / cv_lb_correlation.py exactly.

MODEL
-----
Cell space (n=15,715 cells; full = cells with complete history in the fit window):
    TWS(c,t) = mu_c + trend_c(t) + FAST(c,t) + eps(c,t)
    a(c,t)   = TWS - mu - trend          (detrended anomaly)
    scores   s_k(t) = v_k^T a_t  = x_k(t) + e_k(t)
        x_k: persistent factor (AR(1), phi_k, stationary var Vx_k)
        e_k: month-white observation noise in score space, var Rn_k
    Vx_k, Rn_k identified from the lag structure of the scores:
        C_l = Cov(s_t, s_{t+l}) = Vx_k * phi_k^l  (white noise cancels at l>=1)
        V_s = Vx_k + Rn_k
    => phi_k from log-linear fit of C_l on l (l=1..4), Vx_k = exp(intercept),
       Rn_k = V_s - Vx_k.  This fixes Task-11 bug #2 (R measured in score space).

OBSERVATIONS
------------
  TWS anchor (full field visible):  y = V^T (AF[a] - Dtil(a)) = x + e + d
      R_anchor_k = Rn_k + VarD_k, VarD_k = variance of the D-tilde error projected
      on factor k, estimated from the scatter of y across the era's anchors:
      Var_anchor(y_k) - Vx_k - Rn_k  (floored at 0, smoothed over k).
  Covariates (every month with rows): z = V^T (cov_field_m - S_era)
      per-factor regression z_k = H_k x_k + r_k fitted on ALL fit months
      (fixes Task-11 bug #1: no anchor-only calibration).

FILTER
------
Forward Kalman (diagonal in factor space), per-month gap-aware propagation
(P <- phi^{2g} P + Vx(1-phi^{2g})), cov updates each observed month, TWS anchor
updates at anchor months. Backward information = RTS smoother from the next anchor
(also implemented: independent backward filter + precision blend, for comparison).
"""
import numpy as np
import pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']


# --------------------------------------------------------------------------- data
def load_train():
    train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVS)
    train['time'] = pd.to_datetime(train['time'])
    for c in ['TWS_t', 'target'] + COVS:
        train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
    train['cc'] = (train['lat'].round(2).astype(str) + '_' + train['lon'].round(2).astype(str)) \
        .astype('category').cat.codes.astype('int32')
    train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
    train['t_abs'] = (train['ym'] // 100) * 12 + (train['ym'] % 100) - 1
    return train


def load_test(train):
    test = pd.read_csv(f'{DATA}/Test (2).csv')
    test['time'] = pd.to_datetime(test['time'])
    for c in ['TWS_t'] + COVS:
        test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
    codes = train[['cc', 'lat', 'lon']].drop_duplicates('cc').sort_values('cc')
    pos = {(int(round(float(la) * 1000)), int(round(float(lo) * 1000))): int(cc)
           for cc, la, lo in zip(codes['cc'], codes['lat'], codes['lon'])}
    test['cc'] = [pos.get((int(round(float(a) * 1000)), int(round(float(b) * 1000))), -1)
                  for a, b in zip(test['lat'], test['lon'])]
    assert (test['cc'] >= 0).all()
    test['ym'] = test['time'].dt.year * 100 + test['time'].dt.month
    test['t_abs'] = (test['ym'] // 100) * 12 + (test['ym'] % 100) - 1
    test['masked'] = test['TWS_t_masked'].astype(bool)
    return test


# --------------------------------------------------------------------------- factor model
class FactorModel:
    """PCA basis + per-factor dynamics + cov observation model, built on fit rows only.

    rows: DataFrame of train rows restricted to the fit period (year<=2012 for CV,
          all train for the test anchor-LOO harness).
    """

    def __init__(self, rows, n_cells, Kmax=None, verbose=True):
        self.n_cells = n_cells
        yms = np.sort(rows['ym'].unique())
        self.months = np.array([(int(v) // 100) * 12 + (int(v) % 100) - 1 for v in yms])
        T = len(yms)
        ym_to_i = {int(v): i for i, v in enumerate(yms)}

        # ---- scalar infra (same formulas as cv_lb_correlation.py) ----
        self.mu_c = rows.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
        self.clim = rows.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')
        Z = rows[COVS].values.astype('float32')
        yv = rows['TWS_t'].values.astype('float32')
        Z1 = np.column_stack([Z, np.ones(len(Z))])
        okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
        self.coef = np.linalg.solve(Z1[okz].T @ Z1[okz] + 1e-2 * np.eye(6), Z1[okz].T @ yv[okz])

        F = np.full((T, n_cells), np.nan, dtype=np.float32)
        F[rows['ym'].map(ym_to_i).values, rows['cc'].values] = rows['TWS_t'].values
        F64 = F.astype(np.float64)
        ok_t = np.isfinite(F64)
        t_mat = np.where(ok_t, self.months[:, None].astype(np.float64), np.nan)
        self.tbar_c = np.nanmean(t_mat, axis=0)
        td = t_mat - self.tbar_c[None, :]
        sxy = np.nansum(td * F64, axis=0)
        sxx = np.nansum(td * td, axis=0)
        self.beta_c = np.where((sxx > 100) & (ok_t.sum(axis=0) >= 24),
                               sxy / np.where(sxx > 0, sxx, 1), 0.0).astype('float32')

        # ---- detrended anomaly field ----
        TD = self.months[:, None].astype(np.float64) - self.tbar_c[None, :]
        A = F64 - self.mu_c[None, :] - TD * self.beta_c[None, :]
        self.full = ~np.isnan(A).any(axis=0)
        self.nfull = int(self.full.sum())
        A_f = A[:, self.full]
        A_f = A_f - A_f.mean(axis=0, keepdims=True)
        if verbose:
            print(f"[FactorModel] fit months={T} full cells={self.nfull}/{n_cells}", flush=True)

        # ---- PCA basis (SVD), keep up to rank ----
        rank = min(T - 1, self.nfull)
        Kall = rank if Kmax is None else min(Kmax, rank)
        _, S, Vt = np.linalg.svd(A_f, full_matrices=False)
        self.V = Vt[:Kall].T.copy()                # (nfull, Kall)
        self.scores = A_f @ self.V                 # (T, Kall) = noisy factor scores
        self.evr = (S ** 2 / (S ** 2).sum())
        if verbose:
            print(f"[FactorModel] rank={rank}, K={Kall}, cumEV(K)={self.evr[:Kall].sum():.4f}", flush=True)

        # ---- per-factor moments: phi, Vx, Rn from lag structure ----
        sc = self.scores - self.scores.mean(axis=0, keepdims=True)
        n = T
        self.Vs = (sc ** 2).sum(axis=0) / (n - 1)                 # Var(s_k)
        # lag covariance C_l over (t, t+l) both-present pairs (any gap in between is fine)
        max_lag = 4
        Cl = np.zeros((max_lag, Kall))
        for l in range(1, max_lag + 1):
            a, b = sc[:-l], sc[l:]
            m = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            am, bm = a[m] - a[m].mean(axis=0), b[m] - b[m].mean(axis=0)
            Cl[l - 1] = (am * bm).sum(axis=0) / max(m.sum() - 1, 1)
        # white-noise-in-score per factor from tail of the mode spectrum (fallback)
        # primary: Rn = Vs - Vx from the lag fit
        phi = np.zeros(Kall)
        Vx = np.zeros(Kall)
        for k in range(Kall):
            c = Cl[:, k]
            if c[0] <= 0 or not np.isfinite(c[0]):
                continue
            ls = np.arange(1, max_lag + 1)
            pos = c > 0
            if pos.sum() >= 2:
                b1, b0 = np.polyfit(ls[pos], np.log(c[pos]), 1)
                phi_k = float(np.clip(np.exp(b1), 0.02, 0.99))
                Vx_k = float(np.exp(b0))
            else:
                phi_k = 0.0
                Vx_k = 0.0
            phi[k], Vx[k] = phi_k, Vx_k
        Vx = np.clip(Vx, 0.0, self.Vs)
        Rn = np.clip(self.Vs - Vx, 1e-8, None)
        self.phi, self.Vx, self.Rn = phi, Vx, Rn
        self.q = Vx * (1.0 - phi ** 2)
        if verbose:
            print(f"[FactorModel] phi: top10={np.round(phi[:10], 3)}", flush=True)
            print(f"[FactorModel] Vx:  top10={np.round(Vx[:10], 1)}", flush=True)
            print(f"[FactorModel] Rn:  top10={np.round(Rn[:10], 3)}  mean(tail50)={Rn[-50:].mean():.3f}", flush=True)

        # ---- innovation cross-correlations (VAR check) ----
        # residuals of per-factor AR(1) on consecutive-month pairs
        ia, ib = [], []
        for i in range(T - 1):
            if self.months[i + 1] - self.months[i] == 1:
                ia.append(i); ib.append(i + 1)
        ia, ib = np.array(ia), np.array(ib)
        resid = sc[ib] - phi[None, :] * sc[ia]
        resid = resid - resid.mean(axis=0, keepdims=True)
        C_res = (resid.T @ resid) / max(len(ia) - 1, 1)
        d = np.sqrt(np.diag(C_res))
        Ccorr = C_res / np.outer(d, d)
        off = Ccorr - np.diag(np.diag(Ccorr))
        self.innov_corr = Ccorr
        if verbose:
            k50 = min(50, Kall)
            o50 = off[:k50, :k50]
            print(f"[FactorModel] innov x-corr (K={k50}): mean|off|={np.abs(o50).mean():.3f} "
                  f"max|off|={np.abs(o50).max():.3f}  (VAR needed if >> 0.15)", flush=True)

        # ---- cov observation model (pooled cov regression -> score space) ----
        cov_est = np.full(len(rows), np.nan, dtype='float32')
        okr = np.isfinite(Z).all(axis=1)
        cov_est[okr] = np.column_stack([Z[okr], np.ones(okr.sum())]) @ self.coef
        # per-month cov field (cell space)
        cov_field = np.full((T, n_cells), np.nan, dtype=np.float32)
        cov_field[rows['ym'].map(ym_to_i).values[okr], rows['cc'].values[okr]] = cov_est[okr]
        cov_field = cov_field - self.mu_c[None, :]
        S_fit = np.nanmean(cov_field, axis=0)
        z_fit = (cov_field - S_fit[None, :])[:, self.full] @ self.V      # (T, Kall)
        self.z_fit = z_fit
        self.S_fit = S_fit
        self.cov_field_fit = cov_field

        # ---- per-factor H, Rcov: z_k = H_k x_k + r_k on ALL fit months ----
        zm = z_fit - z_fit.mean(axis=0, keepdims=True)
        Czs = (zm * sc).sum(axis=0) / (n - 1)
        Vz = (zm ** 2).sum(axis=0) / (n - 1)
        H = np.zeros(Kall)
        Rcov = Vz.copy()
        sig = 2.5 / np.sqrt(n)
        for k in range(Kall):
            if Vx[k] <= 0:
                continue
            rho = Czs[k] / np.sqrt(Vz[k] * self.Vs[k])
            if abs(rho) < sig:
                continue
            H[k] = Czs[k] / Vx[k]
            Rcov[k] = max(Vz[k] - H[k] * Czs[k], 0.02 * Vz[k])
        self.H, self.Rcov = H, Rcov
        if verbose:
            act = H != 0
            print(f"[FactorModel] cov obs: active factors={act.sum()}/{Kall}; "
                  f"H top10={np.round(H[:10], 3)}", flush=True)
            print(f"[FactorModel] corr(z,s) top10={np.round((Czs/np.sqrt(Vz*self.Vs))[:10], 3)}", flush=True)

        # ---- extended loadings for non-full cells (regress on scores) ----
        # v_c,k = Cov(a_c, s_k) / Vs_k  (scores are mutually orthogonal over time)
        nf = ~self.full
        self.V_ext = np.zeros((n_cells, Kall), dtype=np.float64)
        self.V_ext[self.full] = self.V
        if nf.any():
            A_nf = A[:, nf]
            msk = np.isfinite(A_nf)
            X = np.where(msk, A_nf, 0.0)
            counts = msk.sum(axis=0)
            num = sc.T @ X                                   # (K, n_nf)
            den = np.maximum(counts[None, :] - 1, 1)
            cov_cs = num / den
            okc = counts >= 24
            with np.errstate(divide='ignore', invalid='ignore'):
                vload = np.where(okc[None, :], cov_cs / np.maximum(self.Vs[:, None], 1e-6), 0.0)
            # kill tail factors with no persistent signal
            vload *= (Vx[:, None] > 4 * Rn[:, None])
            self.V_ext[nf] = vload.T
            if verbose:
                print(f"[FactorModel] non-full cells: {int(nf.sum())}, loadings fit for "
                      f"{int(okc.sum())}", flush=True)

    # -------------------------------------------------- per-K parameter slice
    def params(self, K):
        K = min(K, self.V.shape[1])
        return dict(K=K, V=self.V[:, :K].copy(), V_ext=self.V_ext[:, :K].copy(),
                    phi=self.phi[:K].copy(), Vx=self.Vx[:K].copy(), Rn=self.Rn[:K].copy(),
                    q=self.q[:K].copy(), H=self.H[:K].copy(), Rcov=self.Rcov[:K].copy(),
                    full=self.full)

    def trendex(self, t):
        return (np.float64(t) - self.tbar_c) * self.beta_c


# --------------------------------------------------------------------------- era helpers
def era_cov_scores(fm, era_rows, era_months):
    """Build cell-space cov fields + score-space z observations for an era (val or test).

    Returns dict: month -> dict(cov_field (n_cells,), z (Kall,)) plus S_era.
    """
    n_cells = fm.n_cells
    Zv = era_rows[COVS].values.astype('float32')
    okv = np.isfinite(Zv).all(axis=1)
    cov_est = np.full(len(era_rows), np.nan, dtype='float32')
    cov_est[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ fm.coef
    cov_field = {}
    zsc = {}
    ta = era_rows['t_abs'].values
    cc = era_rows['cc'].values
    for m in era_months:
        selm = (ta == m) & okv
        f = np.full(n_cells, np.nan, dtype=np.float32)
        f[cc[selm]] = cov_est[selm]
        f = f - fm.mu_c
        cov_field[int(m)] = f
    S = np.nanmean(np.array([cov_field[int(m)] for m in era_months]), axis=0)
    for m in era_months:
        g = cov_field[int(m)] - S
        zsc[int(m)] = fm.V.T @ g[fm.full]
    return dict(cov_field=cov_field, S=S, z=zsc)


def anchor_fields(era_rows, anchors, mu_c, n_cells, tws_col='TWS_t', visible=None):
    """AF[a] = TWS field at anchor a minus mu_c."""
    ta = era_rows['t_abs'].values
    cc = era_rows['cc'].values
    tws = era_rows[tws_col].values
    AF = {}
    for a in anchors:
        sel = (ta == a)
        if visible is not None:
            sel = sel & visible
        f = np.full(n_cells, np.nan, dtype=np.float32)
        f[cc[sel]] = tws[sel]
        AF[int(a)] = f - mu_c
    return AF


def estimate_VarD(ys, Vx, Rn, smooth=9, scale=1.0):
    """Per-factor variance of the D-tilde error from anchor-observation scatter.

    ys: (n_anchor, K) anchor score observations y_a = x_a + e_a + d_a.
    Var_anchor(y_k) ~ Vx_k + Rn_k + VarD_k  =>  VarD_k = scatter - Vx - Rn (>=0, smoothed).
    """
    var_anchor = ys.var(axis=0, ddof=1) if len(ys) > 1 else np.zeros(ys.shape[1])
    raw = var_anchor - Vx - Rn
    raw = np.clip(raw, 0.0, None)
    if smooth > 1:
        pad = smooth // 2
        rp = np.concatenate([raw[:pad][::-1], raw, raw[-pad:][::-1]]) if len(raw) > pad else raw
        ker = np.ones(smooth) / smooth
        raw = np.convolve(rp, ker, mode='valid')
    return raw * scale


# --------------------------------------------------------------------------- filters
def fwd_filter(P, x0, P0, months, z_obs, anchor_obs, phi, Vx, q, H=None, Rcov=None,
               use_cov=True):
    """Forward Kalman over `months` (sorted t_abs after the init month).

    P: params dict (for V_ext-independent fields; phi/Vx/q are (K,) arrays).
    x0, P0: initial state mean / var (K,).
    months: list of t_abs to process (gaps handled via phi**g).
    z_obs: dict month -> z (K,) or None.
    anchor_obs: dict month -> y (K,) full-field TWS observation (already D-tilde-subtracted),
        with R_anchor (K,) given as `R_anchor` dict month -> (K,).
    Returns dict month -> (x_filt, P_filt, x_pred, P_pred).
    """
    x = x0.astype(np.float64).copy()
    Pv = P0.astype(np.float64).copy()
    prev = None
    out = {}
    R_anchor = anchor_obs['R'] if anchor_obs is not None else {}
    for m in months:
        if prev is None:
            g = 1
        else:
            g = m - prev
        phig = phi ** g
        x = phig * x
        Pv = (phi ** (2 * g)) * Pv + Vx * (1 - phi ** (2 * g))
        xp, Pp = x.copy(), Pv.copy()
        if use_cov and z_obs is not None and m in z_obs and H is not None:
            z = z_obs[m]
            Kg = Pv * H / (H * H * Pv + Rcov)
            Kg = np.where(H != 0, Kg, 0.0)
            x = x + Kg * (z - H * x)
            Pv = (1 - Kg * H) * Pv
        if anchor_obs is not None and m in anchor_obs.get('y', {}):
            y = anchor_obs['y'][m]
            Ra = R_anchor[m]
            Kg = Pv / (Pv + Ra)
            x = x + Kg * (y - x)
            Pv = (1 - Kg) * Pv
        out[m] = (x, Pv, xp, Pp)
        prev = m
    return out


def rts_smooth(fwd, months, phi, Vx, q):
    """RTS smoother over the forward-pass months. Returns dict month -> x_smooth."""
    months = sorted(months)
    xs = {}
    x = fwd[months[-1]][0].copy()
    xs[months[-1]] = x
    for i in range(len(months) - 2, -1, -1):
        m, mn = months[i], months[i + 1]
        g = mn - m
        phig = phi ** g
        xf, Pf, xp, Pp = fwd[m]
        Pp_nxt = fwd[mn][3]
        G = Pf * phig / np.maximum(Pp_nxt, 1e-12)
        x = xf + G * (x - fwd[mn][2])
        xs[m] = x
    return xs


def bwd_filter(P, xk, Pk, k_anchor, months, z_obs, phi, Vx, q, H=None, Rcov=None,
               use_cov=True):
    """Independent backward filter from anchor k down through `months` (descending).

    Stationary AR(1) is time-reversible with the same phi (E[x_t|x_{t+1}] = phi x_{t+1}).
    Returns dict month -> (x, Pv) for months in `months` (which must include k_anchor first).
    """
    x = xk.astype(np.float64).copy()
    Pv = Pk.astype(np.float64).copy()
    out = {}
    prev = None
    for m in months:
        if prev is None:
            g = 0
        else:
            g = prev - m
        if g > 0:
            phig = phi ** g
            x = phig * x
            Pv = (phi ** (2 * g)) * Pv + Vx * (1 - phi ** (2 * g))
        if use_cov and z_obs is not None and m in z_obs and H is not None:
            z = z_obs[m]
            Kg = Pv * H / (H * H * Pv + Rcov)
            Kg = np.where(H != 0, Kg, 0.0)
            x = x + Kg * (z - H * x)
            Pv = (1 - Kg * H) * Pv
        out[m] = (x.copy(), Pv.copy())
        prev = m
    return out


def rmse(pred, truth):
    m = np.isfinite(pred) & np.isfinite(truth)
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2))), int(m.sum())
