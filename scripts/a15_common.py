"""
a15_common.py — V8 latent-factor predictor infrastructure (Agent 15).

Generator model (a14a forensics, treated as ground truth):
  TRAIN era: TWS(c,t) = mu_c + trend_c(t) + F(c,t)
  TEST era : TWS(c,t) = mu_c + D(c) + s_c*(t-t0) + F(c,t)   (new offset + new slope field)
  F = rank~136 factor field, per-mode AR(1): f_k(t+1) = phi_k f_k(t) + w_k(t+1)
  The detrended field is NOISELESS (per-cell resid std after top-100 PCs = 0.072):
  the only "noise" is the dynamical innovation w (per-cell std 0.43-0.47).
  Covariates observe F per mode (SPEI_12/SOIL strongest); the cov->mode coupling is
  NONSTATIONARY -> all cov observation fits use recency weighting.

Conventions (identical to submission_v4.py / cv_lb_correlation.py):
  cc cell ids from train lat/lon; t_abs = year*12 + (month-1); mu_c, clim, global cov
  regression coef, per-cell OLS beta_c on t_abs, D-tilde = W1*Dhat + W2*S + W3*trendex.

Kalman: per mode k a 2-component state [fast (AR1 phi_k), d (static D-tilde error)].
  Anchor month: noiseless observation of fast+d  (y = V^T(A_anchor - mu - Dtil)).
  Cov months  : z_jk = H_jk * fast_k + noise  (5 covs, diagonal R).
"""
import numpy as np
import pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']


# ------------------------------------------------------------------ loading
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


def month_field(rows, month, n_cells, valcol, okmask=None):
    """(n_cells,) field of valcol at t_abs==month (NaN elsewhere)."""
    ta = rows['t_abs'].values
    sel = (ta == month)
    if okmask is not None:
        sel = sel & okmask
    f = np.full(n_cells, np.nan, dtype=np.float32)
    f[rows['cc'].values[sel]] = rows[valcol].values[sel]
    return f


# ------------------------------------------------------------------ factor core
class FactorCore:
    """Factor model fitted on `rows` (train rows of the fit window).

    K : number of modes retained (<= rank of fit window).
    rec_edges : (y0, y1) — recency weight 1 for year<y0, 2 in [y0,y1), 3 for >=y1.
    """

    def __init__(self, rows, n_cells, K=100, rec_edges=(2010, 2013), H_sig=2.5,
                 H_window=None, verbose=True):
        self.n_cells = n_cells
        yms = np.sort(rows['ym'].unique())
        self.months = np.array([(int(v) // 100) * 12 + (int(v) % 100) - 1 for v in yms])
        T = len(yms)
        ym_to_i = {int(v): i for i, v in enumerate(yms)}

        # ---- scalar infra (v4 conventions) ----
        self.mu_c = rows.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
        self.clim = rows.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype('float32')  # (C, J)
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

        # ---- detrended anomaly field, SVD ----
        TD = self.months[:, None].astype(np.float64) - self.tbar_c[None, :]
        A = F64 - self.mu_c[None, :] - TD * self.beta_c[None, :]
        self.full = ~np.isnan(A).any(axis=0)
        self.nfull = int(self.full.sum())
        A_f = A[:, self.full]
        A_f = A_f - A_f.mean(axis=0, keepdims=True)
        rank = min(T - 1, self.nfull)
        K = min(K, rank)
        self.K = K
        _, _, Vt = np.linalg.svd(A_f, full_matrices=False)
        self.V = Vt[:K].T.copy()                     # (nfull, K)
        self.S = A_f @ self.V                        # (T, K) factor scores (noiseless)
        if verbose:
            print(f"[core] fit months={T} full cells={self.nfull}/{n_cells} K={K}", flush=True)

        # ---- per-mode dynamics on calendar-consecutive pairs ----
        self.ip = np.where(np.diff(self.months) == 1)[0]
        ia, ib = self.ip, self.ip + 1
        num = (self.S[ia] * self.S[ib]).sum(axis=0)
        den = (self.S[ia] ** 2).sum(axis=0)
        self.phi = np.clip(num / np.maximum(den, 1e-9), -0.60, 0.995)
        self.Vx = (self.S ** 2).sum(axis=0) / (T - 1)          # stationary variance per mode
        self.q = self.Vx * (1.0 - self.phi ** 2)               # innovation variance per mode
        if verbose:
            print(f"[core] phi top10={np.round(self.phi[:10], 3)}  n_neg={int((self.phi < 0).sum())}", flush=True)

        # ---- extended loadings for non-full cells (regression on scores) ----
        nf = ~self.full
        self.V_ext = np.zeros((n_cells, K), dtype=np.float64)
        self.V_ext[self.full] = self.V
        if nf.any():
            A_nf = A[:, nf]
            msk = np.isfinite(A_nf)
            X = np.where(msk, A_nf, 0.0)
            counts = msk.sum(axis=0)
            cov_cs = (self.S.T @ X) / np.maximum(counts[None, :] - 1, 1)   # (K, n_nf)
            okc = counts >= 24
            with np.errstate(divide='ignore', invalid='ignore'):
                vload = np.where(okc[None, :], cov_cs / np.maximum(self.Vx[:, None], 1e-9), 0.0)
            self.V_ext[nf] = vload.T

        # ---- recency weights per fit month (for cov-observation fits) ----
        yr = self.months // 12
        w = np.ones(T)
        w[yr >= rec_edges[0]] = 2.0
        w[yr >= rec_edges[1]] = 3.0
        if H_window is not None:
            w = np.where(yr >= H_window, 1.0, 0.0)   # era-local fit (0 weight before)
        self.wrec = w

        # ---- cov deviation scores on the fit window ----
        # z_jk(t) = v_k^T (cov_j field(t) - fit-window mean of cov_j field)
        # (missing cov cells -> 0 deviation, i.e. imputed at era mean)
        J = len(COVS)
        self.Zj = np.zeros((J, T, K))
        for j, c in enumerate(COVS):
            Fc = np.full((T, n_cells), np.nan, dtype=np.float32)
            Fc[rows['ym'].map(ym_to_i).values, rows['cc'].values] = rows[c].values
            dev = Fc[:, self.full].astype(np.float64)
            dev = np.where(np.isfinite(dev), dev, 0.0)
            dev = dev - dev.mean(axis=0, keepdims=True)
            self.Zj[j] = dev @ self.V

        # ---- per-mode-per-cov observation model z_jk = H_jk * f_k + noise ----
        # recency-weighted, significance-shrunk
        self.H = np.zeros((K, J))
        self.Rcov = np.ones((K, J))
        neff = float((w.sum() ** 2) / (w ** 2).sum())
        for j in range(J):
            z = self.Zj[j]                       # (T, K)
            wz = w[:, None]
            Cf = (wz * z * self.S).sum(axis=0) / (wz * self.S ** 2).sum(axis=0)   # ~cov(z,f)/var(f)
            Vz = (wz * z ** 2).sum(axis=0) / w.sum()
            Vf = (wz * self.S ** 2).sum(axis=0) / w.sum()
            r = Cf * np.sqrt(Vf / np.maximum(Vz, 1e-12))                          # corr(z,f)
            tstat = np.abs(r) * np.sqrt(np.maximum(neff - 2, 1) / np.maximum(1 - r ** 2, 1e-9))
            keep = tstat > H_sig
            Hj = np.where(keep, Cf, 0.0)
            Rj = np.maximum(Vz - Hj * Cf, 0.02 * Vz)
            self.H[:, j] = Hj
            self.Rcov[:, j] = Rj
        if verbose:
            act = (self.H != 0)
            print(f"[core] cov-obs active modes: " +
                  " ".join(f"{c[:6]}:{act[:, j].sum()}" for j, c in enumerate(COVS)), flush=True)

    # -------------------------------------------------- era structures
    def trendex(self, t):
        return (np.float64(t) - self.tbar_c) * self.beta_c

    def era_cov(self, era_rows):
        """Cov fields for every era month (dict month -> (J+1, n_cells):
        [0] combined global-reg cov field - mu ; [1..J] per-cov fields).
        Returns (months, fields_dict)."""
        n_cells = self.n_cells
        Zv = era_rows[COVS].values.astype('float32')
        okv = np.isfinite(Zv).all(axis=1)
        cov_est = np.full(len(era_rows), np.nan, dtype='float32')
        cov_est[okv] = np.column_stack([Zv[okv], np.ones(okv.sum())]) @ self.coef
        months = np.sort(era_rows['t_abs'].unique())
        ta = era_rows['t_abs'].values
        cc = era_rows['cc'].values
        fields = {}
        for m in months:
            selm = (ta == m) & okv
            fm = np.full((len(COVS) + 1, n_cells), np.nan, dtype=np.float32)
            fm[0][cc[selm]] = cov_est[selm]
            for j, c in enumerate(COVS):
                fm[j + 1][cc[selm]] = era_rows[c].values[selm]
            fields[int(m)] = fm - self.mu_c[None, :]
        return months, fields

    def zscores(self, fields, months):
        """z_jk(m) with era-mean removal. Returns dict month -> (J, K)."""
        J = len(COVS)
        per_cov_mean = np.zeros((J, self.n_cells))
        for j in range(J):
            stack = np.array([fields[int(m)][j + 1] for m in months])
            per_cov_mean[j] = np.nanmean(stack, axis=0)
        z = {}
        for m in months:
            zm = np.zeros((J, self.K))
            for j in range(J):
                dev = fields[int(m)][j + 1] - per_cov_mean[j]
                dev = np.where(np.isfinite(dev), dev, 0.0)
                zm[j] = self.V.T @ dev[self.full]
            z[int(m)] = zm
        return z

    def anchor_y(self, Afield, Dtil_field):
        """y = V^T (Afield - Dtil) on full cells. Afield/Dtil: (n_cells,) anomaly fields."""
        g = (Afield - Dtil_field)[self.full]
        g = np.where(np.isfinite(g), g, 0.0)
        return self.V.T @ g


def smooth_series(x, w=9):
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.concatenate([x[:pad][::-1], x, x[-pad:][::-1]]) if len(x) > pad else x
    return np.convolve(xp, np.ones(w) / w, mode='valid')


def estimate_VarD(Y, gaps, phi, Vx, smooth=9, scale=1.0):
    """VarD_k from anchor observations. Y: (n_anchor, K) anchor y's; gaps: list of
    (i, j, gap) consecutive-anchor pairs.  E[y_i*y_j] = phi^gap*Vx + VarD  =>
    VarD = mean_pair_products - mean(phi^gap)*Vx (clipped >=0, smoothed)."""
    if len(gaps) == 0 or Y.shape[0] < 2:
        return np.zeros(Y.shape[1])
    prods = np.array([Y[i] * Y[j] for i, j, g in gaps])
    fast_share = np.array([phi ** g for i, j, g in gaps]).mean(axis=0) * Vx
    raw = prods.mean(axis=0) - fast_share
    raw = np.clip(raw, 0.0, None)
    raw = smooth_series(raw, smooth)
    # second estimator: Var over anchors - Vx (valid when fast states decorrelate)
    raw2 = np.clip(Y.var(axis=0, ddof=1) - Vx, 0.0, None)
    raw2 = smooth_series(raw2, smooth)
    out = np.maximum(raw, 0.35 * raw2) * scale
    return out


# ------------------------------------------------------------------ per-mode 2-comp Kalman
class ModeKF:
    """2-component (fast, static d) Kalman per mode, vectorized over K modes.

    State: x = [xf, xd]; P = [[a, b], [b, c]] per mode.
      transition (gap g): xf <- phi^g xf ; a <- phi^{2g}a + Vx(1-phi^{2g}); b <- phi^g b; c <- c + qd*g
      anchor obs: y = xf + xd (noiseless)
      cov obs: z_j = H_j xf + v_j,  v ~ diag(Rcov_j)   (J-dim obs per mode)
    """

    def __init__(self, phi, Vx, q, H, Rcov, VarD, qd=None):
        self.phi = np.asarray(phi, float)
        self.Vx = np.asarray(Vx, float)
        self.q = np.asarray(q, float)
        self.H = np.asarray(H, float)          # (K, J)
        self.Rcov = np.asarray(Rcov, float)    # (K, J)
        self.VarD = np.asarray(VarD, float)
        self.qd = np.zeros_like(self.phi) if qd is None else np.asarray(qd, float)
        self.K = len(self.phi)
        self.reset()

    def reset(self):
        self.xf = np.zeros(self.K)
        self.xd = np.zeros(self.K)
        self.a = self.Vx.copy()
        self.b = np.zeros(self.K)
        self.c = self.VarD.copy()

    def anchor_update(self, y):
        s = self.a + self.c + 2 * self.b
        s = np.maximum(s, 1e-12)
        nu = y - (self.xf + self.xd)
        g0 = (self.a + self.b) / s
        g1 = (self.b + self.c) / s
        self.xf = self.xf + g0 * nu
        self.xd = self.xd + g1 * nu
        h0 = self.a + self.b
        h1 = self.b + self.c
        self.a = self.a - g0 * h0
        self.b = self.b - g0 * h1
        self.c = self.c - g1 * h1

    def step(self, gap=1):
        phig = self.phi ** gap
        phi2g = phig ** 2
        self.xf = phig * self.xf
        self.a = phi2g * self.a + self.Vx * (1 - phi2g)
        self.b = phig * self.b
        self.c = self.c + self.qd * gap

    def cov_update(self, z):
        """z: (J, K) cov scores at this month (era-mean removed).
        obs model: z_j = H_j * xf + v_j,  v ~ diag(Rcov)."""
        H = self.H                       # (K, J)
        R = self.Rcov
        J = H.shape[1]
        if J == 0 or not (H != 0).any():
            return
        a, b = self.a.copy(), self.b.copy()
        # innovation covariance S[j,l] = H_j a H_l + R_j delta_jl  -> (K, J, J)
        S = (H * a[:, None])[:, :, None] * H[:, None, :]
        idx = np.arange(J)
        S[:, idx, idx] += R
        nu = z.T - H * self.xf[:, None]                # (K, J)
        # w = S^{-1} H  (K, J)
        w = np.linalg.solve(S + 1e-12 * np.eye(J)[None], H[:, :, None])[:, :, 0]
        wn = (w * nu).sum(axis=1)                      # (K,)
        u = (w * H).sum(axis=1)                        # (K,) = H^T S^{-1} H
        self.xf = self.xf + a * wn
        self.xd = self.xd + b * wn
        self.a = a - a * a * u
        self.b = b - a * b * u
        self.c = self.c - b * b * u

    def mean_var(self):
        return self.xf + self.xd, self.a + self.c + 2 * self.b


def rmse(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), int(ok.sum())
