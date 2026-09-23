"""a14d_common.py — shared infrastructure for the k=0 specialist (Agent 14-d).

Honest CV protocol (strict test-structure clone):
  - FIT period  : train rows with year <= 2012  (mu_c, clim, beta_c, models)
  - VAL period  : train rows 2013-2015
  - val "in-file months" = all train months in 2013-2015 (covariates visible at all of them,
    exactly like Test.csv where covs exist at all 18 in-file months)
  - val k=0 rows = rows at VAL ANCHOR months = val months whose calendar month is in
    {1,6,7,9,11,12} (the calendar months whose test mask fraction <= 0.5, i.e. the months
    where Test has fully-unmasked anchor instances)
  - covs(t+1) visible for a val k=0 row iff anchor cal-month in {1,6,11,12} (test anchors
    2016-01, 2016-06, 2016-12, 2018-11 all have their next calendar month in Test.csv)
    AND t+1 exists in train. Anchors at cal {7,9} (test: 2018-07, 2015-09) NEVER see covs(t+1).
  - at val anchor months the whole TWS_t field is observable (matches test anchors);
    months before/after anchors are NOT observable except covs (test: no anchor has t-1 in Test)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']
ANCHOR_CAL = {1, 6, 7, 9, 11, 12}          # calendar months of test anchors (mask frac <= 0.5)
NEXT_CAL = {1, 6, 11, 12}                   # anchors whose t+1 IS in Test.csv (2016-01/06/12, 2018-11)
NO_NEXT_CAL = {7, 9}                        # 2015-09, 2018-07: t+1 NOT in Test.csv


def load_train():
    tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVS)
    tr['time'] = pd.to_datetime(tr['time'])
    for c in ['TWS_t', 'target'] + COVS:
        tr[c] = pd.to_numeric(tr[c], errors='coerce').astype('float32')
    tr['cc'] = (tr['lat'].round(2).astype(str) + '_' + tr['lon'].round(2).astype(str)) \
        .astype('category').cat.codes.astype('int32')
    tr['ym'] = tr['time'].dt.year * 100 + tr['time'].dt.month
    tr['t_abs'] = (tr['ym'] // 100) * 12 + (tr['ym'] % 100) - 1
    tr['cal_mon'] = tr['time'].dt.month
    return tr


def cell_coords(tr):
    coords = tr[['cc', 'lat', 'lon']].drop_duplicates('cc').sort_values('cc')
    return coords['lat'].values.astype(np.float64), coords['lon'].values.astype(np.float64)


def build_nb_graph(lat, lon):
    """1-degree neighbor graph. ring1 = (lat±1,lon),(lat,lon±1); box2 = Chebyshev<=2 minus self."""
    n = len(lat)

    def key(la, lo):
        lo = ((lo + 180.0) % 360.0) - 180.0
        return (int(round(la * 10)), int(round(lo * 10)))

    k2i = {}
    for i in range(n):
        k2i[key(lat[i], lon[i])] = i
    offs1 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offs2 = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)
             if not (dx == 0 and dy == 0)]
    nb1 = np.full((n, 4), -1, np.int32)
    nb2 = np.full((n, len(offs2)), -1, np.int32)
    for i in range(n):
        la, lo = lat[i], lon[i]
        for j, (dx, dy) in enumerate(offs1):
            nb1[i, j] = k2i.get(key(la + dx, lo + dy), -1)
        for j, (dx, dy) in enumerate(offs2):
            nb2[i, j] = k2i.get(key(la + dx, lo + dy), -1)
    return nb1, nb2


class Fields:
    """Per-month cell-indexed fields: TWS and covariates, for every month present in `df`."""

    def __init__(self, df, n_cells):
        months = np.sort(df['t_abs'].unique())
        self.months = months
        self.midx = {int(m): i for i, m in enumerate(months)}
        T = len(months)
        self.TWS = np.full((T, n_cells), np.nan, np.float32)
        self.COV = np.full((T, n_cells, len(COVS)), np.nan, np.float32)
        mi = df['t_abs'].map(self.midx).values
        ci = df['cc'].values
        self.TWS[mi, ci] = df['TWS_t'].values
        for j, c in enumerate(COVS):
            self.COV[mi, ci, j] = df[c].values

    def has(self, t_abs):
        return int(t_abs) in self.midx


def nb_stats_matrix(F, nb1, nb2):
    """Neighbor stats for a (T, n) field matrix -> dict of (T, n) float32 arrays."""
    T, n = F.shape
    n1 = nb1.shape[1]
    out = {k: np.full((T, n), np.nan, np.float32)
           for k in ['nb_mean', 'nb_std', 'nb_min', 'nb_max', 'nb_cnt', 'box2_mean']}
    ok1 = nb1 >= 0
    ok2 = nb2 >= 0
    idx1 = np.clip(nb1, 0, n - 1)
    idx2 = np.clip(nb2, 0, n - 1)
    for t in range(T):
        f = F[t]
        g1 = f[idx1]
        g1[~ok1] = np.nan
        g2 = f[idx2]
        g2[~ok2] = np.nan
        with np.errstate(invalid='ignore'):
            out['nb_mean'][t] = np.nanmean(g1, axis=1)
            out['nb_std'][t] = np.nanstd(g1, axis=1)
            out['nb_min'][t] = np.nanmin(g1, axis=1)
            out['nb_max'][t] = np.nanmax(g1, axis=1)
            out['nb_cnt'][t] = np.isfinite(g1).sum(axis=1)
            out['box2_mean'][t] = np.nanmean(g2, axis=1)
    # all-NaN rows produce NaN already; nb_cnt stays 0
    return out


class Infra:
    """Fit-period (<=2012) cell statistics: mu_c, clim, beta_c, tbar_c (+ optional PC basis V)."""

    def __init__(self, fit_df, n_cells, months_all, K=None):
        months = np.sort(fit_df['t_abs'].unique())
        midx = {int(m): i for i, m in enumerate(months)}
        T = len(months)
        F = np.full((T, n_cells), np.nan, np.float32)
        mi = fit_df['t_abs'].map(midx).values
        F[mi, fit_df['cc'].values] = fit_df['TWS_t'].values
        t_abs_m = np.array([(int(m) // 12) * 12 + (int(m) % 12) for m in months], np.float64)
        self.mu_c = np.nanmean(F, axis=0).astype(np.float32)
        self.clim = fit_df.groupby('cc')[COVS].mean().reindex(range(n_cells)).values.astype(np.float32)
        # per-cell OLS slope on t_abs
        F64 = F.astype(np.float64)
        ok = np.isfinite(F64)
        t_mat = np.where(ok, t_abs_m[:, None], np.nan)
        self.tbar_c = np.nanmean(t_mat, axis=0)
        td = t_mat - self.tbar_c[None, :]
        sxx = np.nansum(td * td, axis=0)
        sxy = np.nansum(td * F64, axis=0)
        nok = ok.sum(axis=0)
        self.beta_c = np.where((sxx > 100) & (nok >= 24), sxy / np.where(sxx > 0, sxx, 1), 0.0).astype(np.float32)
        self.n_obs_c = nok.astype(np.int32)
        # optional: top-K right singular vectors of the detrended fit anomaly field
        self.V = None
        if K is not None:
            full = nok == T
            A = F64 - self.mu_c[None, :] - (t_abs_m[:, None] - self.tbar_c[None, :]) * self.beta_c[None, :]
            Ad = A[:, full]
            Ad = Ad - Ad.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Ad, full_matrices=False)
            self.V = Vt[:K].T.astype(np.float32)
            self.full_mask = full

    def trendex(self, t_abs):
        return ((np.float64(t_abs) - self.tbar_c) * self.beta_c).astype(np.float32)


# ---------------- strict val protocol ----------------
def val_anchor_info(val_df):
    """Return DataFrame of val anchor months + has_next flags."""
    months = np.sort(val_df['t_abs'].unique())
    train_months = set(int(m) for m in months)
    rows = []
    for m in months:
        m = int(m)
        cal = (m % 12) + 1
        if cal not in ANCHOR_CAL:
            continue
        nxt_ok = (cal in NEXT_CAL) and ((m + 1) in train_months)
        rows.append(dict(t_abs=m, cal=cal, has_next=bool(nxt_ok),
                         reason=('no-next-by-cal' if cal in NO_NEXT_CAL
                                 else ('no-next-gap' if not nxt_ok else 'next-ok'))))
    return pd.DataFrame(rows)


def rmse(p, y):
    p = np.asarray(p, np.float64); y = np.asarray(y, np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), int(ok.sum())


def wt_rmse_components(pred_by_group, y_by_group, w_with=0.668, w_without=0.332):
    """Test-weighted overall RMSE from subgroup MSEs (test mix: 66.8% with / 33.2% without covs(t+1))."""
    mses = {}
    for g, (p, y) in pred_by_group.items():
        r, n = rmse(p, y)
        mses[g] = (r, n, float(np.mean((np.asarray(p)[np.isfinite(p) & np.isfinite(y)]
                                        - np.asarray(y)[np.isfinite(p) & np.isfinite(y)]) ** 2)))
    overall = np.sqrt(w_with * mses.get('with', (0, 0, 0))[2] + w_without * mses.get('without', (0, 0, 0))[2])
    return overall, mses
