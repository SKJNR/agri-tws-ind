"""
Shared pipeline module: data loading, feature matrices, smoothed climatology,
anchor chaining, feature construction for both validation and submission.

Core design (from experiments 1-6):
  Model A (k=0 / unmasked): 1-month LightGBM on TWS_t + covariates
  Model B (k>=1 / masked): two-stage — per-k linear (smoothed climatology delta space)
                           + residual LightGBM
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
COVARS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t', 'month_sin', 'month_cos']
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
VAL_START = pd.Timestamp('2013-01-01')


def month_ym_int(year, month):
    return year * 100 + month


def load_train():
    train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t'] + COVARS + ['target'])
    train['time'] = pd.to_datetime(train['time'])
    for c in ['TWS_t', 'target'] + COVARS:
        train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
    train['lat'] = train['lat'].astype('float32')
    train['lon'] = train['lon'].astype('float32')
    train['cell'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str))
    train['cell_code'] = train['cell'].astype('category').cat.codes.astype('int32')
    train = train.sort_values(['cell_code', 'time']).reset_index(drop=True)
    train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
    train['cal_m'] = train['time'].dt.month.astype('int8')
    return train


class GridContext:
    """All per-cell / per-month structures needed for feature building."""

    def __init__(self, train, clim_cutoff=VAL_START):
        ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
        train = train.copy()
        train['midx'] = ym_idx.astype('int32')
        self.train = train
        self.n_cells = int(train['cell_code'].max()) + 1
        self.n_months = len(ym_codes)
        self.ym_codes = ym_codes
        self.ym_to_midx = {int(ym): i for i, ym in enumerate(ym_codes)}
        self.midx_to_ym = {i: int(ym) for i, ym in enumerate(ym_codes)}

        n_cells, n_months = self.n_cells, self.n_months
        # monthly cell matrices
        self.M_tws = np.full((n_months, n_cells), np.nan, dtype=np.float32)
        self.M_spei1 = np.full((n_months, n_cells), np.nan, dtype=np.float32)
        self.M_tws[train['midx'].values, train['cell_code'].values] = train['TWS_t'].values
        self.M_spei1[train['midx'].values, train['cell_code'].values] = train['SPEI_01_t'].values
        # cumsum for spei_sum over calendar range (am+1..rm)
        S = np.where(np.isnan(self.M_spei1), 0, self.M_spei1)
        C = (~np.isnan(self.M_spei1)).astype(np.int32)
        self.CSf = np.vstack([np.zeros((1, n_cells), np.float32), np.cumsum(S, axis=0)])
        self.CCf = np.vstack([np.zeros((1, n_cells), np.int32), np.cumsum(C, axis=0)])

        # neighbors (8-neighborhood on 1-deg grid)
        cells = train[['cell_code', 'lat', 'lon']].drop_duplicates().sort_values('cell_code')
        grid = {(la, lo): cc for cc, la, lo in
                zip(cells['cell_code'].values, cells['lat'].values, cells['lon'].values)}
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.nbr_mat = np.zeros((n_cells, 8), dtype=np.int32)
        for cc, la, lo in zip(cells['cell_code'].values, cells['lat'].values, cells['lon'].values):
            ns = [grid.get((la + da, lo + do)) for da, do in offs]
            ns = [n for n in ns if n is not None]
            self.nbr_mat[cc] = (ns + [cc] * (8 - len(ns)))[:8]
        self.N_tws = np.empty_like(self.M_tws)
        for i in range(n_months):
            v = self.M_tws[i][self.nbr_mat]
            with np.errstate(invalid='ignore'):
                self.N_tws[i] = np.nanmean(v, axis=1)

        # smoothed climatology from months < clim_cutoff
        pre = train[train['time'] < clim_cutoff]
        g = pre.groupby(['cell_code', 'cal_m'])['TWS_t'].mean()
        clim_raw = np.full((n_cells, 13), np.nan, dtype=np.float32)
        for (cc, cm), v in g.items():
            clim_raw[cc, cm] = v
        gmc = pre.groupby('cal_m')['TWS_t'].mean()
        fb = np.array([gmc.get(m, 0.0) for m in range(13)], dtype=np.float32)
        clim_raw = np.where(np.isnan(clim_raw), fb[None, :], clim_raw)
        # spatial smoothing (3x3 neighborhood mean)
        clim_sp = np.zeros_like(clim_raw)
        for cc in range(n_cells):
            ns = sorted(set(list(self.nbr_mat[cc][:8])) | {cc})
            clim_sp[cc] = np.mean(clim_raw[ns], axis=0)
        # monthly Gaussian smoothing (circular, sigma~1)
        w = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
        w /= w.sum()
        clim_sm = np.zeros_like(clim_sp)
        for m in range(1, 13):
            idx = [(m - 2 + j - 1) % 12 + 1 for j in range(5)]
            clim_sm[:, m] = (clim_sp[:, idx] * w[None, :]).sum(axis=1)
        self.clim = clim_sm
        # regime features
        cst = pre.groupby('cell_code')['TWS_t'].std().reindex(range(n_cells)).fillna(0.5).values
        self.cell_std = cst.astype('float32')
        self.clim_amp = (np.nanmax(clim_sm, axis=1) - np.nanmin(clim_sm, axis=1)).astype('float32')

        # row-level arrays
        self.row_cell = train['cell_code'].values
        self.row_midx = train['midx'].values
        self.row_cal_m = train['cal_m'].values
        self.row_t1m = ((train['time'].dt.month.values % 12) + 1).astype('int8')
        self.row_year = train['time'].dt.year.values
        self.row_month = train['time'].dt.month.values
        self.row_ym = train['ym'].values
        self.row_lat = train['lat'].values
        self.row_lon = train['lon'].values
        self.row_target = train['target'].values
        self.row_time = train['time'].values
        self.cov_mat = train[COVARS].values.astype('float32')

    def anchor_info(self, idxs, k):
        """Anchor (t-k) lookup for row indices idxs at lag k.
        Returns amap (month idx, -1 if missing), anchor_tws, spei_sum, complete flag."""
        ac = self.row_cell[idxs]
        rm_ = self.row_midx[idxs]
        tot = self.row_year[idxs] * 12 + (self.row_month[idxs] - 1) - k
        aym = (tot // 12) * 100 + tot % 12 + 1
        amap = np.array([self.ym_to_midx.get(int(v), -1) for v in aym], dtype=np.int32)
        has = amap >= 0
        amap_safe = np.where(has, amap, 0)
        anchor_tws = self.M_tws[amap_safe, ac]
        anchor_tws = np.where(has, anchor_tws, np.nan)
        cnt = self.CCf[rm_ + 1, ac] - self.CCf[amap_safe + 1, ac]
        spei_sum = self.CSf[rm_ + 1, ac] - self.CSf[amap_safe + 1, ac]
        spei_sum = np.where(has, spei_sum, np.nan)
        complete = has & (cnt == k)
        return amap_safe, has, anchor_tws, spei_sum, complete

    def features_B(self, idxs, k, ac, rm_, amap, anchor_tws, spei_sum):
        """Model B features + stage-1 linear design matrix."""
        acm = ((self.row_year[idxs] * 12 + self.row_month[idxs] - 1 - k) % 12) + 1
        clim_a = self.clim[ac, acm]
        clim_t1 = self.clim[ac, self.row_t1m[idxs]]
        anchor_anom = anchor_tws - clim_a
        nb_anchor_anom = self.N_tws[amap, ac] - clim_a
        dClim = clim_t1 - clim_a
        lin = np.column_stack([dClim, anchor_anom, spei_sum, nb_anchor_anom,
                               np.ones(len(idxs), np.float32)])
        X = np.column_stack([
            anchor_tws, anchor_anom, np.full(len(idxs), k, dtype=np.float32), spei_sum,
            nb_anchor_anom, clim_t1, clim_a,
            self.cov_mat[idxs], self.row_lat[idxs], self.row_lon[idxs],
            self.cell_std[ac], self.clim_amp[ac],
        ]).astype(np.float32)
        return X, lin, clim_t1

    def features_A(self, idxs):
        """Model A (1-month) features."""
        ac = self.row_cell[idxs]
        X = np.column_stack([
            self.M_tws[self.row_midx[idxs], ac],
            self.cov_mat[idxs],
        ]).astype(np.float32)
        return X


def build_val_indices(g):
    """Validation rows: anchors 2013-01..2015-02 every other month, k=0..6.
    Returns list of (idxs, k)."""
    rows_mask = (g.row_time >= np.datetime64(VAL_START))
    times_w = sorted(g.train.loc[rows_mask, 'time'].unique())
    anchor_times = times_w[::2]
    out = []
    for t in anchor_times:
        t = pd.Timestamp(t)
        for k in range(7):
            tot = t.year * 12 + (t.month - 1) + k
            tym = month_ym_int(tot // 12, tot % 12 + 1)
            if tym not in g.ym_to_midx:
                continue
            sel = np.where(rows_mask & (g.row_ym == tym))[0]
            if len(sel) == 0:
                continue
            out.append((sel, k))
    return out
