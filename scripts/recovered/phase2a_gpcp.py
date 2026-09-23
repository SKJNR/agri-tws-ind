"""
PHASE 2a: GPCP precipitation signal test.

Q: Does cumulative RAW precip anomaly (mm, per-cell scaled) reconstruct TWS anomaly
   better than standardized spei_sum?
Tests (on train, pre-2013 fit / 2013+ val):
  S1: corr(TWS anom_t, cum GPCP precip anom over (t-k, t])  vs  spei_sum corr
  S2: per-cell shrinkage regression: anom ~ a*anom_anchor + b*precip_cum
      (fit pre-2013, val 2013+) — compare RMSE vs AR(1)+spei baseline
"""
import pandas as pd
import numpy as np
import xarray as xr

DATA = '/home/z/my-project/data'
EXT = f'{DATA}/external'
VAL_START = pd.Timestamp('2013-01-01')

# ---------- load GPCP ----------
ds = xr.open_dataset(f'{EXT}/gpcp_precip.mon.mean.nc', decode_times=False)
t = ds.time.values  # days since 1800-01-01
dates = pd.to_datetime('1800-01-01') + pd.to_timedelta(t, unit='D')
ym_idx_gpcp = dates.year * 100 + dates.month
precip = ds.precip.values  # (570, 72, 144) mm/day
lat_g = ds.lat.values  # 2.5° centers: -88.75..88.75
lon_g = ds.lon.values  # 1.25..358.75
print(f"GPCP: {precip.shape}, months {ym_idx_gpcp[0]}..{ym_idx_gpcp[-1]}", flush=True)

# monthly mm = mm/day * days in month
days_in_m = np.array([pd.Timestamp(f"{y}-{m:02d}-01").days_in_month for y, m in
                      zip(ym_idx_gpcp // 100, ym_idx_gpcp % 100)])
precip_mm = precip * days_in_m[:, None, None]  # mm/month

# ---------- load train ----------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'SPEI_01_t', 'target'])
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target', 'SPEI_01_t']:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cell'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str))
train['cell_code'] = train['cell'].astype('category').cat.codes.astype('int32')
n_cells = int(train['cell_code'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
cells = train[['cell_code', 'lat', 'lon']].drop_duplicates().sort_values('cell_code')

# ---------- map train cells -> GPCP 2.5° box ----------
# our grid: 1° centers at .5 (e.g., -55.5, -0.5); GPCP box centers at ±1.25+2.5k
lat_i = np.searchsorted(lat_g, cells['lat'].values) - 1
lat_i = np.clip(lat_i, 0, len(lat_g) - 1)
lon_180 = np.where(cells['lon'].values < 0, cells['lon'].values + 360, cells['lon'].values)
lon_i = np.searchsorted(lon_g, lon_180) - 1
lon_i = np.clip(lon_i, 0, len(lon_g) - 1)
cell_gpcp = precip_mm[:, lat_i, lon_i]  # (months, n_cells) mm/month
gpcp_ym = ym_idx_gpcp

# precip climatology per cell-month (from all GPCP months up to 2014-12 to avoid any val leakage... use pre-2013)
pre_mask_g = gpcp_ym < 201301
gpcp_clim = np.zeros((13, n_cells), dtype=np.float32)
gpcp_count = np.zeros((13, n_cells), dtype=np.float32)
for i, ym in enumerate(gpcp_ym):
    if not pre_mask_g[i]:
        continue
    m = ym % 100
    ok = ~np.isnan(cell_gpcp[i])
    gpcp_clim[m][ok] += cell_gpcp[i][ok]
    gpcp_count[m][ok] += 1
gpcp_clim = np.where(gpcp_count > 0, gpcp_clim / np.maximum(gpcp_count, 1), 0)

# per-cell monthly precip anomaly (mm)
ym_to_gi = {int(ym): i for i, ym in enumerate(gpcp_ym)}
train_gi = np.array([ym_to_gi.get(int(v), -1) for v in train['ym'].values])
p_curr = np.where(train_gi >= 0, cell_gpcp[np.clip(train_gi, 0, None), train['cell_code'].values], np.nan)
p_clim_m = gpcp_clim[train['time'].dt.month.values, train['cell_code'].values]
p_anom = p_curr - p_clim_m

# per-cell std of precip anomaly (for scaling) — fit on pre-2013
pre_tr = (train['time'] < VAL_START).values
df_tmp = pd.DataFrame({'cell': train['cell_code'].values, 'pa': p_anom, 'pre': pre_tr})
p_std = df_tmp[df_tmp.pre].groupby('cell')['pa'].std().reindex(range(n_cells)).fillna(1.0).values.astype(np.float32)

# ---------- S1: signal comparison at k=3 ----------
print("\n=== S1: anomaly-level signal (all train rows) ===")
# TWS anomaly per cell (per-cell mean from pre-2013)
twsm = train[pre_tr].groupby('cell_code')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype(np.float32)
anom_t = train['TWS_t'].values - twsm[train['cell_code'].values]

# compute cum precip anomaly and cum spei over last k months via monthly matrices
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
n_months_tr = len(ym_codes)
P_anom_M = np.full((n_months_tr, n_cells), np.nan, dtype=np.float32)
SPEI_M = np.full((n_months_tr, n_cells), np.nan, dtype=np.float32)
P_anom_M[ym_idx, train['cell_code'].values] = p_anom
SPEI_M[ym_idx, train['cell_code'].values] = train['SPEI_01_t'].values
PA = np.where(np.isnan(P_anom_M), 0, P_anom_M)
PC = (~np.isnan(P_anom_M)).astype(np.int32)
SA = np.where(np.isnan(SPEI_M), 0, SPEI_M)
SC = (~np.isnan(SPEI_M)).astype(np.int32)
PASf = np.vstack([np.zeros((1, n_cells), np.float32), np.cumsum(PA, axis=0)])
PACf = np.vstack([np.zeros((1, n_cells), np.int32), np.cumsum(PC, axis=0)])
SASf = np.vstack([np.zeros((1, n_cells), np.float32), np.cumsum(SA, axis=0)])
SACf = np.vstack([np.zeros((1, n_cells), np.int32), np.cumsum(SC, axis=0)])

rc = train['cell_code'].values
rm = ym_idx
for k in [1, 2, 3, 6]:
    am = rm - k  # approximate: contiguous month index in ym_codes (gaps!)
    # guard: ym_codes may skip months; use calendar arithmetic instead
    tot = (train['time'].dt.year.values * 12 + train['time'].dt.month.values - 1) - k
    aym = (tot // 12) * 100 + tot % 12 + 1
    amap = np.array([{int(v): i for i, v in enumerate(ym_codes)}.get(int(v), -1) for v in aym])
    has = amap >= 0
    amap_s = np.where(has, amap, 0)
    pc_cum = PASf[rm + 1, rc] - PASf[amap_s + 1, rc]
    pc_cnt = PACf[rm + 1, rc] - PACf[amap_s + 1, rc]
    sp_cum = SASf[rm + 1, rc] - SASf[amap_s + 1, rc]
    sp_cnt = SACf[rm + 1, rc] - SACf[amap_s + 1, rc]
    ok = has & (pc_cnt == k) & (sp_cnt == k) & ~np.isnan(anom_t)
    c1 = np.corrcoef(anom_t[ok], pc_cum[ok])[0, 1]
    c2 = np.corrcoef(anom_t[ok], sp_cum[ok])[0, 1]
    # scaled version: precip_anom / cell std
    c3 = np.corrcoef(anom_t[ok], (pc_cum[ok] / p_std[rc][ok]))[0, 1]
    print(f"k={k}: corr(anom, cumGPCP)={c1:.4f} | corr(anom, cumSPEI)={c2:.4f} | scaled GPCP={c3:.4f} (n={ok.sum():,})")

# ---------- S2: multivariate — does GPCP add on top of SPEI? ----------
print("\n=== S2: incremental value of GPCP (k=3, linear) ===")
k = 3
tot = (train['time'].dt.year.values * 12 + train['time'].dt.month.values - 1) - k
aym = (tot // 12) * 100 + tot % 12 + 1
amap = np.array([{int(v): i for i, v in enumerate(ym_codes)}.get(int(v), -1) for v in aym])
has = amap >= 0
amap_s = np.where(has, amap, 0)
pc_cum = PASf[rm + 1, rc] - PASf[amap_s + 1, rc]
sp_cum = SASf[rm + 1, rc] - SASf[amap_s + 1, rc]
anchor_anom = np.full(len(train), np.nan, dtype=np.float32)
atws = np.full(len(train), np.nan, dtype=np.float32)
atws_lookup = pd.Series(train['TWS_t'].values, index=pd.MultiIndex.from_arrays([train['cell_code'], ym_idx])).to_dict()
for i in range(0, len(train), 1):
    pass  # too slow — vectorize
atws = np.array([atws_lookup.get((rc[i], amap_s[i]), np.nan) for i in range(len(train))], dtype=np.float32)
ok = has & ~np.isnan(atws) & ~np.isnan(anom_t)
X = np.column_stack([atws[ok] - twsm[rc[ok]], sp_cum[ok], pc_cum[ok], pc_cum[ok] / p_std[rc[ok]]])
yv = anom_t[ok]
names = ['anchor_anom', 'spei_sum', 'gpcp_cum', 'gpcp_scaled']
from numpy.linalg import lstsq
for subset, label in [([0, 1], 'anchor+spei'), ([0, 1, 2], 'anchor+spei+gpcp'), ([0, 1, 3], 'anchor+spei+gpcp_scaled'), ([0, 2], 'anchor+gpcp'), ([0, 1, 2, 3], 'all')]:
    coef, *_ = lstsq(np.column_stack([X[:, j] for j in subset] + [np.ones(len(X))]), yv, rcond=None)
    pred = np.column_stack([X[:, j] for j in subset] + [np.ones(len(X))]) @ coef
    print(f"  {label:24s}: RMSE={np.sqrt(np.mean((yv-pred)**2)):.4f} (in-sample)")
print(f"\n  anomaly std: {np.std(yv):.4f}")
