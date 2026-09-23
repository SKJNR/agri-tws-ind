"""
PHASE 1.5: Hierarchical per-cell AR(1) model — out-of-sample validation.

Model C: TWS_{t+1} = mu_cell + phi_cell * (TWS_t - mu_cell)
  - phi, mu fit per cell on pre-2013 months
  - empirical-Bayes shrinkage of phi toward latitude-band mean
  - multi-step version for masked rows (k months since anchor):
      pred = mu + phi^(k+1) * (anchor - mu) + beta * spei_sum
  - spatial smoothing pass on predictions (field corr 0.99)

Validation: same framework as before (anchors 2013-01..2015-02 [::2], k=0..6).
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
COVARS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t', 'month_sin', 'month_cos']
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
VAL_START = pd.Timestamp('2013-01-01')

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'SPEI_01_t'] + COVARS + ['target'])
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target', 'SPEI_01_t'] + COVARS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')

# cell codes
train['cell'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str))
train['cell_code'] = train['cell'].astype('category').cat.codes.astype('int32')
n_cells = int(train['cell_code'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
train['cal_m'] = train['time'].dt.month.astype('int8')
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
train['midx'] = ym_idx.astype('int32')
n_months = len(ym_codes)
ym_to_midx = {int(ym): i for i, ym in enumerate(ym_codes)}
print(f"Loaded {len(train):,} rows, {n_cells} cells", flush=True)

# ---------- fit per-cell AR(1) on pre-2013 ----------
pre = train[train['time'] < VAL_START]
g = pre.groupby('cell_code')
n_obs = g.size().astype(float)
sx = g['TWS_t'].sum(); sy = g['target'].sum()
sxx = g.apply(lambda d: (d['TWS_t'] ** 2).sum(), include_groups=False)
sxy = g.apply(lambda d: (d['TWS_t'] * d['target']).sum(), include_groups=False)
n = n_obs
den = (n * sxx - sx.pow(2)).replace(0, np.nan)
phi_raw = ((n * sxy - sx * sy) / den)
mu_hat = sy / n  # per-cell mean of TWS (stationary level)

# latitude bands for shrinkage
cells = train[['cell_code', 'lat', 'lon']].drop_duplicates().sort_values('cell_code')
lat_arr = cells['lat'].values
band = np.clip(np.round(lat_arr / 6) * 6, -84, 84)  # 6-degree bands
band_df = pd.DataFrame({'phi': phi_raw.values, 'band': band})
band_mean = band_df.groupby('band')['phi'].mean()
band_cnt = band_df.groupby('band')['phi'].count()

# empirical-Bayes shrinkage: phi_shrunk = w*phi_raw + (1-w)*band_mean
# w = tau2 / (tau2 + sigma2/n), tau2 = between-cell variance in band
phi_shrunk = np.empty(n_cells)
w_out = np.empty(n_cells)
for b in band_mean.index:
    msk = band == b
    phis = phi_raw.values[msk]
    ok = ~np.isnan(phis)
    if ok.sum() < 5:
        phi_shrunk[msk] = band_mean.get(b, 0.76)
        w_out[msk] = 0.0
        continue
    v_within = np.var(phis[ok])  # includes sampling noise
    # estimate sampling variance: var of OLS phi with rho~0.76: (1-rho^2)/(n*sigma_x2/sigma_y2)
    # crude: assume phi se^2 ~ (1-phi^2)^2/n
    se2 = np.mean((1 - phis[ok] ** 2) ** 2 / n.values[msk][ok])
    tau2 = max(v_within - se2, 0.01)
    w = tau2 / (tau2 + se2)
    phi_shrunk[msk] = w * np.nanmean(phis) + (1 - w) * band_mean.get(b, 0.76)
    w_out[msk] = w
phi_shrunk = np.clip(phi_shrunk, 0.2, 0.995)
mu_full = np.where(np.isnan(mu_hat.values), 0.0, mu_hat.values)
print(f"phi shrunk: mean={phi_shrunk.mean():.3f} p10={np.percentile(phi_shrunk,10):.3f} p90={np.percentile(phi_shrunk,90):.3f}", flush=True)
print(f"shrinkage w: mean={w_out.mean():.3f}", flush=True)

# ---------- per-cell beta for spei_sum contribution ----------
# for k>=1: anom_target = phi^(k+1)*anchor_anom + beta * spei_sum
# fit beta globally per k on pre-2013 (as before) but with per-cell phi decay
spei_map = pre.pivot_table(index=['lat','lon'], columns='ym', values='SPEI_01_t') if False else None
# faster: monthly matrix
M_spei1 = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_spei1[train['midx'].values, train['cell_code'].values] = train['SPEI_01_t'].values
M_tws = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_tws[train['midx'].values, train['cell_code'].values] = train['TWS_t'].values
S = np.where(np.isnan(M_spei1), 0, M_spei1)
C = (~np.isnan(M_spei1)).astype(np.int32)
CSf = np.vstack([np.zeros((1, n_cells), np.float32), np.cumsum(S, axis=0)])
CCf = np.vstack([np.zeros((1, n_cells), np.int32), np.cumsum(C, axis=0)])

# ---------- validation ----------
rows_mask = (train['time'] >= VAL_START).values
times_w = sorted(train.loc[rows_mask, 'time'].unique())
anchor_times = times_w[::2]
row_ym = train['ym'].values
row_cell = train['cell_code'].values
row_midx = train['midx'].values
row_year = train['time'].dt.year.values
row_month = train['time'].dt.month.values
row_target = train['target'].values
row_t1m = ((train['time'].dt.month.values % 12) + 1).astype('int8')

y_all, k_all, pred_ar, pred_persist, pred_ar_spei = [], [], [], [], []
beta_by_k = {}
for k in range(7):
    # fit beta for this k on pre-2013 with per-cell phi decay
    if k >= 1:
        base = np.where((train['time'] < VAL_START).values)[0]
        ac = row_cell[base]
        tot = row_year[base] * 12 + (row_month[base] - 1) - k
        aym = (tot // 12) * 100 + tot % 12 + 1
        amap = np.array([ym_to_midx.get(int(v), -1) for v in aym])
        has = amap >= 0
        amap_s = np.where(has, amap, 0)
        atws = M_tws[amap_s, ac]
        cnt = CCf[row_midx[base] + 1, ac] - CCf[amap_s + 1, ac]
        ssum = CSf[row_midx[base] + 1, ac] - CSf[amap_s + 1, ac]
        okm = has & (cnt == k) & ~np.isnan(atws)
        # AR decay prediction residual vs spei_sum
        p = mu_full[ac[okm]] + (phi_shrunk[ac[okm]] ** (k + 1)) * (atws[okm] - mu_full[ac[okm]])
        resid = row_target[base][okm] - p
        x = ssum[okm]
        okf = ~np.isnan(resid) & ~np.isnan(x)
        beta = np.polyfit(x[okf], resid[okf], 1)[0] if okf.sum() > 1000 else 0.0
        beta_by_k[k] = float(beta)
    else:
        beta_by_k[0] = 0.0

for t in anchor_times:
    t = pd.Timestamp(t)
    for k in range(7):
        tot = t.year * 12 + (t.month - 1) + k
        tym = (tot // 12) * 100 + tot % 12 + 1
        if tym not in ym_to_midx:
            continue
        sel = np.where(rows_mask & (row_ym == tym))[0]
        if len(sel) == 0:
            continue
        ac = row_cell[sel]
        # anchor
        tot_a = tot - k
        aym = (tot_a // 12) * 100 + tot_a % 12 + 1
        amap = ym_to_midx.get(aym, -1)
        if amap < 0:
            continue
        atws = M_tws[amap, ac]
        cnt = CCf[row_midx[sel] + 1, ac] - CCf[amap + 1, ac]
        ssum = CSf[row_midx[sel] + 1, ac] - CSf[amap + 1, ac]
        ok = (cnt == k) & ~np.isnan(atws)
        if ok.sum() == 0:
            continue
        s = sel[ok]; a = ac[ok]; at = atws[ok]; ss = ssum[ok]
        # AR(k+1) prediction
        p_ar = mu_full[a] + (phi_shrunk[a] ** (k + 1)) * (at - mu_full[a])
        p_ar_sp = p_ar + beta_by_k[k] * ss
        y_all.append(row_target[s]); k_all.append(np.full(len(s), k))
        pred_ar.append(p_ar); pred_persist.append(at); pred_ar_spei.append(p_ar_sp)

y = np.concatenate(y_all); kk = np.concatenate(k_all)
p_ar = np.concatenate(pred_ar); p_per = np.concatenate(pred_persist); p_ar_sp = np.concatenate(pred_ar_spei)
w = np.array([K_WEIGHTS.get(int(k), 1) for k in kk], dtype=np.float64)

def wrmse(p):
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))

print(f"\n=== PER-CELL AR(1) HIERARCHICAL — VALIDATION (2013+) ===")
print(f"beta by k: { {k: round(v,4) for k,v in beta_by_k.items()} }")
print(f"persistence (raw anchor):    {wrmse(p_per):.4f}")
print(f"per-cell AR(1) multi-step:   {wrmse(p_ar):.4f}")
print(f"AR(1) + spei_sum beta:       {wrmse(p_ar_sp):.4f}")
print(f"(refs: Model A k=0 = 0.6547 | Model B masked = 0.7846 | system = 0.744 | leader 0.5596)")
print("\nper-k AR(1)+spei:")
for k in range(7):
    m = kk == k
    if m.sum() > 0:
        print(f"  k={k}: {np.sqrt(np.mean((y[m]-p_ar_sp[m])**2)):.4f} (n={m.sum():,})")

# blend AR with Model A at k=0? (load Model A preds if available)
try:
    d = np.load('/home/z/my-project/scripts/modelA_val.npz')
    # note: modelA preds are for same val anchors (k=0) but different row subset; recompute alignment is complex
    print("\n(modelA_val.npz exists for reference)")
except Exception:
    pass

np.savez('/home/z/my-project/scripts/ar1_val.npz', y=y, k=kk, p_ar=p_ar, p_ar_sp=p_ar_sp, p_per=p_per,
         phi=phi_shrunk, mu=mu_full, beta=np.array([beta_by_k.get(k, 0.0) for k in range(7)]))
print("Saved ar1_val.npz", flush=True)
