"""
LEVEL-SPACE KALMAN (no climatology — the generator has no seasonality).

State: L_t = TWS level per cell
  L_{t+1} = mu_c + phi_c*(L_t - mu_c) + w,  w ~ N(0, q_c)
  mu_c = per-cell mean (138 obs, reliable), phi_c shrunk, q_c shrunk
Obs: z_t = b'·[S01,S03,S06,S12,SM] + c  (pooled coefs, RAW cov values) = L_t + v

k=0 rows: pred = mu_c + phi_c*(TWS_t - mu_c)
masked rows: Kalman chain from anchor, pred = mu + phi*xh
Validation: block structure 2013+ (same as run_B / kalman.py for comparability).
"""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
VAL_START = pd.Timestamp('2013-01-01')
K_WEIGHTS = {0: 6, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1}
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target'] + COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = int(train['cc'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)
train['midx'] = ym_idx.astype('int32')
n_months = len(ym_codes)
ym_to_midx = {int(v): i for i, v in enumerate(ym_codes)}
row_cc = train['cc'].values
row_ym = train['ym'].values
row_target = train['target'].values
pre_mask = (train['time'] < VAL_START).values
pre = train[pre_mask]
print(f"Loaded {len(train):,}", flush=True)

# ---------- parameters ----------
PHI_G = 0.77
g = pre.groupby('cc')
n_ = g.size().astype(float).reindex(range(n_cells)).fillna(0)
mu_c = g['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype(np.float32)
sx_ = g['TWS_t'].sum().reindex(range(n_cells)).fillna(0)
sy_ = g['target'].sum().reindex(range(n_cells)).fillna(0)
sxx_ = g.apply(lambda d: (d['TWS_t'] ** 2).sum(), include_groups=False).reindex(range(n_cells)).fillna(0)
sxy_ = g.apply(lambda d: (d['TWS_t'] * d['target']).sum(), include_groups=False).reindex(range(n_cells)).fillna(0)
# phi on demeaned data: sum((x-mu)(y-mu)) / sum((x-mu)^2)
scx = sx_ - n_ * mu_c
scxx = sxx_ - n_ * mu_c**2
scxy = sxy_ - (sx_ * sy_) / n_
phi_raw = (scxy / scxx.replace(0, np.nan)).clip(0.2, 0.99)
lam = n_ / (n_ + 50.0)
phi_c = (lam * phi_raw.fillna(PHI_G) + (1 - lam) * PHI_G).values.astype(np.float32)
# q per cell
q_raw = pre.groupby('cc').apply(lambda d: np.nanvar(d['target'] - PHI_G * d['TWS_t'] - (1 - PHI_G) * d['TWS_t'].mean()), include_groups=False)
q_raw = q_raw.reindex(range(n_cells))
Q_G = float(np.var(pre['target'] - PHI_G * pre['TWS_t']))
lam_q = n_ / (n_ + 50.0)
q_c = (lam_q * q_raw.fillna(Q_G).clip(0.01, 2.0) + (1 - lam_q) * Q_G).values.astype(np.float32)
print(f"phi_c mean={phi_c.mean():.3f} | q_c mean={q_c.mean():.4f} | Q_G={Q_G:.4f}", flush=True)

# observation model: pooled ridge, RAW covs -> TWS level
cov_mat = train[COVS].values.astype('float32')
A = np.column_stack([cov_mat[pre_mask], np.ones(pre_mask.sum())])
b = train['TWS_t'].values[pre_mask]
coef_obs = np.linalg.solve(A.T @ A + 10.0 * np.eye(6), A.T @ b)
resid = b - A @ coef_obs
R_G = float(np.var(resid))
print(f"obs R (in-sample): {R_G:.4f}", flush=True)
# honest R (temporal split)
m_fit = (train['time'] < '2010-01-01').values
m_ev = ((train['time'] >= '2010-01-01') & pre_mask).values
Ah = np.column_stack([cov_mat[m_fit], np.ones(m_fit.sum())])
bh = train['TWS_t'].values[m_fit]
ch = np.linalg.solve(Ah.T @ Ah + 10*np.eye(6), Ah.T @ bh)
Ae = np.column_stack([cov_mat[m_ev], np.ones(m_ev.sum())])
R_honest = float(np.var(train['TWS_t'].values[m_ev] - Ae @ ch))
print(f"obs R (honest):    {R_honest:.4f}", flush=True)

# monthly matrices
M_L = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_L[ym_idx, row_cc] = train['TWS_t'].values
z_all = cov_mat @ coef_obs[:5] + coef_obs[5]
M_z = np.full((n_months, n_cells), np.nan, dtype=np.float32)
M_z[ym_idx, row_cc] = z_all
cov_ok = ~np.isnan(cov_mat).any(axis=1)
M_zok = np.zeros((n_months, n_cells), dtype=bool)
M_zok[ym_idx, row_cc] = cov_ok

# ---------- validation ----------
rows_mask = (train['time'] >= VAL_START).values
times_w = sorted(train.loc[rows_mask, 'time'].unique())
anchor_times = times_w[::2]


def run_val(R_mult, use_obs=True):
    y_true, y_pred, k_list = [], [], []
    R_eff = R_mult * R_G
    for t in anchor_times:
        t = pd.Timestamp(t)
        a_midx = ym_to_midx[t.year * 100 + t.month]
        L_anchor = M_L[a_midx].copy()
        xh = np.where(np.isnan(L_anchor), 0.0, L_anchor).astype(np.float32)
        P = np.where(~np.isnan(L_anchor), 0.0, 1e6).astype(np.float32)
        for k in range(1, 7):
            tot = t.year * 12 + (t.month - 1) + k
            tym = (tot // 12) * 100 + tot % 12 + 1
            if tym not in ym_to_midx:
                continue
            t_midx = ym_to_midx[tym]
            # predict
            xh = mu_c + phi_c * (xh - mu_c)
            P = phi_c ** 2 * P + q_c
            if use_obs:
                z = M_z[t_midx]
                ok = M_zok[t_midx] & np.isfinite(z) & (P < 1e5)
                K = np.where(ok, P / (P + R_eff), 0.0).astype(np.float32)
                xh = np.where(ok, xh + K * (z - xh), xh)
                P = np.where(ok, (1 - K) * P, P)
            sel = np.where(rows_mask & (row_ym == tym))[0]
            if len(sel) == 0:
                continue
            cc = row_cc[sel]
            pred = mu_c[cc] + phi_c[cc] * (xh[cc] - mu_c[cc])
            y_true.append(row_target[sel])
            y_pred.append(pred)
            k_list.append(np.full(len(sel), k))
    y = np.concatenate(y_true); p = np.concatenate(y_pred); kk = np.concatenate(k_list)
    w = np.array([K_WEIGHTS.get(int(k), 1) for k in kk], dtype=np.float64)
    wr = np.sqrt(np.average((y - p) ** 2, weights=w))
    pk = {k: np.sqrt(np.mean((y[kk == k] - p[kk == k]) ** 2)) for k in range(1, 7) if (kk == k).sum()}
    return wr, pk, y, p, kk


print("\n=== LEVEL-SPACE KALMAN (k=1..6, block val) ===", flush=True)
for Rm, lab in [(1e18, 'no-obs (pure AR decay)'), (4, 'R x4'), (2, 'R x2'), (1, 'R x1'), (R_honest/R_G, 'honest R')]:
    wr, pk, y, p, kk = run_val(Rm)
    print(f"{lab:22s}: weighted={wr:.4f} | " + " ".join(f"k{k}={pk[k]:.3f}" for k in sorted(pk)))

# k=0: per-cell AR(1) in level space on same val anchors
y0, p0 = [], []
for t in anchor_times:
    t = pd.Timestamp(t)
    a_midx = ym_to_midx[t.year * 100 + t.month]
    sel = np.where(rows_mask & (row_ym == t.year * 100 + t.month))[0]
    if len(sel) == 0:
        continue
    cc = row_cc[sel]
    L = M_L[a_midx][cc]
    ok = ~np.isnan(L)
    y0.append(row_target[sel][ok])
    p0.append((mu_c[cc][ok] + phi_c[cc][ok] * (L[ok] - mu_c[cc][ok])))
y0 = np.concatenate(y0); p0 = np.concatenate(p0)
print(f"\nk=0 per-cell AR(1) level-space: {np.sqrt(np.mean((y0-p0)**2)):.4f} (n={len(y0):,})")
print("(refs: run_A LGBM k=0 = 0.6547 | AR(1) global all-rows = 0.6379 | floor ~0.62)")
