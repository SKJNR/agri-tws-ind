"""
SUBMISSION v3: push persistence higher + isolate k=0 model choice.

LB feedback:
  v1 (trees ~0.77 eff, with covs):        0.8059
  v2b (Kalman 0.97/0.84, with covs):       0.7962  <- best
  v2c (pure decay 0.95/0.82, no covs):     0.8337

Signal: HIGHER persistence + cov-obs Kalman = winning direction.

v3 variants (all with cov-obs Kalman, k=0 = pure AR slope=lam*phi):
  v3a: (0.97, 0.84) — isolate: is the v2b k=0 LGBM blend helping or hurting?
  v3b: (0.99, 0.86) — push persistence higher
  v3c: (0.995, 0.87) — extreme push
"""
import numpy as np
import pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t']

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time', 'lat', 'lon', 'TWS_t', 'target'] + COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t', 'target'] + COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
n_cells = int(train['cc'].max()) + 1
train['ym'] = train['time'].dt.year * 100 + train['time'].dt.month
ym_codes, ym_idx = np.unique(train['ym'].values, return_inverse=True)

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t'] + COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc', 'lat', 'lon']].drop_duplicates()
cell_map = {(round(float(la), 1), round(float(lo), 1)): int(cc) for cc, la, lo in zip(codes['cc'], codes['lat'], codes['lon'])}
test['cc'] = [cell_map.get((round(float(la), 1), round(float(lo), 1)), -1) for la, lo in zip(test['lat'], test['lon'])]
assert (test['cc'] >= 0).all()
test['ym'] = test['time'].dt.year * 100 + test['time'].dt.month
test['masked'] = test['TWS_t_masked'].astype(bool)

# parameters from ALL train months
mu_c = train.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).fillna(0).values.astype('float32')
var_tot = float(np.var(train['TWS_t']))
cov_mat = train[COVS].values.astype('float32')
A = np.column_stack([cov_mat, np.ones(len(cov_mat))])
b = train['TWS_t'].values
coef_z = np.linalg.solve(A.T @ A + 10.0 * np.eye(6), A.T @ b)
# honest R: fit 2002-2010, eval 2010-2015
mf = (train['time'] < '2010-01-01').values
Ah = np.column_stack([cov_mat[mf], np.ones(mf.sum())])
ch = np.linalg.solve(Ah.T @ Ah + 10 * np.eye(6), Ah.T @ b[mf])
me = ~mf
Ae = np.column_stack([cov_mat[me], np.ones(me.sum())])
R_h = float(np.var(b[me] - Ae @ ch))
print(f"var_tot={var_tot:.4f} R_h={R_h:.4f}", flush=True)

# combined month axis (train + test)
all_yms = sorted(set(list(ym_codes.astype(int)) + [int(v) for v in test['ym'].unique()]))
ym_to_combo = {v: i for i, v in enumerate(all_yms)}
M_L = np.full((len(all_yms), n_cells), np.nan, dtype=np.float32)
M_Z = np.full((len(all_yms), n_cells), np.nan, dtype=np.float32)
M_ZOK = np.zeros((len(all_yms), n_cells), dtype=bool)
M_L[ym_idx, train['cc'].values] = train['TWS_t'].values
zt = cov_mat @ coef_z[:5] + coef_z[5]
M_Z[ym_idx, train['cc'].values] = zt
M_ZOK[ym_idx, train['cc'].values] = ~np.isnan(cov_mat).any(axis=1)
test_cov = test[COVS].values.astype('float32')
z_test = test_cov @ coef_z[:5] + coef_z[5]
ti = np.array([ym_to_combo[int(v)] for v in test['ym'].values])
M_Z[ti, test['cc'].values] = z_test
M_ZOK[ti, test['cc'].values] = ~np.isnan(test_cov).any(axis=1)
unm = ~test['masked'].values
M_L[ti[unm], test['cc'].values[unm]] = test['TWS_t'].values[unm]

# anchor structure
month_mr = test.groupby('ym')['masked'].mean()
anchors = sorted(month_mr[month_mr < 0.5].index)
yms_arr = test['ym'].values
row_anchor = np.empty(len(test), dtype=np.int64)
for i, ym in enumerate(yms_arr):
    a = anchors[0]
    for cand in anchors:
        if cand <= ym: a = cand
    row_anchor[i] = a

def kalman_predict(phi, lam):
    P_init = (1 - lam) * var_tot
    q = lam * var_tot * (1 - phi ** 2)
    pred = np.full(len(test), np.nan, dtype=np.float64)
    for a in anchors:
        a_ci = ym_to_combo[int(a)]
        L_anchor = M_L[a_ci]
        anom0 = L_anchor - mu_c
        xh = np.where(np.isfinite(anom0), lam * anom0, 0.0).astype(np.float32)
        P = np.where(np.isfinite(anom0), P_init, 1e6).astype(np.float32)
        for k in range(1, 8):
            tot = (a // 100) * 12 + (a % 100) - 1 + k
            tym = (tot // 12) * 100 + tot % 12 + 1
            if tym not in ym_to_combo: continue
            t_ci = ym_to_combo[int(tym)]
            xh = phi * xh
            P = phi ** 2 * P + q
            z = M_Z[t_ci]
            ok = M_ZOK[t_ci] & np.isfinite(z) & (P < 1e5)
            K = np.where(ok, P / (P + R_h), 0.0).astype(np.float32)
            xh = np.where(ok, xh + K * (z - mu_c - xh), xh)
            P = np.where(ok, (1 - K) * P, P)
            sel = np.where((row_anchor == a) & (yms_arr == tym) & test['masked'].values)[0]
            if len(sel) == 0: continue
            cc = test['cc'].values[sel]
            pred[sel] = mu_c[cc] + phi * xh[cc]
    return pred

sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
unm_mask = ~test['masked'].values
cc_test = test['cc'].values
tws_test = test['TWS_t'].values
m_mask = test['masked'].values

def assemble(masked_pred, phi, lam, tag):
    slope0 = lam * phi
    p_ar = mu_c[cc_test] + slope0 * (tws_test - mu_c[cc_test])
    pred = np.empty(len(test), dtype=np.float64)
    pred[unm_mask] = p_ar[unm_mask]  # pure AR at k=0
    pred[m_mask] = masked_pred[m_mask]
    out = sub[['ID']].copy()
    out['Target'] = pred.astype(np.float32)
    out.to_csv(f'{DL}/submission_{tag}.csv', index=False)
    print(f"saved {tag}: phi={phi} lam={lam} | mean={pred.mean():.4f} std={pred.std():.4f} "
          f"(unm std={pred[unm_mask].std():.3f}, masked std={pred[m_mask].std():.3f})", flush=True)

# v3a: same persistence as v2b BUT pure-AR k=0 (isolates LGBM blend effect vs v2b)
assemble(kalman_predict(0.97, 0.84), 0.97, 0.84, 'v3a')
# v3b: push persistence higher
assemble(kalman_predict(0.99, 0.86), 0.99, 0.86, 'v3b')
# v3c: extreme push
assemble(kalman_predict(0.995, 0.87), 0.995, 0.87, 'v3c')
print("\nDone. 3 v3 variants saved.", flush=True)
