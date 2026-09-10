"""
Final prediction: build test features, run Model A (unmasked) + Model B avg (masked),
write submission CSV.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import json

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/scripts'
DL = '/home/z/my-project/download'
COVARS = ['SPEI_01_t', 'SPEI_03_t', 'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t', 'month_sin', 'month_cos']

# ---------- load context ----------
ctx = np.load(f'{OUT}/final_context.npz')
clim = ctx['clim']; cell_std = ctx['cell_std']; clim_amp = ctx['clim_amp']
nbr_mat = ctx['nbr_mat']; lat_by_cell = ctx['lat_by_cell']; lon_by_cell = ctx['lon_by_cell']
n_cells = len(lat_by_cell)
cell_code_map = {(round(float(la), 1), round(float(lo), 1)): i for i, (la, lo) in enumerate(zip(lat_by_cell, lon_by_cell))}
with open(f'{OUT}/final_coefs.json') as f:
    coefs = {int(k): np.array(v) for k, v in json.load(f).items()}

mA = lgb.Booster(model_file=f'{OUT}/final_modelA.txt')
mB1 = lgb.Booster(model_file=f'{OUT}/final_modelB1.txt')
mB2 = lgb.Booster(model_file=f'{OUT}/final_modelB2.txt')

# ---------- load test ----------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t'] + COVARS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
test['lat'] = test['lat'].astype('float32')
test['lon'] = test['lon'].astype('float32')
test['cell_code'] = [cell_code_map.get((round(float(la), 1), round(float(lo), 1)), -1)
                     for la, lo in zip(test['lat'], test['lon'])]
assert (test['cell_code'] >= 0).all(), "unmapped test cells!"
test['ym'] = test['time'].dt.year * 100 + test['time'].dt.month
test['cal_m'] = test['time'].dt.month
print(f"Test rows: {len(test):,}", flush=True)

# ---------- anchor structure ----------
unmasked_mask = ~test['TWS_t_masked'].astype(bool)
# anchor months = months where most rows unmasked
month_maskrate = test.groupby('ym')['TWS_t_masked'].mean()
anchor_yms = set(month_maskrate[month_maskrate < 0.5].index)
print(f"Anchor months: {sorted(anchor_yms)}", flush=True)
t1m = ((test['time'].dt.month.values % 12) + 1).astype('int8')
test['t1m'] = t1m

# per-row k and anchor ym
def ym_add(ym, k):
    y, m = ym // 100, ym % 100
    tot = y * 12 + (m - 1) + k
    return (tot // 12) * 100 + tot % 12 + 1

anchor_yms_sorted = sorted(anchor_yms)
row_ym = test['ym'].values
anchor_ym_arr = np.empty(len(test), dtype=np.int64)
k_arr = np.empty(len(test), dtype=np.int32)
for i, ym in enumerate(row_ym):
    a = None
    for cand in anchor_yms_sorted:
        if cand <= ym:
            a = cand
        else:
            break
    if a is None:
        a = anchor_yms_sorted[0]
    anchor_ym_arr[i] = a
    dy = (ym // 100 - a // 100) * 12 + (ym % 100 - a % 100)
    k_arr[i] = dy
# sanity: masked rows must have k>=1
mm_check = test['TWS_t_masked'].astype(bool).values
assert (k_arr[mm_check] >= 1).all(), "masked rows with k=0!"
print(f"k distribution: {pd.Series(k_arr).value_counts().sort_index().to_dict()}", flush=True)
print(f"unmasked rows total: {(~mm_check).sum():,} (of which k>0: {((~mm_check) & (k_arr>0)).sum():,})", flush=True)

# ---------- lookups from test ----------
# anchor TWS: (cell, anchor_ym) -> TWS_t from unmasked rows
un = test[unmasked_mask]
anchor_tws_map = {}
for cc, ym, tws in zip(un['cell_code'].values, un['ym'].values, un['TWS_t'].values):
    anchor_tws_map[(int(cc), int(ym))] = float(tws)
# SPEI_01: (cell, ym) -> value (all rows)
spei_map = {}
for cc, ym, s in zip(test['cell_code'].values, test['ym'].values, test['SPEI_01_t'].values):
    spei_map[(int(cc), int(ym))] = float(s)
# neighbor TWS at anchor: per anchor ym, cell -> TWS then neighbor mean
anchor_nb_map = {}
for ym in anchor_yms_sorted:
    sub = un[un['ym'] == ym]
    tws_by_cell = np.full(n_cells, np.nan, dtype=np.float32)
    tws_by_cell[sub['cell_code'].values] = sub['TWS_t'].values
    nb = np.empty(n_cells, dtype=np.float32)
    for c in range(n_cells):
        v = tws_by_cell[nbr_mat[c]]
        ok = ~np.isnan(v)
        nb[c] = v[ok].mean() if ok.any() else np.nan
    anchor_nb_map[ym] = nb
print(f"Lookups built", flush=True)

# ---------- build features ----------
n = len(test)
cc_arr = test['cell_code'].values.astype(np.int64)
clim_a = np.empty(n, dtype=np.float32)
clim_t1 = np.empty(n, dtype=np.float32)
anchor_tws = np.empty(n, dtype=np.float32)
nb_anchor = np.empty(n, dtype=np.float32)
spei_sum = np.empty(n, dtype=np.float32)

for i in range(n):
    cc = int(cc_arr[i]); a = int(anchor_ym_arr[i]); k = int(k_arr[i])
    # anchor cal month
    acm = a % 100
    clim_a[i] = clim[cc, acm]
    clim_t1[i] = clim[cc, int(t1m[i])]
    at = anchor_tws_map.get((cc, a), np.nan)
    anchor_tws[i] = at if not np.isnan(at) else clim_a[i]  # fallback
    nb_anchor[i] = anchor_nb_map[a][cc]
    # spei_sum over (a, ym]
    s = 0.0
    ym = a
    for _ in range(k):
        ym = ym_add(ym, 1)
        s += spei_map.get((cc, ym), 0.0)
    spei_sum[i] = s

anchor_anom = anchor_tws - clim_a
nb_anchor_anom = nb_anchor - clim_a
nb_anchor_anom = np.where(np.isnan(nb_anchor_anom), 0.0, nb_anchor_anom)
k_f = k_arr.astype(np.float32)

cov = test[COVARS].values.astype(np.float32)
lat_f = test['lat'].values.astype(np.float32)
lon_f = test['lon'].values.astype(np.float32)
cstd = cell_std[cc_arr]
camp = clim_amp[cc_arr]

X_B = np.column_stack([
    anchor_tws, anchor_anom, k_f, spei_sum, nb_anchor_anom, clim_t1, clim_a,
    cov, lat_f, lon_f, cstd, camp,
]).astype(np.float32)
assert not np.isnan(X_B).any(), "NaN in X_B"

# ---------- predict ----------
pred = np.empty(n, dtype=np.float64)
# Model A: ALL unmasked rows (incl. stragglers in masked months)
unm = (~mm_check)
XA = np.column_stack([test['TWS_t'].values, cov]).astype(np.float32)
pred[unm] = mA.predict(XA[unm])
# Model B: masked rows
mm = mm_check
pB1 = mB1.predict(X_B[mm])
lin = np.column_stack([(clim_t1 - clim_a)[mm], anchor_anom[mm], spei_sum[mm], nb_anchor_anom[mm],
                       np.ones(int(mm.sum()), np.float32)])
s1 = np.empty(mm.sum(), dtype=np.float64)
kv = k_arr[mm]
for k in range(1, 7):
    msk = kv == k
    if msk.any():
        s1[msk] = lin[msk].astype(np.float64) @ coefs[k]
pB2 = s1 + mB2.predict(X_B[mm])
pred[mm] = 0.5 * pB1 + 0.5 * pB2
print(f"Predictions done. mean={pred.mean():.4f} std={pred.std():.4f} nan={np.isnan(pred).sum()}", flush=True)

# ---------- submission ----------
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
assert len(sub) == len(test)
pred_df = pd.DataFrame({'ID': test['ID'], 'Target': pred.astype(np.float32)})
sub_out = sub[['ID']].merge(pred_df, on='ID', how='left')
assert sub_out['Target'].notna().all(), "missing predictions!"
sub_out.to_csv(f'{DL}/submission_v1.csv', index=False)
print(f"Submission saved: {DL}/submission_v1.csv ({len(sub_out):,} rows)", flush=True)

# quick stats per k
test['k'] = k_arr
test['pred'] = pred
print("\nPrediction stats by k:")
print(test.groupby('k')['pred'].agg(['count', 'mean', 'std']).round(4).to_string())
