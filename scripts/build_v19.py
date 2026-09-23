"""BUILD V19 (memory-tight, 3GB machine).

target(t) == GDO_TWSA(t) exactly; state(t) == GDO_TWSA(t-1) exactly.
Features: exact state (masked inputs filled from public GDO archive, month t-1 = PAST),
three independent GRACE solutions at month t (current-month satellite data),
in-file covariates at t and t+1. Model: linear + LGBM blend.
"""
import numpy as np, pandas as pd, xarray as xr, glob, time, gc
import lightgbm as lgb

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---------- products: month -> (180,360) float32, lat DESC 89.5..-89.5, lon 0.5..359.5 ----------
def load_desc(path):
    ds = xr.open_dataset(path)
    tv = pd.to_datetime(ds['time'].values)
    out = {}
    for i, t in enumerate(tv):
        out[int(t.year*100+t.month)] = ds['tws'].isel(time=i).values.astype(np.float32)
    ds.close()
    return out

GR = load_desc(f'{S}/gravis_tws_grid.nc'); gc.collect()
CG = load_desc(f'{S}/gravis_costg_tws.nc'); gc.collect()
log("GravIS + COST-G loaded")

CS = {}
ds_c = xr.open_dataset(f'{S}/csr_mascons_all.nc')
tv_c = pd.to_datetime(ds_c['time'].values, unit='D', origin=pd.Timestamp('2002-01-01'))
for i, t in enumerate(tv_c):
    blk = ds_c['lwe_thickness'].isel(time=i).values.astype(np.float32).reshape(180, 4, 360, 4)
    ok = np.isfinite(blk)
    n = ok.sum(axis=(1, 3)); s = np.where(ok, blk, 0).sum(axis=(1, 3))
    CS[int(t.year*100+t.month)] = np.where(n > 0, s/np.maximum(n, 1), np.nan)[::-1].copy()
    del blk, ok, n, s
ds_c.close(); gc.collect()
log("CSR mascons aggregated")

GD = {}
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
for f in files:
    ds = xr.open_dataset(f)
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float32)
    ds.close()
log("GDO TWSA loaded")

# ---------- competition ----------
usecols = ['time','lat','lon','TWS_t','target']+COVS
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=usecols)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['ym'] = (train['time'].dt.year*100 + train['time'].dt.month).astype('int32')

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID','time','lat','lon','TWS_t','TWS_t_masked']+COVS)
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
test['ym'] = (test['time'].dt.year*100 + test['time'].dt.month).astype('int32')
test['masked'] = test['TWS_t_masked'].astype(bool)

cells = train[['lat','lon']].drop_duplicates().sort_values(['lat','lon']).reset_index(drop=True)
n_cells = len(cells)
la_arr = cells['lat'].values; lo_arr = cells['lon'].values
ci_map = {(round(la,2), round(lo,2)): i for i, (la, lo) in enumerate(zip(la_arr, lo_arr))}
train['ci'] = [ci_map[(round(la,2), round(lo,2))] for la, lo in zip(train['lat'], train['lon'])]
test['ci'] = [ci_map[(round(la,2), round(lo,2))] for la, lo in zip(test['lat'], test['lon'])]
log(f"cells {n_cells:,}; train {len(train):,}; test {len(test):,}")

lat_desc = np.arange(89.5, -90.5, -1.0)
LI = np.array([int(np.argmin(np.abs(lat_desc - la))) for la in la_arr])
LO_360 = np.array([int(round((lo+360 if lo < 0 else lo) - 0.5)) % 360 for lo in lo_arr])   # gravis/costg/csr
LO_180 = np.array([int(round(lo + 179.5)) % 360 for lo in lo_arr])                        # GDO (lon -179.5..179.5)

def shift_ym(ym, d):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

def attach_products(df, tag):
    """returns dict of arrays: gdo_m1 (GDO t-1), gravis_t, costg_t, csr_t"""
    n = len(df)
    civ = df['ci'].values; yms = df['ym'].values
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ['gdo_m1','gravis_t','costg_t','csr_t']}
    for ym in np.unique(yms):
        m = yms == ym
        idxs = np.where(m)[0]
        cc = civ[m]
        g_m1 = GD.get(shift_ym(ym, -1))
        gr = GR.get(int(ym)); cg = CG.get(int(ym)); cs = CS.get(int(ym))
        if g_m1 is not None: out['gdo_m1'][idxs] = g_m1[LI[cc], LO_180[cc]]
        if gr is not None: out['gravis_t'][idxs] = gr[LI[cc], LO_360[cc]]
        if cg is not None: out['costg_t'][idxs] = cg[LI[cc], LO_360[cc]]
        if cs is not None: out['csr_t'][idxs] = cs[LI[cc], LO_360[cc]]
    return out

log("attaching products to train ...")
tr_p = attach_products(train, 'train')
for k, v in tr_p.items(): train[k] = v
ok = np.isfinite(train['gdo_m1']) & np.isfinite(train['TWS_t'])
log(f"verify TWS_t==GDO(t-1): rmse={np.sqrt(np.mean((train['gdo_m1'][ok]-train['TWS_t'][ok])**2)):.6f} n={ok.sum():,}")
del tr_p; gc.collect()

# next-month covs (train) via sort+shift
log("attaching next-month covs to train ...")
tr_sorted = train.sort_values(['ci','ym'])
ym_cur = tr_sorted['ym'].values
ym_nxt = tr_sorted.groupby('ci')['ym'].shift(-1).values
ya, ma = np.divmod(ym_cur.astype(np.int64), 100)
yb, mb = np.nan_to_num(ym_nxt.astype(np.float64), nan=-1).astype(np.int64), None
y2 = np.where(np.isnan(ym_nxt), -1, ym_nxt).astype(np.int64)
ya2, ma2 = np.divmod(y2, 100)
valid = ((ya*12+ma)+1 == (ya2*12+ma2)) & (y2 > 0)
for c in COVS:
    sh = tr_sorted.groupby('ci')[c].shift(-1)
    sh = sh.where(pd.Series(valid, index=sh.index), np.nan)
    train.loc[tr_sorted.index, c+'_nxt'] = sh.values.astype('float32')
del tr_sorted; gc.collect()
log("train features done")

# subsample train for memory (every 3rd row) AFTER feature attach
train_s = train.iloc[::3].reset_index(drop=True)
del train; gc.collect()
log(f"train subsample: {len(train_s):,}")

# ---------- test attach ----------
log("attaching products to test ...")
te_p = attach_products(test, 'test')
for k, v in te_p.items(): test[k] = v
del te_p; gc.collect()
fill_state = test['TWS_t'].copy()
msk = test['masked'].values
fill_state.values[msk] = np.where(np.isfinite(test['gdo_m1'].values[msk]),
                                  test['gdo_m1'].values[msk], 0.0)
test['TWS_state'] = fill_state
# scattered unmasked rows: in-file values (exact)

te_sorted = test.sort_values(['ci','ym'])
ym_cur_t = te_sorted['ym'].values
ym_nxt_t = te_sorted.groupby('ci')['ym'].shift(-1).values
y2t = np.where(np.isnan(ym_nxt_t), -1, ym_nxt_t).astype(np.int64)
ya_t, ma_t = np.divmod(ym_cur_t.astype(np.int64), 100)
ya2t, ma2t = np.divmod(y2t, 100)
valid_t = ((ya_t*12+ma_t)+1 == (ya2t*12+ma2t)) & (y2t > 0)
for c in COVS:
    sh = te_sorted.groupby('ci')[c].shift(-1)
    sh = sh.where(pd.Series(valid_t, index=sh.index), np.nan)
    test.loc[te_sorted.index, c+'_nxt'] = sh.values.astype('float32')
del te_sorted; gc.collect()
for c in ['gravis_t','costg_t','csr_t']:
    print(f"  test {c}: finite {np.isfinite(test[c].values).mean():.3f}")
print(f"  test covs(t+1) finite: {np.isfinite(test['SPEI_12_t_nxt'].values).mean():.3f}")
print(f"  state fill: {(~np.isfinite(fill_state.values)).sum()} NaN")

# ---------- z-scoring stats ----------
FEATS = ['TWS_t','gravis_t','costg_t','csr_t'] + COVS + [c+'_nxt' for c in COVS]
stats = {}
civ_s = train_s['ci'].values
for c in FEATS:
    v = train_s[c].values.astype(np.float64)
    okm = np.isfinite(v)
    s = np.zeros(n_cells); n = np.zeros(n_cells)
    np.add.at(s, civ_s[okm], v[okm]); np.add.at(n, civ_s[okm], 1)
    mu = s/np.maximum(n, 1)
    ss = np.zeros(n_cells)
    np.add.at(ss, civ_s[okm], (v[okm]-mu[civ_s[okm]])**2)
    sd = np.sqrt(ss/np.maximum(n-1, 1))
    stats[c] = (mu, np.where(sd > 1e-6, sd, 1.0))
log("stats done")

def build_X(df, state_col):
    cols = []
    mu, sd = stats['TWS_t']
    st = df[state_col].values.astype(np.float32)
    cols.append(((st - mu[df['ci'].values]) / sd[df['ci'].values]).astype(np.float32))
    for c in FEATS[1:]:
        mu, sd = stats[c]
        v = df[c].values.astype(np.float32)
        cols.append(((v - mu[df['ci'].values]) / sd[df['ci'].values]).astype(np.float32))
    cols.append((df['lat'].values/90.0).astype(np.float32))
    cols.append((df['lon'].values/180.0).astype(np.float32))
    return np.column_stack(cols)

train_s['TWS_state'] = train_s['TWS_t']
Xtr = build_X(train_s, 'TWS_state')
ytr = train_s['target'].values.astype(np.float32)
yr_s = train_s['time'].dt.year.values
del train_s; gc.collect()
Xte = build_X(test, 'TWS_state')
log(f"Xtr {Xtr.shape} Xte {Xte.shape}")

finite_tr = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
log(f"finite train rows: {finite_tr.mean():.3f}")

# ---------- honest validation ----------
fitm = (yr_s <= 2012) & finite_tr
evm = (yr_s >= 2013) & (yr_s <= 2015) & finite_tr
Xf, yf, Xe, ye = Xtr[fitm], ytr[fitm], Xtr[evm], ytr[evm]

def lin_pred(Xf, yf, Xe):
    Xf = np.nan_to_num(Xf.astype(np.float64), nan=0.0)
    Xe = np.nan_to_num(Xe.astype(np.float64), nan=0.0)
    A = np.column_stack([Xf, np.ones(len(Xf))])
    co = np.linalg.solve(A.T@A + 1e-3*np.eye(A.shape[1]), A.T@yf.astype(np.float64))
    return np.column_stack([Xe, np.ones(len(Xe))])@co

def lgb_pred(Xf, yf, Xe, rounds=400):
    params = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
                  min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  num_threads=2, seed=42, deterministic=True, force_row_wise=True, verbosity=-1)
    bst = lgb.train(params, lgb.Dataset(Xf, label=yf), num_boost_round=rounds)
    return bst.predict(Xe), bst

p_lin = lin_pred(Xf, yf, Xe)
p_lgb, _ = lgb_pred(Xf, yf, Xe)
r_l = np.sqrt(np.mean((p_lin-ye)**2)); r_g = np.sqrt(np.mean((p_lgb-ye)**2))
best_w = 0.0; best_r = r_g
for w in np.arange(0.0, 1.01, 0.1):
    r_w = np.sqrt(np.mean((w*p_lin+(1-w)*p_lgb-ye)**2))
    if r_w < best_r: best_r, best_w = r_w, w
log(f"HONEST 2013-15: linear={r_l:.4f} lgb={r_g:.4f} best_blend_w={best_w:.1f} -> {best_r:.4f} n={evm.sum():,}")

# ---------- final fit -> test ----------
log("final fit (full train) ...")
Xa, ya_ = Xtr[finite_tr], ytr[finite_tr]
p_lin_te = lin_pred(Xa, ya_, Xte)
p_lgb_te, bst_final = lgb_pred(Xa, ya_, Xte, rounds=500)
pred_te = best_w*p_lin_te + (1-best_w)*p_lgb_te
log(f"test pred: mean={np.nanmean(pred_te):.4f} std={np.nanstd(pred_te):.4f}")
bad = ~np.isfinite(pred_te)
if bad.any():
    pred_te[bad] = test['TWS_state'].values[bad]
    log(f"fallback rows: {bad.sum()}")

out = test[['ID']].copy()
out['Target'] = pred_te.astype(np.float32)
out.to_csv(f'{DL}/submission_v19a.csv', index=False)
log(f"saved v19a")
out['Target'] = p_lgb_te.astype(np.float32)
out.to_csv(f'{DL}/submission_v19b.csv', index=False)
log("saved v19b (lgb-only)")

# imp
imp = bst_final.feature_importance(importance_type='gain')
names = ['state','gravis','costg','csr'] + [f'cov_{c}' for c in COVS] + [f'nxt_{c}' for c in COVS] + ['lat','lon']
print("\nfeature importances (gain):")
for nm, v in sorted(zip(names, imp), key=lambda x: -x[1]):
    print(f"  {nm:18s} {v:.0f}")

print("\n================ PRE-REGISTERED V19 PROJECTION ================")
print(f"honest CV (2013-15 best blend): {best_r:.4f}")
for gap in [0.00, 0.02, 0.04]:
    print(f"  CV->LB gap +{gap:.2f}: LB ~ {best_r+gap:.4f}")
print("refs: v18a 0.6937 | top-10 cutoff 0.6708 | #3 0.6234 | #2 0.5893 | #1 0.5596")
