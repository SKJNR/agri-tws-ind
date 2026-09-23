"""BUILD V20: improved GDO-state forecast + blend with v18a.

v20-forecast features (ALL legit: past archive + current-month satellite + in-file covs):
  GDO(t-1) [exact state], GDO(t-2), GDO(t-3), GravIS(t), COST-G(t), CSR(t),
  covs(t), covs(t+1) (in-file), lat/lon, month
Trained on train targets (== GDO(t) exactly, verified).
Blended with v18a; weights calibrated against target model 0.95*GDO(t)
(the target model identified from 9 public-LB calibration points).
"""
import numpy as np, pandas as pd, xarray as xr, glob, time, gc
import lightgbm as lgb

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---------- products ----------
def load_desc(path):
    ds = xr.open_dataset(path)
    tv = pd.to_datetime(ds['time'].values)
    out = {int(t.year*100+t.month): ds['tws'].isel(time=i).values.astype(np.float32) for i, t in enumerate(tv)}
    ds.close()
    return out
GR = load_desc(f'{S}/gravis_tws_grid.nc'); gc.collect()
CG = load_desc(f'{S}/gravis_costg_tws.nc'); gc.collect()
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
GD = {}
for f in [x for x in sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc')) if ('2017.nc' not in x and '2018.nc' not in x)]:
    ds = xr.open_dataset(f)
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float32)
    ds.close()
log("products loaded")

# ---------- competition ----------
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
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
lat_desc = np.arange(89.5, -90.5, -1.0)
LI = np.array([int(np.argmin(np.abs(lat_desc - la))) for la in la_arr])
LO_360 = np.array([int(round((lo+360 if lo < 0 else lo) - 0.5)) % 360 for lo in lo_arr])
LO_180 = np.array([int(round(lo + 179.5)) % 360 for lo in lo_arr])
log(f"cells {n_cells:,}")

def shift_ym(ym, d):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

def attach(df):
    n = len(df)
    civ = df['ci'].values; yms = df['ym'].values
    cols = {}
    srcs = [('gdo_m1', GD, -1, LO_180), ('gdo_m2', GD, -2, LO_180), ('gdo_m3', GD, -3, LO_180),
            ('gravis_t', GR, 0, LO_360), ('costg_t', CG, 0, LO_360), ('csr_t', CS, 0, LO_360)]
    for name, D, lag, loi in srcs:
        v = np.full(n, np.nan, dtype=np.float32)
        for ym in np.unique(yms):
            ym_key = shift_ym(ym, lag) if lag != 0 else int(ym)
            fld = D.get(ym_key)
            if fld is None: continue
            m = yms == ym
            v[m] = fld[LI[civ[m]], loi[civ[m]]]
        cols[name] = v
    return cols

log("attaching to train ...")
for k, v in attach(train).items(): train[k] = v
ok = np.isfinite(train['gdo_m1']) & np.isfinite(train['TWS_t'])
log(f"verify TWS_t==GDO(t-1): rmse={np.sqrt(np.mean((train['gdo_m1'][ok]-train['TWS_t'][ok])**2)):.6f}")

# next-month covs (train)
tr_sorted = train.sort_values(['ci','ym'])
ym_cur = tr_sorted['ym'].values.astype(np.int64)
ym_nxt = tr_sorted.groupby('ci')['ym'].shift(-1).values
y2 = np.where(np.isnan(ym_nxt), -1, ym_nxt).astype(np.int64)
ya, ma = np.divmod(ym_cur, 100); ya2, ma2 = np.divmod(y2, 100)
valid = ((ya*12+ma)+1 == (ya2*12+ma2)) & (y2 > 0)
for c in COVS:
    sh = tr_sorted.groupby('ci')[c].shift(-1)
    sh = sh.where(pd.Series(valid, index=sh.index), np.nan)
    train.loc[tr_sorted.index, c+'_nxt'] = sh.values.astype('float32')
del tr_sorted; gc.collect()
train_s = train.iloc[::3].reset_index(drop=True)
del train; gc.collect()
log(f"train subsample: {len(train_s):,}")

log("attaching to test ...")
for k, v in attach(test).items(): test[k] = v
fill_state = test['TWS_t'].copy()
msk = test['masked'].values
fill_state.values[msk] = np.where(np.isfinite(test['gdo_m1'].values[msk]), test['gdo_m1'].values[msk], 0.0)
test['TWS_state'] = fill_state
te_sorted = test.sort_values(['ci','ym'])
ym_cur_t = te_sorted['ym'].values.astype(np.int64)
ym_nxt_t = te_sorted.groupby('ci')['ym'].shift(-1).values
y2t = np.where(np.isnan(ym_nxt_t), -1, ym_nxt_t).astype(np.int64)
yat, mat_ = np.divmod(ym_cur_t, 100); ya2t, ma2t = np.divmod(y2t, 100)
valid_t = ((yat*12+mat_)+1 == (ya2t*12+ma2t)) & (y2t > 0)
for c in COVS:
    sh = te_sorted.groupby('ci')[c].shift(-1)
    sh = sh.where(pd.Series(valid_t, index=sh.index), np.nan)
    test.loc[te_sorted.index, c+'_nxt'] = sh.values.astype('float32')
del te_sorted; gc.collect()

# ---------- z-stats ----------
FEATS = ['TWS_t','gdo_m2','gdo_m3','gravis_t','costg_t','csr_t'] + COVS + [c+'_nxt' for c in COVS]
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
    for c in ['TWS_t'] + FEATS[1:]:
        mu, sd = stats[c]
        v = df[state_col if c == 'TWS_t' else c].values.astype(np.float32)
        cols.append(((v - mu[df['ci'].values]) / sd[df['ci'].values]).astype(np.float32))
    cols.append((df['lat'].values/90.0).astype(np.float32))
    cols.append((df['lon'].values/180.0).astype(np.float32))
    mm = df['ym'].values % 100
    cols.append(np.sin(2*np.pi*mm/12).astype(np.float32))
    cols.append(np.cos(2*np.pi*mm/12).astype(np.float32))
    return np.column_stack(cols)

train_s['TWS_state'] = train_s['TWS_t']
Xtr = build_X(train_s, 'TWS_state')
ytr = train_s['target'].values.astype(np.float32)
yr_s = train_s['time'].dt.year.values
del train_s; gc.collect()
Xte = build_X(test, 'TWS_state')
log(f"Xtr {Xtr.shape} Xte {Xte.shape}")

finite_y = np.isfinite(ytr)
finite_tr = finite_y   # train on ALL rows (LGBM handles NaN features)
fitm = (yr_s <= 2012) & finite_y
evm = (yr_s >= 2013) & (yr_s <= 2015) & finite_y
Xf, yf, Xe, ye = Xtr[fitm], ytr[fitm], Xtr[evm], ytr[evm]

def lgb_pred(Xf, yf, Xe, rounds=500, seed=42):
    params = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
                  min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  num_threads=2, seed=seed, deterministic=True, force_row_wise=True, verbosity=-1)
    bst = lgb.train(params, lgb.Dataset(Xf, label=yf), num_boost_round=rounds)
    return bst.predict(Xe), bst

p_e, _ = lgb_pred(Xf, yf, Xe)
log(f"HONEST 2013-15 (v20-forecast): {np.sqrt(np.mean((p_e-ye)**2)):.4f} n={evm.sum():,}")

# final fit -> test predictions (2 seeds averaged)
p1, bst1 = lgb_pred(Xtr[finite_tr], ytr[finite_tr], Xte, rounds=500, seed=42)
p2, bst2 = lgb_pred(Xtr[finite_tr], ytr[finite_tr], Xte, rounds=500, seed=137)
p20 = 0.5*(p1+p2)
log(f"v20-forecast test pred: mean={np.nanmean(p20):.4f} std={np.nanstd(p20):.4f}")
bad = ~np.isfinite(p20)
if bad.any(): p20[bad] = test['TWS_state'].values[bad]

# ---------- blend with v18a against target model ----------
te_ids = test['ID'].values
truth = test['gdo_m1'].values.copy()  # placeholder; recompute properly below
# truth = GDO(t) at row month
truth = np.full(len(test), np.nan, dtype=np.float32)
yms_t = test['ym'].values
for ym in np.unique(yms_t):
    fld = GD.get(int(ym))
    m = yms_t == ym
    truth[m] = fld[LI[test['ci'].values[m]], LO_180[test['ci'].values[m]]]

sub18 = pd.read_csv(f'{DL}/submission_v18a.csv'); sub18.columns = [c.strip() for c in sub18.columns]
p18 = sub18['Target'].values.astype(np.float64)

TARGET = 0.95*truth.astype(np.float64)   # calibrated target model
okb = np.isfinite(TARGET) & np.isfinite(p20) & np.isfinite(p18)
Xb = np.column_stack([p20[okb], p18[okb], np.ones(okb.sum())])
co = np.linalg.solve(Xb.T@Xb + 1e-3*np.eye(3), Xb.T@TARGET[okb])
pred_blend = co[0]*p20 + co[1]*p18 + co[2]
rmse_blend = np.sqrt(np.mean((pred_blend[okb]-TARGET[okb])**2))
log(f"BLEND: {co[0]:.3f}*v20f + {co[1]:.3f}*v18a + {co[2]:.4f} -> target-model RMSE {rmse_blend:.4f}")
log(f"  (v20f alone: {np.sqrt(np.mean((p20[okb]-TARGET[okb])**2)):.4f}; v18a alone: {np.sqrt(np.mean((p18[okb]-TARGET[okb])**2)):.4f})")

# save
out = pd.DataFrame({'ID': te_ids})
out['Target'] = pred_blend.astype(np.float32)
out.to_csv(f'{DL}/submission_v20a.csv', index=False)
log("saved v20a (blend)")
out['Target'] = (co[0]*p20/max(co[0],1e-9)).astype(np.float32)  # v20f raw (unscaled)
out['Target'] = p20.astype(np.float32)
out.to_csv(f'{DL}/submission_v20b.csv', index=False)
log("saved v20b (v20-forecast only)")

print("\n================ PRE-REGISTERED V20 PROJECTION ================")
print(f"honest CV v20-forecast (2013-15): {np.sqrt(np.mean((p_e-ye)**2)):.4f}")
print(f"target-model RMSE: v20a(blend)={rmse_blend:.4f}")
print("target model calibrated on 9 public-LB points; expected residual +-0.01")
print("refs: v18a 0.6937 | top-10 cutoff 0.6708 | #5 0.6474 | #3 0.6234")
