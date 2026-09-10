"""
Final training: Model A (k=0) + Model B (k>=1, B1 direct + B2 two-stage) on ALL train months.
Saves models + context arrays for the prediction script.
"""
import sys
sys.path.insert(0, '/home/z/my-project/scripts')
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import json
from pipeline import GridContext, load_train, COVARS

t00 = time.time()
train = load_train()
g = GridContext(train, clim_cutoff=pd.Timestamp('2015-09-01'))  # all train months
print(f"Context ({time.time()-t00:.0f}s)", flush=True)

# ---------- Model A: all rows ----------
Xa = g.features_A(np.arange(len(g.train)))
ya = g.row_target.astype('float32')
mA = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=80,
                       min_child_samples=100, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1, max_bin=127)
mA.fit(Xa, ya)
mA.booster_.save_model('/home/z/my-project/scripts/final_modelA.txt')
del Xa, ya
print(f"Model A done ({time.time()-t00:.0f}s)", flush=True)

# ---------- Model B training examples: k=1..6, all months ----------
rng = np.random.RandomState(42)
base = np.arange(len(g.train))
Xs, lins, ys, ks = [], [], [], []
for k in range(1, 7):
    amap, has, atws, ssum, complete = g.anchor_info(base, k)
    ok = has & complete & ~np.isnan(atws)
    idxs = base[ok]
    amap_k, atws_k, ssum_k = amap[ok], atws[ok], ssum[ok]
    if len(idxs) > 350_000:
        sel = rng.choice(len(idxs), 350_000, replace=False)
        idxs, amap_k, atws_k, ssum_k = idxs[sel], amap_k[sel], atws_k[sel], ssum_k[sel]
    Xk, link, _ = g.features_B(idxs, k, g.row_cell[idxs], g.row_midx[idxs], amap_k, atws_k, ssum_k)
    Xs.append(Xk); lins.append(link); ys.append(g.row_target[idxs].astype('float32'))
    ks.append(np.full(len(Xk), k, dtype=np.int8))
    print(f"  k={k}: {len(Xk):,}", flush=True)
X_train = np.vstack(Xs); lin_train = np.vstack(lins)
y_train = np.concatenate(ys); k_train = np.concatenate(ks)
del Xs, lins, ys, ks
print(f"Train B: {X_train.shape} ({time.time()-t00:.0f}s)", flush=True)

# ---------- B1: direct ----------
mB1 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=80,
                        min_child_samples=100, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1, max_bin=127)
mB1.fit(X_train, y_train)
mB1.booster_.save_model('/home/z/my-project/scripts/final_modelB1.txt')
print(f"B1 done ({time.time()-t00:.0f}s)", flush=True)

# ---------- B2: two-stage ----------
coefs = {}
for k in range(1, 7):
    msk = k_train == k
    coef, *_ = np.linalg.lstsq(lin_train[msk].astype(np.float64), y_train[msk].astype(np.float64), rcond=None)
    coefs[k] = coef
    print(f"  s1 k={k}: dClim={coef[0]:.3f} anom={coef[1]:.3f} spei={coef[2]:.3f} nb={coef[3]:.3f} c={coef[4]:.3f}", flush=True)
s1_train = np.empty(len(y_train), dtype=np.float64)
for k in range(1, 7):
    msk = k_train == k
    s1_train[msk] = lin_train[msk].astype(np.float64) @ coefs[k]
res_train = (y_train - s1_train).astype(np.float32)

mB2 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=80,
                        min_child_samples=100, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1, max_bin=127)
mB2.fit(X_train, res_train)
mB2.booster_.save_model('/home/z/my-project/scripts/final_modelB2.txt')
print(f"B2 done ({time.time()-t00:.0f}s)", flush=True)

# ---------- save context arrays for prediction ----------
cells = g.train[['cell_code', 'lat', 'lon']].drop_duplicates().sort_values('cell_code')
np.savez('/home/z/my-project/scripts/final_context.npz',
         clim=g.clim, cell_std=g.cell_std, clim_amp=g.clim_amp, nbr_mat=g.nbr_mat,
         lat_by_cell=cells['lat'].values.astype('float32'),
         lon_by_cell=cells['lon'].values.astype('float32'))
with open('/home/z/my-project/scripts/final_coefs.json', 'w') as f:
    json.dump({str(k): list(map(float, v)) for k, v in coefs.items()}, f)
print(f"Context saved ({time.time()-t00:.0f}s)", flush=True)
print("ALL DONE", flush=True)
