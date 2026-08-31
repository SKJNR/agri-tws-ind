"""
Run 1: Model A (k=0 / 1-month) validation with ablations.
A0: persistence | A1: LGBM direct (8 feats) | A2: LGBM delta | A3: + smoothed anomaly feats
"""
import sys
sys.path.insert(0, '/home/z/my-project/scripts')
import numpy as np
import lightgbm as lgb
import time
from pipeline import GridContext, load_train, build_val_indices, COVARS

t00 = time.time()
train = load_train()
g = GridContext(train, clim_cutoff=__import__('pandas').Timestamp('2013-01-01'))
print(f"Context built ({time.time()-t00:.0f}s)", flush=True)

# val k=0 rows
val_parts = [idxs for idxs, k in build_val_indices(g) if k == 0]
vidx = np.concatenate(val_parts)
# train rows: pre-2013
tr_mask = g.row_time < np.datetime64('2013-01-01')
tidx = np.where(tr_mask)[0]
print(f"Model A train: {len(tidx):,}, val k=0: {len(vidx):,}", flush=True)

ytr = g.row_target[tidx]
yva = g.row_target[vidx]
tws_va = g.M_tws[g.row_midx[vidx], g.row_cell[vidx]]
p0 = np.sqrt(np.mean((yva - tws_va) ** 2))
print(f"A0 persistence: {p0:.4f}", flush=True)


def train_lgb(Xtr, ytr_, delta_base=None, tag=''):
    if delta_base is not None:
        yt = ytr_ - delta_base
    else:
        yt = ytr_
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=80,
                          min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1,
                          max_bin=127)
    m.fit(Xtr, yt)
    return m


# A1: direct, 8 feats (TWS_t + covars)
Xtr = g.features_A(tidx)
Xva = g.features_A(vidx)
m1 = train_lgb(Xtr, ytr, tag='A1')
p1 = m1.predict(Xva)
print(f"A1 LGBM direct: {np.sqrt(np.mean((yva-p1)**2)):.4f} ({time.time()-t00:.0f}s)", flush=True)

# A2: delta (predict target - TWS_t)
tws_tr = Xtr[:, 0]
m2 = train_lgb(Xtr, ytr, delta_base=tws_tr, tag='A2')
p2 = m2.predict(Xva) + tws_va
print(f"A2 LGBM delta: {np.sqrt(np.mean((yva-p2)**2)):.4f}", flush=True)

# A3: + smoothed clim anomaly features (TWS_t - clim_t, clim_t1)
ac_tr, ac_va = g.row_cell[tidx], g.row_cell[vidx]
anom_tr = Xtr[:, 0] - g.clim[ac_tr, g.row_cal_m[tidx]]
anom_va = tws_va - g.clim[ac_va, g.row_cal_m[vidx]]
clim_t1_tr = g.clim[ac_tr, g.row_t1m[tidx]]
clim_t1_va = g.clim[ac_va, g.row_t1m[vidx]]
Xtr3 = np.column_stack([Xtr, anom_tr, clim_t1_tr, clim_t1_tr - g.clim[ac_tr, g.row_cal_m[tidx]]])
Xva3 = np.column_stack([Xva, anom_va, clim_t1_va, clim_t1_va - g.clim[ac_va, g.row_cal_m[vidx]]])
m3 = train_lgb(Xtr3, ytr, tag='A3')
p3 = m3.predict(Xva3)
print(f"A3 +smoothed-anom feats: {np.sqrt(np.mean((yva-p3)**2)):.4f}", flush=True)

# A4: blend A1+A2
for wgt in [0.3, 0.5, 0.7]:
    pb = wgt * p1 + (1 - wgt) * p2
    print(f"A4 blend A1/A2 w={wgt}: {np.sqrt(np.mean((yva-pb)**2)):.4f}", flush=True)

np.savez('/home/z/my-project/scripts/modelA_val.npz', p1=p1, p2=p2, p3=p3, y=yva)
print(f"Done ({time.time()-t00:.0f}s). BEST: see above", flush=True)
