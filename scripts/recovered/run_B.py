"""
Model B validation: k>=1 (masked-row) forecasting.
B1: direct multi-k LGBM (EXP1-C style, smoothed climatology + spei_sum), absolute target
B2: two-stage (per-k linear stage1 + residual LGBM) — EXP6 design
Combined system: Model A (k=0, 0.6547) + best B (k>=1), test k-weights.
"""
import sys
sys.path.insert(0, '/home/z/my-project/scripts')
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from pipeline import GridContext, load_train, build_val_indices, COVARS, K_WEIGHTS

t00 = time.time()
train = load_train()
g = GridContext(train, clim_cutoff=pd.Timestamp('2013-01-01'))
print(f"Context ({time.time()-t00:.0f}s)", flush=True)

# ---------- training examples: k=1..6, months < 2013-01 ----------
rng = np.random.RandomState(42)
base = np.where(g.row_time < np.datetime64('2013-01-01'))[0]
Xs, lins, ys, ks = [], [], [], []
for k in range(1, 7):
    amap, has, atws, ssum, complete = g.anchor_info(base, k)
    ok = has & complete & ~np.isnan(atws)
    idxs = base[ok]
    amap_k, atws_k, ssum_k = amap[ok], atws[ok], ssum[ok]
    if len(idxs) > 250_000:
        sel = rng.choice(len(idxs), 250_000, replace=False)
        idxs, amap_k, atws_k, ssum_k = idxs[sel], amap_k[sel], atws_k[sel], ssum_k[sel]
    Xk, link, _ = g.features_B(idxs, k, g.row_cell[idxs], g.row_midx[idxs], amap_k, atws_k, ssum_k)
    Xs.append(Xk); lins.append(link); ys.append(g.row_target[idxs].astype('float32'))
    ks.append(np.full(len(Xk), k, dtype=np.int8))
    print(f"  k={k}: {len(Xk):,}", flush=True)
X_train = np.vstack(Xs); lin_train = np.vstack(lins)
y_train = np.concatenate(ys); k_train = np.concatenate(ks)
del Xs, lins, ys, ks
print(f"Train B: {X_train.shape} ({time.time()-t00:.0f}s)", flush=True)

# ---------- validation rows: k=1..6 ----------
val_list = [(idxs, k) for idxs, k in build_val_indices(g) if k >= 1]
Xv_parts, linv_parts, yv_parts, kv_parts = [], [], [], []
for idxs, k in val_list:
    amap, has, atws, ssum, complete = g.anchor_info(idxs, k)
    ok = has & complete & ~np.isnan(atws)
    idxs = idxs[ok]
    if len(idxs) == 0:
        continue
    Xk, link, _ = g.features_B(idxs, k, g.row_cell[idxs], g.row_midx[idxs], amap[ok], atws[ok], ssum[ok])
    Xv_parts.append(Xk); linv_parts.append(link)
    yv_parts.append(g.row_target[idxs].astype('float32'))
    kv_parts.append(np.full(len(Xk), k, dtype=np.int8))
Xv = np.vstack(Xv_parts); lin_v = np.vstack(linv_parts)
yv = np.concatenate(yv_parts); vk = np.concatenate(kv_parts)
print(f"Val B: {Xv.shape} ({time.time()-t00:.0f}s)", flush=True)

# ---------- B1: direct multi-k ----------
FEATS_B = ['tws_anchor', 'anchor_anom', 'k', 'spei_sum', 'nb_anchor_anom',
           'clim_t1', 'clim_anchor'] + COVARS + ['lat', 'lon', 'cell_std', 'clim_amp']
m1 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=80,
                       min_child_samples=100, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1, max_bin=127)
m1.fit(X_train, y_train)
p1 = m1.predict(Xv)
print(f"B1 trained ({time.time()-t00:.0f}s)", flush=True)

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

m2 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, num_leaves=80,
                       min_child_samples=100, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1, max_bin=127)
m2.fit(X_train, res_train)
s1_val = np.empty(len(yv), dtype=np.float64)
for k in range(1, 7):
    msk = vk == k
    s1_val[msk] = lin_v[msk].astype(np.float64) @ coefs[k]
p2 = s1_val + m2.predict(Xv)
print(f"B2 trained ({time.time()-t00:.0f}s)", flush=True)

# ---------- evaluate ----------
wv = np.array([K_WEIGHTS.get(int(k), 1) for k in vk], dtype=np.float64)


def wrmse(p):
    return np.sqrt(np.average((yv - p) ** 2, weights=wv))


print(f"\nB1 direct: {wrmse(p1):.4f}")
print(f"B2 two-stage: {wrmse(p2):.4f}")
p12 = 0.5 * (p1 + p2)
print(f"B1+B2 avg: {wrmse(p12):.4f}")
for nm, p in [('B1', p1), ('B2', p2), ('B12', p12)]:
    per = {k: np.sqrt(np.mean((yv[vk == k] - p[vk == k]) ** 2)) for k in range(1, 7) if (vk == k).sum() > 0}
    print(f"  {nm} per-k: " + " ".join(f"k={k}:{v:.3f}" for k, v in per.items()))

# combined system estimate: Model A (0.6547 at k=0) + B at k>=1
MODEL_A_K0 = 0.6547
for nm, p in [('B1', p1), ('B2', p2), ('B12', p12)]:
    total = K_WEIGHTS[0] * MODEL_A_K0 ** 2
    for k in range(1, 7):
        msk = vk == k
        if msk.sum() > 0:
            total += K_WEIGHTS[k] * np.mean((yv[msk] - p[msk]) ** 2)
    print(f"COMBINED A+{nm}: {np.sqrt(total / sum(K_WEIGHTS.values())):.4f}")

np.savez('/home/z/my-project/scripts/modelB_val.npz', p1=p1, p2=p2, y=yv, k=vk)
import json
with open('/home/z/my-project/scripts/modelB_coefs.json', 'w') as f:
    json.dump({str(k): list(map(float, v)) for k, v in coefs.items()}, f)
m1.booster_.save_model('/home/z/my-project/scripts/modelB1_val.txt')
m2.booster_.save_model('/home/z/my-project/scripts/modelB2_val.txt')
print(f"Done ({time.time()-t00:.0f}s)", flush=True)
