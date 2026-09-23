"""2-b step 6: (F) noise-floor: local roughness of train fields vs anchor fields -> lambda_train vs lambda_test.
Also honest out-of-sample rank of the field."""
import numpy as np, pandas as pd
d = np.load('/home/z/my-project/scripts/tb_cache.npz')
A_dt, yms, mu_c = d['A_dt'], d['yms'], d['mu_c']
lat_c, lon_c = d['lat_c'], d['lon_c']
T, n_cells = A_dt.shape

# neighbor structure over ALL cells
key = {(int(round(a*10)), int(round(b*10))): i for i,(a,b) in enumerate(zip(lat_c, lon_c))}
nbrs = {}
for (a,b), i in key.items():
    js = [key.get((a+da, b+db)) for da,db in [(10,0),(-10,0),(0,10),(0,-10)]]
    js = [j for j in js if j is not None]
    if len(js)>=3: nbrs[i] = js
idx_nb = np.array(sorted(nbrs.keys()))
print(f"cells with >=3 neighbors: {len(idx_nb)} / {n_cells}")

def roughness(field):
    """field: (n_cells,) -> array of (x_c - mean nbrs) over valid cells (all nbrs finite)"""
    diffs = []
    for i in idx_nb:
        js = nbrs[i]
        vals = field[js]
        if np.isnan(vals).any() or np.isnan(field[i]): continue
        diffs.append(field[i] - np.mean(vals))
    return np.array(diffs)

# ---------- load test anchors ----------
DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = pd.to_datetime(train['time']).dt.year*100 + pd.to_datetime(train['time']).dt.month
pos_map = {(int(round(float(la)*10)), int(round(float(lo)*10))): int(cc) for cc,la,lo in
           zip(train.drop_duplicates('cc').sort_values('cc')['cc'], train.drop_duplicates('cc').sort_values('cc')['lat'], train.drop_duplicates('cc').sort_values('cc')['lon'])}

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['ym'] = pd.to_datetime(test['time']).dt.year*100 + pd.to_datetime(test['time']).dt.month
test['cc'] = [pos_map.get((int(round(float(a)*10)), int(round(float(b)*10))), -1) for a,b in zip(test['lat'], test['lon'])]
test['masked'] = test['TWS_t_masked'].astype(bool)
for c in ['TWS_t']+COVS: test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')

anchor_yms = sorted(test[~test['masked']].groupby('ym')['masked'].count().index.tolist())
# keep only ~fully-unmasked months
cnt = test.groupby('ym')['masked'].agg(['size','sum'])
anchor_yms = [int(y) for y in cnt[(cnt['sum']==0)].index]
print("anchor months:", anchor_yms)

A_anchor = {}   # anchor anomaly fields (TWS - mu_c)
for y in anchor_yms:
    sub = test[(test['ym']==y) & (~test['masked'])]
    f = np.full(n_cells, np.nan, dtype=np.float64)
    f[sub['cc'].values] = sub['TWS_t'].values
    A_anchor[y] = f - mu_c

# ---------- roughness: train months ----------
F_tr = np.full((T, n_cells), np.nan)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
F_tr[train['ym'].map(ym_to_i).values, train['cc'].values] = pd.to_numeric(train['TWS_t'], errors='coerce').values
# train anomaly relative to per-cell mean (roughness is invariant to static offsets anyway)
rough_tr = []
for t in range(T):
    f = F_tr[t]
    r = roughness(f - np.nanmean(f))
    if len(r) < 10000: continue
    rough_tr.append(np.mean(r**2))
rough_tr = np.array(rough_tr)
print(f"\ntrain months roughness E[(x-mean nbrs)^2]: mean={rough_tr.mean():.5f} std={rough_tr.std():.5f} (n={len(rough_tr)})")

rough_an = []
for y in anchor_yms:
    f = A_anchor[y] + mu_c
    r = roughness(f - np.nanmean(f))
    rough_an.append(np.mean(r**2))
    print(f"anchor {y}: roughness={np.mean(r**2):.5f}")

# Also roughness of train detrended field for reference (same statistic on A_dt)
rough_dt = []
for t in range(T):
    f = A_dt[t]
    r = roughness(f - np.nanmean(f))
    if len(r) < 10000: continue
    rough_dt.append(np.mean(r**2))
rough_dt = np.array(rough_dt)
print(f"train DETRENDED field roughness: mean={rough_dt.mean():.5f}")

# roughness of (anchor - D-hat): isolates FAST+noise
Dhat = np.nanmean(np.stack([A_anchor[y] for y in anchor_yms]), axis=0)
print(f"\nD-hat std: {np.nanstd(Dhat):.4f}")
for y in anchor_yms:
    f = A_anchor[y] - Dhat
    r = roughness(f)
    print(f"anchor {y} minus D-hat: roughness={np.mean(r**2):.5f} (n={len(r)}), field std={np.nanstd(f):.4f}")

# roughness of train field minus per-cell mean AND minus 25-month local smooth? simpler: A_dt - its 5-PC projection
full = ~np.isnan(A_dt).any(axis=0)
A_f = A_dt[:, full] - np.nanmean(A_dt[:, full], axis=0)
U,S_,Vt = np.linalg.svd(A_f, full_matrices=False)
for K in [5, 20, 100]:
    rec = (U[:,:K]*S_[:K]) @ Vt[:K]
    resid = A_f - rec
    # map back to all cells
    rr = []
    for t in range(T):
        f = np.full(n_cells, np.nan); f[full] = resid[t]
        r = roughness(f)
        if len(r) >= 10000: rr.append(np.mean(r**2))
    print(f"train residual roughness after removing top-{K} PCs: mean={np.mean(rr):.5f}")

# ---------- honest out-of-sample rank ----------
print("\nHONEST out-of-sample rank: PCA fit on 2002-2010, coverage of 2011-2015 field variance")
t_split = int(np.searchsorted(yms, 201101))
Atr = A_f[:t_split] - A_f[:t_split].mean(0)
Ate = A_f[t_split:] - A_f[:t_split].mean(0)
U2,S2,V2 = np.linalg.svd(Atr, full_matrices=False)
for K in [10, 30, 50, 100, 200, 300]:
    rec = Ate @ V2[:K].T @ V2[:K]
    frac = 1 - (Ate-rec).var()/Ate.var()
    print(f"  K={K:4d}: covers {frac*100:.1f}% of eval-window variance (noise-free coverage ~ {((frac-0.16)/0.84)*100:.1f}% of signal)")
