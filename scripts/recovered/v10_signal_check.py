"""V10 global signal check — does ERA5-Land deep soil moisture add signal WORLDWIDE?
(agents 15-a measured Europe/N-Africa + Amazon only; this validates globally)
1. Train era: per-cell corr of deseasonalized sm4/sm3 vs TWS anomaly (vs given covs).
2. Test era nowcast (observed test rows, incl. D): per-cell OLS, given covs vs +sm3/sm4.
"""
import numpy as np, pandas as pd

npz = np.load('/home/z/my-project/scripts/v10_sm34_glb.npz')
sm3, sm4, ym_npz = npz['sm3'], npz['sm4'], npz['ym']
DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

def load(fname):
    df = pd.read_csv(f'{DATA}/{fname}', usecols=['time','lat','lon','TWS_t']+COVS)
    df['ym'] = pd.to_datetime(df['time']).dt.year*100 + pd.to_datetime(df['time']).dt.month
    return df

def attach(df):
    la = np.round(df['lat'].values,1); lo = np.round(df['lon'].values,1)
    i = np.floor(la + 56.0).astype(int); j = np.floor(lo + 180.0).astype(int)
    y = df['ym'].values//100; m = df['ym'].values%100
    t = (y-2002)*12 + (m-1)
    df['sm3'] = sm3[t, i, j]; df['sm4'] = sm4[t, i, j]
    return df

train = attach(load('Train (1).csv'))
train['mon'] = train['ym'].values % 100
print(f"train rows with sm4: {train['sm4'].notna().mean():.4f}")

# per-cell deseasonalized correlations (train era)
train['TWSa'] = np.nan
for c in ['sm3','sm4','TWS_t']+COVS:
    train[f'{c}a'] = np.nan
g = train.groupby(['lat','lon','mon'])
for c in ['sm3','sm4','TWS_t']+COVS:
    train[f'{c}a'] = train[c] - g[c].transform('mean')

def pcell_corr(x, y):
    d = train[[x, y]].dropna()
    return d.groupby(train.loc[d.index, ['lat','lon']].apply(lambda r: f"{r.lat}_{r.lon}", axis=1)) \
            .apply(lambda z: z[x].corr(z[y]) if len(z) > 60 else np.nan)
# faster: aggregate with numpy by cell code
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes
def fast_corr(a, b):
    d = train[[a, b, 'cc']].dropna()
    va, vb, cc = d[a].values, d[b].values, d['cc'].values
    order = np.argsort(cc, kind='stable'); va, vb, cc = va[order], vb[order], cc[order]
    bounds = np.flatnonzero(np.diff(cc)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(cc)]])
    out = np.full(starts.shape[0], np.nan)
    for k, (s, e) in enumerate(zip(starts, ends)):
        if e - s > 60:
            x, yv = va[s:e], vb[s:e]
            sx = x.std(); sy = yv.std()
            if sx > 1e-9 and sy > 1e-9:
                out[k] = np.corrcoef(x, yv)[0, 1]
    return out

for c, lab in [('sm4a','sm4 (100-289cm)'), ('sm3a','sm3 (28-100cm)'), ('SOIL_MOISTURE_ta','given SOIL'), ('SPEI_12_t a'.replace(' ',''),'SPEI_12')]:
    key = c if not c.endswith(' ') else 'SPEI_12_ta'
    r = fast_corr(key, 'TWS_t a'.replace(' ',''))
    print(f"corr({lab:18s}, TWS_anom): pct10/25/50/75/90 = {np.nanpercentile(r,10):.3f}/{np.nanpercentile(r,25):.3f}/{np.nanpercentile(r,50):.3f}/{np.nanpercentile(r,75):.3f}/{np.nanpercentile(r,90):.3f}")

# ---- test-era nowcast (observed rows only; incl. D) ----
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked']+COVS)
test['ym'] = pd.to_datetime(test['time']).dt.year*100 + pd.to_datetime(test['time']).dt.month
test = attach(test)
tobs = test[test['TWS_t_masked'].astype(bool) == False].dropna(subset=['TWS_t','sm3','sm4'])
print(f"\ntest observed rows with sm3/4: {len(tobs)}")

# per-cell trend slope from train
tr_g = train.groupby(['lat','lon'])
trend = {}
for key, grp in tr_g:
    t = (grp['ym'].values//100-2002)*12 + grp['ym'].values%100 - 1
    yv = grp['TWS_t'].values
    if len(t) > 60 and np.std(t) > 1:
        trend[key] = np.polyfit(t.astype(float), yv, 1)[0]
tobs['t_abs'] = (tobs['ym'].values//100-2002)*12 + tobs['ym'].values%100 - 1
tobs['slope'] = [trend.get((la,lo), 0.0) for la,lo in zip(tobs['lat'],tobs['lon'])]
tobs['trendex'] = tobs['slope'] * (tobs['t_abs'] - 150)   # centered approx
tobs['cc2'] = (tobs['lat'].round(2).astype(str)+'_'+tobs['lon'].round(2).astype(str))

def oos(feats, label):
    se = 0.0; n = 0
    trmap = {k: gdf for k, gdf in train.groupby(['lat','lon'])}
    for key, gte in tobs.groupby(['lat','lon']):
        gtr = trmap.get(key)
        if gtr is None or len(gtr) < 60: continue
        tt = (gtr['ym'].values//100-2002)*12 + gtr['ym'].values%100 - 1
        ytr = gtr['TWS_t'].values
        sl = np.polyfit(tt.astype(float), ytr, 1)[0] if np.std(tt)>1 else 0.0
        Xtr = np.column_stack([gtr[c].values if c != 'trendex' else sl*(tt-150) for c in feats] + [np.ones(len(gtr))])
        Xte = np.column_stack([gte[c].values if c != 'trendex' else gte['trendex'].values for c in feats] + [np.ones(len(gte))])
        yte = gte['TWS_t'].values
        m = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        if m.sum() < 60: continue
        w = np.linalg.lstsq(Xtr[m], ytr[m], rcond=None)[0]
        me = np.isfinite(Xte).all(1) & np.isfinite(yte)
        if me.sum() == 0: continue
        se += ((yte[me] - Xte[me] @ w)**2).sum(); n += me.sum()
    print(f"  nowcast {label:28s}: OOS RMSE {np.sqrt(se/n):.4f} (n={n})")

print(f"  target std (observed test TWS): {tobs['TWS_t'].std():.4f}")
oos(['trendex'], 'trendex only')
oos(COVS, 'given covs only')
oos(['sm4'], 'sm4 only')
oos(['trendex']+COVS, 'trendex + given covs')
oos(['trendex']+COVS+['sm3','sm4'], 'trendex + given + sm3/sm4')
oos(['trendex']+COVS+['sm3','sm4'], 'trendex + given + sm3/sm4')
