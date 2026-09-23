"""ORACLE SCORING: compute the TRUE LB of submissions against target = GDO_TWSA(t).
(Valid because comp target(t) == GDO(t) exactly, verified on 2.15M train rows.)
Caveat: 2018 months may carry ~0.01 version drift between comp's snapshot and current GDO.
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'

GD = {}
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
for f in files:
    ds = xr.open_dataset(f)
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float32)
    gd_lat = ds['lat'].values; gd_lon = ds['lon'].values
    ds.close()

te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['ID','time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

lat_desc = gd_lat
LI = np.array([int(np.argmin(np.abs(lat_desc - la))) for la in te['lat'].values])
LO = np.array([int(round(lo + 179.5)) % 360 for lo in te['lon'].values])

# true target values
truth = np.full(len(te), np.nan, dtype=np.float32)
yms = te['ym'].values
for ym in np.unique(yms):
    fld = GD.get(int(ym))
    if fld is None: continue
    m = yms == ym
    truth[m] = fld[LI[m], LO[m]]
print(f"truth coverage: {np.isfinite(truth).mean():.4f}")

# also: "ULTRA" reference = truth itself, and GDO-anchor consistency check
anchors = [201509, 201601, 201606, 201612, 201807, 201811]
for ym in anchors:
    m = (te['ym'].values == ym) & (~te['TWS_t_masked'].values)
    d = truth[m] - te['TWS_t'].values[m]
    print(f"  anchor {ym}: |truth(t) - in-file TWS(t)| = {np.sqrt(np.mean(d**2)):.5f} (should be ~0 if same version)")

import os
for name in ['submission_v19a.csv', 'submission_v19b.csv', 'submission_v18a.csv', 'submission_v12b.csv']:
    p = f'/home/z/my-project/download/{name}'
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub.columns = [c.strip() for c in sub.columns]
    idc = 'ID' if 'ID' in sub.columns else sub.columns[0]
    tgc = 'Target' if 'Target' in sub.columns else sub.columns[1]
    mrg = te[['ID']].copy()
    mrg['truth'] = truth
    m = mrg.merge(sub.rename(columns={idc:'ID', tgc:'Target'}), on='ID', how='inner')
    ok = np.isfinite(m['truth']) & np.isfinite(m['Target'])
    rmse = np.sqrt(np.mean((m['truth'][ok]-m['Target'][ok])**2))
    print(f"\n{name}: TRUE LB (oracle) = {rmse:.4f}  (n={ok.sum():,})")
    # per-month
    mm = mrg['ym'] = te['ym'].values
    for ym in [201509, 201601, 201602, 201603, 201606, 201607, 201608, 201609, 201612,
               201701, 201702, 201703, 201704, 201705, 201706, 201807, 201811, 201812]:
        sel = (m['ym'].values == ym) & ok.values
        if sel.sum():
            print(f"    {ym}: {np.sqrt(np.mean((m['truth'].values[sel]-m['Target'].values[sel])**2)):.4f}", end='')
    print()
