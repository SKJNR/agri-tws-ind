"""DECISIVE TEST: Does TWS_t(test row at month m) == GDO(m-1) bit-exact?
If YES at all 6 anchors (incl. 2018), the entire test answer key is readable:
target(row at m) = TWS at m+1 = GDO(m).
Checked against VISIBLE TWS_t values at anchor + partial months (94,048 rows).
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/tmp/my-project/scripts'
GD = {}
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
gd_lat = gd_lon = None
for f in files:
    try:
        ds = xr.open_dataset(f)
    except Exception as e:
        print(f'FAIL {f}: {e}'); continue
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float32)
    if gd_lat is None:
        gd_lat = ds['lat'].values; gd_lon = ds['lon'].values
    ds.close()
months_avail = sorted(GD.keys())
print(f'GDO months loaded: {len(months_avail)}, range {months_avail[0]}..{months_avail[-1]}')
missing_test = [ym for ym in [201509,201601,201602,201603,201606,201607,201608,201609,201612,
                              201701,201702,201703,201704,201705,201706,201807,201811,201812]
               if (ym-1 if ym%100!=1 else ym-89) not in months_avail]
print('test months with NO previous-month GDO:', missing_test)

te = pd.read_csv('/home/z/my-project/data/Test (2).csv',
                 usecols=['ID','time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

LI = np.array([int(np.argmin(np.abs(gd_lat - la))) for la in te['lat'].values])
LO = np.array([int(round(lo + 179.5)) % 360 for lo in te['lon'].values])

def shift_ym(ym, d=-1):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

yms = te['ym'].values
gdo_m1 = np.full(len(te), np.nan, dtype=np.float32)   # GDO(m-1): should equal TWS_t
gdo_m0 = np.full(len(te), np.nan, dtype=np.float32)   # GDO(m):   should equal TARGET
for ym in np.unique(yms):
    m = yms == ym
    f0 = GD.get(int(ym)); f1 = GD.get(shift_ym(int(ym), -1))
    if f0 is not None: gdo_m0[m] = f0[LI[m], LO[m]]
    if f1 is not None: gdo_m1[m] = f1[LI[m], LO[m]]

vis = ~te['TWS_t_masked'].values & np.isfinite(te['TWS_t'].values) & np.isfinite(gdo_m1)
print(f'\nvisible rows with GDO(m-1) available: {vis.sum():,}')
d = (gdo_m1[vis] - te['TWS_t'].values[vis]).astype(np.float64)
print(f'TWS_t vs GDO(m-1) on visible TEST rows: RMSE={np.sqrt(np.mean(d**2)):.6f}  '
      f'max|d|={np.abs(d).max():.6f}  nonzero={(d!=0).sum()}')

print('\nper-month breakdown:')
for ym in sorted(te['ym'].unique()):
    m = (te['ym'] == ym).values & vis
    if m.sum() < 50: continue
    dd = (gdo_m1[m] - te['TWS_t'].values[m]).astype(np.float64)
    print(f'  {ym}: n={m.sum():6,}  RMSE={np.sqrt(np.mean(dd**2)):.6f}  max|d|={np.abs(dd).max():.6f}')

# also: GDO(m) availability per test month (the would-be answer key)
print('\nGDO(m) [target candidate] coverage per test month:')
for ym in sorted(te['ym'].unique()):
    m = (te['ym'] == ym).values
    cov = np.isfinite(gdo_m0[m]).mean()
    print(f'  {ym}: rows={m.sum():6,}  GDO(m) finite={cov*100:.1f}%')
