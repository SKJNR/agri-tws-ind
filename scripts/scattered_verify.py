"""VERIFY the comp's GDO version on the 417 scattered unmasked cells in MASKED months.
comp TWS_t(masked month t) should == GDO(t-1) at those cells.
This tests the 2017 block specifically (is the current GDO's volatility in the comp too?)."""
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

def shift_ym(ym, d=-1):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

# scattered unmasked rows in masked months
mfrac = te.groupby('ym')['TWS_t_masked'].mean()
masked_months = [int(m) for m in mfrac[mfrac > 0.9].index]
scat = te[(te['ym'].isin(masked_months)) & (~te['TWS_t_masked'])]
print(f"scattered unmasked rows: {len(scat)} across months {sorted(scat['ym'].unique())}")

print("\ncomp TWS_t(t) vs GDO(t-1) on scattered cells:")
for ym in sorted(scat['ym'].unique()):
    sub = scat[scat['ym'] == ym]
    ym2 = shift_ym(ym, -1)
    fld = GD.get(ym2)
    if fld is None:
        print(f"  {ym}: GDO({ym2}) MISSING"); continue
    gv = []
    for la, lo in zip(sub['lat'].values, sub['lon'].values):
        li = int(np.argmin(np.abs(gd_lat - la)))
        lo_i = int(round(lo + 179.5)) % 360
        gv.append(fld[li, lo_i])
    gv = np.array(gv); cv = sub['TWS_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    rm = np.sqrt(np.mean((gv[ok]-cv[ok])**2))
    print(f"  {ym} (n={ok.sum():3d}): rmse vs GDO({ym2}) = {rm:.5f}  corr={np.corrcoef(gv[ok],cv[ok])[0,1]:+.4f}")
