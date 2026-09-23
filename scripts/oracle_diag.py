"""DIAGNOSE the oracle scoring: baselines + per-month + alignment checks."""
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

LI = np.array([int(np.argmin(np.abs(gd_lat - la))) for la in te['lat'].values])
LO = np.array([int(round(lo + 179.5)) % 360 for lo in te['lon'].values])

yms = te['ym'].values
truth = np.full(len(te), np.nan, dtype=np.float32)
truth_m1 = np.full(len(te), np.nan, dtype=np.float32)
def shift_ym(ym, d):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1
for ym in np.unique(yms):
    fld = GD.get(int(ym)); fld_m1 = GD.get(shift_ym(ym, -1))
    m = yms == ym
    if fld is not None: truth[m] = fld[LI[m], LO[m]]
    if fld_m1 is not None: truth_m1[m] = fld_m1[LI[m], LO[m]]

df = te[['ID','ym','TWS_t_masked']].copy()
df['truth'] = truth
df['state_gdo_m1'] = truth_m1

sub = pd.read_csv('/home/z/my-project/download/submission_v19a.csv')
sub.columns = [c.strip() for c in sub.columns]
df['pred'] = sub['Target'].values

# baselines
ok = np.isfinite(df['truth'])
print("=== baselines (oracle) ===")
print(f"  ULTRA pred=truth:          {np.sqrt(np.mean((df['truth'][ok]-df['truth'][ok])**2)):.5f}")
okm1 = ok & np.isfinite(df['state_gdo_m1'])
print(f"  persistence GDO(t-1):      {np.sqrt(np.mean((df['truth'][okm1]-df['state_gdo_m1'][okm1])**2)):.4f}")
print(f"  v19a:                      {np.sqrt(np.mean((df['truth'][ok]-df['pred'][ok])**2)):.4f}")
print(f"  corr(pred, truth):         {np.corrcoef(df['pred'][ok], df['truth'][ok])[0,1]:.4f}")

# per-month
print("\n=== per-month oracle RMSE ===")
for ym in sorted(df['ym'].unique()):
    m = (df['ym'].values == ym) & ok.values
    if m.sum():
        rm = np.sqrt(np.mean((df['truth'].values[m]-df['pred'].values[m])**2))
        rm_p = np.sqrt(np.mean((df['truth'].values[m]-df['state_gdo_m1'].values[m])**2)) if np.isfinite(df['state_gdo_m1'].values[m]).all() else np.nan
        print(f"  {ym}: v19a={rm:.4f}  persistence={rm_p:.4f}  n={m.sum():,}")

# spot check one cell-month for alignment
print("\n=== spot checks ===")
for ym in [201601, 201602]:
    m = (df['ym'].values == ym) & ok.values
    i = np.where(m)[0][:3]
    for j in i:
        print(f"  ym={ym} id={df['ID'].values[j][:30]} truth={df['truth'].values[j]:+.4f} "
              f"pred={df['pred'].values[j]:+.4f} state_m1={df['state_gdo_m1'].values[j]:+.4f}")
