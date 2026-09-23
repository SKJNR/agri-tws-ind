"""LAG ALIGNMENT TEST: do GravIS/COSTG/CSR fields align with GDO at the same
file-month (X) or the previous file-month (X-1)?

- If GravIS(X) ~= GDO(X):  for a competition row at month m, the feature
  'gravis_t' (file-month m) = GDO(m) = THE TARGET  -> future GRACE info (ILLEGAL lane)
- If GravIS(X) ~= GDO(X-1): feature = the state month              -> gray lane

Also verify against visible test-anchor TWS_t (= GDO(m-1), bit-exact).
"""
import numpy as np, pandas as pd, xarray as xr, glob

S = '/tmp/my-project/scripts'
DATA = '/home/z/my-project/data'

GD = {}
files = sorted(glob.glob(f'{S}/gdo_twsa/twsan_*.nc'))
files = [f for f in files if ('2017.nc' not in f and '2018.nc' not in f)]
for f in files:
    ds = xr.open_dataset(f)
    for i, t in enumerate(pd.to_datetime(ds['time'].values)):
        GD[int(t.year*100+t.month)] = ds['twsan'].isel(time=i).values[0].astype(np.float64)
    gd_lat = ds['lat'].values; gd_lon = ds['lon'].values
    ds.close()

def load_gravis(path):
    ds = xr.open_dataset(path)
    out = {}
    tv = pd.to_datetime(ds['time'].values)
    vname = 'tws' if 'tws' in ds.data_vars else list(ds.data_vars)[0]
    for i, t in enumerate(tv):
        out[int(t.year*100+t.month)] = ds[vname].isel(time=i).values.astype(np.float64)
    lat = ds['lat'].values; lon = ds['lon'].values
    ds.close()
    return out, lat, lon

GR, gr_lat, gr_lon = load_gravis(f'{S}/gravis_tws_grid.nc')
CG, cg_lat, cg_lon = load_gravis(f'{S}/gravis_costg_tws.nc')
print('GravIS months:', len(GR), 'range', min(GR), max(GR))
print('COSTG months:', len(CG), 'range', min(CG), max(CG))

# use test cells
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['lat','lon','time','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time'])
te['ym'] = te['time'].dt.year*100 + te['time'].dt.month
LI = np.array([int(np.argmin(np.abs(gd_lat - la))) for la in te['lat'].values])
LO = np.array([int(round(lo + 179.5)) % 360 for lo in te['lon'].values])
grLI = np.array([int(np.argmin(np.abs(gr_lat - la))) for la in te['lat'].values])
grLO = np.array([int(round((lo+360 if lo < 0 else lo) - 0.5)) % 360 for lo in te['lon'].values])
cgLI = np.array([int(np.argmin(np.abs(cg_lat - la))) for la in te['lat'].values])
cgLO = np.array([int(round((lo+360 if lo < 0 else lo) - 0.5)) % 360 for lo in te['lon'].values])

def shift_ym(ym, d):
    y, m = divmod(int(ym), 100)
    t = y*12 + (m-1) + d
    return (t//12)*100 + (t % 12) + 1

# take a set of months where all products exist; compare field-level correlations
test_months = [201509, 201601, 201606, 201612, 201701, 201703, 201807, 201811]
print('\nField-level corr over test cells (rows of visible+masked cells at that month):')
print(f'{"month":>7} {"corr(GR(X),GDO(X))":>20} {"corr(GR(X),GDO(X-1))":>22} {"corr(CG(X),GDO(X))":>20} {"corr(CG(X),GDO(X-1))":>22}')
for ym in test_months:
    rowm = (te['ym'] == ym).values
    f_g0 = GD.get(ym); f_gm1 = GD.get(shift_ym(ym, -1))
    f_gr = GR.get(ym); f_cg = CG.get(ym)
    if f_g0 is None or f_gm1 is None or f_gr is None or f_cg is None:
        print(f'{ym:>7}  missing'); continue
    g0 = f_g0[LI[rowm], LO[rowm]]; gm1 = f_gm1[LI[rowm], LO[rowm]]
    gr = f_gr[grLI[rowm], grLO[rowm]]; cg = f_cg[cgLI[rowm], cgLO[rowm]]
    ok = np.isfinite(g0) & np.isfinite(gm1) & np.isfinite(gr) & np.isfinite(cg)
    c1 = np.corrcoef(gr[ok], g0[ok])[0, 1]
    c2 = np.corrcoef(gr[ok], gm1[ok])[0, 1]
    c3 = np.corrcoef(cg[ok], g0[ok])[0, 1]
    c4 = np.corrcoef(cg[ok], gm1[ok])[0, 1]
    print(f'{ym:>7} {c1:>20.4f} {c2:>22.4f} {c3:>20.4f} {c4:>22.4f}')

# And: what does anchor TWS_t (= GDO(m-1)) correlate with?
print('\nAnchor visible TWS_t vs products at same file-month:')
for ym in [201509, 201601, 201606, 201612, 201807, 201811]:
    rowm = (te['ym'] == ym).values & (~te['TWS_t_masked'].values)
    tws = te['TWS_t'].values[rowm]
    f_gr = GR.get(ym); f_cg = CG.get(ym)
    if f_gr is None: continue
    gr = f_gr[grLI[rowm], grLO[rowm]]; cg = f_cg[cgLI[rowm], cgLO[rowm]] if f_cg is not None else np.full(len(tws), np.nan)
    ok = np.isfinite(tws) & np.isfinite(gr) & np.isfinite(cg)
    print(f'  {ym}: corr(TWS_t, GravIS(X))={np.corrcoef(tws[ok], gr[ok])[0,1]:+.4f}  '
          f'corr(TWS_t, COSTG(X))={np.corrcoef(tws[ok], cg[ok])[0,1]:+.4f}')
