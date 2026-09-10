"""GravIS deep match: per-cell time-series corr + time-shift test + monthly global mean corr."""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
ds = xr.open_dataset(f'{S}/gravis_tws_grid.nc')
tws = ds['tws'].values.astype(np.float64)
tv = pd.to_datetime(ds['time'].values)
lat_g = ds['lat'].values; lon_g = ds['lon'].values
ym_idx = {t.year*100+t.month: i for i, t in enumerate(tv)}
lat_pos = {v: i for i, v in enumerate(lat_g)}

def gval(la, lo, ym):
    if ym not in ym_idx: return np.nan
    glo = lo if lo >= 0 else lo + 360
    li = lat_pos.get(round(float(la), 1))
    if li is None: li = int(np.argmin(np.abs(lat_g - la)))
    return tws[ym_idx[ym], li, int(round((glo - 0.5))) % 360]

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month

# ---- per-cell time-series corr (sample) ----
cells = tr[['lat','lon']].drop_duplicates().sample(2000, random_state=0)
rs, slopes = [], []
for la, lo in cells.values:
    sub = tr[(tr['lat'] == la) & (tr['lon'] == lo)]
    gv = np.array([gval(la, lo, ym) for ym in sub['ym'].values])
    cv = sub['TWS_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    if ok.sum() >= 60:
        r = np.corrcoef(gv[ok], cv[ok])[0,1]
        rs.append(r)
        sl = np.polyfit(gv[ok], cv[ok], 1)[0]
        slopes.append(sl)
rs = np.array(rs); slopes = np.array(slopes)
print("per-cell time-series corr vs GravIS (2000 cells, train):")
print(f"  median={np.median(rs):.3f} mean={np.mean(rs):.3f} p10={np.percentile(rs,10):.3f} p90={np.percentile(rs,90):.3f}")
print(f"  slope: median={np.median(slopes):.3f} (comp std / gravis std)")

# ---- time-shift test at anchors ----
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month
def nxt(ym):
    y, m = divmod(ym, 100); return (y+1)*100+1 if m == 12 else ym+1
def prv(ym):
    y, m = divmod(ym, 100); return (y-1)*100+12 if m == 1 else ym-1

for anchor in [201601, 201612, 201811]:
    sub = te[(te['ym'] == anchor) & (~te['TWS_t_masked'])]
    c = sub['TWS_t'].values
    line = f"  anchor {anchor}:"
    for label, ym in [('t-2', prv(prv(anchor))), ('t-1', prv(anchor)), ('t', anchor), ('t+1', nxt(anchor)), ('t+2', nxt(nxt(anchor)))]:
        gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv)
        if ok.sum() > 100:
            line += f"  {label}={np.corrcoef(c[ok], gv[ok])[0,1]:+.3f}"
    print(line)

# ---- monthly global-mean time series corr (train) ----
gm_comp = tr.groupby('ym')['TWS_t'].mean()
gm_grav = []
for ym in gm_comp.index:
    i = ym_idx.get(int(ym))
    if i is not None:
        gm_grav.append((ym, np.nanmean(tws[i])))
gm = np.array([v for _, v in gm_grav]); gmv = np.array([y for y, _ in gm_grav])
both = pd.Series(gm, index=gmv).reindex(gm_comp.index).dropna()
print(f"\nglobal-mean time corr (train months): {np.corrcoef(both.values, gm_comp.loc[both.index].values)[0,1]:+.4f}")

# ---- spatial corr every train year (drift over time?) ----
for yr in [2003, 2006, 2009, 2012, 2015]:
    yms = [y for y in tr['ym'].unique() if y // 100 == yr][:3]
    rr = []
    for ym in yms:
        sub = tr[tr['ym'] == ym]
        gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
        ok = np.isfinite(gv)
        rr.append(np.corrcoef(sub['TWS_t'].values[ok], gv[ok])[0,1])
    print(f"  {yr}: spatial corr = {np.mean(rr):+.3f}")
