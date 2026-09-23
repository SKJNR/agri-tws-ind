"""CSR RL06.3 mascon match test vs competition TWS."""
import numpy as np, pandas as pd, xarray as xr

S = '/home/z/my-project/scripts'
DATA = '/home/z/my-project/data'
ds = xr.open_dataset(f'{S}/csr_mascons_all.nc')
tvar = ds['time']
tvals = pd.to_datetime(tvar.values, unit='D', origin=pd.Timestamp('2002-01-01'))
lwe = ds['lwe_thickness'].values.astype(np.float32)   # (258, 720, 1440)
lat_c = ds['lat'].values; lon_c = ds['lon'].values
print(f"CSR mascons: {lwe.shape}, {tvals[0].date()}..{tvals[-1].date()}")
print("lat:", lat_c[:3], "...", lat_c[-3:], " lon:", lon_c[:3], "...", lon_c[-3:])
ym_idx = {t.year*100+t.month: i for i, t in enumerate(tvals)}

# aggregate 0.25 -> 1 deg: block mean
def to1deg_fast(lwe):
    T, la_n, lo_n = lwe.shape
    la1 = np.arange(720 // 4) * 4
    lo1 = np.arange(1440 // 4) * 4
    out = np.full((T, 180, 360), np.nan, dtype=np.float32)
    # block means with nan handling
    for i, li in enumerate(la1):
        for j, lo in enumerate(lo1):
            blk = lwe[:, li:li+4, lo:lo+4]
            n = np.isfinite(blk).sum(axis=(1, 2))
            s = np.where(np.isfinite(blk), blk, 0).sum(axis=(1, 2))
            out[:, i, j] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out
print("aggregating to 1deg ...")
L1 = to1deg_fast(lwe)
# lat grid: CSR lat ASCENDING -89.875..89.875 (0.25 centers); block i (rows 4i..4i+3)
# covers lat -89.875+0.25*4i .. -89.875+0.25*(4i+3); center = -89.875 + 0.25*(4i+1.5)
lat1 = np.array([-89.875 + 0.25*(4*i+1.5) for i in range(180)])  # -89.5, -88.5, ..., 89.5
lat1_pos = {round(float(v), 1): i for i, v in enumerate(lat1)}
lon1 = np.array([0.125 + 0.25*(4*j+1.5) for j in range(360)])  # 0.5, 1.5, ..., 359.5
print("lat1[0], lat1[-1]:", lat1[0], lat1[-1], "lon1[0], lon1[-1]:", lon1[0], lon1[-1])

def gval(la, lo, ym):
    if ym not in ym_idx: return np.nan
    glo = lo if lo >= 0 else lo + 360
    li = lat1_pos.get(round(float(la), 1))
    if li is None: li = int(np.argmin(np.abs(lat1 - la)))
    lo_i = int(round((glo - 0.5))) % 360
    return L1[ym_idx[ym], li, lo_i]

# month coverage
csr_months = set(ym_idx.keys())
print(f"\nCSR mascon months 200204..201812 available: {len([m for m in csr_months if 200204 <= m <= 201812])}")

te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
te['time'] = pd.to_datetime(te['time']); te['ym'] = te['time'].dt.year*100 + te['time'].dt.month

print("\n=== spatial corr at test anchors (CSR mascons) ===")
for ym in [201509, 201601, 201606, 201612, 201807, 201811]:
    sub = te[(te['ym'] == ym) & (~te['TWS_t_masked'])]
    gv = np.array([gval(la, lo, ym) for la, lo in zip(sub['lat'].values, sub['lon'].values)])
    c = sub['TWS_t'].values
    ok = np.isfinite(gv)
    if ok.sum() > 100:
        print(f"  {ym}: corr={np.corrcoef(c[ok], gv[ok])[0,1]:+.4f} n={ok.sum():,}")

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month

print("\n=== per-cell time-series corr (1500 cells, train) ===")
cells = tr[['lat','lon']].drop_duplicates().sample(1500, random_state=0)
rs = []
for la, lo in cells.values:
    sub = tr[(tr['lat'] == la) & (tr['lon'] == lo)]
    gv = np.array([gval(la, lo, ym) for ym in sub['ym'].values])
    cv = sub['TWS_t'].values
    ok = np.isfinite(gv) & np.isfinite(cv)
    if ok.sum() >= 60: rs.append(np.corrcoef(gv[ok], cv[ok])[0,1])
rs = np.array(rs)
print(f"  median={np.median(rs):.3f} mean={np.mean(rs):.3f} p10={np.percentile(rs,10):.3f} p90={np.percentile(rs,90):.3f}")

print("\n=== month coverage vs competition ===")
comp_months = set(tr['ym'].unique()) | set(te['ym'].unique())
full = set(y*100+m for y in range(2002, 2019) for m in range(1, 13) if 200204 <= y*100+m <= 201812)
missing_comp = full - comp_months
csr_missing = full - csr_months
print(f"comp-missing: {len(missing_comp)}  csr-missing: {len(csr_missing)}")
print(f"comp-missing AND csr has data: {sorted(missing_comp - csr_missing)}")
print(f"comp has data AND csr missing: {sorted(comp_months & csr_missing)}")
