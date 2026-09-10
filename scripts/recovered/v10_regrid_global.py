"""V10 regrid — CORRECTED. Native ERA5-Land 0.1° centers are at x.0..x.9 (integer-offset).
Competition 1° cell centered at x.5 must receive native [x.0, x.1, ..., x.9]:
  i = floor(lat + 56.0)   (LAT_C[0] = -55.5; floor(v - c0 + 0.5))
  j = floor(lon + 180.0)  (LON_C[0] = -179.5)
Output: scripts/v10_sm34_glb.npz {sm3, sm4: [T=205, 140, 360], lat(140), lon(360), ym}
T = 2002-01..2019-01. NaN-safe: finite-count denominator per bin (mask time-constant verified).
"""
import numpy as np, pandas as pd, xarray as xr, glob, os

LAT_C = np.arange(-55.5, 84.0, 1.0)    # 140
LON_C = np.arange(-179.5, 180.0, 1.0)  # 360 (full ring; competition uses 358 of them)
NLAT, NLON, T = len(LAT_C), len(LON_C), 205
N = NLAT * NLON

sm3 = np.full((T, NLAT, NLON), np.nan, dtype=np.float32)
sm4 = np.full((T, NLAT, NLON), np.nan, dtype=np.float32)

files = sorted(glob.glob('/home/z/my-project/scripts/v10_glb_sm34_*.nc'))
print(f"found {len(files)} regional files"); assert len(files) == 16

for f in files:
    ds = xr.open_dataset(f)
    la = ds.latitude.values; lo = ds.longitude.values
    times = pd.to_datetime(ds.valid_time.values)
    mi = np.array([(t.year - 2002) * 12 + (t.month - 1) for t in times])
    keep = (mi >= 0) & (mi < T); mi_k = mi[keep]
    # full-bin mapping: native [x.0..x.9] -> bin center x.5
    # NOTE: netCDF lat/lon carry float dust (~1e-12, direction varies by box) — round to 0.1° first
    la_c = np.round(la, 1); lo_c = np.round(lo, 1)
    i_bin = np.floor(la_c + 56.0).astype(np.int64)   # -55.5+0.5 = -56 offset base
    j_bin = np.floor(lo_c + 180.0).astype(np.int64)
    ok_la = (i_bin >= 0) & (i_bin < NLAT)
    ok_lo = (j_bin >= 0) & (j_bin < NLON)
    OK = ok_la[:, None] & ok_lo[None, :]           # [nlat, nlon]
    flat = (i_bin[:, None] * NLON + j_bin[None, :])[OK]   # [K]
    for var, arr in [('swvl3', sm3), ('swvl4', sm4)]:
        fv = ds[var].values[keep]                 # [T_k, nlat, nlon]
        vals = fv.reshape(len(mi_k), -1)[:, OK.ravel()]   # [T_k, K]
        fin = np.isfinite(vals)
        # per-time finite counts (mask can vary by month)
        sums = np.zeros((len(mi_k), N), dtype=np.float64)
        cnts_t = np.zeros((len(mi_k), N), dtype=np.float64)
        rows_t = np.arange(len(mi_k))
        np.add.at(sums, (rows_t[:, None], flat[None, :].astype(np.int64)),
                  np.where(fin, vals, 0.0))
        np.add.at(cnts_t, (rows_t[:, None], flat[None, :].astype(np.int64)),
                  fin.astype(np.float64))
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = sums / cnts_t
        mean[cnts_t == 0] = np.nan
        tgt = arr.reshape(T, N)
        rows = np.repeat(mi_k, N); cols = np.tile(np.arange(N), len(mi_k))
        upd = mean.reshape(-1)
        m = np.isfinite(upd)
        tgt[rows[m], cols[m]] = upd[m]
    ds.close()
    print(f"{os.path.basename(f)}: stitched ({int(OK.sum())} native cells)")

cov = np.isfinite(sm4).mean()
print(f"sm4 finite coverage: {cov:.3f}")
ym = np.array([(2002 + i // 12) * 100 + (i % 12) + 1 for i in range(T)])
np.savez_compressed('/home/z/my-project/scripts/v10_sm34_glb.npz',
                    sm3=sm3, sm4=sm4, lat=LAT_C, lon=LON_C, ym=ym)
print("saved scripts/v10_sm34_glb.npz")
