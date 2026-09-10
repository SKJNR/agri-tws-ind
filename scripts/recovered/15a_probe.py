import numpy as np, pandas as pd, xarray as xr

da  = xr.open_dataset("/home/z/my-project/scripts/15a_era5_extract/data_stream-moda_stepType-avgad.nc")  # tp
da2 = xr.open_dataset("/home/z/my-project/scripts/15a_era5_extract/data_stream-moda_stepType-avgua.nc")  # sd
dl  = xr.open_dataset("/home/z/my-project/scripts/15a_era5land_sm34_2002_2018.nc")                       # swvl3/4
tp, sd, sm3, sm4 = da["tp"], da2["sd"], dl["swvl3"], dl["swvl4"]

def to1deg(field):
    la = field.latitude.values; lo = field.longitude.values; f = field.values
    lat_c = np.arange(np.floor(la.min())+0.5, np.ceil(la.max()), 1.0)
    lon_c = np.arange(np.floor(lo.min())+0.5, np.ceil(lo.max()), 1.0)
    out = np.full((f.shape[0], len(lat_c), len(lon_c)), np.nan)
    for i,lc in enumerate(lat_c):
        mi = (la>=lc-0.5)&(la<lc+0.5)
        for j,oc in enumerate(lon_c):
            mj = (lo>=oc-0.5)&(lo<oc+0.5)
            sub = f[:, mi, :][:, :, mj]
            if sub.size: out[:,i,j] = np.nanmean(sub, axis=(1,2))
    return lat_c, lon_c, out

lat_c, lon_c, TP = to1deg(tp); _,_, SD = to1deg(sd); _,_, SM3 = to1deg(sm3); _,_, SM4 = to1deg(sm4)
times = pd.to_datetime(da.valid_time.values)
ym_index = {f"{t.year:04d}-{t.month:02d}": i for i,t in enumerate(times)}
print("ERA5 1deg:", TP.shape, times[0].date(), "->", times[-1].date())

tr = pd.read_csv("/home/z/my-project/data/Train (1).csv")
box = tr[(tr.lat>=30)&(tr.lat<72)&(tr.lon>=-15)&(tr.lon<45)].copy()
box["ym"] = box["time"].astype(str).str.slice(0,7)
box["mi"] = box["ym"].map(ym_index).fillna(-1).astype(int)
box = box[box.mi>=0]
lati = {round(v,1): i for i,v in enumerate(lat_c)}; lonj = {round(v,1): j for j,v in enumerate(lon_c)}
box["i"] = box.lat.round(1).map(lati); box["j"] = box.lon.round(1).map(lonj)
box = box.dropna(subset=["i","j"]); box["i"]=box.i.astype(int); box["j"]=box.j.astype(int)
for nm, F in [("tp",TP),("sd",SD),("sm3",SM3),("sm4",SM4)]:
    box[nm] = F[box.mi.values, box.i.values, box.j.values]
box = box.dropna(subset=["tp","sd","sm3","sm4","TWS_t"])
print("matched rows:", len(box), "cells:", box.groupby(['lat','lon']).ngroups)

# per-cell anomalies + detrend
cols = ["TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm3","sm4"]
pieces = []
for key, g in box.groupby(["lat","lon"]):
    t = g["mi"].values.astype(float); tt = t - t.mean()
    d = {"lat": key[0], "lon": key[1]}
    for c in cols:
        y = g[c].values; mu = y.mean()
        d[c+"_an"] = y - mu
        b = np.polyfit(tt, y-mu, 1)[0] if len(g)>=24 else 0.0
        d[c+"_dt"] = y - mu - b*tt
    d["mi"] = g["mi"].values; d["ym"] = g["ym"].values
    pieces.append(pd.DataFrame(d))
B = pd.concat(pieces)
print("anomaly cells:", len(pieces))

def corr(a,b):
    m=~np.isnan(a)&~np.isnan(b)
    if m.sum()<50: return np.nan
    return np.corrcoef(a[m],b[m])[0,1]

print("\n=== per-cell corr with detrended TWS anomaly (fast state), 2002-2015 train era ===")
for c in ["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm3","sm4"]:
    per = B.groupby(["lat","lon"]).apply(lambda g: corr(g[c+"_dt"], g["TWS_t_dt"]), include_groups=False).dropna()
    q = per.quantile([.25,.5,.75]).values
    print(f"{c:16s} median {q[1]:+.3f}  IQR [{q[0]:+.3f},{q[2]:+.3f}]  n={len(per)}")

# snowy cells
sdvar = B.groupby(["lat","lon"])["sd_dt"].std()
snowy = sdvar[sdvar > np.percentile(sdvar.dropna(), 75)].index
sub = B[B.set_index(["lat","lon"]).index.isin(snowy)]
print(f"\n=== SNOWY cells (top-quartile sd variability, n={len(snowy)}), lat median {np.median([k[0] for k in snowy]):.0f} ===")
for c in ["SPEI_01_t","SPEI_03_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd"]:
    per = sub.groupby(["lat","lon"]).apply(lambda g: corr(g[c+"_dt"], g["TWS_t_dt"]), include_groups=False).dropna()
    print(f"  {c:16s} median {per.median():+.3f}")

# incremental R2 beyond given covariates
from numpy.linalg import lstsq
BASE = ["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
def r2_pair(g, extra):
    y = g["TWS_t_dt"].values
    Xb = np.column_stack([g[c+"_dt"].values for c in BASE] + [np.ones(len(g))])
    m = ~np.isnan(y)
    for c in BASE+extra: m &= ~np.isnan(g[c+"_dt"].values)
    y2=y[m]; Xb2=Xb[m]
    r2b = 1 - ((y2-Xb2@lstsq(Xb2,y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    if extra:
        Xe = np.column_stack([Xb2]+[g[c+"_dt"].values[m] for c in extra])
        r2e = 1 - ((y2-Xe@lstsq(Xe,y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    else: r2e = r2b
    return pd.Series({"base":r2b, "ext":r2e})

print("\n=== INCREMENTAL per-cell R2 on detrended TWS (train 2002-2015) ===")
for name, extra in [("none",[]), ("tp",["tp"]), ("sd",["sd"]), ("sm3+sm4",["sm3","sm4"]), ("ALL",["tp","sd","sm3","sm4"])]:
    out = B.groupby(["lat","lon"]).apply(lambda g: r2_pair(g, extra), include_groups=False)
    print(f"  +{name:10s}: base {out['base'].median():.3f} -> {out['ext'].median():.3f} | median gain {(out['ext']-out['base']).median():+.4f} | mean gain {(out['ext']-out['base']).mean():+.4f} | cells gain>0.05: {((out['ext']-out['base'])>0.05).mean()*100:.0f}%")

# surface SM vs deep SM alone
out = B.groupby(["lat","lon"]).apply(lambda g: r2_pair_pair(g), include_groups=False) if False else None
def r2_sm(g):
    y = g["TWS_t_dt"].values
    Xs = np.column_stack([g["SOIL_MOISTURE_dt"].values, np.ones(len(g))])
    Xd = np.column_stack([g["SOIL_MOISTURE_dt"].values, g["sm3_dt"].values, g["sm4_dt"].values, np.ones(len(g))])
    m = ~np.isnan(y)&~np.isnan(Xs).all(1)&~np.isnan(Xd).all(1)
    y2=y[m]
    r2s = 1-((y2-Xs[m]@lstsq(Xs[m],y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    r2d = 1-((y2-Xd[m]@lstsq(Xd[m],y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    return pd.Series({"surf":r2s,"deep":r2d})
out = B.groupby(["lat","lon"]).apply(r2_sm, include_groups=False)
print(f"\n  SOIL_MOISTURE(given) alone: median R2 {out['surf'].median():.3f} -> +ERA5-Land sm3/sm4: {out['deep'].median():.3f} (gain {(out['deep']-out['surf']).median():+.4f})")

B.to_parquet("/home/z/my-project/scripts/15a_box.parquet")
print("\nsaved 15a_box.parquet")
