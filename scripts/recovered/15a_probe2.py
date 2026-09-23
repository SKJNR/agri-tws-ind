import numpy as np, pandas as pd, xarray as xr
from numpy.linalg import lstsq

# --- load all external fields, regrid to 1deg ---
def to1deg(field):
    la = field.latitude.values; lo = field.longitude.values; f = field.values
    lat_c = np.arange(np.floor(la.min())+0.5, np.ceil(la.max()), 1.0)
    lon_c = np.arange(np.floor(lo.min())+0.5, np.ceil(lo.max()), 1.0)
    out = np.full((f.shape[0], len(lat_c), len(lon_c)), np.nan)
    for i,lc in enumerate(lat_c):
        mi=(la>=lc-0.5)&(la<lc+0.5)
        for j,oc in enumerate(lon_c):
            mj=(lo>=oc-0.5)&(lo<oc+0.5)
            sub=f[:,mi,:][:,:,mj]
            if sub.size: out[:,i,j]=np.nanmean(sub,axis=(1,2))
    return lat_c, lon_c, out

da  = xr.open_dataset("/home/z/my-project/scripts/15a_era5_extract/data_stream-moda_stepType-avgad.nc")
da2 = xr.open_dataset("/home/z/my-project/scripts/15a_era5_extract/data_stream-moda_stepType-avgua.nc")
dl  = xr.open_dataset("/home/z/my-project/scripts/15a_era5land_sm34_2002_2018.nc")
dl2 = xr.open_dataset("/home/z/my-project/scripts/15a_era5land_sm12_2002_2018.nc")
lat_c, lon_c, TP = to1deg(da["tp"]); _,_,SD = to1deg(da2["sd"])
_,_,SM3 = to1deg(dl["swvl3"]); _,_,SM4 = to1deg(dl["swvl4"])
_,_,SM1 = to1deg(dl2["swvl1"]); _,_,SM2 = to1deg(dl2["swvl2"])
times = pd.to_datetime(da.valid_time.values)
ym_index = {f"{t.year:04d}-{t.month:02d}": i for i,t in enumerate(times)}

tr = pd.read_csv("/home/z/my-project/data/Train (1).csv")
box = tr[(tr.lat>=30)&(tr.lat<72)&(tr.lon>=-15)&(tr.lon<45)].copy()
box["ym"] = box["time"].astype(str).str.slice(0,7)
box["mi"] = box["ym"].map(ym_index).fillna(-1).astype(int)
box = box[box.mi>=0]
lati = {round(v,1): i for i,v in enumerate(lat_c)}; lonj = {round(v,1): j for j,v in enumerate(lon_c)}
box["i"]=box.lat.round(1).map(lati); box["j"]=box.lon.round(1).map(lonj)
box=box.dropna(subset=["i","j"]); box["i"]=box.i.astype(int); box["j"]=box.j.astype(int)
for nm,F in [("tp",TP),("sd",SD),("sm1",SM1),("sm2",SM2),("sm3",SM3),("sm4",SM4)]:
    box[nm]=F[box.mi.values, box.i.values, box.j.values]
box=box.dropna(subset=["tp","sd","sm1","sm2","sm3","sm4","TWS_t"])
print("rows:", len(box), "cells:", box.groupby(['lat','lon']).ngroups)

# --- per-cell anomalies: (a) mean-removed [level], (b) monthly-clim-removed [deseason], (c) deseason+detrend ---
cols = ["TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm1","sm2","sm3","sm4"]
pieces=[]
for key,g in box.groupby(["lat","lon"]):
    d={"lat":key[0],"lon":key[1],"mi":g["mi"].values}
    for c in cols:
        y=g[c].values.copy()
        d[c+"_an"]=y-y.mean()                                    # level anomaly (mean removed)
        mo=g["mi"].values%12
        clim=np.array([np.nanmean(y[mo==m]) for m in range(12)])
        ds_=y-clim[mo]                                            # deseasonalized
        d[c+"_ds"]=ds_
        t=g["mi"].values.astype(float); tt=t-t.mean()
        b=np.polyfit(tt,ds_,1)[0] if len(g)>=36 else 0.0
        d[c+"_dd"]=ds_-b*tt                                      # deseason + detrend
    pieces.append(pd.DataFrame(d))
B=pd.concat(pieces)

def corr(a,b):
    m=~np.isnan(a)&~np.isnan(b)
    if m.sum()<50: return np.nan
    return np.corrcoef(a[m],b[m])[0,1]

BASE=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
def r2_pair(g, extra, suffix):
    y=g["TWS_t"+suffix].values
    m=~np.isnan(y)
    for c in BASE+extra: m&=~np.isnan(g[c+suffix].values)
    Xb=np.column_stack([g[c+suffix].values[m] for c in BASE]+[np.ones(m.sum())]); y2=y[m]
    r2b=1-((y2-Xb@lstsq(Xb,y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    if extra:
        Xe=np.column_stack([Xb]+[g[c+suffix].values[m] for c in extra])
        r2e=1-((y2-Xe@lstsq(Xe,y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    else: r2e=r2b
    return pd.Series({"base":r2b,"ext":r2e})

for suffix,label in [("_ds","DESEASONALIZED (monthly clim removed)"),("_an","LEVEL ANOMALY (incl trend+interannual = SLOW+FAST)")]:
    print(f"\n=== per-cell corr with TWS, {label} ===")
    for c in ["SPEI_01_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm1","sm2","sm3","sm4"]:
        per=B.groupby(["lat","lon"]).apply(lambda g,c=c,s=suffix: corr(g[c+s],g["TWS_t"+s]),include_groups=False).dropna()
        print(f"  {c:16s} median {per.median():+.3f}  IQR [{per.quantile(.25):+.3f},{per.quantile(.75):+.3f}]")
    print(f"  --- incremental R2 beyond 5 given covs ({label}) ---")
    for name,extra in [("none",[]),("tp",["tp"]),("sd",["sd"]),("sm3+sm4",["sm3","sm4"]),
                       ("sm1..sm4",["sm1","sm2","sm3","sm4"]),("ALL",["tp","sd","sm1","sm2","sm3","sm4"])]:
        out=B.groupby(["lat","lon"]).apply(lambda g,e=extra,s=suffix: r2_pair(g,e,s),include_groups=False)
        print(f"    +{name:9s}: base {out['base'].median():.3f} -> {out['ext'].median():.3f} | median gain {(out['ext']-out['base']).median():+.4f} | mean gain {(out['ext']-out['base']).mean():+.4f} | >0.05 in {((out['ext']-out['base'])>0.05).mean()*100:.0f}% cells")

# provenance: is given SOIL_MOISTURE == ERA5-Land swvl1 (deseasonalized)?
per=B.groupby(["lat","lon"]).apply(lambda g: corr(g["SOIL_MOISTURE_t_ds"],g["sm1_ds"]),include_groups=False).dropna()
print(f"\nPROVENANCE corr(given SOIL_MOISTURE, ERA5-Land swvl1 deseasonalized): median {per.median():+.3f}, IQR [{per.quantile(.25):+.3f},{per.quantile(.75):+.3f}]")
per=B.groupby(["lat","lon"]).apply(lambda g: corr(g["SOIL_MOISTURE_t_an"],g["sm1_an"]),include_groups=False).dropna()
print(f"PROVENANCE corr(given SOIL, swvl1 level-anom): median {per.median():+.3f}")

# snowy cells deseasonalized
sdvar=B.groupby(["lat","lon"])["sd_an"].std()
snowy=sdvar[sdvar>np.percentile(sdvar.dropna(),75)].index
sub=B[B.set_index(["lat","lon"]).index.isin(snowy)]
print(f"\n=== SNOWY cells deseasonalized (n={len(snowy)}) ===")
for c in ["SPEI_12_t","SOIL_MOISTURE_t","sd","sm4"]:
    per=sub.groupby(["lat","lon"]).apply(lambda g,c=c: corr(g[c+"_ds"],g["TWS_t_ds"]),include_groups=False).dropna()
    print(f"  {c:16s} median {per.median():+.3f}")
B.to_parquet("/home/z/my-project/scripts/15a_box2.parquet"); print("saved 15a_box2.parquet")
