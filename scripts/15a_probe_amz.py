import numpy as np, pandas as pd, xarray as xr
from numpy.linalg import lstsq

def to1deg(field):
    la=field.latitude.values; lo=field.longitude.values; f=field.values
    lat_c=np.arange(np.floor(la.min())+0.5, np.ceil(la.max()),1.0)
    lon_c=np.arange(np.floor(lo.min())+0.5, np.ceil(lo.max()),1.0)
    out=np.full((f.shape[0],len(lat_c),len(lon_c)),np.nan)
    for i,lc in enumerate(lat_c):
        mi=(la>=lc-0.5)&(la<lc+0.5)
        for j,oc in enumerate(lon_c):
            mj=(lo>=oc-0.5)&(lo<oc+0.5)
            sub=f[:,mi,:][:,:,mj]
            if sub.size: out[:,i,j]=np.nanmean(sub,axis=(1,2))
    return lat_c,lon_c,out

dl=xr.open_dataset("/home/z/my-project/scripts/15a_era5land_amz_sm34.nc")
lat_c,lon_c,SM3=to1deg(dl["swvl3"]); _,_,SM4=to1deg(dl["swvl4"])
times=pd.to_datetime(dl.valid_time.values)
ym_index={f"{t.year:04d}-{t.month:02d}":i for i,t in enumerate(times)}
print("Amazon ext grid:",SM4.shape, lat_c[0],"..",lat_c[-1], lon_c[0],"..",lon_c[-1])

tr=pd.read_csv("/home/z/my-project/data/Train (1).csv")
box=tr[(tr.lat>=-20)&(tr.lat<10)&(tr.lon>=-80)&(tr.lon<-45)].copy()
box["ym"]=box["time"].astype(str).str.slice(0,7)
box["mi"]=box["ym"].map(ym_index).fillna(-1).astype(int); box=box[box.mi>=0]
lati={round(v,1):i for i,v in enumerate(lat_c)}; lonj={round(v,1):j for j,v in enumerate(lon_c)}
box["i"]=box.lat.round(1).map(lati); box["j"]=box.lon.round(1).map(lonj)
box=box.dropna(subset=["i","j"]); box["i"]=box.i.astype(int); box["j"]=box.j.astype(int)
box["sm3"]=SM3[box.mi.values,box.i.values,box.j.values]; box["sm4"]=SM4[box.mi.values,box.i.values,box.j.values]
box=box.dropna(subset=["sm3","sm4","TWS_t"])
print("train rows:",len(box),"cells:",box.groupby(['lat','lon']).ngroups)

def corr(a,b):
    m=~np.isnan(a)&~np.isnan(b)
    if m.sum()<50: return np.nan
    return np.corrcoef(a[m],b[m])[0,1]

BASE=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
pieces=[]
for key,g in box.groupby(["lat","lon"]):
    d={"lat":key[0],"lon":key[1]}
    for c in ["TWS_t"]+BASE+["sm3","sm4"]:
        y=g[c].values; mo=g["mi"].values%12
        clim=np.array([np.nanmean(y[mo==m]) for m in range(12)])
        d[c]=y-clim[mo]
    d["mi"]=g["mi"].values
    pieces.append(pd.DataFrame(d))
B=pd.concat(pieces)
print("\n=== Amazon: per-cell corr with deseasonalized TWS (train era) ===")
for c in BASE+["sm3","sm4"]:
    per=B.groupby(["lat","lon"]).apply(lambda g,c=c: corr(g[c],g["TWS_t"]),include_groups=False).dropna()
    print(f"  {c:16s} median {per.median():+.3f}  IQR [{per.quantile(.25):+.3f},{per.quantile(.75):+.3f}]")

def r2_pair(g,extra):
    y=g["TWS_t"].values; m=~np.isnan(y)
    for c in BASE+extra: m&=~np.isnan(g[c].values)
    Xb=np.column_stack([g[c].values[m] for c in BASE]+[np.ones(m.sum())]); y2=y[m]
    r2b=1-((y2-Xb@lstsq(Xb,y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    if extra:
        Xe=np.column_stack([Xb]+[g[c].values[m] for c in extra])
        r2e=1-((y2-Xe@lstsq(Xe,y2,rcond=None)[0])**2).sum()/((y2-y2.mean())**2).sum()
    else: r2e=r2b
    return pd.Series({"base":r2b,"ext":r2e})
for name,extra in [("none",[]),("sm3+sm4",["sm3","sm4"])]:
    out=B.groupby(["lat","lon"]).apply(lambda g,e=extra: r2_pair(g,e),include_groups=False)
    print(f"  +{name:8s}: base {out['base'].median():.3f} -> {out['ext'].median():.3f} | median gain {(out['ext']-out['base']).median():+.4f} | mean gain {(out['ext']-out['base']).mean():+.4f}")
