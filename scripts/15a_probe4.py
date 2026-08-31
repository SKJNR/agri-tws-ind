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

dl =xr.open_dataset("/home/z/my-project/scripts/15a_era5land_sm34_2002_2018.nc")
_,_,SM3=to1deg(dl["swvl3"]); _,_,SM4=to1deg(dl["swvl4"])
times=pd.to_datetime(dl.valid_time.values)
ym_index={f"{t.year:04d}-{t.month:02d}":i for i,t in enumerate(times)}

def attach(df):
    df=df.copy(); df["ym"]=df["time"].astype(str).str.slice(0,7)
    df["mi"]=df["ym"].map(ym_index).fillna(-1).astype(int)
    df=df[df.mi>=0]
    return df

tr=pd.read_csv("/home/z/my-project/data/Train (1).csv")
box=attach(tr[(tr.lat>=30)&(tr.lat<72)&(tr.lon>=-15)&(tr.lon<45)])
te=pd.read_csv("/home/z/my-project/data/Test (2).csv")
tobs=attach(te[(te.lat>=30)&(te.lat<72)&(te.lon>=-15)&(te.lon<45)]).dropna(subset=["TWS_t"])
lat_c=np.arange(30.5,72,1.0); lon_c=np.arange(-14.5,45,1.0)
lati={round(v,1):i for i,v in enumerate(lat_c)}; lonj={round(v,1):j for j,v in enumerate(lon_c)}
def add_ext(df):
    df=df.dropna(subset=["lat"]).copy()
    df["i"]=df.lat.round(1).map(lati); df["j"]=df.lon.round(1).map(lonj)
    df=df.dropna(subset=["i","j"]); df["i"]=df.i.astype(int); df["j"]=df.j.astype(int)
    df["sm3"]=SM3[df.mi.values,df.i.values,df.j.values]
    df["sm4"]=SM4[df.mi.values,df.i.values,df.j.values]
    return df.dropna(subset=["sm3","sm4"])
box=add_ext(box); tobs=add_ext(tobs)
print("train:",len(box),"test obs:",len(tobs))

BASE=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]
# build features: train-era climatology removed; trendex feature from train TWS slope
rows_tr=[]; rows_te=[]
for key,g in box.groupby(["lat","lon"]):
    mo=g["mi"].values%12
    cl={"TWS":np.array([np.nanmean(g["TWS_t"].values[mo==m]) for m in range(12)])}
    for c in BASE+["sm3","sm4"]:
        cl[c]=np.array([np.nanmean(g[c].values[mo==m]) for m in range(12)])
    d={c:g[c].values-cl[c][mo] for c in BASE+["sm3","sm4"]}
    d["TWS"]=g["TWS_t"].values-cl["TWS"][mo]
    t=g["mi"].values.astype(float); mu=g["TWS_t"].mean()
    b=np.polyfit(t-mu*0,g["TWS_t"].values-mu,1)[0]
    d["trendex"]=b*(t-t.mean())
    d["trend_slope"]=b
    d["lat"]=key[0]; d["lon"]=key[1]; d["mi"]=g["mi"].values
    rows_tr.append(d)
    gte=tobs[(tobs.lat==key[0])&(tobs.lon==key[1])]
    if len(gte)>0:
        mote=gte["mi"].values%12
        de={c:gte[c].values-cl[c][mote] for c in BASE+["sm3","sm4"]}
        de["TWS"]=gte["TWS_t"].values-cl["TWS"][mote]
        te_t=gte["mi"].values.astype(float)
        de["trendex"]=b*(te_t-t.mean())
        de["lat"]=key[0]; de["lon"]=key[1]; de["mi"]=gte["mi"].values
        rows_te.append(de)
TR=pd.DataFrame([r for d in rows_tr for r in [dict(zip(d.keys(),[v if np.isscalar(v) else v[k] for k,v in []]))]] ) if False else None
# flatten properly
def flat(rows):
    out=[]
    for d in rows:
        n=len(d["mi"])
        rec={k:(v if np.isscalar(v) else None) for k,v in d.items()}
        df=pd.DataFrame({k:(np.full(n,v) if np.isscalar(v) else v) for k,v in d.items()})
        out.append(df)
    return pd.concat(out)
TR=flat(rows_tr); TE=flat(rows_te)
TR.to_csv("/home/z/my-project/scripts/15a_TR.csv",index=False); TE.to_csv("/home/z/my-project/scripts/15a_TE.csv",index=False)

def oos(train_df,test_df,feats,label):
    se=0;n=0
    for key,gtr in train_df.groupby(["lat","lon"]):
        gte=test_df[(test_df.lat==key[0])&(test_df.lon==key[1])]
        if len(gte)<1: continue
        Xtr=np.column_stack([gtr[c].values for c in feats]+[np.ones(len(gtr))])
        ytr=gtr["TWS"].values
        m=~np.isnan(ytr)&~np.isnan(Xtr).any(1)
        if m.sum()<60: continue
        w=lstsq(Xtr[m],ytr[m],rcond=None)[0]
        Xte=np.column_stack([gte[c].values for c in feats]+[np.ones(len(gte))])
        yte=gte["TWS"].values
        me=~np.isnan(yte)&~np.isnan(Xte).any(1)
        if me.sum()==0: continue
        se+=((yte[me]-Xte[me]@w)**2).sum(); n+=me.sum()
    print(f"  {label:34s} OOS RMSE {np.sqrt(se/n):.4f} (n={n})")
    return np.sqrt(se/n)

print("\n=== TEST-era nowcast (anchors+partials, incl D): combined feature sets ===")
print("  target std (observed test TWS_ds): %.4f" % TE["TWS"].std())
oos(TR,TE,["trendex"],"trendex only")
oos(TR,TE,BASE,"given covs only")
oos(TR,TE,["sm4"],"sm4 only")
oos(TR,TE,["trendex"]+BASE,"trendex + given")
oos(TR,TE,["trendex","sm3","sm4"],"trendex + sm3sm4")
oos(TR,TE,["trendex"]+BASE+["sm3","sm4"],"trendex + given + sm3sm4")

# per-cell R2 gain split by climate zone (lat bands)
print("\n=== R2 of nowcast by latitude band (given vs given+sm3sm4+trendex) ===")
for lo,hi in [(30,40),(40,50),(50,60),(60,72)]:
    r1=oos(TR[(TR.lat>=lo)&(TR.lat<hi)],TE[(TE.lat>=lo)&(TE.lat<hi)],["trendex"]+BASE,"")
    r2=oos(TR[(TR.lat>=lo)&(TR.lat<hi)],TE[(TE.lat>=lo)&(TE.lat<hi)],["trendex"]+BASE+["sm3","sm4"],"")
    print(f"  lat {lo}-{hi}: {r1:.4f} -> {r2:.4f} (delta {r2-r1:+.4f})")
