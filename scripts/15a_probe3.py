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

da =xr.open_dataset("/home/z/my-project/scripts/15a_era5_extract/data_stream-moda_stepType-avgad.nc")
da2=xr.open_dataset("/home/z/my-project/scripts/15a_era5_extract/data_stream-moda_stepType-avgua.nc")
dl =xr.open_dataset("/home/z/my-project/scripts/15a_era5land_sm34_2002_2018.nc")
dl2=xr.open_dataset("/home/z/my-project/scripts/15a_era5land_sm12_2002_2018.nc")
lat_c,lon_c,TP=to1deg(da["tp"]); _,_,SD=to1deg(da2["sd"])
_,_,SM3=to1deg(dl["swvl3"]); _,_,SM4=to1deg(dl["swvl4"])
_,_,SM1=to1deg(dl2["swvl1"]); _,_,SM2=to1deg(dl2["swvl2"])
times=pd.to_datetime(da.valid_time.values)
ym_index={f"{t.year:04d}-{t.month:02d}":i for i,t in enumerate(times)}

def attach(df):
    df=df.copy(); df["ym"]=df["time"].astype(str).str.slice(0,7)
    df["mi"]=df["ym"].map(ym_index).fillna(-1).astype(int)
    df=df[df.mi>=0]
    lati={round(v,1):i for i,v in enumerate(lat_c)}; lonj={round(v,1):j for j,v in enumerate(lon_c)}
    df["i"]=df.lat.round(1).map(lati); df["j"]=df.lon.round(1).map(lonj)
    df=df.dropna(subset=["i","j"]); df["i"]=df.i.astype(int); df["j"]=df.j.astype(int)
    for nm,F in [("tp",TP),("sd",SD),("sm1",SM1),("sm2",SM2),("sm3",SM3),("sm4",SM4)]:
        df[nm]=F[df.mi.values,df.i.values,df.j.values]
    return df.dropna(subset=["tp","sd","sm1","sm2","sm3","sm4"])

tr=pd.read_csv("/home/z/my-project/data/Train (1).csv")
box=attach(tr[(tr.lat>=30)&(tr.lat<72)&(tr.lon>=-15)&(tr.lon<45)])
te=pd.read_csv("/home/z/my-project/data/Test (2).csv")
tbox=attach(te[(te.lat>=30)&(te.lat<72)&(te.lon>=-15)&(te.lon<45)])
# keep rows where TWS_t is observed (anchors + partial)
tobs=tbox.dropna(subset=["TWS_t"])
print("train rows:",len(box),"cells:",box.groupby(['lat','lon']).ngroups,
      "| test rows w/ TWS:",len(tobs),"cells:",tobs.groupby(['lat','lon']).ngroups,
      "| test months:",sorted(tobs.ym.unique()))

# per-cell monthly climatology from FULL 2002-2018 external record (uses test-era external months
# only to define seasonality — legal, but for honesty compute ALSO from train-only)
def build(df, clim_source):
    # clim_source: dict (lat,lon,col)->12 clim over external record
    pieces=[]
    for key,g in df.groupby(["lat","lon"]):
        d={"lat":key[0],"lon":key[1]}
        for c in ["TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm1","sm2","sm3","sm4"]:
            if c not in g.columns: continue
            y=g[c].values
            mo=g["mi"].values%12
            clim=np.array([np.nanmean(y[mo==m]) for m in range(12)]) if key in clim_source[c] else None
            if clim is None:
                clim=np.array([np.nanmean(y[mo==m]) for m in range(12)])
            d[c]=y-clim[mo]
        d["ym"]=g["ym"].values; d["mi"]=g["mi"].values
        pieces.append(pd.DataFrame(d))
    return pd.concat(pieces)

# climatology from TRAIN era only (2002-2015) for both train and test -> leak-free
clim_train={}
for key,g in box.groupby(["lat","lon"]):
    for c in ["TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm1","sm2","sm3","sm4"]:
        mo=g["mi"].values%12
        clim_train.setdefault(c,{})[key]=np.array([np.nanmean(g[c].values[mo==m]) for m in range(12)])

def ds_(df):
    pieces=[]
    for key,g in df.groupby(["lat","lon"]):
        d={"lat":key[0],"lon":key[1]}
        for c in ["TWS_t","SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","tp","sd","sm1","sm2","sm3","sm4"]:
            y=g[c].values; mo=g["mi"].values%12; clim=clim_train[c][key]
            d[c]=y-clim[mo]
        d["ym"]=g["ym"].values; d["mi"]=g["mi"].values
        pieces.append(pd.DataFrame(d))
    return pd.concat(pieces)

TR=ds_(box); TE=ds_(tobs)
BASE=["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]

def oos(train_df, test_df, feats, label):
    # fit per-cell on train_df, evaluate on test_df
    se=[]; n=0
    for key,gtr in train_df.groupby(["lat","lon"]):
        gte=test_df[(test_df.lat==key[0])&(test_df.lon==key[1])]
        if len(gte)<1: continue
        Xtr=np.column_stack([gtr[c].values for c in feats]+[np.ones(len(gtr))])
        ytr=gtr["TWS_t"].values
        m=~np.isnan(ytr)&~np.isnan(Xtr).any(1)
        if m.sum()<60: continue
        w=lstsq(Xtr[m],ytr[m],rcond=None)[0]
        Xte=np.column_stack([gte[c].values for c in feats]+[np.ones(len(gte))])
        yte=gte["TWS_t"].values
        me=~np.isnan(yte)&~np.isnan(Xte).any(1)
        if me.sum()==0: continue
        pred=Xte[me]@w
        se.append(((yte[me]-pred)**2).sum()); n+=me.sum()
    rmse=np.sqrt(np.sum(se)/n)
    print(f"  {label:28s} OOS RMSE {rmse:.4f}  (n={n})")
    return rmse

print("\n=== TEST 0: baselines on test-era observed TWS ===")
# baselines: mu-only and trend-extrapolation
se_mu=0; se_tr=0; n=0
for key,gtr in box.groupby(["lat","lon"]):
    gte=tobs[(tobs.lat==key[0])&(tobs.lon==key[1])]
    if len(gte)<1: continue
    yte=gte["TWS_t"].values
    mu=gtr["TWS_t"].mean()
    t=gtr["mi"].values.astype(float); tt=t-t.mean()
    b=np.polyfit(tt,gtr["TWS_t"].values-mu,1)[0]
    tte=gte["mi"].values.astype(float); tte=tte-t.mean()
    se_mu+=((yte-mu)**2).sum(); se_tr+=((yte-mu-b*tte)**2).sum(); n+=len(yte)
print(f"  mu only   RMSE {np.sqrt(se_mu/n):.4f}")
print(f"  mu+trendex RMSE {np.sqrt(se_tr/n):.4f}  (n={n})")

print("\n=== TEST 1: honest OOS, train 2002-2015 -> predict TEST-era observed TWS (anchors+partials, D-included) ===")
for feats,label in [(BASE,"given covs (5)"),
                    (["sm4"],"sm4 only"),
                    (BASE+["sm3","sm4"],"given + sm3+sm4"),
                    (BASE+["sm1","sm2","sm3","sm4"],"given + sm1-4"),
                    (BASE+["tp","sd","sm1","sm2","sm3","sm4"],"given + ALL ext")]:
    oos(TR,TE,feats,label)

print("\n=== TEST 2: within-train temporal split 2002-2010 -> 2011-2015 (fast-state tracking, no D) ===")
TR_fit=TR[TR.mi< (2011-2002)*12]; TR_val=TR[TR.mi>=(2011-2002)*12]
for feats,label in [(BASE,"given covs (5)"),
                    (["sm4"],"sm4 only"),
                    (BASE+["sm3","sm4"],"given + sm3+sm4"),
                    (BASE+["tp","sd","sm1","sm2","sm3","sm4"],"given + ALL ext")]:
    oos(TR_fit,TR_val,feats,label)

print("\n=== TEST 3: residual std after regression (train era, in-sample) vs noise floor 0.456 ===")
for feats,label in [(BASE,"given covs"),(BASE+["sm3","sm4"],"+sm3sm4"),(BASE+["tp","sd","sm1","sm2","sm3","sm4"],"+ALL ext")]:
    res_std=[]
    for key,g in TR.groupby(["lat","lon"]):
        X=np.column_stack([g[c].values for c in feats]+[np.ones(len(g))]); y=g["TWS_t"].values
        m=~np.isnan(y)&~np.isnan(X).any(1)
        if m.sum()<60: continue
        r=y[m]-X[m]@lstsq(X[m],y[m],rcond=None)[0]
        res_std.append(r.std())
    res_std=np.array(res_std)
    print(f"  {label:28s} median residual std {np.median(res_std):.4f}  (target TWS_ds std ~ {TR.groupby(['lat','lon'])['TWS_t'].std().median():.4f})")
