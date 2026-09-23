import pandas as pd, numpy as np
t = pd.read_csv("/home/z/my-project/data/Test (2).csv", usecols=["lat","lon"])
lons = np.sort(t.lon.unique()); lats = np.sort(t.lat.unique())
# find the 3-degree lon gaps
gaps = np.where(np.diff(lons) > 1.5)[0]
for g in gaps:
    print("lon gap:", lons[g], "->", lons[g+1], "(missing:", [round(x,1) for x in np.arange(lons[g]+1, lons[g+1], 1.0)], ")")
# expected full .5 grid
full_lons = np.arange(-179.5, 180, 1.0)
missing = set(np.round(full_lons,1)) - set(np.round(lons,1))
print("missing lons from full 1deg .5-centered grid:", sorted(missing))
# lat similarly
print("full lat range count:", len(np.arange(-55.5, 84, 1.0)), "actual:", len(lats))
# check cell (lat,lon) pairs = land mask; count per lat
cells = t[['lat','lon']].drop_duplicates()
per_lat = cells.groupby('lat').size()
print("\nlat bands with <= 50 cells:", dict(per_lat[per_lat<=50]))
# hemisphere summary
print("N-hemisphere cells:", (cells.lat>0).sum(), " S-hemisphere:", (cells.lat<0).sum())
# check the 3-deg gap location in lat: which lats have cells at the missing lon region
if gaps:
    gl = lons[gaps[0]]+1
    near = cells[(cells.lon>gl-2)&(cells.lon<gl+2)]
    print("\ncells near gap:", len(near))
