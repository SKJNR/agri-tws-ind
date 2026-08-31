import pandas as pd, numpy as np
t = pd.read_csv("/home/z/my-project/data/Test (2).csv")
tr = pd.read_csv("/home/z/my-project/data/Train (1).csv", usecols=["lat","lon"])
print("Test rows:", len(t), "Train rows:", len(tr))
print("Test cols:", list(t.columns))
print("\n--- GRID ---")
lats = np.sort(t.lat.unique()); lons = np.sort(t.lon.unique())
print("n unique lat:", len(lats), "n unique lon:", len(lons), "n cells:", t.groupby(['lat','lon']).ngroups)
print("lat range:", lats.min(), "to", lats.max())
print("lon range:", lons.min(), "to", lons.max())
dlat = np.diff(lats); dlon = np.diff(lons)
print("lat spacing: min %.6f max %.6f, unique(rounded 6): %s" % (dlat.min(), dlat.max(), np.unique(np.round(dlat,6))[:10]))
print("lon spacing: min %.6f max %.6f, unique(rounded 6): %s" % (dlon.min(), dlon.max(), np.unique(np.round(dlon,6))[:10]))
# check alignment to grid multiples
lat_mod = np.abs(lats*4 - np.round(lats*4)).max()
lon_mod = np.abs(lons*4 - np.round(lons*4)).max()
print("max |lat*4 - round| =", lat_mod, " max |lon*4 - round| =", lon_mod)
print("lat values sample:", lats[:8], "...", lats[-4:])
print("lon values sample:", lons[:8], "...", lons[-4:])
# grid completeness
full = len(np.arange(lats.min(), lats.max()+0.25, 0.25)) * len(np.arange(lons.min(), lons.max()+0.25, 0.25))
print("full 0.25 box product would be:", full, "-> coverage fraction: %.4f" % (len(t.groupby(['lat','lon']))/full))
# train/test cell identity
cells_t = set(map(tuple, t[['lat','lon']].drop_duplicates().values))
cells_tr = set(map(tuple, tr[['lat','lon']].drop_duplicates().values))
print("cells in test not train:", len(cells_t - cells_tr), " train not test:", len(cells_tr - cells_t))
# latitude bands present
lat_count = t.groupby('lat').size()
print("\ncells per lat band: min %d max %d median %.0f, n lat bands with >0 cells: %d" % (lat_count.min(), lat_count.max(), lat_count.median(), (lat_count>0).sum()))
