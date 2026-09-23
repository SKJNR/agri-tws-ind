"""Check our grid coverage + time range to size the ERA5-Land download request."""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon'])
print(f"train rows: {len(train):,}")
print(f"train time range: {train['time'].min()} .. {train['time'].max()}")

cells = train[['lat','lon']].drop_duplicates()
print(f"unique cells: {len(cells):,}")
print(f"lat range: {cells['lat'].min():.2f} .. {cells['lat'].max():.2f}")
print(f"lon range: {cells['lon'].min():.2f} .. {cells['lon'].max():.2f}")

# resolution check
lats = np.sort(cells['lat'].unique())
lons = np.sort(cells['lon'].unique())
dlat = np.diff(lats)
dlon = np.diff(lons)
dlat = dlat[dlat > 1e-6]
dlon = dlon[dlon > 1e-6]
print(f"n unique lats: {len(lats)}, n unique lons: {len(lons)}")
print(f"lat step: {dlat.min():.4f}..{dlat.max():.4f} (median {np.median(dlat):.4f})")
print(f"lon step: {dlon.min():.4f}..{dlon.max():.4f} (median {np.median(dlon):.4f})")

test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon'])
print(f"\ntest rows: {len(test):,}")
print(f"test time range: {test['time'].min()} .. {test['time'].max()}")
tcells = test[['lat','lon']].drop_duplicates()
print(f"test unique cells: {len(tcells):,}")
extra = set(map(tuple, tcells.values)) - set(map(tuple, cells.values))
print(f"test cells not in train: {len(extra)}")

# monthly coverage of test
test['time'] = pd.to_datetime(test['time'])
tm = test.groupby(test['time'].dt.to_period('M')).size()
print(f"\ntest months ({len(tm)}): {sorted([str(p) for p in tm.index])}")
