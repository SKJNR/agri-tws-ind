"""
Final real-vs-synthetic determination with NaN-safe spatial analysis.
Tests:
1. Grid extent + completeness
2. Spatial smoothness of trend & seasonal fields (real: smooth; synthetic: noisy)
3. Seasonal amplitude field spatial coherence (real GRACE: Amazon/monsoon belt >> deserts)
4. White-noise vs spatially-coherent residual structure
"""
import pandas as pd
import numpy as np

train = pd.read_csv('/home/z/my-project/data/Train (1).csv',
                    usecols=['time', 'lat', 'lon', 'TWS_t'])
train['y'] = pd.to_datetime(train['time']).dt.year
train['m'] = pd.to_datetime(train['time']).dt.month

lats = np.sort(train.lat.unique()); lons = np.sort(train.lon.unique())
print(f'Grid: lat [{lats.min()} .. {lats.max()}] n={len(lats)}, '
      f'lon [{lons.min()} .. {lons.max()}] n={len(lons)}, '
      f'cells={len(train.groupby(["lat","lon"]))}')

# ---- per-cell trend & seasonal amplitude (NaN-safe) ----
def cell_stats(d):
    t = d['y'] + d['m'] / 12.0
    slope = np.polyfit(t, d['TWS_t'], 1)[0]
    clim = d.groupby('m')['TWS_t'].mean()
    ampl = clim.max() - clim.min()
    return pd.Series({'slope': slope, 'seas_ampl': ampl})

cs = train.groupby(['lat', 'lon']).apply(cell_stats, include_groups=False).reset_index()
print(f'\ncells with stats: {len(cs)}')

# ---- spatial smoothness via neighbor correlation on the scattered grid ----
def neighbor_corr(field):
    # join each cell to its (lat+1, lon) and (lat, lon+1) neighbors
    f = field.set_index(['lat', 'lon']).slope if 'slope' in field else None
    d = field.set_index(['lat', 'lon'])
    vals = d.iloc[:, -1]  # last column
    out = []
    for dlat, dlon in [(1, 0), (0, 1)]:
        a = vals.rename('a').reset_index()
        a['lat'] = a['lat'] - dlat; a['lon'] = a['lon'] - dlon
        b = vals.rename('b').reset_index()
        mg = a.merge(b, on=['lat', 'lon'])
        if len(mg) > 100:
            out.append(np.corrcoef(mg.a, mg.b)[0, 1])
    return out

tr_c = neighbor_corr(cs[['lat', 'lon', 'slope']])
se_c = neighbor_corr(cs[['lat', 'lon', 'seas_ampl']])
print(f'Trend field neighbor corr: {[f"{c:.3f}" for c in tr_c]}  (real GRACE: 0.7-0.95)')
print(f'Seasonal-amp field neighbor corr: {[f"{c:.3f}" for c in se_c]}  (real GRACE: 0.6-0.9)')

# ---- regional seasonal amplitude (real GRACE fingerprints) ----
def reg(name, lat_r, lon_r):
    sel = cs[(cs.lat >= lat_r[0]) & (cs.lat <= lat_r[1]) &
             (cs.lon >= lon_r[0]) & (cs.lon <= lon_r[1])]
    print(f'{name:20s} seasonal amplitude mean {sel.seas_ampl.mean():.3f}  (n={len(sel)})')

print('\nSeasonal amplitude by region (real GRACE: Amazon 2-4x Sahara):')
reg('Amazon', [-10, 0], [-70, -50])
reg('Sahara', [18, 30], [-10, 30])
reg('Monsoon India', [8, 28], [70, 95])
reg('Greenland', [60, 84], [-73, -12])
reg('Global', [-90, 90], [-180, 180])

# ---- residual whiteness: after removing cell mean + trend + monthly clim,
#      is the residual spatially coherent month to month? ----
piv = train.pivot_table(index=['lat', 'lon'], columns=['y', 'm'], values='TWS_t')
# cell-demeaned field
F = piv.values  # (cells, months)
Fd = F - np.nanmean(F, axis=1, keepdims=True)
# spatial corr of the demeaned field between adjacent cells (same month)
idx = piv.index.to_frame(index=False)
ordr = np.lexsort((idx.lon, idx.lat))
Fd_o = Fd[ordr]; idx_o = idx.iloc[ordr].reset_index(drop=True)
same_col = idx_o.lon.values[1:] == idx_o.lon.values[:-1]
pair_c = np.corrcoef(Fd_o[:-1][same_col], Fd_o[1:][same_col])[0, 1]
print(f'\nDemeaned-field adjacent-cell (1deg lon) corr: {pair_c:.4f}')
print('(real GRACE with added white noise: still high ~0.8+ because signal dominates;')
print(' pure synthetic + white noise: similar; LOW only if field is mostly white noise)')
