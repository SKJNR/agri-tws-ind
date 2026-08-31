"""
Is the competition TWS data REAL GRACE (or real-calibrated), or fully synthetic?
Real GRACE has known physical fingerprints:
- Greenland / Antarctica: strong NEGATIVE trend (ice mass loss, -100+ Gt/yr)
- Alaska glaciers: negative trend
- Amazon: strong seasonal cycle + 2010/2015-16 drought signals
- Sahara/deserts: near-zero variance
If the per-cell trend field matches these geographic patterns -> data is real
(or real-anchored) -> external TWS products could match targets (exploit possible).
If trends are spatially random -> synthetic -> external TWS cannot match.
"""
import pandas as pd
import numpy as np

train = pd.read_csv('/home/z/my-project/data/Train (1).csv',
                    usecols=['time', 'lat', 'lon', 'TWS_t'])
train['y'] = pd.to_datetime(train['time']).dt.year
train['m'] = pd.to_datetime(train['time']).dt.month

# per-cell linear trend (2002-2015)
g = train.groupby(['lat', 'lon'])
trend = g.apply(lambda d: np.polyfit(d['y'] + d['m'] / 12.0, d['TWS_t'], 1)[0], include_groups=False)
trend = trend.rename('slope').reset_index()

def region(name, lat_r, lon_r):
    sel = trend[(trend.lat >= lat_r[0]) & (trend.lat <= lat_r[1]) &
                (trend.lon >= lon_r[0]) & (trend.lon <= lon_r[1])]
    print(f'{name:22s} n={len(sel):5d}  mean slope {sel.slope.mean():+.5f}/yr  '
          f'frac<0 {100*(sel.slope<0).mean():4.1f}%')
    return sel

print('=== Per-cell TWS trend by region (2002-2015) ===')
print('(real GRACE expectation: Greenland/Alaska strongly NEGATIVE, ~-0.02..-0.05/yr anomaly units)')
region('Greenland',        [60, 84], [-73, -12])
region('Antarctica',       [-90, -60], [-180, 180])
region('Alaska',           [55, 72], [-170, -130])
region('Amazon basin',     [-15, 5], [-75, -45])
region('Sahara',           [16, 32], [-15, 35])
region('Congo basin',      [-10, 5], [12, 32])
region('Global',           [-90, 90], [-180, 180])

# spatial autocorrelation of the trend field (real GRACE trends are spatially smooth)
piv = trend.pivot(index='lat', columns='lon', values='slope').values
print('\n=== Spatial smoothness of trend field ===')
# correlation between adjacent lat bands
c_adj = np.corrcoef(piv[:-1].ravel(), piv[1:].ravel())[0, 1]
c_shift2 = np.corrcoef(piv[:-2].ravel(), piv[2:].ravel())[0, 1]
print(f'corr(trend rows, trend rows+1deg lat): {c_adj:.4f}')
print(f'corr(trend rows, trend rows+2deg lat): {c_shift2:.4f}')
print('(real GRACE: >0.8; random synthetic: ~0)')

# same for TWS variance field
var = g['TWS_t'].std().rename('std').reset_index()
pv2 = var.pivot(index='lat', columns='lon', values='std').values
c_v = np.corrcoef(pv2[:-1].ravel(), pv2[1:].ravel())[0, 1]
print(f'corr(std rows, std rows+1deg lat): {c_v:.4f}')

# month-of-year seasonality strength (Amazon should be strong)
gm = train.groupby(['lat', 'lon', 'm'])['TWS_t'].mean().reset_index()
seas = gm.groupby(['lat', 'lon'])['TWS_t'].std().rename('seas_std').reset_index()
sel_amz = seas[(seas.lat >= -10) & (seas.lat <= 0) & (seas.lon >= -70) & (seas.lon <= -50)]
sel_sah = seas[(seas.lat >= 18) & (seas.lat <= 30) & (seas.lon >= -10) & (seas.lon <= 30)]
print(f'\nAmazon seasonality std: {sel_amz.seas_std.mean():.4f} (real GRACE: large, ~0.3+)')
print(f'Sahara seasonality std: {sel_sah.seas_std.mean():.4f} (real GRACE: small, ~0.05)')
