"""GEOGRAPHIC FINGERPRINT TEST: does the TWS trend/amplitude pattern match real GRACE?

Real GRACE 2002-2018 mass change hotspots (negative trends):
  - Greenland (-250+ Gt/yr), Alaska glaciers, Canadian Arctic, Antarctica (excluded here?)
  - N. India groundwater (-40 Gt/yr), Middle East (Iran), Cuyo Argentina, California Central Valley,
  - Caspian Sea region, Aral Sea
Positive/oscillating: Amazon (huge seasonal amplitude ~15-30 cm), Congo, Niger, Mississippi.

Tests:
  F1: lat/lon coverage of the competition grid.
  F2: per-cell linear trend (2002-2015 train) mapped geographically — top negative trends
      should sit at Greenland/Alaska/India/Iran/Argentina/California.
  F3: seasonal amplitude by region (Amazon largest).
  F4: The 2010-2011 Amazon flood/drought signature in the fast field.
  F5: mu_c geographic pattern (full-record standardization residue).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
tr['ym'] = tr['time'].dt.year*100 + tr['time'].dt.month
tr['t_abs'] = (tr['ym']//100)*12 + (tr['ym']%100) - 1

print("=== F1: grid coverage ===")
print(f"lat: [{tr['lat'].min()}, {tr['lat'].max()}]  lon: [{tr['lon'].min()}, {tr['lon'].max()}]")
cells = tr[['lat','lon']].drop_duplicates()
print(f"cells: {len(cells):,}")

# per-cell linear trend over train
g = tr.groupby(['lat','lon'])
out = []
for (la, lo), sub in g:
    if len(sub) < 60: continue
    x = sub['t_abs'].values.astype(float); y = sub['TWS_t'].values
    b = np.polyfit(x, y, 1)[0]
    # seasonal amplitude: std of monthly climatology
    mu_m = sub.groupby('ym')['TWS_t'].mean()
    out.append((la, lo, b*12, y.std(), len(sub)))
df = pd.DataFrame(out, columns=['lat','lon','trend_per_yr','std','n'])
print(f"\n=== F2: per-cell trend (per year), n={len(df)} ===")
print(df['trend_per_yr'].describe().round(4).to_string())

def region(name, latr, lonr):
    m = (df.lat>=latr[0])&(df.lat<=latr[1])&(df.lon>=lonr[0])&(df.lon<=lonr[1])
    s = df[m]
    if len(s)==0: print(f"  {name:28s}: no cells"); return
    print(f"  {name:28s}: n={len(s):4d} trend={s['trend_per_yr'].mean():+.3f}/yr (median {s['trend_per_yr'].median():+.3f}) std={s['std'].mean():.3f}")

print("\n--- known GRACE mass-loss hotspots (expect NEGATIVE trends) ---")
region("Greenland", (59, 84), (-75, -15))
region("Alaska", (55, 72), (-170, -130))
region("N India/Pakistan", (24, 34), (72, 84))
region("Middle East (Iran)", (25, 40), (44, 62))
region("Cuyo Argentina", (-38, -30), (-72, -64))
region("California", (33, 41), (-124, -118))
region("Caspian/Aral region", (36, 48), (46, 60))
print("\n--- stable / oscillating regions (expect ~0 trend) ---")
region("Sahara", (18, 30), (-10, 30))
region("Amazon", (-12, 2), (-75, -50))
region("Congo", (-6, 4), (12, 30))
region("Siberia", (55, 70), (60, 120))
region("Europe", (40, 55), (0, 25))
region("Australia", (-38, -25), (115, 150))
region("US Midwest", (35, 48), (-100, -85))
region("Patagonia", (-55, -40), (-75, -65))

print("\n=== F3: seasonal amplitude by region (monthly climatology std) ===")
# quick: detrended monthly std proxy = std of (TWS - trend) per cell, by region
for name, latr, lonr in [("Amazon", (-12,2), (-75,-50)), ("Sahara", (18,30), (-10,30)),
                          ("Congo", (-6,4), (12,30)), ("Siberia", (55,70), (60,120)),
                          ("Greenland", (59,84), (-75,-15))]:
    m = (df.lat>=latr[0])&(df.lat<=latr[1])&(df.lon>=lonr[0])&(df.lon<=lonr[1])
    if m.sum(): print(f"  {name:12s}: std={df[m]['std'].mean():.3f}")

print("\n=== F4: biggest |trend| cells (top 15) — geographic sanity ===")
d2 = df.reindex(df['trend_per_yr'].abs().sort_values(ascending=False).index)
print(d2.head(15).round(3).to_string(index=False))
print("\n=== F4b: biggest seasonal-amplitude cells (top 10) ===")
d3 = df.reindex(df['std'].sort_values(ascending=False).index)
print(d3.head(10).round(3).to_string(index=False))
