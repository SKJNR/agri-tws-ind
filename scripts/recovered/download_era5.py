"""Full ERA5 monthly download: 9 variables, 2002-2019, global, regridded to 1° server-side.

Variables (why):
  volumetric_soil_water_layer_1..4 : soil water 0-7/7-28/28-100/100-289cm — deep layers track
                                     groundwater = the D component we can't currently see
  snow_depth_water_equivalent      : snow storage — TWS component in N latitudes, missing entirely
  total_precipitation              : direct TWS input
  evaporation                      : direct TWS loss
  runoff                           : lateral TWS movement
  2m_temperature                   : drought driver (temp spikes, per competition brief)

Output: data/era5_full.nc (~500MB), then aggregate_era5.py turns it into
data/era5_grid.parquet keyed by (ym, cc).

Run: nohup python3 scripts/download_era5.py > download/era5_download.log 2>&1 &
"""
import cdsapi
import time, sys, os

OUT = '/home/z/my-project/data/era5_full.nc'
VARS = [
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'snow_depth',
    'total_precipitation',
    'evaporation',
    'runoff',
    '2m_temperature',
]
YEARS = [str(y) for y in range(2002, 2020)]   # covers train (2002-2015) + test (2015-2018) + t+1 of last test month (2019-01)
MONTHS = [f'{m:02d}' for m in range(1, 13)]

req = {
    'product_type': 'monthly_averaged_reanalysis',
    'variable': VARS,
    'year': YEARS,
    'month': MONTHS,
    'time': '00:00',
    'data_format': 'netcdf',
    'grid': [1.0, 1.0],
}

print(f"requesting {len(VARS)} vars x {len(YEARS)} years x 12 months = {len(VARS)*len(YEARS)*12} fields @1deg", flush=True)

c = cdsapi.Client(timeout=120, quiet=False, progress=True)
licence_wait_rounds = 0
attempt = 0
while True:
    attempt += 1
    try:
        t0 = time.time()
        c.retrieve('reanalysis-era5-single-levels-monthly-means', req, OUT)
        print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min -> {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB) ===", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n[attempt {attempt}] failed: {e}", flush=True)
        if 'licence' in str(e).lower():
            licence_wait_rounds += 1
            if licence_wait_rounds > 20:
                print("giving up after 20 licence waits (~40 min)", flush=True)
                sys.exit(1)
            print("Licence not accepted yet — waiting 120s for user to accept, then retrying...", flush=True)
            time.sleep(120)
        elif 'queue' in str(e).lower() or '429' in str(e):
            print("rate-limited/queued — retrying in 180s...", flush=True)
            time.sleep(180)
        elif attempt < 8:
            print("retrying in 60s...", flush=True)
            time.sleep(60)
        else:
            sys.exit(1)
