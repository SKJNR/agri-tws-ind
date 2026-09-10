"""Tiny CDS test request: 1 variable, 1 month, regridded to 1° — verifies
(a) auth works, (b) server-side regridding accepted, (c) grid centers match our x.5 cells."""
import cdsapi
import xarray as xr

c = cdsapi.Client(timeout=60, quiet=False)
try:
    c.retrieve('reanalysis-era5-single-levels-monthly-means', {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'snow_depth_water_equivalent',
        'year': '2015',
        'month': '09',
        'time': '00:00',
        'data_format': 'netcdf',
        'grid': [1.0, 1.0],   # server-side regrid to 1°x1°
    }, '/home/z/my-project/data/cds_test.nc')
    print("\n=== DOWNLOAD OK ===")
except Exception as e:
    print(f"\n=== FAILED: {e} ===")
    raise SystemExit(1)

# inspect the file
try:
    ds = xr.open_dataset('/home/z/my-project/data/cds_test.nc')
    print(ds)
    lat = ds[sorted(ds.coords, key=lambda k: k.lower())[0] if 'lat' not in ds.coords and 'latitude' not in ds.coords else ('lat' if 'lat' in ds.coords else 'latitude')].values
    lon = ds['lon' if 'lon' in ds.coords else 'longitude'].values
    print(f"\nlat: {lat.min()} .. {lat.max()}  n={len(lat)}  step={lat[1]-lat[0]:.3f}")
    print(f"lon: {lon.min()} .. {lon.max()}  n={len(lon)}  step={lon[1]-lon[0]:.3f}")
    print(f"lat centers at .5? {abs(lat[0] - round(lat[0])):.3f} offset")
    print(f"first lats: {lat[:5]}")
    print(f"first lons: {lon[:5]}")
except Exception as e:
    print(f"inspect failed: {e}")
