"""Test which variable names exist in reanalysis-era5-single-levels-monthly-means."""
import cdsapi

c = cdsapi.Client(timeout=60, quiet=True, progress=False)

candidates = {
    'volumetric_soil_water_layer_1': 'swvl1',
    'volumetric_soil_water_layer_2': 'swvl2',
    'volumetric_soil_water_layer_3': 'swvl3',
    'volumetric_soil_water_layer_4': 'swvl4',
    'snow_depth': 'sde (m weq)',
    'snow_depth_water_equivalent': 'sdwe?',
    'total_precipitation': 'tp',
    'evaporation': 'e',
    'runoff': 'ro',
    '2m_temperature': 't2m',
}

ok, bad = [], []
for var in candidates:
    try:
        c.retrieve('reanalysis-era5-single-levels-monthly-means', {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': var,
            'year': '2015', 'month': '09', 'time': '00:00',
            'data_format': 'netcdf', 'grid': [1.0, 1.0],
        }, f'/home/z/my-project/data/var_test.nc')
        ok.append(var)
        print(f"OK    {var}")
    except Exception as e:
        bad.append(var)
        msg = str(e).split('\n')[0][:80]
        print(f"FAIL  {var}: {msg}")

print(f"\nvalid: {ok}")
print(f"invalid: {bad}")
