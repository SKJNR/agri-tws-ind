import cdsapi, time
c = cdsapi.Client(quiet=True, progress=False)
req = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"],
    "year": [str(y) for y in range(2002, 2019)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [72, -15, 30, 45],
}
t0=time.time(); print("C: submitting ERA5-Land sm1+sm2 ...", flush=True)
try:
    c.retrieve("reanalysis-era5-land-monthly-means", req, "/home/z/my-project/scripts/15a_era5land_sm12_2002_2018.nc")
    print("C DONE in %.1f s" % (time.time()-t0), flush=True)
except Exception as e:
    print("C FAILED: %r" % e, flush=True)
