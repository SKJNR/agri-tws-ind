import cdsapi, time
c = cdsapi.Client(quiet=True, progress=False)
req = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4"],
    "year": [str(y) for y in range(2002, 2019)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [72, -15, 30, 45],
}
t0 = time.time(); print("B: submitting ERA5-Land sm3+sm4 2002-2018 ...", flush=True)
try:
    c.retrieve("reanalysis-era5-land-monthly-means", req, "/home/z/my-project/scripts/15a_era5land_sm34_2002_2018.nc")
    print("B DONE in %.1f s" % (time.time()-t0), flush=True)
except Exception as e:
    print("B FAILED after %.1f s: %r" % (time.time()-t0, e), flush=True)
