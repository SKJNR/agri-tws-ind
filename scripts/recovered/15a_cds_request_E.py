import cdsapi, time
c = cdsapi.Client(quiet=True, progress=False)
req = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["volumetric_soil_water_layer_4"],
    "year": ["2015"], "month": ["01"], "time": ["00:00"],
    "data_format": "netcdf", "download_format": "unarchived",
    "grid": [1.0, 1.0],
}
t0=time.time(); print("E: testing server-side 1deg grid global swvl4 ...", flush=True)
try:
    c.retrieve("reanalysis-era5-land-monthly-means", req, "/home/z/my-project/scripts/15a_test_grid1.nc")
    print("E DONE in %.1f s" % (time.time()-t0), flush=True)
except Exception as e:
    print("E FAILED: %r" % (e,), flush=True)
