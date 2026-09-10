import cdsapi, time
c = cdsapi.Client(quiet=True, progress=False)
req = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["total_precipitation", "snow_depth"],
    "year": [str(y) for y in range(2002, 2019)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [72, -15, 30, 45],
}
t0 = time.time(); print("A: submitting ERA5 tp+sd 2002-2018 ...", flush=True)
try:
    c.retrieve("reanalysis-era5-single-levels-monthly-means", req, "/home/z/my-project/scripts/15a_era5_tp_sd_2002_2018.nc")
    print("A DONE in %.1f s" % (time.time()-t0), flush=True)
except Exception as e:
    print("A FAILED after %.1f s: %r" % (time.time()-t0, e), flush=True)
