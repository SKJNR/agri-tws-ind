import cdsapi, sys, time
c = cdsapi.Client(quiet=True, progress=False)
req = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["total_precipitation"],
    "year": ["2015"],
    "month": ["12"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [60, -15, 30, 45],   # Europe + Med + N-Africa slice
}
t0 = time.time()
print("submitting request...", flush=True)
try:
    r = c.retrieve("reanalysis-era5-single-levels-monthly-means", req, "/home/z/my-project/scripts/15a_era5_tp_201512.nc")
    print("DONE in %.1f s" % (time.time()-t0), flush=True)
    print("result:", r, flush=True)
except Exception as e:
    print("FAILED after %.1f s: %r" % (time.time()-t0, e), flush=True)
    sys.exit(1)
