import xarray as xr
ds = xr.open_dataset("/home/z/my-project/scripts/15a_era5_tp_201512.nc")
print(ds)
print("\n--- details ---")
print("vars:", list(ds.data_vars))
tp = ds[list(ds.data_vars)[0]]
print("units:", tp.attrs.get("units"), "| dims:", dict(tp.sizes))
print("lat range:", float(ds.latitude.min()), "to", float(ds.latitude.max()), "| step:", float(ds.latitude[0]-ds.latitude[1]))
print("lon range:", float(ds.longitude.min()), "to", float(ds.longitude.max()))
print("time:", ds.valid_time.values if "valid_time" in ds else ds.time.values)
print("tp mean over region (m/day):", float(tp.mean()), "=> mm/month:", float(tp.mean())*1000*31)
