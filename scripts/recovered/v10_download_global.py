"""V10 — global ERA5-Land swvl3+swvl4 download (background).
16 boxes tiling lat [-56,84] x lon [-180,180], 2002-01..2019-01 monthly.
4 parallel workers, each 4 sequential CDS requests. Native 0.1° (regrid ourselves).
Then v10_regrid_global.py block-means to the competition 1° grid (centers x.5).
"""
import cdsapi, time, sys, os

BOXES = []
for n, w, s, e in [(84, None, 50, None), (50, None, 15, None), (15, None, -20, None), (-20, None, -56, None)]:
    for we in [(-180, -90), (-90, 0), (0, 90), (90, 180)]:
        BOXES.append((n, we[0], s, we[1]))

YEARS = [str(y) for y in range(2002, 2020)]  # 2002..2019 (2019-01 included)
MONTHS = [f"{m:02d}" for m in range(1, 13)]

def worker(box_ids):
    c = cdsapi.Client(quiet=True, progress=False, retry_max=2, sleep_max=60)
    for i in box_ids:
        n, w, s, e = BOXES[i]
        out = f"/home/z/my-project/scripts/v10_glb_sm34_{i:02d}.nc"
        if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
            print(f"[box {i:02d}] already done", flush=True); continue
        req = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": ["volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4"],
            "year": YEARS, "month": MONTHS, "time": ["00:00"],
            "data_format": "netcdf", "download_format": "unarchived",
            "area": [n, w, s, e],
        }
        t0 = time.time()
        print(f"[box {i:02d}] N{n} W{w} S{s} E{e} submitting...", flush=True)
        part = out + ".part"
        for attempt in range(2):
            try:
                c.retrieve("reanalysis-era5-land-monthly-means", req, part)
                sz = os.path.getsize(part)
                if sz < 1_000_000:
                    raise RuntimeError(f"file too small: {sz}")
                os.rename(part, out)
                print(f"[box {i:02d}] DONE {sz/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)
                break
            except Exception as ex:
                print(f"[box {i:02d}] attempt {attempt+1} FAILED: {ex!r}", flush=True)
                time.sleep(10)

if __name__ == "__main__":
    wid = int(sys.argv[1])
    ids = [wid, wid+4, wid+8, wid+12]  # worker w handles boxes w, w+4, w+8, w+12
    worker(ids)
    print(f"worker {wid} ALL DONE", flush=True)
