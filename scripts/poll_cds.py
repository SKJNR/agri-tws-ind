"""Poll CDS request status by request ID (short-lived, safe to re-run)."""
import sys, json, urllib.request

BASE = 'https://cds.climate.copernicus.eu/api'
KEY = 'c6cc105f-2c38-4e66-90f0-89e66692c6a7'
RID = sys.argv[1] if len(sys.argv) > 1 else 'a41b0df1-32a5-465c-82cc-600f858cefd8'

req = urllib.request.Request(
    f'{BASE}/retrieve/v1/requests/{RID}',
    headers={'PRIVATE-TOKEN': KEY},
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    print(json.dumps(d, indent=2)[:2000])
except Exception as e:
    print(f"status check failed: {e}")
    # try alternate endpoint
    try:
        req2 = urllib.request.Request(
            f'{BASE}/retrieve/v1/processes/reanalysis-era5-single-levels-monthly-means/execution',
            headers={'PRIVATE-TOKEN': KEY},
        )
        print("alt endpoint not applicable for polling; use queue listing")
    except Exception as e2:
        print(f"alt failed: {e2}")
