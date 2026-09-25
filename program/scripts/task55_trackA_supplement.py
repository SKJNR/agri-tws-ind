#!/usr/bin/env python3
"""Task-55 Track A supplement: negative-value distribution shape, per-state
shares, DEM retry (50 wells, backoff), Track-B 3-station sample with concrete
readings. Within Addendum-11 pre-registered scope (per-class breakdown,
report-only)."""
import json, time, urllib.request
import duckdb

CSV = "/tmp/gwl_download/gwl_data.csv"
CUT = "2022-12-31"
LAB = "'Andhra Pradesh','Telangana'"
con = duckdb.connect()
out = {}

# 1. negative distribution shape (row level)
neg = con.execute(f"""
SELECT quantile_cont(gwl_value, [0.01, 0.25, 0.50, 0.75, 0.99]) ,
       count(*),
       avg(CASE WHEN gwl_value > -5 THEN 1 ELSE 0 END),
       avg(CASE WHEN gwl_value > -2 THEN 1 ELSE 0 END)
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({LAB}) AND gwl_value < 0
""").fetchone()
qs = neg[0]
out["negatives_row_level"] = {
    "n": neg[1], "q01": round(qs[0],2), "q25": round(qs[1],2), "q50": round(qs[2],2),
    "q75": round(qs[3],2), "q99": round(qs[4],2),
    "share_gt_minus5": round(neg[2],4), "share_gt_minus2": round(neg[3],4)}
print("negatives row-level:", out["negatives_row_level"])

# 2. per-state negative shares + medians
for st in ("Andhra Pradesh", "Telangana"):
    r = con.execute(f"""
    SELECT count(*), avg(CASE WHEN gwl_value < 0 THEN 1 ELSE 0 END), median(gwl_value)
    FROM read_csv_auto('{CSV}')
    WHERE date <= DATE '{CUT}' AND state = '{st}' AND gwl_value IS NOT NULL
    """).fetchone()
    out[f"state_{st.replace(' ','_')}"] = {"rows": r[0], "neg_share": round(r[1],4),
                                           "median_gwl": round(r[2],2)}
    print(st, out[f"state_{st.replace(' ','_')}"])

# 3. what is the elevation column? distribution + per-state medians
ev = con.execute(f"""
SELECT median(elevation), quantile_cont(elevation, [0.1, 0.9]), count(*),
       avg(CASE WHEN elevation IS NULL THEN 1 ELSE 0 END)
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({LAB})
""").fetchone()
out["elevation_column"] = {"median": round(float(ev[0]),2),
                           "q10_q90": [round(float(ev[1][0]),2), round(float(ev[1][1]),2)],
                           "rows": ev[2], "null_share": round(float(ev[3]),4)}
print("elevation column:", out["elevation_column"])
# terrain sanity: distinct lat/lon grid — plateau wells should show DEM ~300m;
# internal check: median abs file-elev vs expected terrain is out of scope without DEM.

# 3b. elevation column constancy (internal, decisive)
const = con.execute(f"""
SELECT count(DISTINCT elevation), min(elevation), max(elevation)
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({LAB}) AND elevation IS NOT NULL
""").fetchone()
out["elevation_column"]["distinct_values"] = const[0]
out["elevation_column"]["min_max"] = [round(float(const[1]),2), round(float(const[2]),2)]
print("elevation distinct values:", const)

# 3c. TG sign-mixing: flipped SUBSERIES vs flipped WELLS
mix = con.execute(f"""
SELECT count(*) FILTER (WHERE pos>0 AND neg>0) wells_mixed_sign,
       count(*) FILTER (WHERE pos=0 AND neg>0) wells_all_negative,
       count(*) FILTER (WHERE neg=0 AND pos>0) wells_all_positive,
       count(*) total
FROM (
  SELECT station_code,
         count(*) FILTER (WHERE gwl_value > 0) pos,
         count(*) FILTER (WHERE gwl_value < 0) neg
  FROM read_csv_auto('{CSV}')
  WHERE date <= DATE '{CUT}' AND state = 'Telangana' AND gwl_value IS NOT NULL
  GROUP BY station_code)
""").fetchone()
out["tg_sign_mixing"] = {"wells_mixed_sign": mix[0], "wells_all_negative": mix[1],
                          "wells_all_positive": mix[2], "wells_total": mix[3]}
print("TG sign mixing:", out["tg_sign_mixing"])

# 3d. TG positive-mode vs negative-mode medians
modes = con.execute(f"""
SELECT median(gwl_value) FILTER (WHERE gwl_value > 0),
       median(gwl_value) FILTER (WHERE gwl_value < 0),
       quantile_cont(gwl_value, 0.5)
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state = 'Telangana' AND gwl_value IS NOT NULL
""").fetchone()
out["tg_modes"] = {"positive_mode_median_m": round(float(modes[0]),2),
                   "negative_mode_median_m": round(float(modes[1]),2),
                   "overall_median_m": round(float(modes[2]),2)}
print("TG modes:", out["tg_modes"])

# 3e. TG mixing by year (is the flip a vintage?)
by_year = con.execute(f"""
SELECT year(date), count(*), round(avg(CASE WHEN gwl_value<0 THEN 1.0 ELSE 0.0 END),3)
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state = 'Telangana' AND gwl_value IS NOT NULL
GROUP BY 1 ORDER BY 1
""").fetchall()
out["tg_neg_share_by_year"] = {int(y): [int(n), float(s)] for y, n, s in by_year}
print("TG neg share by year (first/last 5):", by_year[:5], "...", by_year[-5:])

# 3f. exact flip onset month (TG 2019-2022, month level)
onset = con.execute(f"""
SELECT strftime(date, '%Y-%m') ym, count(*) n,
       round(avg(CASE WHEN gwl_value<0 THEN 1.0 ELSE 0.0 END),3) neg_share,
       round(median(gwl_value),2) med
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state = 'Telangana' AND gwl_value IS NOT NULL
  AND date >= DATE '2019-01-01'
GROUP BY 1 ORDER BY 1
""").fetchall()
out["tg_monthly_2019_2022"] = {ym: [int(n), float(s), float(m)] for ym, n, s, m in onset}
print("TG monthly (2019-2022):")
for ym, n, s, m in onset:
    print(f"  {ym} n={n:6d} neg%={s:5.1%} med={m:7.2f}")
w = con.execute(f"""
SELECT station_code, any_value(latitude) lat, any_value(longitude) lon,
       any_value(elevation) elevation, median(gwl_value) med_gwl
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({LAB}) AND latitude IS NOT NULL
GROUP BY station_code ORDER BY md5(station_code || '42') LIMIT 20
""").fetch_df()
dem = None
for attempt in range(3):
    try:
        lats = ",".join(f"{v:.5f}" for v in w.lat)
        lons = ",".join(f"{v:.5f}" for v in w.lon)
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
        with urllib.request.urlopen(url, timeout=60) as r:
            dem = json.load(r)["elevation"]
        break
    except Exception as e:
        print(f"DEM attempt {attempt+1} failed: {e}")
        time.sleep(20 * (attempt + 1))
if dem is not None:
    w["dem"] = dem
    diff = (w.elevation.astype(float) - w.dem.astype(float)).abs()
    out["dem_crosscheck"] = {
        "n": len(w), "median_abs_diff_m": round(float(diff.median()),1),
        "median_dem_terrain_m": round(float(w.dem.median()),1),
        "median_file_elev_m": round(float(w.elevation.astype(float).median()),1),
        "sample_station_vs_file_vs_dem": [
            [str(r.station_code), float(r.elevation), float(r.dem)] for _, r in w.head(5).iterrows()]}
    print("DEM crosscheck:", out["dem_crosscheck"])
else:
    out["dem_crosscheck"] = None
    print("DEM crosscheck: BLOCKED (all retries failed)")

# 5. Track-B sample: 3 concrete stations with a concrete dated reading each
tb = con.execute(f"""
WITH picked AS (
  SELECT station_code FROM read_csv_auto('{CSV}')
  WHERE date <= DATE '{CUT}' AND state IN ({LAB}) AND gwl_value <= -50
  GROUP BY station_code ORDER BY md5(station_code) LIMIT 1
)
SELECT 'QC-failure exemplar (row <= -50 m)' why, station_code FROM picked
UNION ALL
SELECT 'small-negative-median well', 'TSGWD_1590'
UNION ALL
SELECT 'typical-positive well', station_code FROM (
  SELECT station_code, median(gwl_value) m, count(*) n
  FROM read_csv_auto('{CSV}')
  WHERE date <= DATE '{CUT}' AND state IN ({LAB})
  GROUP BY station_code HAVING n >= 40 AND m BETWEEN 7 AND 9
  ORDER BY md5(station_code || '7') LIMIT 1)
""").fetchall()
rows = []
for why, st in tb:
    r = con.execute(f"""
    SELECT station_code, district, state, date, gwl_value, latitude, longitude
    FROM read_csv_auto('{CSV}')
    WHERE station_code = '{st}' AND date <= DATE '{CUT}'
    ORDER BY date DESC LIMIT 1
    """).fetchone()
    rows.append({"why": why, "station_code": r[0], "district": r[1], "state": r[2],
                 "latest_open_date": str(r[3])[:10], "gwl_value": round(float(r[4]),2),
                 "lat": round(float(r[5]),4), "lon": round(float(r[6]),4)})
out["track_b_sample"] = rows
print("Track-B:", json.dumps(rows, indent=1))

# merge into existing artifact
path = "/home/z/my-project/program/data/manifests/task55_datum_report.json"
rep = json.load(open(path))
rep["supplement"] = out
with open(path, "w") as f:
    json.dump(rep, f, indent=1)
print("artifact updated:", path)
