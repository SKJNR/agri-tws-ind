#!/usr/bin/env python3
"""task52_verify_outputs.py — independent verification of Day-1 outputs."""
import sys
import pandas as pd
import requests

AGRI = "/home/z/my-project/download/agri_tws_ind/"
fail = 0


def check(label, cond, detail=""):
    global fail
    print("%s %s %s" % ("PASS" if cond else "FAIL", label, detail))
    if not cond:
        fail += 1


full = pd.read_csv(AGRI + "ap_ts_weather_openmeteo_full59.csv", parse_dates=["date"])
check("full59 rows", len(full) == 273524, "%d (59 x 4636)" % len(full))
check("full59 districts", full["district"].nunique() == 59,
      str(full["district"].nunique()))
counts = full.groupby("state")["district"].nunique().to_dict()
check("AP=26 TS=33", counts.get("AP") == 26 and counts.get("TS") == 33, str(counts))
check("date range", str(full["date"].min().date()) == "2014-01-01"
      and str(full["date"].max().date()) == "2026-09-10",
      "%s..%s" % (full["date"].min().date(), full["date"].max().date()))
per = full.groupby("district").size()
check("all districts 4636 rows", (per == 4636).all(),
      "min=%d max=%d" % (per.min(), per.max()))
cols = ["date", "precipitation_sum", "temperature_2m_max", "temperature_2m_min",
        "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean",
        "district", "state"]
check("columns", list(full.columns) == cols, str(list(full.columns)))
nan_by_var = full[cols[1:6]].isna().sum().to_dict()
check("zero NaNs all vars", sum(nan_by_var.values()) == 0, str(nan_by_var))
dupes = full.duplicated(subset=["district", "date"]).sum()
check("no duplicate (district,date)", dupes == 0, str(dupes))

ten = pd.read_csv(AGRI + "ap_ts_weather_raw_10yrs.csv", parse_dates=["date"])
check("qwen10 rows", len(ten) == 36520, "%d (10 x 3652)" % len(ten))
check("qwen10 districts", ten["district"].nunique() == 10,
      str(ten["district"].nunique()))
check("qwen10 ends 2023-12-31", str(ten["date"].max().date()) == "2023-12-31",
      str(ten["date"].max().date()))
check("qwen10 starts 2014-01-01", str(ten["date"].min().date()) == "2014-01-01")

# live API cross-check: Anantapur 2020-07-15, same params incl. timezone
row = full[(full["district"] == "Anantapur")
           & (full["date"] == "2020-07-15")].iloc[0]
r = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={"latitude": 14.68, "longitude": 77.6, "start_date": "2020-07-15",
            "end_date": "2020-07-15",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,"
                     "et0_fao_evapotranspiration,soil_moisture_0_to_7cm_mean",
            "timezone": "Asia/Kolkata"}, timeout=30)
d = r.json()["daily"]
api = {"precipitation_sum": d["precipitation_sum"][0],
       "temperature_2m_max": d["temperature_2m_max"][0],
       "temperature_2m_min": d["temperature_2m_min"][0],
       "et0_fao_evapotranspiration": d["et0_fao_evapotranspiration"][0],
       "soil_moisture_0_to_7cm_mean": d["soil_moisture_0_to_7cm_mean"][0]}
mism = {k: (api[k], row[k]) for k in api
        if api[k] is None or abs(float(api[k]) - float(row[k])) > 1e-6}
check("API value cross-check (Anantapur 2020-07-15)", not mism,
      "api=%s csv=%s" % (api, {k: row[k] for k in api}) if mism else str(api))

print("\n%s (failures: %d)" % ("ALL CHECKS PASSED" if fail == 0 else "FAILURES", fail))
sys.exit(1 if fail else 0)
