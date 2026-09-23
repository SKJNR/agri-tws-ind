#!/usr/bin/env python3
"""
fetch_data.py — Day-1 Sprint foundation data (AGRI-TWS-IND-v1)
================================================================
Hardened fork of Qwen's Day-1 script (Open-Meteo Archive API, free, no
key). Fixes and upgrades relative to the original:

 1. BUG FIX — window: original end_date 2023-12-31 leaves the frozen
    protocol's TEST YEARS (2024-25) undownloadable and captures no
    operational current data. Fixed: 2014-01-01 -> 2026-09-10.
    (The referee protocol needs train <=2019 / val 2020-22 / test
    2023-25 — the original window would have made test unscorable.)
 2. Retries with exponential backoff (5s/15s/45s) + clear error text.
 3. Per-district cache (.om_cache/) — RESUMABLE: re-run continues where
    it left off; a failure at district 40 loses nothing.
 4. Data-quality audit + manifest (row counts, NaN counts, date gaps,
    per-district elevation from the API).
 5. Full sprint scope: all 59 districts (AP 26 + TS 33), Qwen's original
    10 run first so the exact Day-1 task completes early.
 6. --probe mode: 5-day validation call BEFORE the long run, so any API
    contract change surfaces in 5 seconds, not after 10 minutes.

USAGE (on your machine):
    pip install requests pandas
    python fetch_data.py --probe      # validate first (~5 s)
    python fetch_data.py              # all 59 districts (~8-12 min)
    python fetch_data.py --quick      # Qwen's exact 10 districts only

OUTPUTS (written to the current directory):
    ap_ts_weather_raw_10yrs.csv        — Qwen's Day-2-compatible file
                                          (10 districts, 2014-01-01..2023-12-31,
                                          same columns/filename Qwen specced)
    ap_ts_weather_openmeteo_full59.csv — full sprint foundation
                                          (59 districts, 2014-01-01..2026-09-10)
    weather_download_manifest.md       — QA audit + provenance
    districts_ap_ts.csv                — the 59-district registry
                                          (coordinates flagged QWEN_PROVIDED vs
                                          GLM_APPROX pending LGD/KVK verification)
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2014-01-01"
END_DATE = "2026-09-10"          # protocol test years 2024-25 + operational tail
QUICK_END = "2023-12-31"         # Qwen's original window (Day-2 compatibility)
DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean",
]
TIMEZONE = "Asia/Kolkata"
CACHE_DIR = Path(".om_cache")
SLEEP_BETWEEN = 2.0              # politeness; free tier allows 600 req/min

# ---------------------------------------------------------------------------
# District registry: 59 districts (AP 26 + TS 33)
#   coord_source: QWEN_PROVIDED = from the Day-1 task (approx centroids);
#                 GLM_APPROX    = approximate centroids, pending verification
#                 against LGD/GADM in the loader task (flagged, not silent).
# ---------------------------------------------------------------------------

DISTRICTS = [
    # --- Qwen's 10 pilot districts (exact, run first) ---
    ("AP", "Anantapur", 14.68, 77.60, "QWEN_PROVIDED"),
    ("AP", "Kurnool", 15.82, 78.03, "QWEN_PROVIDED"),
    ("AP", "Guntur", 16.30, 80.44, "QWEN_PROVIDED"),
    ("AP", "Krishna", 16.50, 80.64, "QWEN_PROVIDED"),
    ("AP", "Chittoor", 13.21, 79.11, "QWEN_PROVIDED"),
    ("TS", "Nizamabad", 18.67, 78.10, "QWEN_PROVIDED"),
    ("TS", "Karimnagar", 18.43, 79.12, "QWEN_PROVIDED"),
    ("TS", "Warangal", 17.97, 79.59, "QWEN_PROVIDED"),
    ("TS", "Khammam", 17.24, 80.15, "QWEN_PROVIDED"),
    ("TS", "Mahabubnagar", 16.73, 77.99, "QWEN_PROVIDED"),
    # --- Andhra Pradesh, remaining 16 ---
    ("AP", "Alluri Sitharama Raju", 18.32, 82.88, "GLM_APPROX"),
    ("AP", "Anakapalli", 17.55, 82.95, "GLM_APPROX"),  # corrected 2026-09-15: old (17.39,83.01) = masked fill pixel, soil=0 from 2017-01 (Task 54, DECISION_LOG)
    ("AP", "Annamayya", 14.05, 78.75, "GLM_APPROX"),
    ("AP", "Bapatla", 15.91, 80.47, "GLM_APPROX"),
    ("AP", "Dr B R Ambedkar Konaseema", 16.58, 81.98, "GLM_APPROX"),
    ("AP", "East Godavari", 17.00, 81.78, "GLM_APPROX"),
    ("AP", "Eluru", 16.71, 81.10, "GLM_APPROX"),
    ("AP", "Kakinada", 16.99, 82.25, "GLM_APPROX"),
    ("AP", "Nandyal", 15.48, 78.48, "GLM_APPROX"),
    ("AP", "NTR", 16.51, 80.65, "GLM_APPROX"),
    ("AP", "Palnadu", 16.24, 80.05, "GLM_APPROX"),
    ("AP", "Parvathipuram Manyam", 18.78, 83.43, "GLM_APPROX"),
    ("AP", "Prakasam", 15.51, 80.05, "GLM_APPROX"),
    ("AP", "Sri Potti Sriramulu Nellore", 14.44, 79.99, "GLM_APPROX"),
    ("AP", "Srikakulam", 18.30, 83.90, "GLM_APPROX"),
    ("AP", "Sri Sathya Sai", 14.41, 77.81, "GLM_APPROX"),
    ("AP", "Tirupati", 13.63, 79.42, "GLM_APPROX"),
    ("AP", "Visakhapatnam", 17.69, 83.22, "GLM_APPROX"),
    ("AP", "Vizianagaram", 18.11, 83.41, "GLM_APPROX"),
    ("AP", "West Godavari", 16.55, 81.53, "GLM_APPROX"),
    ("AP", "YSR Kadapa", 14.47, 78.82, "GLM_APPROX"),
    # --- Telangana, remaining 23 ---
    ("TS", "Adilabad", 19.66, 78.53, "GLM_APPROX"),
    ("TS", "Bhadradri Kothagudem", 17.55, 80.75, "GLM_APPROX"),
    ("TS", "Hanumakonda", 18.00, 79.58, "GLM_APPROX"),
    ("TS", "Hyderabad", 17.39, 78.49, "GLM_APPROX"),
    ("TS", "Jagtial", 18.79, 78.93, "GLM_APPROX"),
    ("TS", "Jangaon", 17.72, 79.18, "GLM_APPROX"),
    ("TS", "Jayashankar Bhupalpally", 18.19, 80.08, "GLM_APPROX"),
    ("TS", "Jogulamba Gadwal", 16.23, 77.79, "GLM_APPROX"),
    ("TS", "Kamareddy", 18.32, 78.33, "GLM_APPROX"),
    ("TS", "Kumuram Bheem Asifabad", 19.36, 79.29, "GLM_APPROX"),
    ("TS", "Mahabubabad", 17.60, 79.87, "GLM_APPROX"),
    ("TS", "Mancherial", 18.87, 79.44, "GLM_APPROX"),
    ("TS", "Medak", 18.05, 78.27, "GLM_APPROX"),
    ("TS", "Medchal Malikajgiri", 17.53, 78.48, "GLM_APPROX"),
    ("TS", "Mulugu", 18.22, 80.02, "GLM_APPROX"),
    ("TS", "Nagarkurnool", 16.48, 78.33, "GLM_APPROX"),
    ("TS", "Nalgonda", 17.05, 79.27, "GLM_APPROX"),
    ("TS", "Narayanpet", 16.75, 77.50, "GLM_APPROX"),
    ("TS", "Nirmal", 19.10, 78.34, "GLM_APPROX"),
    ("TS", "Peddapalli", 18.61, 79.37, "GLM_APPROX"),
    ("TS", "Rajanna Sircilla", 18.38, 78.83, "GLM_APPROX"),
    ("TS", "Rangareddy", 17.30, 78.35, "GLM_APPROX"),
    ("TS", "Sangareddy", 17.62, 78.08, "GLM_APPROX"),
    ("TS", "Siddipet", 18.10, 78.85, "GLM_APPROX"),
    ("TS", "Suryapet", 17.14, 79.62, "GLM_APPROX"),
    ("TS", "Vikarabad", 17.34, 77.90, "GLM_APPROX"),
    ("TS", "Wanaparthy", 16.37, 78.07, "GLM_APPROX"),
    ("TS", "Yadadri Bhuvanagiri", 17.51, 78.89, "GLM_APPROX"),
]


def quick_districts():
    return [d for d in DISTRICTS if d[4] == "QWEN_PROVIDED"]


def slug(state, name):
    return (state + "_" + name).replace(" ", "_").replace(".", "").replace(",", "")


# ---------------------------------------------------------------------------
# Fetch with retries + cache
# ---------------------------------------------------------------------------

def fetch_district(lat, lon, name, state, start, end, retries=3):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    delays = [5, 15, 45]
    for attempt in range(retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            data = r.json()
            if "daily" not in data or "time" not in data.get("daily", {}):
                reason = data.get("reason", json.dumps(data)[:200])
                if attempt < retries:
                    print("  ! %s: API error (%s) — retry %d" % (name, reason, attempt + 1))
                    time.sleep(delays[attempt])
                    continue
                print("  x %s: API error, giving up: %s" % (name, reason))
                return None
            df = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df["district"] = name
            df["state"] = state
            meta = {"elevation": data.get("elevation"),
                    "latitude": data.get("latitude"), "longitude": data.get("longitude"),
                    "generationtime_ms": data.get("generationtime_ms")}
            return df, meta
        except requests.exceptions.ConnectTimeout:
            if attempt < retries:
                print("  ! %s: connection timeout — retry %d" % (name, attempt + 1))
                time.sleep(delays[attempt])
                continue
            print("  x %s: cannot reach %s from this machine/network." % (name, API_URL))
            print("    If this repeats for all districts: the network blocks the host.")
            print("    GLM's sandbox hits exactly this (egress policy) — run locally instead.")
            return None
        except Exception as e:
            if attempt < retries:
                print("  ! %s: %s — retry %d" % (name, e, attempt + 1))
                time.sleep(delays[attempt])
                continue
            print("  x %s: failed after retries: %s" % (name, e))
            return None
    return None


def get_district_frame(state, name, lat, lon, start, end):
    """Cache-aware fetch: skips complete cached districts."""
    cache = CACHE_DIR / (slug(state, name) + ".csv")
    expected = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    if cache.exists():
        try:
            df = pd.read_csv(cache, parse_dates=["date"])
            if len(df) == expected and df["date"].min() == pd.Timestamp(start):
                print("  = %s: cached (%d rows) — skip" % (name, len(df)))
                return df
            print("  ~ %s: cache incomplete (%d/%d rows) — refetch"
                  % (name, len(df), expected))
        except Exception:
            print("  ~ %s: cache unreadable — refetch" % name)
    out = fetch_district(lat, lon, name, state, start, end)
    if out is None:
        return None
    df, meta = out
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    with open(str(cache).replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f)
    return df


# ---------------------------------------------------------------------------
# QA + outputs
# ---------------------------------------------------------------------------

def audit(frames, start, end):
    expected = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    rows = []
    for (state, name, lat, lon, src), df in frames.items():
        meta_path = CACHE_DIR / (slug(state, name) + ".meta.json")
        elev = None
        if meta_path.exists():
            try:
                elev = json.loads(meta_path.read_text()).get("elevation")
            except Exception:
                pass
        nans = {c: int(df[c].isna().sum()) for c in DAILY_VARS}
        gaps = int(expected - df["date"].nunique())
        rows.append({"state": state, "district": name, "lat": lat, "lon": lon,
                     "coord_source": src, "elevation_m": elev,
                     "rows": len(df), "expected": expected, "date_gaps": gaps,
                     "nan_precip": nans["precipitation_sum"],
                     "nan_tmax": nans["temperature_2m_max"],
                     "nan_tmin": nans["temperature_2m_min"],
                     "nan_et0": nans["et0_fao_evapotranspiration"],
                     "nan_soil": nans["soil_moisture_0_to_7cm_mean"],
                     "first": str(df["date"].min().date()),
                     "last": str(df["date"].max().date())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Day-1 Open-Meteo foundation download")
    ap.add_argument("--probe", action="store_true", help="5-day validation call, then exit")
    ap.add_argument("--quick", action="store_true", help="only Qwen's 10 pilot districts")
    ap.add_argument("--export-registry", action="store_true",
                    help="write districts_ap_ts.csv and exit (no network)")
    args = ap.parse_args()

    if args.export_registry:
        pd.DataFrame(DISTRICTS, columns=["state", "district", "lat", "lon",
                                         "coord_source"]).to_csv("districts_ap_ts.csv", index=False)
        print("districts_ap_ts.csv written (%d districts)" % len(DISTRICTS))
        return 0

    if args.probe:
        print("PROBE: 5-day validation call to %s ..." % API_URL)
        out = fetch_district(14.68, 77.60, "Anantapur-probe", "AP",
                             "2026-08-25", "2026-09-10")
        if out is None:
            print("PROBE FAILED — do not start the long run until this passes.")
            return 1
        df, meta = out
        print("PROBE OK — %d days returned, vars: %s" % (len(df), list(df.columns)))
        print("  elevation=%s m; sample:" % meta.get("elevation"))
        print(df.head(3).to_string(index=False))
        print("If recent days show NaN: normal (ERA5T trailing availability).")
        return 0

    todo = quick_districts() if args.quick else DISTRICTS
    print("=" * 70)
    print("Day-1 foundation download — %d districts, %s .. %s"
          % (len(todo), START_DATE, END_DATE))
    print("Resumable: safe to re-run after any interruption. Ctrl+C aborts safely.")
    print("=" * 70)

    frames = {}
    t0 = time.time()
    for i, (state, name, lat, lon, src) in enumerate(todo, 1):
        print("[%2d/%d] %s (%s)" % (i, len(todo), name, state))
        df = get_district_frame(state, name, lat, lon, START_DATE, END_DATE)
        if df is not None:
            frames[(state, name, lat, lon, src)] = df
        time.sleep(SLEEP_BETWEEN)

    if not frames:
        print("\nNo data fetched. If every district hit connection timeouts,")
        print("this network blocks archive-api.open-meteo.com (GLM's sandbox does).")
        print("Run this script from a normal home/office connection.")
        return 1

    full = pd.concat(frames.values(), ignore_index=True)
    full = full.sort_values(["state", "district", "date"]).reset_index(drop=True)
    full_file = "ap_ts_weather_openmeteo_full59.csv"
    full.to_csv(full_file, index=False)
    print("\nSaved %s (%d rows, %d districts)"
          % (full_file, len(full), full["district"].nunique()))

    # Qwen's Day-2-compatible file: 10 pilot districts, original window.
    qn = [d for d in frames if d[4] == "QWEN_PROVIDED"]
    if qn:
        sub = pd.concat([frames[d] for d in qn], ignore_index=True)
        sub = sub[sub["date"] <= pd.Timestamp(QUICK_END)]
        sub = sub.sort_values(["state", "district", "date"]).reset_index(drop=True)
        sub.to_csv("ap_ts_weather_raw_10yrs.csv", index=False)
        print("Saved ap_ts_weather_raw_10yrs.csv (%d rows, %d districts, thru %s)"
              % (len(sub), sub["district"].nunique(), QUICK_END))

    # QA manifest
    qa = audit(frames, START_DATE, END_DATE)
    md = ["# weather_download_manifest.md — Day-1 foundation download",
          "",
          "- API: %s (Open-Meteo Archive; ERA5/ERA5-Land reanalysis)" % API_URL,
          "- window: %s .. %s | timezone: %s" % (START_DATE, END_DATE, TIMEZONE),
          "- variables: %s" % ", ".join(DAILY_VARS),
          "- fetched: %s | districts: %d" % (dt.datetime.now().isoformat(timespec="seconds"),
                                             len(frames)),
          "- trailing-day NaNs = ERA5T preliminary availability (normal);",
          "  trailing consolidated months finalize ~2-3 months later (LATENCY_TABLE L5/L13 rule).",
          "",
          "| state | district | coord_source | elev_m | rows | gaps | NaN(precip/tmax/tmin/et0/soil) | first | last |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in qa.iterrows():
        md.append("| %s | %s | %s | %s | %d | %d | %d/%d/%d/%d/%d | %s | %s |"
                  % (r["state"], r["district"], r["coord_source"], r["elevation_m"],
                     r["rows"], r["date_gaps"], r["nan_precip"], r["nan_tmax"],
                     r["nan_tmin"], r["nan_et0"], r["nan_soil"], r["first"], r["last"]))
    md += ["",
           "coord_source: QWEN_PROVIDED = Day-1 task coordinates (approx centroids);",
           "GLM_APPROX = approximate centroids pending LGD/GADM verification (loader task).",
           "",
           "Sprint ruling (DECISION_LOG, 2026-09-15): this CSV is loader work, not an",
           "evaluation (T7 unaffected). Test years 2023-25 stay untouched until the R7-SS5",
           "order completes: D1.1-empirical -> regime-map freeze -> baseline ladder.",
           "First LGBM run = SMOKE TEST on train<=2019 + val 2020-22 only, logged first."]
    open("weather_download_manifest.md", "w").write("\n".join(md))
    print("Saved weather_download_manifest.md (QA audit)")

    bad = qa[(qa["date_gaps"] > 0) | (qa[["nan_precip", "nan_soil"]] > 0).any(axis=1)]
    print("\nQA: %d/%d districts clean; %d with gaps/NaNs (see manifest)."
          % (len(qa) - len(bad), len(qa), len(bad)))
    print("Done in %.1f min. Reply to Qwen: 'Script ran successfully, CSV is saved.'"
          % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
