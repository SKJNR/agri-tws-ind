#!/usr/bin/env python3
"""
task54_fix_anakapalli.py — repair the dead sea-pixel at Anakapalli.

DEFECT (found by task54_qc_and_feasibility.py):
  Registry point (17.39, 83.01) resolves to a SEA pixel (API elevation
  0.0 m); ERA5-Land returns soil_moisture = 0.000 for 3539/4636 days
  (fill pattern, not physics). Precip/tmax were usable, soil was garbage.

FIX (logged registry correction, GLM_APPROX -> GLM_APPROX):
  New point (17.55, 82.95): in-district, inland, elev 45 m, live soil
  (probe: 0 zeros, mean 0.145, 2020 kharif sane). Refetch all 5 vars for
  the full window at the new point, replace district cache, then the
  caller re-runs fetch_data.py to rebuild outputs offline.
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

AGRI = Path("/home/z/my-project/download/agri_tws_ind")
sys.path.insert(0, str(AGRI))
import fetch_data as fd  # noqa: E402

STATE, NAME = "AP", "Anakapalli"
NEW_LAT, NEW_LON = 17.55, 82.95
CHUNK_DIR = fd.CACHE_DIR / "chunks"
URL = "https://archive-api.open-meteo.com/v1/archive"
RETRY = [3, 8, 20, None]


def fetch(lat, lon, start, end):
    params = {"latitude": lat, "longitude": lon, "start_date": start,
              "end_date": end, "daily": ",".join(fd.DAILY_VARS),
              "timezone": fd.TIMEZONE}
    for delay in RETRY:
        try:
            r = requests.get(URL, params=params, timeout=30)
            data = r.json()
            daily = data.get("daily") or {}
            if "time" not in daily:
                print("  API miss:", str(data)[:120])
            else:
                df = pd.DataFrame(daily).rename(columns={"time": "date"})
                df["date"] = pd.to_datetime(df["date"])
                if str(df["date"].min().date()) == start:
                    df["district"] = NAME
                    df["state"] = STATE
                    meta = {"elevation": data.get("elevation"),
                            "latitude": data.get("latitude"),
                            "longitude": data.get("longitude"),
                            "generationtime_ms": data.get("generationtime_ms")}
                    return df, meta
                print("  start mismatch:", df["date"].min().date())
        except Exception as e:
            print("  %s — retrying" % str(e)[:70])
        if delay:
            time.sleep(delay)
    return None


def chunks(start, end, days=730):
    s, cur, out = dt.date.fromisoformat(start), dt.date.fromisoformat(start), []
    e = dt.date.fromisoformat(end)
    while cur <= e:
        nxt = min(cur + dt.timedelta(days=days - 1), e)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + dt.timedelta(days=1)
    return out


def main():
    expected = (dt.date.fromisoformat(fd.END_DATE)
                - dt.date.fromisoformat(fd.START_DATE)).days + 1
    slugname = fd.slug(STATE, NAME)
    pieces, meta = [], None
    for cs, ce in chunks(fd.START_DATE, fd.END_DATE):
        out = fetch(NEW_LAT, NEW_LON, cs, ce)
        if out is None:
            print("FAILED at chunk %s..%s — re-run this script (resumable off)"
                  % (cs, ce))
            return 1
        cdf, meta = out
        pieces.append(cdf)
        # persist chunk cache (replaces old-point chunks)
        cdf.to_csv(CHUNK_DIR / ("%s__%s_%s.csv" % (slugname, cs, ce)), index=False)
        (CHUNK_DIR / ("%s__%s_%s.meta.json" % (slugname, cs, ce))
         ).write_text(json.dumps(meta))
        print("  chunk %s..%s ok (%d rows)" % (cs, ce, len(cdf)))
        time.sleep(0.5)

    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["date"] >= pd.Timestamp(fd.START_DATE))
            & (df["date"] <= pd.Timestamp(fd.END_DATE))]
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    all_dates = pd.date_range(fd.START_DATE, fd.END_DATE, freq="D")
    pad = [d for d in all_dates if d not in set(df["date"])]
    if pad:
        print("  WARNING: %d missing dates — padding NaN" % len(pad))
        pdf = pd.DataFrame({"date": pad})
        for c in fd.DAILY_VARS:
            pdf[c] = float("nan")
        pdf["district"], pdf["state"] = NAME, STATE
        df = pd.concat([df, pdf], ignore_index=True).sort_values("date").reset_index(drop=True)

    # verification gates for the fix
    assert len(df) == expected, "row count %d != %d" % (len(df), expected)
    assert df["date"].min() == pd.Timestamp(fd.START_DATE)
    soil = df["soil_moisture_0_to_7cm_mean"]
    n_zero = int((soil == 0).sum())
    n_nan = int(df[fd.DAILY_VARS].isna().sum().sum())
    # exact-zero CAN be a legit ERA5-Land dry-down (Anantapur 2019: 83 zero
    # days in a real drought). The FILL signature is LONG consecutive runs
    # (old point: 3653 straight days). Detector: max run length.
    s = soil.reset_index(drop=True)
    run = best = 0
    for v in s:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    print("new-pixel soil: zeros=%d (was 3539), max-zero-run=%d d (was 3653), "
          "min=%.4f, mean=%.3f | NaNs=%d"
          % (n_zero, best, soil.min(), soil.mean(), n_nan))
    assert n_nan == 0, "NaNs introduced"
    assert best <= 45, "long zero run = masked fill, not physics"
    assert meta.get("elevation", 0) > 0, "elevation must be land (>0 m)"

    df.to_csv(fd.CACHE_DIR / (slugname + ".csv"), index=False)
    (fd.CACHE_DIR / (slugname + ".meta.json")).write_text(json.dumps(meta))
    print("district cache replaced: %s (%d rows, elev %s m)"
          % (slugname, len(df), meta.get("elevation")))
    print("NEXT: update registry coords in fetch_data.py, then re-run it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
