#!/usr/bin/env python3
"""
task52_sandbox_fetch_driver.py — sandbox execution shim for Day-1 download
============================================================================
WHY THIS EXISTS (Task 52):
  - Task 51 found archive-api.open-meteo.com hard-blocked from the sandbox.
    TODAY the block is gone, but the egress path is INTERMITTENT: ~15% of
    requests stall with 0 bytes received until timeout (30 s each). Fast
    requests answer in <1.5 s. Parallel workers make stalls overlap instead
    of blocking the queue.
  - The delivered fetch_data.py does ONE request per district for the full
    12.7-year window — that shape stalls badly. Small 2-year chunks are
    fast, so we chunk (7 per district).
  - Background processes are killed between sandbox commands, so the run
    is sliced into bounded time budgets; everything is resumable at the
    CHUNK level (.om_cache/chunks/) and district level (.om_cache/*.csv).

WHAT IT DOES:
  - Imports DISTRICTS/constants from the delivered fetch_data.py (unchanged).
  - Thread pool (--workers, default 5): fetches chunks concurrently; per-
    chunk timeout 30 s, retries 3/8/20 s; chunk cached atomically to
    .om_cache/chunks/<slug>__<start>_<end>.csv (+ .meta.json).
  - When all chunks of a district exist: stitch -> clamp to window -> pad
    trailing shortfalls with NaN (ERA5T availability, logged) -> write
    .om_cache/<slug>.csv + <slug>.meta.json in EXACTLY the format the
    delivered script expects.
  - After all 59 districts cached: run the ORIGINAL fetch_data.py — every
    district is a cache hit; it assembles the final CSVs + QA manifest
    offline.

USAGE:
  python task52_sandbox_fetch_driver.py --budget-sec 400   # repeat until done
  python task52_sandbox_fetch_driver.py --status           # just report
"""

import argparse
import datetime as dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

AGRI_DIR = Path("/home/z/my-project/download/agri_tws_ind")
sys.path.insert(0, str(AGRI_DIR))
import fetch_data as fd  # noqa: E402  (delivered script: DISTRICTS, constants)

CHUNK_DAYS = 730
CHUNK_TIMEOUT = 30
RETRY_DELAYS = [3, 8, 20]
CHUNK_DIR = fd.CACHE_DIR / "chunks"

_tls = threading.local()
_deadline = None  # global run deadline (set in main)


def p(msg):
    print(msg, flush=True)


def get_session():
    if not hasattr(_tls, "session"):
        _tls.session = requests.Session()
    return _tls.session


def chunk_bounds(start, end, days=CHUNK_DAYS):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    out = []
    cur = s
    while cur <= e:
        nxt = min(cur + dt.timedelta(days=days - 1), e)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + dt.timedelta(days=1)
    return out


def fetch_chunk(lat, lon, name, start, end):
    """One small-window request. Returns (df, meta) or None."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(fd.DAILY_VARS),
        "timezone": fd.TIMEZONE,
    }
    delays = RETRY_DELAYS + [None]
    for attempt, delay in enumerate(delays):
        if _deadline and time.time() > _deadline:
            return None
        try:
            r = get_session().get(fd.API_URL, params=params, timeout=CHUNK_TIMEOUT)
            data = r.json()
            daily = data.get("daily") or {}
            if "time" not in daily:
                reason = data.get("reason", json.dumps(data)[:150])
                p("    ! %s %s..%s API: %s" % (name, start, end, reason))
            else:
                df = pd.DataFrame(daily).rename(columns={"time": "date"})
                df["date"] = pd.to_datetime(df["date"])
                if df["date"].min() != pd.Timestamp(start):
                    p("    ! %s %s..%s starts %s not %s"
                      % (name, start, end, df["date"].min().date(), start))
                else:
                    df["district"] = name.split("|")[0]
                    df["state"] = name.split("|")[1]
                    meta = {"elevation": data.get("elevation"),
                            "latitude": data.get("latitude"),
                            "longitude": data.get("longitude"),
                            "generationtime_ms": data.get("generationtime_ms")}
                    return df, meta
        except Exception as e:
            p("    ! %s %s..%s %s" % (name, start, end, str(e)[:90]))
        if delay is None:
            return None
        time.sleep(delay)
    return None


def chunk_path(state, name, cs, ce):
    return CHUNK_DIR / ("%s__%s_%s.csv" % (fd.slug(state, name), cs, ce))


def worker(task):
    """Fetch one chunk if not cached. Returns status string."""
    state, name, lat, lon, cs, ce = task
    cpath = chunk_path(state, name, cs, ce)
    if cpath.exists():
        return "cached"
    if _deadline and time.time() > _deadline:
        return "skipped"
    out = fetch_chunk(lat, lon, "%s|%s" % (name, state), cs, ce)
    if out is None:
        return "failed"
    cdf, meta = out
    cdf.to_csv(cpath, index=False)
    (CHUNK_DIR / ("%s__%s_%s.meta.json" % (fd.slug(state, name), cs, ce))
     ).write_text(json.dumps(meta))
    return "ok"


def district_complete(state, name, expected):
    path = fd.CACHE_DIR / (fd.slug(state, name) + ".csv")
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        return (len(df) == expected
                and df["date"].min() == pd.Timestamp(fd.START_DATE))
    except Exception:
        return False


def stitch(state, name, bounds, expected):
    """Combine cached chunks into the district CSV; True if complete."""
    slugname = fd.slug(state, name)
    pieces, metas, missing = [], None, 0
    for cs, ce in bounds:
        cpath = chunk_path(state, name, cs, ce)
        mpath = CHUNK_DIR / ("%s__%s_%s.meta.json" % (slugname, cs, ce))
        if not cpath.exists():
            missing += 1
            continue
        pieces.append(pd.read_csv(cpath, parse_dates=["date"]))
        if mpath.exists():
            try:
                metas = json.loads(mpath.read_text())
            except Exception:
                pass
    if missing:
        p("  ~ %s: %d/%d chunks present — deferred" % (name, len(pieces), len(bounds)))
        return False
    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["date"] >= pd.Timestamp(fd.START_DATE))
            & (df["date"] <= pd.Timestamp(fd.END_DATE))]
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    have_dates = set(df["date"])
    all_dates = pd.date_range(fd.START_DATE, fd.END_DATE, freq="D")
    pad = [d for d in all_dates if d not in have_dates]
    if pad:
        p("  ~ %s: padding %d missing dates with NaN (first %s, last %s)"
          % (name, len(pad), pad[0].date(), pad[-1].date()))
        pad_df = pd.DataFrame({"date": pad})
        for c in fd.DAILY_VARS:
            pad_df[c] = float("nan")
        pad_df["district"] = name
        pad_df["state"] = state
        df = pd.concat([df, pad_df], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
    if len(df) != expected or df["date"].min() != pd.Timestamp(fd.START_DATE):
        p("  x %s: stitched frame bad (%d rows, starts %s) — not caching"
          % (name, len(df), df["date"].min()))
        return False
    df.to_csv(fd.CACHE_DIR / (slugname + ".csv"), index=False)
    if metas:
        (fd.CACHE_DIR / (slugname + ".meta.json")).write_text(json.dumps(metas))
    p("  + %s: COMPLETE (%d rows)" % (name, len(df)))
    return True


def main():
    global _deadline
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-sec", type=int, default=400)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    expected = (dt.date.fromisoformat(fd.END_DATE)
                - dt.date.fromisoformat(fd.START_DATE)).days + 1
    all_bounds = {name: chunk_bounds(fd.START_DATE, fd.END_DATE)
                  for _, name, _, _, _ in fd.DISTRICTS}

    if args.status:
        done = [n for s, n, _, _, _ in fd.DISTRICTS
                if district_complete(s, n, expected)]
        p("STATUS: %d/59 districts complete" % len(done))
        for s, n, _, _, _ in fd.DISTRICTS:
            if n not in done:
                nb = sum(1 for cs, ce in all_bounds[n] if chunk_path(s, n, cs, ce).exists())
                p("  pending %-28s %d/%d chunks" % (n, nb, len(all_bounds[n])))
        return 0

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    fd.CACHE_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    _deadline = t0 + args.budget_sec

    tasks = []
    for (state, name, lat, lon, src) in fd.DISTRICTS:
        if district_complete(state, name, expected):
            continue
        for cs, ce in all_bounds[name]:
            if not chunk_path(state, name, cs, ce).exists():
                tasks.append((state, name, lat, lon, cs, ce))

    p("driver: budget %ds, workers %d, %d chunks todo, %d districts pending"
      % (args.budget_sec, args.workers, len(tasks),
         sum(1 for s, n, _, _, _ in fd.DISTRICTS
             if not district_complete(s, n, expected))))

    counts = {"ok": 0, "cached": 0, "failed": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                counts[fut.result()] += 1
            except Exception as e:
                counts["failed"] += 1
                p("    x worker exception: %s" % str(e)[:90])

    stitched = 0
    for (state, name, lat, lon, src) in fd.DISTRICTS:
        if district_complete(state, name, expected):
            continue
        if stitch(state, name, all_bounds[name], expected):
            stitched += 1

    done_total = sum(1 for s, n, _, _, _ in fd.DISTRICTS
                     if district_complete(s, n, expected))
    p("SUMMARY: %s | stitched %d, %d/59 districts total, %.0fs elapsed"
      % (counts, stitched, done_total, time.time() - t0))
    if done_total == len(fd.DISTRICTS):
        p("ALL DISTRICTS COMPLETE — now run: cd %s && python fetch_data.py"
          % AGRI_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
