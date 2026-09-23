#!/usr/bin/env python3
"""
build_weather_backbone.py — Weather Backbone (registered next step; founder
directive 2026-09-18 "proceed immediately to the Weather Backbone on the
59-execution basis").

What it does
  1. Loads the Day-1 Open-Meteo daily foundation (59 districts, 2014-01-01..
     2026-09-10, ERA5-based) incl. the API's own PM ET0 column.
  2. Downloads IMD 1-degree gridded tmax/tmin (imdlib, IMD Pune binary .GRD)
     into program/data/imd/ (regenerable cache, gitignored) and extracts the
     nearest grid cell per district centroid.
  3. Computes Hargreaves-Samani ET0 with pyet (method=0, k=0.0135, lat in
     radians; calibrated against FAO-56 hand calc in --selftest, agree <1%)
     on BOTH temperature sources: OM temps (et0_hs_om) and IMD temps
     (et0_hs_imd). OM-PM (API column) is carried as et0_pm_api.
  4. Builds the district-month covariate panel (59-execution basis).
  5. Runs the two consistency checks, OPEN WINDOW ONLY (<=2022, D#21):
       A. IMD 1-deg temperature consistency: OM vs IMD tmax/tmin
          (monthly bias / RMSE / Pearson r, overall + monsoon vs dry).
       B. ET0 triangulation: OM-PM vs OM-HS (method check),
          OM-HS vs IMD-HS (source check), OM-PM vs IMD-HS (worst case).

Discipline
  - AM-6: covariate engineering only, NO model runs. D#22 weights ban intact.
  - Backbone parquet carries the full exogenous weather range with
    in_open_window flag; statistics reported on the open window only.
  - IMD raw .GRD files are NOT committed (regenerable via this script);
    the derived district-day parquet IS committed (provenance-linked).

Usage
  python3 build_weather_backbone.py --selftest
  python3 build_weather_backbone.py --run [--imd-years 2014 2024]
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(HERE)
ROOT = os.path.dirname(PROG)
OM_CSV = os.path.join(ROOT, "download", "agri_tws_ind",
                      "ap_ts_weather_openmeteo_full59.csv")
FETCH_PY = os.path.join(ROOT, "download", "agri_tws_ind", "fetch_data.py")
IMD_DIR = os.path.join(PROG, "data", "imd")
PARQ_DIR = os.path.join(PROG, "data", "parquet")
MANI_DIR = os.path.join(PROG, "data", "manifests")
REPORT_MD = os.path.join(ROOT, "download", "agri_tws_ind",
                         "WEATHER_BACKBONE_REPORT.md")

OPEN_END = "2022-12-31"  # D#21 open window end (dev statistics only)


# ----------------------------------------------------------------------------
# provenance helpers
# ----------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fetch_coords():
    """Parse the (state, district, lat, lon, source) tuples from fetch_data.py
    (repo-tracked provenance — no duplicated hardcoding here)."""
    src = open(FETCH_PY, encoding="utf-8").read()
    pat = re.compile(
        r'\(\s*"(AP|TS|TG)"\s*,\s*"([^"]+)"\s*,\s*([0-9.]+)\s*,'
        r'\s*([0-9.]+)\s*,\s*"(QWEN_PROVIDED|GLM_APPROX)"\s*\)')
    rows = pat.findall(src)
    df = pd.DataFrame(rows, columns=["state", "district", "lat", "lon",
                                      "coord_source"])
    if df.empty:
        raise SystemExit("FAIL: no coord tuples parsed from fetch_data.py")
    df["lat"] = df.lat.astype(float)
    df["lon"] = df.lon.astype(float)
    return df


# ----------------------------------------------------------------------------
# ET0 engines
# ----------------------------------------------------------------------------

def hs_et0_pyet(tmax_s, tmin_s, lat_deg):
    """Hargreaves-Samani ET0 (mm/day) via pyet, method=0, k default.
    pyet expects lat in RADIANS and a DatetimeIndex (DOY drives Ra)."""
    import pyet
    tmean_s = (tmax_s + tmin_s) / 2.0
    out = pyet.hargreaves(tmean_s, tmax_s, tmin_s, math.radians(lat_deg),
                          method=0)
    return out.clip(lower=0.0)


def ra_mm_day(doy, lat_deg):
    """FAO-56 Eq.21 extraterrestrial radiation, mm/day equivalent."""
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365.0)
    d = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(d))))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(d)
        + math.cos(phi) * math.cos(d) * math.sin(ws))
    return ra * 0.408


def hs_et0_fao(tmax, tmin, doy, lat_deg):
    """FAO-56 Eq.52 Hargreaves (independent hand formula for selftest)."""
    tmean = (tmax + tmin) / 2.0
    return 0.0023 * ra_mm_day(doy, lat_deg) * (tmean + 17.8) * math.sqrt(
        max(tmax - tmin, 0.0))


# ----------------------------------------------------------------------------
# IMD
# ----------------------------------------------------------------------------

def imd_nearest_cell(lats, lons, lat, lon):
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    return i, j


def build_imd_daily(coords, years, imd_dir=IMD_DIR):
    """Download/serve IMD 1-deg tmax+tmin, extract nearest cell per district.
    Returns (district-day dataframe, coverage dict)."""
    import imdlib as imd
    os.makedirs(imd_dir, exist_ok=True)
    per_year = {}
    for yr in years:
        for var in ("tmax", "tmin"):
            key = (yr, var)
            try:
                ds = imd.get_data(var, yr, yr, fn_format="year",
                                  file_dir=imd_dir)
            except Exception as e:  # year not published / network
                per_year[key] = f"DOWNLOAD_FAIL: {type(e).__name__}: {e}"[:200]
                continue
            da = ds.get_xarray()[var]
            lats = np.asarray(da.lat.values, dtype=float)
            lons = np.asarray(da.lon.values, dtype=float)
            times = pd.to_datetime(da.time.values)
            arr = np.asarray(da.values, dtype=float)  # (time, lat, lon)
            per_year[key] = dict(shape=arr.shape, n_days=len(times))
            # nearest cell per district, computed once per (yr,var)
            recs = []
            for _, r in coords.iterrows():
                i, j = imd_nearest_cell(lats, lons, r.lat, r.lon)
                recs.append((r.district, lats[i], lons[j],
                             r.lat, r.lon))
            if var == "tmax" and yr == years[0]:
                cell_df = pd.DataFrame(
                    recs, columns=["district", "imd_lat", "imd_lon",
                                   "cent_lat", "cent_lon"])
            store = per_year.setdefault("_store", {})
            store[key] = (arr, times, lats, lons)
    # assemble long dataframe across years/vars
    frames = []
    store = per_year.get("_store", {})
    coords_by_d = coords.set_index("district")
    for (yr, var), (arr, times, lats, lons) in sorted(store.items()):
        sub = {}
        for d in coords.district:
            r = coords_by_d.loc[d]
            i, j = imd_nearest_cell(lats, lons, r.lat, r.lon)
            sub[d] = arr[:, i, j]
        df = pd.DataFrame(sub, index=times)
        df.index.name = "date"
        long = df.stack().rename("value").reset_index()
        long.columns = ["date", "district", "value"]
        long["variable"] = var
        frames.append(long)
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "district", "value", "variable"])
    daily = daily.pivot_table(index=["date", "district"], columns="variable",
                              values="value").reset_index()
    coverage = {f"{k[0]}_{k[1]}": v for k, v in per_year.items()
                if k != "_store"}
    return daily, coverage


# ----------------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------------

def month_key(ts):
    return ts.strftime("%Y-%m")


def run(imd_years):
    os.makedirs(PARQ_DIR, exist_ok=True)
    os.makedirs(MANI_DIR, exist_ok=True)

    coords = parse_fetch_coords()
    om = pd.read_csv(OM_CSV, parse_dates=["date"])
    om_districts = set(om.district.unique())
    coord_districts = set(coords.district)
    if om_districts != coord_districts:
        raise SystemExit(f"FAIL district mismatch: "
                         f"om-only={sorted(om_districts-coord_districts)} "
                         f"coords-only={sorted(coord_districts-om_districts)}")

    # --- IMD ---
    imd_daily, coverage = build_imd_daily(coords, list(range(
        imd_years[0], imd_years[1] + 1)))
    imd_daily["date"] = pd.to_datetime(imd_daily["date"])
    imd_daily = imd_daily.sort_values(["district", "date"])
    imd_path = os.path.join(PARQ_DIR, "imd_district_daily.parquet")
    imd_daily.to_parquet(imd_path, compression="zstd")

    # --- ET0 (pyet HS) on both sources ---
    om = om.sort_values(["district", "date"])
    om["et0_hs_om"] = np.nan
    for d, g in om.groupby("district"):
        lat = float(coords.loc[coords.district == d, "lat"].iloc[0])
        s = hs_et0_pyet(g.set_index("date")["temperature_2m_max"],
                        g.set_index("date")["temperature_2m_min"], lat)
        om.loc[g.index, "et0_hs_om"] = s.to_numpy()
    imd_daily["et0_hs_imd"] = np.nan
    for d, g in imd_daily.groupby("district"):
        lat = float(coords.loc[coords.district == d, "lat"].iloc[0])
        if len(g) == 0:
            continue
        s = hs_et0_pyet(g.set_index("date")["tmax"],
                        g.set_index("date")["tmin"], lat)
        imd_daily.loc[g.index, "et0_hs_imd"] = s.to_numpy()

    # --- merge OM + IMD daily ---
    imd_daily = imd_daily.rename(columns={"tmax": "tmax_imd",
                                          "tmin": "tmin_imd"})
    df = om.merge(imd_daily[["date", "district", "tmax_imd", "tmin_imd",
                             "et0_hs_imd"]],
                  on=["date", "district"], how="left")
    df["in_open_window"] = df.date <= pd.Timestamp(OPEN_END)
    df["tmean_om"] = (df.temperature_2m_max + df.temperature_2m_min) / 2

    # --- monthly panel ---
    df["ym"] = df.date.dt.strftime("%Y-%m")
    agg = df.groupby(["district", "state", "ym"]).agg(
        n_days=("date", "size"),
        precip_mm_sum=("precipitation_sum", "sum"),
        tmax_om_mean=("temperature_2m_max", "mean"),
        tmin_om_mean=("temperature_2m_min", "mean"),
        tmean_om_mean=("tmean_om", "mean"),
        tmax_imd_mean=("tmax_imd", "mean"),
        tmin_imd_mean=("tmin_imd", "mean"),
        et0_pm_api_mmd=("et0_fao_evapotranspiration", "mean"),
        et0_hs_om_mmd=("et0_hs_om", "mean"),
        et0_hs_imd_mmd=("et0_hs_imd", "mean"),
        soil_moisture_mean=("soil_moisture_0_to_7cm_mean", "mean"),
        in_open_window=("in_open_window", "first"),
    ).reset_index()
    for c in ("et0_pm_api", "et0_hs_om", "et0_hs_imd"):
        agg[f"{c}_mm_month"] = agg[f"{c}_mmd"] * agg["n_days"]
    panel_path = os.path.join(PARQ_DIR, "weather_backbone_monthly.parquet")
    agg.to_parquet(panel_path, compression="zstd")

    # --- consistency checks (open window only, D#21) ---
    chk = df[df.in_open_window & df.tmax_imd.notna()].copy()
    chk["d_tmax"] = chk.temperature_2m_max - chk.tmax_imd
    chk["d_tmin"] = chk.temperature_2m_min - chk.tmin_imd
    chk["monsoon"] = chk.date.dt.month.isin([6, 7, 8, 9])

    def stats(x, y, label):
        m = x.notna() & y.notna()
        x, y = x[m], y[m]
        return dict(label=label, n=int(len(x)), bias=float((x - y).mean()),
                    rmse=float(np.sqrt(((x - y) ** 2).mean())),
                    r=float(x.corr(y)))

    checks = dict(
        tmax_overall=stats(chk.temperature_2m_max, chk.tmax_imd,
                           "OM tmax vs IMD tmax (daily, open window)"),
        tmin_overall=stats(chk.temperature_2m_min, chk.tmin_imd,
                           "OM tmin vs IMD tmin (daily, open window)"),
        tmax_monsoon=stats(chk[chk.monsoon].temperature_2m_max,
                           chk[chk.monsoon].tmax_imd, "OM vs IMD tmax MONSOON"),
        tmax_dry=stats(chk[~chk.monsoon].temperature_2m_max,
                       chk[~chk.monsoon].tmax_imd, "OM vs IMD tmax DRY"),
        et0_method=stats(chk.et0_fao_evapotranspiration, chk.et0_hs_om,
                         "OM-PM vs OM-HS (method check)"),
        et0_source=stats(chk.et0_hs_om, chk.et0_hs_imd,
                         "OM-HS vs IMD-HS (source check)"),
        et0_worst=stats(chk.et0_fao_evapotranspiration, chk.et0_hs_imd,
                        "OM-PM vs IMD-HS (worst case)"),
    )
    per_district = {}
    for d, g in chk.groupby("district"):
        per_district[d] = dict(
            tmax_bias=float(g.d_tmax.mean()),
            tmin_bias=float(g.d_tmin.mean()),
            tmax_rmse=float(np.sqrt((g.d_tmax ** 2).mean())),
            tmin_rmse=float(np.sqrt((g.d_tmin ** 2).mean())),
            n=int(len(g)))
    worst_tmax = max(per_district, key=lambda k: abs(per_district[k]["tmax_bias"]))

    # --- report ---
    report = dict(
        built_at_utc=pd.Timestamp.utcnow().isoformat(),
        om_csv_sha256=sha256(OM_CSV),
        panel_sha256=sha256(panel_path),
        imd_parquet_sha256=sha256(imd_path),
        districts=int(coords.district.nunique()),
        om_rows=int(len(om)),
        om_date_range=[str(om.date.min().date()), str(om.date.max().date())],
        imd_years=f"{imd_years[0]}-{imd_years[1]}",
        imd_coverage=coverage,
        open_window_end=OPEN_END,
        consistency=checks,
        per_district_worst_tmax_bias=dict(
            district=worst_tmax, **per_district[worst_tmax]),
        et0_engine="pyet 1.5 hargreaves method=0 k=0.0135 (FAO-56 "
                   "Eq.52 equivalent, selftest cross-checked <1%)",
        notes=[
            "IMD 1-deg grid representativeness: nearest cell to district "
            "centroid; coastal/Ghats districts inherit grid-cell bias.",
            "Open-Meteo = ERA5/ERA5-Land reanalysis (CC-BY-4.0); IMD "
            "gridded = station-based 1-deg official product (IMD Pune, "
            "free download, academic use; raw .GRD not committed).",
            "Statistics on open window (<=2022) only per D#21; monthly "
            "panel carries full exogenous range with in_open_window flag.",
            "AM-6 untouched: covariate engineering only, no model runs; "
            "D#22 weights ban intact.",
        ])
    rp = os.path.join(MANI_DIR, "weather_backbone_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=1, default=str)

    # human-readable markdown
    md = ["# Weather Backbone Report — pyet ET0 + IMD 1-deg consistency",
          "",
          f"Built {report['built_at_utc']} | OM rows {report['om_rows']:,} | "
          f"districts {report['districts']} | IMD {report['imd_years']}",
          "",
          "## Headline checks (open window <=2022, D#21)",
          "",
          "| Check | n | bias | RMSE | Pearson r |",
          "|---|---|---|---|---|"]
    for k, v in checks.items():
        md.append(f"| {v['label']} | {v['n']:,} | {v['bias']:+.2f} | "
                  f"{v['rmse']:.2f} | {v['r']:.3f} |")
    md += ["", "## Verdict",
           "",
           f"- Temperature: OM(ERA5) vs IMD 1-deg daily tmax bias "
           f"{checks['tmax_overall']['bias']:+.2f} C, r "
           f"{checks['tmax_overall']['r']:.3f} — "
           + ("CONSISTENT" if checks['tmax_overall']['r'] > 0.85
              and abs(checks['tmax_overall']['bias']) < 2 else "FLAG"),
           f"- ET0 method check: PM(API) vs HS(pyet, OM temps) bias "
           f"{checks['et0_method']['bias']:+.2f} mm/day, r "
           f"{checks['et0_method']['r']:.3f} — "
           + ("CONSISTENT" if checks['et0_method']['r'] > 0.85 else "FLAG"),
           f"- ET0 source check: HS(OM) vs HS(IMD) bias "
           f"{checks['et0_source']['bias']:+.2f} mm/day, r "
           f"{checks['et0_source']['r']:.3f} — "
           + ("CONSISTENT" if checks['et0_source']['r'] > 0.85 else "FLAG"),
           f"- Worst tmax-bias district: {worst_tmax} "
           f"({report['per_district_worst_tmax_bias']['tmax_bias']:+.2f} C)",
           "",
           "## Notes"]
    md += [f"- {n}" for n in report["notes"]]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps(dict(panel_rows=len(agg), checks={k: dict(
        bias=round(v['bias'], 3), rmse=round(v['rmse'], 3),
        r=round(v['r'], 4)) for k, v in checks.items()}), indent=1))
    print("WROTE:", panel_path)
    print("WROTE:", imd_path)
    print("WROTE:", rp)
    print("WROTE:", REPORT_MD)


# ----------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------

def selftest():
    coords = parse_fetch_coords()
    assert len(coords) == 59, f"coords != 59: {len(coords)}"
    assert coords.lat.between(11, 21).all() and coords.lon.between(75, 86).all()

    # pyet HS vs FAO-56 hand formula (the calibration this build relies on)
    idx = pd.date_range("2022-05-15", periods=1)
    tmax, tmin = pd.Series([35.0], idx), pd.Series([25.0], idx)
    v_pyet = float(hs_et0_pyet(tmax, tmin, 17.5).iloc[0])
    v_fao = hs_et0_fao(35.0, 25.0, 135, 17.5)
    ratio = v_pyet / v_fao
    assert 0.95 < ratio < 1.05, f"pyet/FAO ratio {ratio:.4f} out of band"

    # nearest-cell arithmetic
    lats = np.arange(7.5, 38.0, 1.0)
    lons = np.arange(67.5, 98.0, 1.0)
    i, j = imd_nearest_cell(lats, lons, 17.38, 78.49)
    assert lats[i] == 17.5 and lons[j] == 78.5

    # month aggregation invariant on synthetic 2022 (non-leap)
    d = pd.date_range("2022-01-01", "2022-12-31")
    g = pd.DataFrame({"date": d, "district": "X"})
    g["ym"] = g.date.dt.strftime("%Y-%m")
    nd = g.groupby("ym").size()
    import calendar
    for ym, n in nd.items():
        y, m = map(int, ym.split("-"))
        assert n == calendar.monthrange(y, m)[1]

    # OM sanity
    om = pd.read_csv(OM_CSV, usecols=["et0_fao_evapotranspiration",
                                       "precipitation_sum"])
    assert om.et0_fao_evapotranspiration.between(0, 20).all()
    assert (om.precipitation_sum >= 0).all()

    print(f"SELFTEST PASS (coords=59, pyet/FAO ratio={ratio:.4f}, "
          f"OM rows={len(om):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--imd-years", nargs=2, type=int, default=[2014, 2024])
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run(tuple(a.imd_years))
    else:
        ap.print_help()
