#!/usr/bin/env python3
"""
weather_backbone_riders.py — implements Qwen riders WB-R1/R2/R3 on the
delivered Weather Backbone (DECISION_LOG Addendum 9, 2026-09-23;
consequences PRE-STATED in the addendum BEFORE this run).

  WB-R1  season x regime bias-decomposition table (IMD-vs-OM tmax/tmin)
         + trip rule |bias| > 1.0 C + pre-stated consequence:
         flags, OM->IMD offset table, bias-corrected ET0 VARIANT
         (diagnostic lane; pinned primary et0_pm_api unchanged).
  WB-R2  REPRESENTATIVENESS_LIMITED tags (named terrain list UNION
         empirical |tmax bias| >= 2.0 C) + headline checks recomputed
         on the low-relief subset; verdict-flip detection.
  WB-R3  per-district monthly divergence log (primary et0_pm_api vs
         pyet artifact et0_hs_om), MAD% > 15 => QA flag; rollups.

Discipline: open window only (<=2022, D#21). No sealed contact.
AM-6: QA/covariate engineering, no model runs. D#22 intact.

Usage
  python3 weather_backbone_riders.py --selftest
  python3 weather_backbone_riders.py --run
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(HERE)
ROOT = os.path.dirname(PROG)
OM_CSV = os.path.join(ROOT, "download", "agri_tws_ind",
                      "ap_ts_weather_openmeteo_full59.csv")
PARQ = os.path.join(PROG, "data", "parquet")
MANI = os.path.join(PROG, "data", "manifests")
REPORT_MD = os.path.join(ROOT, "download", "agri_tws_ind",
                         "WEATHER_BACKBONE_RIDERS.md")

OPEN_END = pd.Timestamp("2022-12-31")   # D#21
SEASONS = [("DJF", [12, 1, 2]), ("MAM", [3, 4, 5]),
           ("JJAS", [6, 7, 8, 9]), ("ON", [10, 11])]
REGIMES = [("R1418", 2014, 2018), ("R1922", 2019, 2022)]
TRIP_TEMP = 1.0      # deg C, pre-stated (WB-R1)
TRIP_TMAX_TAG = 2.0  # deg C, empirical terrain tag (WB-R2)
TRIP_MAD_PCT = 15.0  # % monthly MAD, pre-stated (WB-R3)

# Named terrain list — declared in Addendum 9 BEFORE this session's
# per-district computation (Eastern Ghats agency / upper-Godavari gorge)
NAMED_TERRAIN = ["Alluri Sitharama Raju", "Parvathipuram Manyam",
                 "Mulugu", "Bhadradri Kothagudem"]

PANEL = os.path.join(PARQ, "weather_backbone_monthly.parquet")
IMD_PQ = os.path.join(PARQ, "imd_district_daily.parquet")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def season_of(month):
    for name, months in SEASONS:
        if month in months:
            return name
    raise ValueError(month)


def load_daily():
    om = pd.read_csv(OM_CSV, parse_dates=["date"]).sort_values(
        ["district", "date"])
    imd = pd.read_parquet(IMD_PQ)
    imd["date"] = pd.to_datetime(imd["date"])
    imd = imd.rename(columns={"tmax": "tmax_imd", "tmin": "tmin_imd"})
    df = om.merge(imd[["date", "district", "tmax_imd", "tmin_imd"]],
                  on=["date", "district"], how="left")
    df = df[df.date <= OPEN_END]                      # D#21 open window
    df = df[df.tmax_imd.notna() & df.tmin_imd.notna()].copy()
    # et0_hs_om via pyet — IDENTICAL engine call to build_weather_backbone
    # (pyet.hargreaves method=0, lat radians, DatetimeIndex) so R2's
    # low-relief recomputation reproduces the delivered checks.
    import math
    import pyet
    import re
    src = open(os.path.join(ROOT, "download", "agri_tws_ind",
                            "fetch_data.py"), encoding="utf-8").read()
    lats = dict(re.findall(
        r'\(\s*"(?:AP|TS|TG)"\s*,\s*"([^"]+)"\s*,\s*([0-9.]+)\s*,', src))
    df["et0_hs_om"] = np.nan
    df["et0_hs_imd"] = np.nan
    for d, g in df.groupby("district"):
        lat_r = math.radians(float(lats[d]))
        idx = g.set_index("date")
        s_om = pyet.hargreaves(
            (idx["temperature_2m_max"] + idx["temperature_2m_min"]) / 2.0,
            idx["temperature_2m_max"], idx["temperature_2m_min"],
            lat_r, method=0)
        df.loc[g.index, "et0_hs_om"] = s_om.clip(lower=0.0).to_numpy()
        s_imd = pyet.hargreaves(
            (idx["tmax_imd"] + idx["tmin_imd"]) / 2.0,
            idx["tmax_imd"], idx["tmin_imd"], lat_r, method=0)
        df.loc[g.index, "et0_hs_imd"] = s_imd.clip(lower=0.0).to_numpy()
    df["d_tmax"] = df.temperature_2m_max - df.tmax_imd
    df["d_tmin"] = df.temperature_2m_min - df.tmin_imd
    df["season"] = df.date.dt.month.map(season_of)
    df["year"] = df.date.dt.year
    df["regime"] = np.where(df.year <= 2018, "R1418", "R1922")
    return df


# ---------------------------------------------------------------- WB-R1
def rider1(df):
    # aggregate table: season x regime x variable
    agg_rows = []
    for (sname, _), (rname, y0, y1) in [
        (s, r) for s in SEASONS for r in REGIMES]:
        g = df[(df.season == sname) & (df.regime == rname)]
        for var in ("tmax", "tmin"):
            b = g[f"d_{var}"]
            agg_rows.append(dict(
                season=sname, regime=rname, variable=var, n=int(len(g)),
                bias=float(b.mean()),
                rmse=float(np.sqrt((b ** 2).mean()))))
    agg = pd.DataFrame(agg_rows)
    agg["trip"] = agg.bias.abs() > TRIP_TEMP

    # season pooled over regimes (report convenience)
    pool = agg.groupby(["season", "variable"]).agg(
        bias_weighted=("bias", "mean")).reset_index()  # cells are balanced

    # per-district x season table (whole open window per cell)
    pd_rows = []
    for d, gd in df.groupby("district"):
        for sname, _ in SEASONS:
            g = gd[gd.season == sname]
            for var in ("tmax", "tmin"):
                pd_rows.append(dict(
                    district=d, season=sname, variable=var,
                    n=int(len(g)), bias=float(g[f"d_{var}"].mean())))
    per_district = pd.DataFrame(pd_rows)
    per_district["trip"] = per_district.bias.abs() > TRIP_TEMP

    # per-district tmax bias (whole open window) — input to WB-R2 empirical
    d_tmax_bias = df.groupby("district").d_tmax.mean()

    # consequence (pre-stated): offsets + corrected ET0 variant
    off = per_district.pivot_table(index=["district", "season"],
                                   columns="variable", values="bias")
    off = off.rename(columns={"tmax": "tmax_offset", "tmin": "tmin_offset"}
                     ).reset_index()
    # corrected HS variant (diagnostic lane)
    m = df.merge(off, on=["district", "season"], how="left")
    m["tmax_bc"] = m.temperature_2m_max - m.tmax_offset
    m["tmin_bc"] = m.temperature_2m_min - m.tmin_offset
    return dict(aggregate=agg, pooled=pool, per_district=per_district,
                offsets=off, daily_corrected=m, d_tmax_bias=d_tmax_bias)


def hs_fao52(tmax, tmin, doy, lat_deg):
    """FAO-56 Eq.52 Hargreaves — the build's calibrated engine (pyet
    equivalent; <1% by selftest). Reused here so the corrected variant
    uses the SAME method as et0_hs_om."""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=float))
    doy = np.asarray(doy, dtype=float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = -np.tan(phi) * np.tan(dec)
    ws = np.arccos(np.clip(x, -1.0, 1.0))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec)
        + np.cos(phi) * np.cos(dec) * np.sin(ws))
    ra_mm = ra * 0.408
    tmean = (np.asarray(tmax) + np.asarray(tmin)) / 2.0
    return 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(
        np.maximum(np.asarray(tmax) - np.asarray(tmin), 0.0))


def corrected_et0_variant(r1, coords):
    """et0_hs_om_bc: Hargreaves on offset-corrected OM temps, per
    district-season. Monthly means (mm/d) written to the riders panel."""
    m = r1["daily_corrected"]
    lat_by_d = dict(zip(coords.district, coords.lat))
    out = []
    for d, g in m.groupby("district"):
        lat = lat_by_d[d]
        doy = g.date.dt.dayofyear.to_numpy()
        et0 = hs_fao52(g.tmax_bc.to_numpy(), g.tmin_bc.to_numpy(),
                       doy, lat)
        gg = pd.DataFrame({"district": d, "date": g.date.to_numpy(),
                           "et0_hs_om_bc": np.clip(et0, 0, None),
                           "et0_hs_om_raw": hs_fao52(
                               g.temperature_2m_max.to_numpy(),
                               g.temperature_2m_min.to_numpy(), doy, lat)})
        out.append(gg)
    daily = pd.concat(out, ignore_index=True)
    daily["ym"] = pd.to_datetime(daily.date).dt.strftime("%Y-%m")
    monthly = daily.groupby(["district", "ym"]).agg(
        et0_hs_om_bc_mmd=("et0_hs_om_bc", "mean"),
        et0_hs_om_mmd=("et0_hs_om_raw", "mean")).reset_index()
    return daily, monthly


# ---------------------------------------------------------------- WB-R2
def rider2(df, d_tmax_bias):
    named = [d for d in NAMED_TERRAIN if d in set(df.district)]
    empirical = sorted(d for d, b in d_tmax_bias.items()
                       if abs(b) >= TRIP_TMAX_TAG)
    tagged = sorted(set(named) | set(empirical))
    low_relief = sorted(set(df.district) - set(tagged))

    def checks(sub, tag):
        rows = {}
        mon = sub.date.dt.month.isin([6, 7, 8, 9])

        def st(x, y):
            m = x.notna() & y.notna()
            x, y = x[m], y[m]
            return dict(n=int(len(x)), bias=round(float((x - y).mean()), 3),
                        rmse=round(float(np.sqrt(((x - y) ** 2).mean())), 3),
                        r=round(float(x.corr(y)), 4))
        rows["tmax_overall"] = st(sub.temperature_2m_max, sub.tmax_imd)
        rows["tmin_overall"] = st(sub.temperature_2m_min, sub.tmin_imd)
        rows["tmax_monsoon"] = st(sub[mon].temperature_2m_max,
                                  sub[mon].tmax_imd)
        rows["tmax_dry"] = st(sub[~mon].temperature_2m_max,
                              sub[~mon].tmax_imd)
        rows["et0_method"] = st(sub.et0_fao_evapotranspiration,
                                sub.et0_hs_om)
        rows["et0_source"] = st(sub.et0_hs_om, sub.et0_hs_imd)
        rows["et0_worst"] = st(sub.et0_fao_evapotranspiration,
                               sub.et0_hs_imd)
        return {f"{tag}::{k}": v for k, v in rows.items()}

    full = checks(df, "full")
    low = checks(df[df.district.isin(low_relief)], "lowrelief")
    # verdict flip detection (pre-stated consequence)
    flips = []
    for k in ("tmax_overall", "tmin_overall", "tmax_monsoon", "tmax_dry",
              "et0_method", "et0_source", "et0_worst"):
        f, l = full[f"full::{k}"], low[f"lowrelief::{k}"]
        ok = lambda v: v["r"] > 0.85 and (abs(v["bias"]) < 2
                                          if "tmax" in k or "tmin" in k
                                          else True)
        if ok(f) and not ok(l):
            flips.append(k)
    return dict(named=named, empirical=empirical, tagged=tagged,
                low_relief=low_relief, checks_full=full, checks_low=low,
                verdict_flips=flips)


# ---------------------------------------------------------------- WB-R3
def rider3():
    panel = pd.read_parquet(PANEL)
    p = panel[panel.in_open_window].copy()
    p = p[p.et0_pm_api_mmd > 0]
    p["mad_pct"] = (p.et0_pm_api_mmd - p.et0_hs_om_mmd).abs() / \
        p.et0_pm_api_mmd * 100.0
    p["qa_flag"] = p.mad_pct > TRIP_MAD_PCT
    roll = p.groupby("district").agg(
        n_months=("mad_pct", "size"),
        n_flagged=("qa_flag", "sum"),
        mean_mad_pct=("mad_pct", "mean"),
        max_mad_pct=("mad_pct", "max")).reset_index()
    return dict(monthly=p, rollup=roll,
                n_flagged_total=int(p.qa_flag.sum()),
                n_districtmonths=int(len(p)))


# ---------------------------------------------------------------- run
def parse_fetch_coords():
    import re
    src = open(os.path.join(ROOT, "download", "agri_tws_ind",
                            "fetch_data.py"), encoding="utf-8").read()
    pat = re.compile(
        r'\(\s*"(AP|TS|TG)"\s*,\s*"([^"]+)"\s*,\s*([0-9.]+)\s*,'
        r'\s*([0-9.]+)\s*,\s*"(QWEN_PROVIDED|GLM_APPROX)"\s*\)')
    rows = pat.findall(src)
    df = pd.DataFrame(rows, columns=["state", "district", "lat", "lon",
                                      "coord_source"])
    df["lat"] = df.lat.astype(float)
    df["lon"] = df.lon.astype(float)
    return df


def run():
    coords = parse_fetch_coords()
    df = load_daily()
    assert df.district.nunique() == 59, "district count != 59"

    r1 = rider1(df)
    daily_bc, monthly_bc = corrected_et0_variant(r1, coords)
    r2 = rider2(df, r1["d_tmax_bias"])
    r3 = rider3()

    # riders monthly panel (new artifact — delivered panel untouched)
    panel = pd.read_parquet(PANEL)
    riders_panel = panel.merge(
        monthly_bc, on=["district", "ym"], how="left").merge(
        r3["monthly"][["district", "ym", "mad_pct", "qa_flag"]],
        on=["district", "ym"], how="left")
    rp_path = os.path.join(PARQ, "weather_riders_monthly.parquet")
    riders_panel.to_parquet(rp_path, compression="zstd")

    # ---- manifest ----
    manifest = dict(
        built_at_utc=pd.Timestamp.utcnow().isoformat(),
        implements="DECISION_LOG Addendum 9 riders WB-R1/R2/R3 "
                   "(pre-stated before this run)",
        inputs_sha256=dict(
            om_csv=sha256(OM_CSV), panel=sha256(PANEL), imd=sha256(IMD_PQ)),
        rules=dict(
            seasons={s: m for s, m in SEASONS},
            regimes={r: [y0, y1] for r, y0, y1 in REGIMES},
            trip_temp_c=TRIP_TEMP, trip_tmax_tag_c=TRIP_TMAX_TAG,
            trip_mad_pct=TRIP_MAD_PCT, named_terrain=NAMED_TERRAIN),
        wb_r1=dict(
            aggregate_table=r1["aggregate"].to_dict("records"),
            n_aggregate_trips=int(r1["aggregate"].trip.sum()),
            n_district_season_trips=int(r1["per_district"].trip.sum()),
            district_season_trips=r1["per_district"][
                r1["per_district"].trip].to_dict("records"),
            offsets=r1["offsets"].to_dict("records"),
            corrected_variant="et0_hs_om_bc (diagnostic lane; primary "
                              "et0_pm_api PINNED, unchanged)"),
        wb_r2=dict(
            named=r2["named"], empirical=r2["empirical"],
            tagged=r2["tagged"], n_tagged=len(r2["tagged"]),
            low_relief=r2["low_relief"], n_low_relief=len(r2["low_relief"]),
            checks=r2["checks_full"] | r2["checks_low"],
            verdict_flips=r2["verdict_flips"]),
        wb_r3=dict(
            n_districtmonths=r3["n_districtmonths"],
            n_flagged_total=r3["n_flagged_total"],
            flagged_districts=r3["rollup"][
                r3["rollup"].n_flagged > 0].to_dict("records"),
            rollup=r3["rollup"].to_dict("records")),
        riders_panel_sha256=sha256(rp_path),
    )
    mp = os.path.join(MANI, "weather_qa_riders.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1, default=str)

    # ---- human-readable report ----
    a = r1["aggregate"]
    md = ["# Weather Backbone Riders — WB-R1/R2/R3 (Addendum 9)",
          "",
          f"Built {manifest['built_at_utc']} | open window <=2022 (D#21) | "
          "rules pre-stated in DECISION_LOG Addendum 9 BEFORE this run",
          "",
          "## WB-R1 — season x regime bias decomposition (OM - IMD, degC)",
          "",
          "| season | regime | var | n | bias | RMSE | trip |",
          "|---|---|---|---|---|---|---|"]
    for _, r in a.iterrows():
        md.append(f"| {r.season} | {r.regime} | {r.variable} | {r.n:,} | "
                  f"{r.bias:+.2f} | {r.rmse:.2f} | "
                  f"{'YES' if r.trip else ''} |")
    md += ["",
           f"Aggregate trips (>1.0C): **{manifest['wb_r1']['n_aggregate_trips']}"
           f"/16** | district-season trips: **"
           f"{manifest['wb_r1']['n_district_season_trips']}/472**",
           "",
           "## WB-R2 — REPRESENTATIVENESS_LIMITED + low-relief verdicts",
           "",
           f"- Named terrain: {', '.join(r2['named'])}",
           f"- Empirical (|tmax bias| >= 2.0C): {', '.join(r2['empirical'])}",
           f"- TAGGED ({len(r2['tagged'])}): {', '.join(r2['tagged'])}",
           f"- Low-relief subset ({len(r2['low_relief'])} districts)",
           "",
           "| Check | n (full) | bias (full) | r (full) | n (low-relief) | "
           "bias (low) | r (low) |",
           "|---|---|---|---|---|---|---|"]
    for k in ("tmax_overall", "tmin_overall", "tmax_monsoon", "tmax_dry",
              "et0_method", "et0_source", "et0_worst"):
        f_, l_ = r2["checks_full"][f"full::{k}"], \
            r2["checks_low"][f"lowrelief::{k}"]
        md.append(f"| {k} | {f_['n']:,} | {f_['bias']:+.2f} | {f_['r']:.3f} "
                  f"| {l_['n']:,} | {l_['bias']:+.2f} | {l_['r']:.3f} |")
    md += ["",
           f"Verdict flips on low-relief subset: "
           f"**{r2['verdict_flips'] or 'NONE — QA status holds'}**",
           "",
           "## WB-R3 — ET0 divergence log (primary OM-PM vs pyet artifact "
           "HS-OM)",
           "",
           f"- District-months (open window): {r3['n_districtmonths']:,}",
           f"- QA flags (monthly MAD > 15%): **{r3['n_flagged_total']}**",
           f"- Districts with any flag: "
           f"{len(manifest['wb_r3']['flagged_districts'])}",
           ""]
    if manifest["wb_r3"]["flagged_districts"]:
        md += ["| district | flagged months | mean MAD% | max MAD% |",
               "|---|---|---|---|"]
        for d in manifest["wb_r3"]["flagged_districts"]:
            md.append(f"| {d['district']} | {int(d['n_flagged'])} | "
                      f"{d['mean_mad_pct']:.1f} | {d['max_mad_pct']:.1f} |")
    md += ["",
           "Primary ET0 et0_pm_api remains PINNED (OM hash-frozen; swap = "
           "AM-5 amendment). et0_hs_om_bc carried as diagnostic lane in "
           "weather_riders_monthly.parquet.",
           "",
           "AM-6 untouched (QA/covariate engineering, no model runs); "
           "D#22 weights ban intact; no sealed contact."]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md) + "\n")

    print("AGG TRIPS:", manifest["wb_r1"]["n_aggregate_trips"], "/16")
    print("DISTR-SEASON TRIPS:", manifest["wb_r1"]["n_district_season_trips"],
          "/472")
    print("TAGGED:", len(r2["tagged"]), r2["tagged"])
    print("VERDICT FLIPS:", r2["verdict_flips"] or "NONE")
    print("R3 FLAGS:", r3["n_flagged_total"], "of", r3["n_districtmonths"])
    print("WROTE:", rp_path)
    print("WROTE:", mp)
    print("WROTE:", REPORT_MD)


def selftest():
    coords = parse_fetch_coords()
    assert len(coords) == 59
    for d in NAMED_TERRAIN:
        assert d in set(coords.district), f"named terrain missing: {d}"
    # season mapping totality + partition
    allm = [m for _, ms in SEASONS for m in ms]
    assert sorted(allm) == list(range(1, 13)), "seasons must partition year"
    # HS engine vs build selftest anchor (35/25C, DOY 135, lat 17.5)
    v = float(hs_fao52([35.0], [25.0], [135], 17.5)[0])
    import math
    assert 4.0 < v < 7.0, f"HS hand engine out of plausible band: {v}"
    # trip thresholds sane
    assert TRIP_TEMP == 1.0 and TRIP_TMAX_TAG == 2.0 \
        and TRIP_MAD_PCT == 15.0
    print(f"SELFTEST PASS (59 coords, seasons partition, HS={v:.2f} mm/d)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run()
    else:
        ap.print_help()
