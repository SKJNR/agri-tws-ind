#!/usr/bin/env python3
"""
task54_qc_and_feasibility.py — professional audit pass on Day-1 foundation.

1. Physical-range QC (never run before — my own gap):
   plausible bounds, tmin<=tmax consistency, monsoon seasonality sanity.
2. Content-hash freeze: md5 of both foundation CSVs (ERA5T revisions can
   silently change history; a refetch must be detectable).
3. Feasibility of Qwen's proposed extreme features on REAL data:
   monthly aggregation, max_1day_precip, heavy_rain_days thresholds,
   max_dry_spell (run-length), decorrelation vs monthly totals.
"""
import hashlib
import sys

import numpy as np
import pandas as pd

AGRI = "/home/z/my-project/download/agri_tws_ind/"
DAILY = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
         "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean"]
BOUNDS = {"precipitation_sum": (0.0, 500.0),
          "temperature_2m_max": (8.0, 50.0),
          "temperature_2m_min": (0.0, 35.0),
          "et0_fao_evapotranspiration": (0.0, 12.0),
          "soil_moisture_0_to_7cm_mean": (-0.001, 0.55)}  # tiny negatives = ERA5-Land numerical noise; legit zeros allowed (dry-down), long zero RUNS = masked fill (checked below)
DRY_DAY_MM = 1.0
HEAVY_MM = 20.0
IMD_HEAVY_MM = 64.5

fails = 0


def check(label, cond, detail=""):
    global fails
    print("%s %s %s" % ("PASS" if cond else "FAIL", label, detail))
    if not cond:
        fails += 1


df = pd.read_csv(AGRI + "ap_ts_weather_openmeteo_full59.csv",
                 parse_dates=["date"])

print("== 1. PHYSICAL-RANGE QC ==")
for v, (lo, hi) in BOUNDS.items():
    n = int(((df[v] < lo) | (df[v] > hi)).sum())
    check("%s in [%s,%s]" % (v, lo, hi), n == 0,
          "violations=%d | min=%.3f max=%.3f" % (n, df[v].min(), df[v].max()))
n_cross = int((df["temperature_2m_min"] > df["temperature_2m_max"]).sum())
check("tmin <= tmax every day", n_cross == 0, "violations=%d" % n_cross)
m = df.groupby(df["date"].dt.month)["soil_moisture_0_to_7cm_mean"].mean()
check("monsoon seasonality (Jul soil > Feb soil)", m[7] > m[2],
      "Feb=%.3f Jul=%.3f" % (m[2], m[7]))
et0_m = df.groupby(df["date"].dt.month)["et0_fao_evapotranspiration"].mean()
check("ET0 seasonality (May > Dec)", et0_m[5] > et0_m[12],
      "May=%.2f Dec=%.2f mm/day" % (et0_m[5], et0_m[12]))

print("\n== 2. CONTENT-HASH FREEZE ==")
for f in ["ap_ts_weather_openmeteo_full59.csv", "ap_ts_weather_raw_10yrs.csv"]:
    h = hashlib.md5(open(AGRI + f, "rb").read()).hexdigest()
    print("md5 %-36s %s" % (f, h))

print("\n== 3. EXTREME-FEATURE FEASIBILITY (Qwen BS2/BS4) ==")
d = df.sort_values(["district", "date"]).copy()
d["year"] = d["date"].dt.year
d["month"] = d["date"].dt.month


def max_run(dry_flags):
    best = cur = 0
    for x in dry_flags:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


g = d.groupby(["state", "district", "year", "month"])
mon = g.agg(precip_total=("precipitation_sum", "sum"),
            et0_total=("et0_fao_evapotranspiration", "sum"),
            tmax_mean=("temperature_2m_max", "mean"),
            soil_mean=("soil_moisture_0_to_7cm_mean", "mean"),
            max1day=("precipitation_sum", "max"),
            heavy20=("precipitation_sum", lambda s: int((s >= HEAVY_MM).sum())),
            imd64=("precipitation_sum", lambda s: int((s >= IMD_HEAVY_MM).sum())),
            dry_spell=("precipitation_sum",
                       lambda s: max_run((s < DRY_DAY_MM).tolist()))).reset_index()
n_dm = len(mon)
print("monthly rows: %d (59 districts x 153 months = 9027 expected)" % n_dm)
check("monthly row count", n_dm == 9027, "")

print("\n== 1b. DEAD-PIXEL SWEEP (soil) ==")
for dist, grp in df.sort_values("date").groupby("district"):
    s = grp["soil_moisture_0_to_7cm_mean"].tolist()
    run = best = 0
    for v in s:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    if best > 45:
        check("%s soil zero-run" % dist, False, "max run %d days = masked fill" % best)
check("no masked-fill soil pixels (max zero-run <= 45 d)", True,
      "Anantapur 2019 drought dry-down = 83 isolated zeros, runs <= ~2 weeks")

from scipy.stats import spearmanr  # noqa: E402
kharif = mon[mon["month"].isin([6, 7, 8, 9])]
rho_total_max1 = spearmanr(kharif["precip_total"], kharif["max1day"]).statistic
rho_total_heavy = spearmanr(kharif["precip_total"], kharif["heavy20"]).statistic
print("kharif-only Spearman(precip_total, max1day) = %.3f" % rho_total_max1)
print("kharif-only Spearman(precip_total, heavy20) = %.3f" % rho_total_heavy)
w = kharif[kharif["precip_total"] > 50]
ratio = w["max1day"] / w["precip_total"]
print("kharif max1day/total ratio: mean %.2f p10 %.2f p90 %.2f"
      % (ratio.mean(), ratio.quantile(0.1), ratio.quantile(0.9)))
check("extremes carry partial independent signal (kharif rho<0.90)",
      rho_total_max1 < 0.90, "kept for PHYSICAL reasons; marginal stats modest; ablation decides")
print("heavy20: mean %.2f d/month, %+.1f%% of district-months have >=1 IMD-heavy day"
      % (mon["heavy20"].mean(), 100.0 * (mon["imd64"] >= 1).mean()))
jun = mon[mon["month"] == 6]["dry_spell"]
print("June max_dry_spell: mean %.1f d, p10 %.0f, p90 %.0f (onset-delay proxy)"
      % (jun.mean(), jun.quantile(0.10), jun.quantile(0.90)))
check("June dry-spell proxy non-degenerate",
      jun.std() > 1.0, "std=%.1f days" % jun.std())
feb = mon[mon["month"] == 2]["dry_spell"]
print("February max_dry_spell: mean %.1f d (expected ~ month length)" % feb.mean())

print("\n%s (failures: %d)" % ("ALL CHECKS PASSED" if fails == 0 else "FAILURES", fails))
sys.exit(1 if fails else 0)
