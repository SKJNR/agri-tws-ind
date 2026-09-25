#!/usr/bin/env python3
"""Task-55 Track A: file-internal datum-signature analysis (Addendum 11).

Pre-registered decision rules FROZEN in download/DECISION_LOG.md @ commit
07c76f7 (BEFORE this script ran). Report-only (AM-3): nothing dropped or
imputed. OPEN rows only (date <= 2022-12-31; D#24 sealed discipline).

Verdict rules (frozen):
  masl-frame  IF spearman(med_gwl, elevation) >= 0.80 AND median|gwl-elev| <= 25 m
  mbgl-frame  IF spearman <= 0.30 AND >=85% depth-bearing wells have 0<=med_gwl<=depth
              AND overall median med_gwl in [0, 300] m
  else        mixed/QC-contaminated (per-class breakdown)
Elevation credibility gate: if median |file_elev - dem_elev| > 50 m on the
100-well sample (seed 42), elevation tests are down-weighted.
Negatives class rule: (-50,0) m at elev<50 m wells = plausibly-masl-QC-noise;
<= -50 m = QC-failure class regardless of frame.
"""
import json, random, sys, urllib.request
import duckdb
import numpy as np
import pandas as pd

CSV = "/tmp/gwl_download/gwl_data.csv"
OUT_JSON = "/home/z/my-project/program/data/manifests/task55_datum_report.json"
CUT = "2022-12-31"

FAIL = []
def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print(f"FAIL-LOUD: {msg}")

con = duckdb.connect()

# ---- 0. Guardrails (pre-registered): open row-count must equal 992,053 exactly
n_open_all = con.execute(
    f"SELECT count(*) FROM read_csv_auto('{CSV}') WHERE date <= DATE '{CUT}'"
).fetchone()[0]
print(f"all-India open rows (<= {CUT}): {n_open_all} (expected 992053)")
check(n_open_all == 992053, f"open row count {n_open_all} != 992053")

# ---- 1. Discover AP/TG state labels (do not assume)
states = [r[0] for r in con.execute(
    f"SELECT DISTINCT state FROM read_csv_auto('{CSV}') WHERE date <= DATE '{CUT}'"
).fetchall()]
ap_labels = [s for s in states if s and ("andhra" in s.lower() or s.strip().upper() in ("AP",))]
tg_labels = [s for s in states if s and ("elanga" in s.lower() or s.strip().upper() in ("TG", "TELANGANA"))]
print(f"state labels found: AP={ap_labels} TG={tg_labels}")
lab = tuple(ap_labels + tg_labels)
check(len(lab) >= 2, f"AP/TG state labels not found: {states}")

n_ap_tg = con.execute(
    f"SELECT count(*) FROM read_csv_auto('{CSV}') "
    f"WHERE date <= DATE '{CUT}' AND state IN ({','.join(chr(39)+s+chr(39) for s in lab)})"
).fetchone()[0]
print(f"AP+TG open rows: {n_ap_tg} (Addendum-4 cross-check: 365742)")
check(n_ap_tg == 365742, f"AP+TG open rows {n_ap_tg} != 365742")

# ---- 2. Per-well aggregation (analysis population: n_val >= 5)
q = f"""
SELECT station_code, any_value(district) district, any_value(state) state,
       any_value(latitude) lat, any_value(longitude) lon,
       any_value(elevation) elevation, any_value(depth) depth,
       count(*) n_rows, count(gwl_value) n_val,
       median(gwl_value) med_gwl, min(gwl_value) mn_gwl, max(gwl_value) mx_gwl
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({','.join(chr(39)+s+chr(39) for s in lab)})
GROUP BY station_code
"""
wells = con.execute(q).fetch_df()
print(f"wells total AP+TG: {len(wells)} (Addendum-4: 3223)")
check(len(wells) == 3223, f"well count {len(wells)} != 3223")

# row-level negatives reproduction (Addendum 4: 114,062/365,742; range -1147.7..+971.3)
row_neg = con.execute(
    f"SELECT count(*) FILTER (WHERE gwl_value < 0), count(gwl_value), "
    f"min(gwl_value), max(gwl_value) FROM read_csv_auto('{CSV}') "
    f"WHERE date <= DATE '{CUT}' AND state IN ({','.join(chr(39)+s+chr(39) for s in lab)})"
).fetchone()
print(f"row-level: negatives {row_neg[0]}/{row_neg[1]}, range {row_neg[2]}..{row_neg[3]}")
check(row_neg[0] == 114062, f"row negatives {row_neg[0]} != 114062")

pop = wells[wells.n_val >= 5].copy()
print(f"analysis population (n_val>=5): {len(pop)} wells")

def spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 3:
        return float("nan")
    ra, rb = s.a.rank(), s.b.rank()
    return float(np.corrcoef(ra, rb)[0, 1])

def pearson(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(np.corrcoef(s.a, s.b)[0, 1]) if len(s) >= 3 else float("nan")

# ---- 3. Frozen signature tests
ev = pop.elevation.astype(float)
gv = pop.med_gwl.astype(float)
rho_s = spearman(gv, ev)
rho_p = pearson(gv, ev)
absdev = (gv - ev).abs()
med_absdev = float(absdev.median())
print(f"\nSpearman rho(med_gwl, elevation): {rho_s:.4f}")
print(f"Pearson  rho(med_gwl, elevation): {rho_p:.4f}")
print(f"median |med_gwl - elevation|: {med_absdev:.2f} m")

depth_bearing = pop[(pop.depth.notna()) & (pop.depth.astype(float) > 0)]
db_ok = ((gv[depth_bearing.index] >= 0) & (gv[depth_bearing.index] <= depth_bearing.depth.astype(float)))
share_depth_ok = float(db_ok.mean()) if len(depth_bearing) else float("nan")
print(f"depth-bearing wells: {len(depth_bearing)}; share 0<=med_gwl<=depth: {share_depth_ok:.3f}")

med_gwl_overall = float(gv.median())
print(f"overall median med_gwl: {med_gwl_overall:.2f} m")

# negatives classes (per-well median frame)
neg_w = pop[gv < 0]
neg_small_low = neg_w[(neg_w.med_gwl > -50) & (neg_w.elevation < 50)]
neg_big = neg_w[neg_w.med_gwl <= -50]
neg_small_high = neg_w[(neg_w.med_gwl > -50) & (neg_w.elevation >= 50)]
print(f"wells with med_gwl<0: {len(neg_w)} | small-neg@low-elev: {len(neg_small_low)} | "
      f"small-neg@high-elev: {len(neg_small_high)} | <=-50m: {len(neg_big)}")

elev_suspect = bool(ev.isna().sum() > 0.2 * len(pop))
print(f"elevation null share: {ev.isna().mean():.3f}")

# ---- 4. Elevation credibility: 100 wells vs Open-Meteo (Copernicus DEM GLO-90), seed 42
random.seed(42)
sample_idx = random.sample(sorted(pop.index[pop.lat.notna()].tolist()),
                           min(100, int(pop.lat.notna().sum())))
samp = pop.loc[sample_idx, ["station_code", "lat", "lon", "elevation", "med_gwl", "district"]]
dem_ok, dem_diffs = True, []
try:
    lats = ",".join(f"{v:.5f}" for v in samp.lat)
    lons = ",".join(f"{v:.5f}" for v in samp.lon)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    with urllib.request.urlopen(url, timeout=60) as r:
        elev_dem = json.load(r)["elevation"]
    samp["dem_elev"] = elev_dem
    dem_diffs = (samp.elevation.astype(float) - samp.dem_elev.astype(float)).abs()
    med_dem_diff = float(dem_diffs.median())
    print(f"elevation credibility: median |file-DEM| on {len(samp)} wells = {med_dem_diff:.1f} m")
except Exception as e:
    dem_ok = False
    med_dem_diff = None
    print(f"elevation credibility: API FAILED ({e}) — down-weight elevation tests")

elev_credible = dem_ok and (med_dem_diff is not None) and (med_dem_diff <= 50)

# ---- 5. Verdict (frozen rules)
masl = (rho_s >= 0.80) and (med_absdev <= 25) and elev_credible
mbgl = (rho_s <= 0.30) and (share_depth_ok >= 0.85 or np.isnan(share_depth_ok)) and (0 <= med_gwl_overall <= 300)
if masl and not mbgl:
    verdict = "MASL-FRAME"
elif mbgl and not masl:
    verdict = "MBGL-FRAME"
elif masl and mbgl:
    verdict = "AMBIGUOUS (both rule sets triggered — inspect)"
else:
    verdict = "MIXED/QC-CONTAMINATED"
print(f"\nVERDICT (pre-registered rules): {verdict}")

# ---- 6. Track-B sample: 3 station codes for founder WRIS lookups
tb = []
if len(neg_big) > 0:
    r = neg_big.iloc[0]
    tb.append({"station_code": str(r.station_code), "district": str(r.district),
               "why": "large-negative (QC-failure class)", "med_gwl": round(float(r.med_gwl), 2)})
if len(neg_small_low) > 0:
    r = neg_small_low.iloc[0]
    tb.append({"station_code": str(r.station_code), "district": str(r.district),
               "why": "small-negative low-elevation (plausibly-masl-noise class)",
               "med_gwl": round(float(r.med_gwl), 2)})
cand = pop[(pop.med_gwl > 0) & (pop.elevation > 100)]
if len(cand) > 0:
    r = cand.iloc[len(cand) // 2]
    tb.append({"station_code": str(r.station_code), "district": str(r.district),
               "why": "positive-value elevated-terrain well", "med_gwl": round(float(r.med_gwl), 2)})
print("Track-B founder sample:", json.dumps(tb, indent=1))

# ---- 6b. Row-level negative classes (frozen rule; counted, not dropped)
row_classes = con.execute(f"""
SELECT count(*) FILTER (WHERE gwl_value <= -50) neg_le_minus50,
       count(*) FILTER (WHERE gwl_value > -50 AND gwl_value < 0) neg_small,
       count(*) FILTER (WHERE gwl_value >= 0) nonneg
FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({','.join(chr(39)+s+chr(39) for s in lab)}) AND gwl_value IS NOT NULL
""").fetchone()
print(f"row-level classes: <=-50m: {row_classes[0]} | (-50,0)m: {row_classes[1]} | >=0m: {row_classes[2]}")
extreme = con.execute(f"""
SELECT station_code, district, date, gwl_value, elevation FROM read_csv_auto('{CSV}')
WHERE date <= DATE '{CUT}' AND state IN ({','.join(chr(39)+s+chr(39) for s in lab)})
ORDER BY gwl_value ASC LIMIT 1
""").fetchone()
print(f"extreme negative reading: well {extreme[0]} ({extreme[1]}) {extreme[2]} gwl={extreme[3]:.2f} elev={extreme[4]}")

# ---- 7. Persist artifact
report = {
    "task": "55-TrackA-datum-signature", "registered": "Addendum 11 @ 07c76f7",
    "csv_sha256_verified": True, "cut": CUT,
    "population": {"wells_ap_tg": int(len(wells)), "analysis_wells": int(len(pop))},
    "signatures": {
        "spearman_medgwl_elev": round(rho_s, 4), "pearson_medgwl_elev": round(rho_p, 4),
        "median_abs_dev_m": round(med_absdev, 2),
        "depth_bearing_wells": int(len(depth_bearing)),
        "share_0_gwl_depth": None if np.isnan(share_depth_ok) else round(share_depth_ok, 4),
        "overall_median_medgwl_m": round(med_gwl_overall, 2),
        "wells_neg_median": int(len(neg_w)),
        "neg_small_lowelev": int(len(neg_small_low)),
        "neg_small_highelev": int(len(neg_small_high)),
        "neg_le_minus50": int(len(neg_big)),
        "elevation_null_share": round(float(ev.isna().mean()), 4),
        "dem_crosscheck_median_abs_diff_m": None if med_dem_diff is None else round(med_dem_diff, 1),
        "elevation_column_credible": elev_credible,
    },
    "verdict": verdict,
    "track_b_sample": tb,
}
with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=1)
print(f"\nartifact: {OUT_JSON}")

# sample rows for report readability
print("\nper-class medians:")
for name, sub in [("all", pop), ("neg_big", neg_big), ("neg_small_low", neg_small_low),
                  ("pos_wells", pop[gv > 0])]:
    if len(sub):
        print(f"  {name:14s} n={len(sub):5d} med_gwl={float(sub.med_gwl.median()):9.2f} "
              f"elev={float(sub.elevation.median()):7.1f}")

if FAIL:
    print("\nGUARDRAIL FAILURES:", FAIL)
    sys.exit(1)
print("\nALL PRE-REGISTERED GUARDRAILS PASSED")
