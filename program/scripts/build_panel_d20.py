#!/usr/bin/env python3
"""
D#20 + critical-path step — duckdb PANEL on open rows (AGRI-TWS-IND, Task 59 GLM).

Registered position: crosswalk + physical split FIRST -> duckdb panel -> ...
Physical split executed (Task 58). This script builds the panel:
  parquet/panel_open_readings.parquet        reading-level, AP+TG open rows,
                                            district_2026 assigned + flags
  parquet/panel_open_district_month.parquet  district_2026 x year-month aggregates
  manifests/panel_build_report.json          full report (mapping table actually
                                            applied, D#20 sensitivity check,
                                            month census, value quality)

PROVISIONAL ASSIGNMENT (logged amendment to D#20 execution):
  Polygons (LGD/Bhuvan-grade) are NOT in sandbox; point-in-polygon cannot run.
  Empirical label structure (verified this run): district labels are CONSTANT
  per well (0 multi-label wells in 3,223), back-applied across all readings;
  label vintage ~= TG 2019-era / AP pre-2022-era spellings. Assignment is
  therefore an explicit per-label map (all 47 observed labels enumerated;
  unknown label = UNASSIGNED_NAME, fail-loud, never guessed).
  Parent-label wells (old undivided districts) are assigned to the CONTINUING
  district and flagged PARENT_CONTINUE with the child set — the split share is
  unresolvable without polygons (queued: d20_assign_wells.py T-D20-2 run when
  founder exports polygons).

D#24 note: aggregates use OPEN rows (<=2022-12-31) only; no sealed data is
read anywhere in this script (input = parquet/gwl_open.parquet by construction).

No model runs (AM-6). Deterministic: same input -> same outputs (sorted keys).
"""
import argparse, datetime, hashlib, json, os, sys

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(HERE)
DATA = os.path.join(PROG, "data")
SRC_PARQUET = os.path.join(DATA, "parquet", "gwl_open.parquet")
OUT_READINGS = os.path.join(DATA, "parquet", "panel_open_readings.parquet")
OUT_DM = os.path.join(DATA, "parquet", "panel_open_district_month.parquet")
OUT_REPORT = os.path.join(DATA, "manifests", "panel_build_report.json")
BASIS_CSV = os.path.join(PROG, "d20", "district_basis.csv")
LEDGER_JSON = os.path.join(PROG, "d20", "change_ledger.json")

AP_TG_STATES = ("Andhra Pradesh", "Telangana")
# generous AP+TG geographic bounds (report-only flag; assignment is label-based)
BBOX = (12.0, 20.5, 76.0, 85.5)  # lat_min, lat_max, lon_min, lon_max

def norm(s):
    """Normalize a district name: upper, strip, collapse spaces, drop .()'-. chars."""
    s = (s or "").strip().upper()
    out = []
    prev_space = False
    for ch in s:
        if ch in " .()-'":
            if ch == " " and not prev_space and out:
                out.append(" ")
                prev_space = True
            # non-space punctuation: drop entirely
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).replace(" ", "")

# ---------------------------------------------------------------------------
# EXPLICIT LABEL MAP — every label observed in AP+TG open rows (47 total).
# status: DIRECT (label claims a current-basis district, unsplit since label
#         vintage) | PARENT_CONTINUE (old undivided district; assigned to the
#         continuing name; children listed = ambiguity set)
# target: 2026-basis district name (must match district_basis.csv after norm)
# ---------------------------------------------------------------------------
LABEL_MAP = {
    # --- TG: 2016-created children (unsplit since) ---
    "KAMAREDDY":        ("DIRECT", "Kamareddy", []),
    "SIDDIPET":         ("DIRECT", "Siddipet", []),
    "SANGAREDDY":       ("DIRECT", "Sangareddy", []),       # label 'SANGA REDDY'
    "VIKARABAD":        ("DIRECT", "Vikarabad", []),
    "MEDCHALMALKAJGIRI":("DIRECT", "Medchal-Malkajgiri", []),  # label 'MEDCHAL'
    "WANAPARTHY":       ("DIRECT", "Wanaparthy", []),
    "NAGARKURNOOL":     ("DIRECT", "Nagarkurnool", []),
    "JOGULAMBAGADWAL":  ("DIRECT", "Jogulamba Gadwal", []),  # label 'JOGULAMBA(GADWAL)'
    "SURYAPET":         ("DIRECT", "Suryapet", []),
    "JANGAON":          ("DIRECT", "Jangaon", []),
    "MAHABUBABAD":      ("DIRECT", "Mahabubabad", []),
    "JAYASHANKARBHUPALPALLY": ("DIRECT", "Jayashankar Bhupalpally", []),  # 'BHUPALPALLY'
    "KUMURAMBHEEMASIFABAD": ("DIRECT", "Kumuram Bheem Asifabad", []),     # 'KUMURAM BHEEM'
    "MANCHERIAL":       ("DIRECT", "Mancherial", []),
    "NIRMAL":           ("DIRECT", "Nirmal", []),
    "JAGTIAL":          ("DIRECT", "Jagtial", []),           # label 'JAGITYAL'
    "PEDDAPALLI":       ("DIRECT", "Peddapalli", []),
    "RAJANNASIRCILLA":  ("DIRECT", "Rajanna Sircilla", []),  # label 'SIRCILLA'
    "BHADRADRIKOTHAGUDEM": ("DIRECT", "Bhadradri Kothagudem", []),  # 'BHADRADRI'
    # --- TG: 2019-created children ---
    "MULUGU":           ("DIRECT", "Mulugu", []),
    "NARAYANPET":       ("DIRECT", "Narayanpet", []),
    # --- TG: renames ---
    "WARANGALURBAN":    ("DIRECT", "Hanumakonda", []),       # rename ~2018-21
    # --- TG: unsplit ---
    "HYDERABAD":        ("DIRECT", "Hyderabad", []),         # note: Medchal-Malkajgiri Hyderabad-share unverified (ledger)
    # --- TG: pre-2016 parents (continuing + 2016/2019 children) ---
    "ADILABAD":   ("PARENT_CONTINUE", "Adilabad",
                   ["Kumuram Bheem Asifabad", "Mancherial", "Nirmal"]),
    "NIZAMABAD":  ("PARENT_CONTINUE", "Nizamabad", ["Kamareddy"]),
    "KARIMNAGAR": ("PARENT_CONTINUE", "Karimnagar",
                   ["Jagtial", "Peddapalli", "Rajanna Sircilla"]),
    "MEDAK":      ("PARENT_CONTINUE", "Medak", ["Sangareddy", "Siddipet"]),
    "RANGAREDDY": ("PARENT_CONTINUE", "Rangareddy",
                   ["Vikarabad", "Medchal-Malkajgiri"]),
    "MAHBUBNAGAR":("PARENT_CONTINUE", "Mahabubnagar",
                   ["Wanaparthy", "Nagarkurnool", "Jogulamba Gadwal", "Narayanpet"]),
    "NALGONDA":   ("PARENT_CONTINUE", "Nalgonda",
                   ["Suryapet", "Yadadri Bhuvanagiri"]),
    "KHAMMAM":    ("PARENT_CONTINUE", "Khammam", ["Bhadradri Kothagudem"]),
    "WARANGALRURAL": ("PARENT_CONTINUE", "Warangal", ["Mulugu"]),  # 2016 name, 2019 carve
    # --- AP: pre-2022 13-district labels (old spellings; 2022-04-04 children) ---
    "SRIKAKULAM":     ("PARENT_CONTINUE", "Srikakulam", ["Parvathipuram Manyam"]),
    "VIZIANAGARAM":   ("PARENT_CONTINUE", "Vizianagaram", ["Parvathipuram Manyam"]),
    "VISAKHAPATNAM":  ("PARENT_CONTINUE", "Visakhapatnam",
                       ["Alluri Sitharama Raju", "Anakapalli"]),
    "EASTGODAVARI":   ("PARENT_CONTINUE", "East Godavari",
                       ["Kakinada", "Dr. B.R. Ambedkar Konaseema"]),
    "WESTGODAVARI":   ("PARENT_CONTINUE", "West Godavari", ["Eluru"]),
    "KRISHNA":        ("PARENT_CONTINUE", "Krishna", ["NTR"]),
    "GUNTUR":         ("PARENT_CONTINUE", "Guntur", ["Bapatla", "Palnadu"]),
    "PRAKASAM":       ("DIRECT", "Prakasam", []),            # unsplit in 2022
    "NELLORE":        ("DIRECT", "Nellore", []),             # unsplit in 2022
    "KURNOOL":        ("PARENT_CONTINUE", "Kurnool", ["Nandyal"]),
    "ANANTHAPURAMU":  ("PARENT_CONTINUE", "Ananthapuramu", ["Sri Sathya Sai"]),  # 'ANANTAPUR'
    "CHITTOOR":       ("PARENT_CONTINUE", "Chittoor", ["Tirupati"]),
    "YSRKADAPA":      ("PARENT_CONTINUE", "YSR Kadapa", ["Annamayya"]),          # 'CUDDAPAH'
}

# raw label -> normalized key (spellings that differ beyond punctuation)
ALIASES = {
    "SANGA REDDY": "SANGAREDDY",
    "MEDCHAL": "MEDCHALMALKAJGIRI",
    "JOGULAMBA(GADWAL)": "JOGULAMBAGADWAL",
    "BHUPALPALLY": "JAYASHANKARBHUPALPALLY",
    "KUMURAM BHEEM": "KUMURAMBHEEMASIFABAD",
    "JAGITYAL": "JAGTIAL",
    "PEDDAPALLY": "PEDDAPALLI",   # data uses Y-ending variant
    "SIRCILLA": "RAJANNASIRCILLA",
    "BHADRADRI": "BHADRADRIKOTHAGUDEM",
    "ANANTAPUR": "ANANTHAPURAMU",
    "CUDDAPAH": "YSRKADAPA",
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_basis():
    import csv
    basis = []
    with open(BASIS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            basis.append(row["district_name_lgd"].strip())
    return basis

def validate_map(basis):
    """Every LABEL_MAP target and child must exist in the 59-name basis."""
    bnorm = {norm(b): b for b in basis}
    errors = []
    for k, (status, target, children) in LABEL_MAP.items():
        if norm(target) not in bnorm:
            errors.append(f"target {target!r} (label key {k}) not in basis")
        for c in children:
            if norm(c) not in bnorm:
                errors.append(f"child {c!r} (under {target!r}) not in basis")
    if errors:
        sys.exit("FATAL: LABEL_MAP inconsistent with district_basis.csv:\n  "
                 + "\n  ".join(errors))
    return bnorm

def build(con):
    basis = load_basis()
    bnorm = validate_map(basis)

    # ---- per-well label census (constancy is a design assumption -> verified) ----
    wells = con.execute(f"""
        SELECT station_code, district, state,
               COUNT(*) n_rows, MIN(date) dmin, MAX(date) dmax,
               ANY_VALUE(latitude) lat, ANY_VALUE(longitude) lon,
               SUM(CASE WHEN NOT (latitude BETWEEN {BBOX[0]} AND {BBOX[1]}
                              AND longitude BETWEEN {BBOX[2]} AND {BBOX[3]})
                   THEN 1 ELSE 0 END) n_outside_bbox
        FROM read_parquet('{SRC_PARQUET}')
        WHERE state IN ('Andhra Pradesh','Telangana')
        GROUP BY 1,2,3
    """).fetchall()
    multi_label = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT station_code FROM read_parquet('{SRC_PARQUET}')
            WHERE state IN ('Andhra Pradesh','Telangana')
            GROUP BY 1 HAVING COUNT(DISTINCT district) > 1)
    """).fetchone()[0]

    assign = {}     # station_code -> dict
    unmapped = {}   # raw label -> count of wells
    for (wid, label, state, n_rows, dmin, dmax, lat, lon, n_out) in wells:
        n = norm(label)
        # alias on raw label first, then on the normalized form (case variants)
        key = ALIASES.get(label.strip()) or ALIASES.get(n, n)
        ent = LABEL_MAP.get(key)
        if ent is None:
            unmapped[label] = unmapped.get(label, 0) + 1
            assign[wid] = {"district_2026": None, "status": "UNASSIGNED_NAME",
                           "label": label, "children": [], "n_rows": n_rows}
            continue
        status, target, children = ent
        assign[wid] = {"district_2026": target, "status": status, "label": label,
                       "children": children, "n_rows": n_rows,
                       "state": state, "lat": lat, "lon": lon,
                       "outside_bbox_rows": int(n_out or 0),
                       "dmin": str(dmin), "dmax": str(dmax)}

    # ---- reading-level panel with assignment (temp map table + JOIN; no
    #      string-interpolated station codes) ----
    con.execute("""CREATE OR REPLACE TABLE well_map (
        station_code VARCHAR, district_2026 VARCHAR,
        assign_status VARCHAR, split_children VARCHAR)""")
    con.executemany(
        "INSERT INTO well_map VALUES (?, ?, ?, ?)",
        [(wid, a["district_2026"], a["status"], "|".join(a["children"]))
         for wid, a in assign.items()])
    con.execute(f"""
        COPY (
            SELECT r.station_code, r.date, r.gwl_value, r.latitude, r.longitude,
                   r.state,
                   r.district AS district_label_raw,
                   r.village, r.precipitation, r.temperature_2m,
                   r.total_evaporation_sum, r.runoff_sum, r.dominant_class,
                   r.soil_moisture_am, r.soil_moisture_pm, r.sr_b5, r.sr_b4,
                   r.elevation, r.stream_order, r.depth, r.well_type,
                   r.well_aquifer_type, r.litho_lithologic, r.aquifer_0_aquifer,
                   r.litho_supergroup,
                   m.district_2026, m.assign_status, m.split_children
            FROM read_parquet('{SRC_PARQUET}') r
            LEFT JOIN well_map m USING (station_code)
            WHERE r.state IN ('Andhra Pradesh','Telangana')
        ) TO '{OUT_READINGS}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # ---- district x year-month aggregates (2026 basis; open rows only) ----
    con.execute(f"""
        COPY (
            SELECT district_2026, year(date) AS yr, month(date) AS mon,
                   COUNT(*) AS n_readings,
                   COUNT(DISTINCT station_code) AS n_wells,
                   AVG(gwl_value) AS gwl_mean, STDDEV_SAMP(gwl_value) AS gwl_sd,
                   MIN(gwl_value) AS gwl_min, MAX(gwl_value) AS gwl_max,
                   SUM(CASE WHEN gwl_value < 0 THEN 1 ELSE 0 END) AS n_negative,
                   SUM(CASE WHEN gwl_value > 300 THEN 1 ELSE 0 END) AS n_gt300
            FROM read_parquet('{OUT_READINGS}')
            WHERE district_2026 IS NOT NULL
            GROUP BY 1,2,3 ORDER BY 1,2,3
        ) TO '{OUT_DM}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # ---- D#20 sensitivity check (pre-registered; label-based variant) ----
    # (a) label-frame (as-labeled, pooled onto target) vs basis-frame means.
    #     Pooling MUST go through well_map so case/spelling variants of the same
    #     target combine (dict-overwrite per raw label was a bug, caught by the
    #     first run's nonzero diff).
    lab = con.execute(f"""
        SELECT m.district_2026, year(r.date), month(r.date), AVG(r.gwl_value)
        FROM read_parquet('{OUT_READINGS}') r JOIN well_map m USING (station_code)
        WHERE m.district_2026 IS NOT NULL
        GROUP BY 1,2,3 ORDER BY 1,2,3
    """).fetchall()
    bas = con.execute(f"""
        SELECT district_2026, yr, mon, gwl_mean
        FROM read_parquet('{OUT_DM}') ORDER BY 1,2,3
    """).fetchall()
    lab_map = {(d, y, m): v for d, y, m, v in lab}
    bas_map = {(d, y, m): v for d, y, m, v in bas}
    diffs = [abs(lab_map[k] - bas_map[k]) for k in
             set(lab_map) & set(bas_map) if lab_map[k] is not None and bas_map[k] is not None]
    max_diff = max(diffs) if diffs else None
    if max_diff is not None and max_diff < 1e-9:
        max_diff = 0.0  # float-plan noise only
    only_lab = len(set(lab_map) - set(bas_map))
    only_bas = len(set(bas_map) - set(lab_map))
    sens = {
        "metric": "max |district-month mean (label-frame) - (2026-basis frame)|",
        "max_abs_diff": max_diff,
        "n_overlapping_keys": len(diffs),
        "keys_only_in_label_frame": only_lab,
        "keys_only_in_basis_frame": only_bas,
        "interpretation": (
            "Zero by construction for continuing-name keys: the provisional "
            "assignment maps each label 1:1 onto its continuing district. The "
            "genuine uncertainty is NOT measurable without polygons and is "
            "reported as the PARENT_CONTINUE ambiguity share below (wells whose "
            "true 2026 district may be one of the listed children). Polygon "
            "binding (T-D20-2) re-runs this check with true point-in-polygon "
            "assignment; until then this section documents structure, not drift."
        ),
    }

    # (b) ambiguity share + coverage
    status_wells = {}
    for a in assign.values():
        status_wells[a["status"]] = status_wells.get(a["status"], 0) + 1
    parent_wells = [a for a in assign.values() if a["status"] == "PARENT_CONTINUE"]
    parent_rows = sum(a["n_rows"] for a in parent_wells)
    total_rows = sum(a["n_rows"] for a in assign.values())
    covered = sorted({a["district_2026"] for a in assign.values() if a["district_2026"]})
    empty_basis = [b for b in basis if b not in covered]

    # (c) month-of-year census per district (D1.1 feed; open rows only)
    month_census = con.execute(f"""
        SELECT district_2026, mon, SUM(n_readings) AS n
        FROM read_parquet('{OUT_DM}') GROUP BY 1,2 ORDER BY 1,2
    """).fetchall()
    mc = {}
    for d, m, n in month_census:
        mc.setdefault(d, {})[int(m)] = int(n)

    # (d) value quality (open rows, AP+TG)
    vq = con.execute(f"""
        SELECT COUNT(*),
               SUM(CASE WHEN gwl_value < 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN gwl_value > 300 THEN 1 ELSE 0 END),
               MIN(gwl_value), MAX(gwl_value), AVG(gwl_value)
        FROM read_parquet('{OUT_READINGS}')
    """).fetchone()
    hist = con.execute(f"""
        SELECT CASE WHEN gwl_value < -100 THEN '<-100'
                    WHEN gwl_value < 0 THEN '-100..0'
                    WHEN gwl_value < 10 THEN '0..10'
                    WHEN gwl_value < 30 THEN '10..30'
                    WHEN gwl_value < 100 THEN '30..100'
                    WHEN gwl_value <= 300 THEN '100..300'
                    ELSE '>300' END AS bucket, COUNT(*)
        FROM read_parquet('{OUT_READINGS}') GROUP BY 1 ORDER BY 1
    """).fetchall()

    out_bbox_wells = sum(1 for a in assign.values() if a.get("outside_bbox_rows", 0) > 0)

    report = {
        "program": "AGRI-TWS-IND",
        "task": "59-PANEL",
        "decision_refs": "D#20 (basis+ledger), D#21 (open split, input), D#24 (open-only stats)",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inputs": {
            "source_parquet": "parquet/gwl_open.parquet (992,053 open rows)",
            "filter": f"state IN {AP_TG_STATES}",
            "open_rows_ap_tg": total_rows,
            "wells_total": len(assign),
            "wells_multi_label": multi_label,
            "label_constancy_verdict": "VERIFIED CONSTANT" if multi_label == 0 else "VIOLATION - inspect",
        },
        "assignment": {
            "mechanism": "PROVISIONAL explicit label map (polygons not in sandbox; "
                         "T-D20-2 point-in-polygon queued on founder LGD export)",
            "status_well_counts": status_wells,
            "parent_continue_wells": len(parent_wells),
            "parent_continue_rows": parent_rows,
            "parent_continue_share_rows": round(parent_rows / total_rows, 4),
            "unmapped_labels": unmapped,
            "districts_with_wells": len(covered),
            "basis_districts_empty_under_provisional": empty_basis,
            "note_empty": ("children created by reorgs receive wells only after polygon "
                           "binding; parents hold their wells under the continuing name "
                           "with split ambiguity flagged"),
        },
        "outputs": {
            "readings_parquet": os.path.relpath(OUT_READINGS, DATA),
            "readings_sha256": sha256_file(OUT_READINGS),
            "district_month_parquet": os.path.relpath(OUT_DM, DATA),
            "district_month_sha256": sha256_file(OUT_DM),
            "district_month_rows": len(bas),
        },
        "d20_sensitivity_check": sens,
        "month_of_year_census": mc,
        "value_quality_open_rows": {
            "n": vq[0], "n_negative": vq[1], "n_gt300": vq[2],
            "min": vq[3], "max": vq[4], "mean": vq[5],
            "histogram": dict(hist),
            "flags": [
                "negatives ~31% of AP+TG open rows: UNIT/DATUM QUESTION (mbgl vs masl vs "
                "QC failures) - unresolved, goes to Task-55 provenance spot-check "
                "(values/dates/units vs India-WRIS raw) BEFORE D1.1 conclusions rest on it",
                "no NaN-as-float in AP+TG slice (NaN present only in other states)",
                "report-only: nothing dropped, nothing imputed (AM-3)",
            ],
        },
        "coordinate_flags": {
            "wells_with_any_outside_bbox_reading": out_bbox_wells,
            "bbox_used": list(BBOX),
            "treatment": "flag-only; assignment is label-based; polygon binding will "
                         "resolve genuine location questions",
        },
        "covariate_note": ("precipitation/temperature_2m/evaporation/runoff/soil_moisture/"
                           "sr_b5/sr_b4 columns are SoulVision pre-joined covariates, "
                           "carried as REFERENCE ONLY; our covariate foundation stays the "
                           "hash-frozen Open-Meteo pipeline (corrected plan item 4)"),
    }
    return report

def selftest():
    """Plumbing check on synthetic labels (no real data touched)."""
    assert norm("SANGA REDDY") == "SANGAREDDY"
    assert norm("JOGULAMBA(GADWAL)") == "JOGULAMBAGADWAL"
    assert norm("Dr. B.R. Ambedkar Konaseema") == "DRBRAMBEDKARKONASEEMA"
    assert ALIASES["JAGITYAL"] == "JAGTIAL"
    basis = load_basis()
    validate_map(basis)
    # every selftest label resolves or fails loud
    ok = True
    for raw in ["KAMAREDDY", "SANGA REDDY", "JOGULAMBA(GADWAL)", "JAGITYAL",
                "ANANTAPUR", "CUDDAPAH", "WARANGAL URBAN", "MAHBUBNAGAR"]:
        key = ALIASES.get(raw.strip(), norm(raw))
        ok = ok and key in LABEL_MAP
    ok = ok and "TIRUPATI" not in LABEL_MAP  # not an observed label
    # a hypothetical unknown label must be unmapped (fail-loud path)
    ok = ok and norm("ATLANTIS") not in LABEL_MAP
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} (map: {len(LABEL_MAP)} entries, "
          f"basis: {len(basis)} districts)")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    for p in (SRC_PARQUET, BASIS_CSV, LEDGER_JSON):
        if not os.path.isfile(p):
            sys.exit(f"FATAL: missing input {p}")
    con = duckdb.connect()
    report = build(con)
    with open(OUT_REPORT, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({
        "open_rows_ap_tg": report["inputs"]["open_rows_ap_tg"],
        "wells": report["inputs"]["wells_total"],
        "status_wells": report["assignment"]["status_well_counts"],
        "districts_with_wells": report["assignment"]["districts_with_wells"],
        "empty_basis_districts": len(report["assignment"]["basis_districts_empty_under_provisional"]),
        "unmapped_labels": report["assignment"]["unmapped_labels"],
        "sens_max_abs_diff": report["d20_sensitivity_check"]["max_abs_diff"],
        "value_quality": {k: report["value_quality_open_rows"][k]
                          for k in ("n", "n_negative", "min", "max", "mean")},
    }, indent=1, default=str))
    print(f"REPORT -> {OUT_REPORT}")

if __name__ == "__main__":
    main()
