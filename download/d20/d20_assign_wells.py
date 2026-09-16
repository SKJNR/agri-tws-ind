#!/usr/bin/env python3
"""
D#20 — well -> district assignment pipeline (AGRI-TWS-IND, Task 57 GLM).

Assigns each groundwater well (point) to exactly one 2026-vintage district
polygon (AP 26 + TG 33) by point-in-polygon. Acceptance (courier Task 57 B.1):
every well assigned OR explicitly unassigned-with-reason; hashes logged.

Design rules (D#20 / AM-3 / blindfold-by-construction):
  - Input: wells CSV with lat/lon columns; polygons = LGD/Bhuvan-grade
    GeoJSON/Shapefile for the 59 districts (GADM banned; OSM only with a
    logged amendment).
  - Deterministic tie-break: if a point falls in >1 polygon (overlap/slash),
    assign to the polygon with the smallest area, log the conflict.
  - Boundary cases: point-on-edge -> smallest-area rule + log.
  - No defaults, no nearest-district imputation. Unassigned requires a
    reason code: OUTSIDE_LAYER | DEGENERATE_COORD | POLYGON_GAP | NO_POLYGONS.

Run modes:
  --wells <csv> --polygons <geojson> [--out <csv>]
  --selftest   (mock polygons + synthetic wells; validates plumbing incl.
               unassigned-with-reason path — runs WITHOUT gwl_data.csv)

Stdlib-only geometry (ray casting); swap to exactextract/geopandas at scale
if perf demands (registered as exactextract, Apache-2.0, SCOUT Q3-AMEND).
"""
import argparse, csv, hashlib, json, math, os, sys

def ray_cast(pt, ring):
    x, y = pt
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside

def polygon_area(ring):
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(s) / 2.0

def load_polygons(path):
    polys = []
    with open(path) as fh:
        gj = json.load(fh)
    for f in gj.get("features", []):
        geom = f.get("geometry") or {}
        props = f.get("properties") or {}
        name = props.get("district") or props.get("DISTRICT") or props.get("district_name") \
               or props.get("dtname") or props.get("NAME_2") or ""
        state = props.get("state") or props.get("ST_NM") or props.get("stname") or ""
        if geom.get("type") == "Polygon":
            parts = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            parts = geom["coordinates"]
        else:
            continue
        for poly in parts:
            polys.append({"district": name, "state": state,
                          "rings": poly, "area": polygon_area(poly[0])})
    return polys

def assign(pt, polys):
    """Return ((district, state), None) or (None, reason)."""
    lat, lon = pt
    if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180:
        return None, "DEGENERATE_COORD"
    if not polys:
        return None, "NO_POLYGONS"
    hits = [p for p in polys if ray_cast((lon, lat), p["rings"][0])]
    if not hits:
        return None, "OUTSIDE_LAYER"
    if len(hits) == 1:
        return (hits[0]["district"], hits[0]["state"]), None
    hits.sort(key=lambda p: p["area"])  # deterministic: smallest area wins
    return (hits[0]["district"], hits[0]["state"]), f"MULTI_HIT x{len(hits)} (smallest-area rule)"

def detect_cols(header):
    h = [c.strip().lower() for c in header]
    latc = header[h.index("lat")] if "lat" in h else (header[h.index("latitude")] if "latitude" in h else None)
    lonc = header[h.index("lon")] if "lon" in h else (header[h.index("longitude")] if "longitude" in h else None)
    return latc, lonc

def run(wells_csv, polygons_geojson, out_csv):
    polys = load_polygons(polygons_geojson)
    assigned = unassigned = 0
    reasons = {}
    with open(wells_csv, newline="") as fh:
        rdr = csv.DictReader(fh)
        latc, lonc = detect_cols(rdr.fieldnames or [])
        if not (latc and lonc):
            sys.exit(f"FATAL: lat/lon columns not found in {rdr.fieldnames}")
        with open(out_csv, "w", newline="") as out:
            wtr = csv.writer(out)
            wtr.writerow(["well_id", "lat", "lon", "state", "district", "status", "reason"])
            for row in rdr:
                try:
                    lat, lon = float(row[latc]), float(row[lonc])
                except (TypeError, ValueError):
                    lat = lon = float("nan")
                wid = row.get("well_id") or row.get("id") or row.get("station_id") or row.get("site_id") or ""
                hit, reason = assign((lat, lon), polys)
                if hit:
                    assigned += 1
                    wtr.writerow([wid, lat, lon, hit[1], hit[0], "ASSIGNED", reason or ""])
                else:
                    unassigned += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    wtr.writerow([wid, lat, lon, "", "", "UNASSIGNED", reason])
    h = hashlib.sha256(open(out_csv, "rb").read()).hexdigest()
    print(f"assigned={assigned} unassigned={unassigned} reasons={reasons}")
    print(f"out sha256={h}")
    return {"assigned": assigned, "unassigned": unassigned, "reasons": reasons,
            "out": out_csv, "sha256": h, "n_polygons": len(polys)}

def selftest():
    polys = []
    boxes = [
        ("Srikakulam", "AP", 18.3, 84.0), ("Parvathipuram Manyam", "AP", 18.8, 83.5),
        ("Nizamabad", "TS", 18.67, 78.1), ("Kamareddy", "TS", 18.3, 78.3),
    ]
    for name, state, lat, lon in boxes:
        d = 0.25
        ring = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]
        polys.append({"district": name, "state": state, "rings": [ring], "area": polygon_area(ring)})
    # overlap pair: smaller Anakapalli box inside larger Visakhapatnam box
    ring_v = [[79.9, 17.6], [81.1, 17.6], [81.1, 18.1], [79.9, 18.1], [79.9, 17.6]]
    ring_a = [[80.3, 17.7], [80.7, 17.7], [80.7, 17.95], [80.3, 17.95], [80.3, 17.7]]
    polys.append({"district": "Visakhapatnam", "state": "AP", "rings": [ring_v], "area": polygon_area(ring_v)})
    polys.append({"district": "Anakapalli", "state": "AP", "rings": [ring_a], "area": polygon_area(ring_a)})

    tests = [
        ((18.3, 84.0), ("Srikakulam", "AP"), None),                      # centroid self-hit
        ((18.58, 83.0), None, "OUTSIDE_LAYER"),                          # gap between boxes
        ((99.0, 200.0), None, "DEGENERATE_COORD"),                       # bad coords
        ((17.85, 80.5), ("Anakapalli", "AP"), "MULTI_HIT"),              # overlap -> smallest area
    ]
    ok = True
    for pt, expect_hit, expect_reason in tests:
        hit, reason = assign(pt, polys)
        got_hit = hit is not None
        if expect_hit is None:
            passed = (not got_hit) and reason == expect_reason
        else:
            passed = got_hit and hit == expect_hit and (expect_reason is None or (reason or "").startswith(expect_reason))
        print(f"  {pt} -> {hit or reason} {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    hit, reason = assign((17.0, 80.0), [])
    nopass = hit is None and reason == "NO_POLYGONS"
    print(f"  empty-polygon-layer -> {reason} {'PASS' if nopass else 'FAIL'}")
    print(f"SELFTEST {'PASS' if ok and nopass else 'FAIL'}")
    return ok and nopass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells"); ap.add_argument("--polygons"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not (a.wells and a.polygons):
        sys.exit(0 if selftest() else 1)
    res = run(a.wells, a.polygons, a.out or "well_district_assignment.csv")
    json.dump(res, open(os.path.splitext(a.out or "well_district_assignment.csv")[0] + ".summary.json", "w"), indent=2)
