#!/usr/bin/env python3
"""
gwl_data.csv HANDBACK CHECK — founder-local pre-upload integrity + receipt
Program: AGRI-TWS-IND | Task: 57-KIT | Built: 2026-09-15 (GLM)

Run this ON YOUR MACHINE, next to your gwl_data.csv copy, BEFORE uploading:

    python3 gwl_handback_check.py gwl_data.csv
    python3 gwl_handback_check.py gwl_data.csv --gzip     # if upload is size-capped

It produces gwl_data.csv.receipt.json (a few KB, safe to paste in chat).
The receipt proves (a) which exact file you have, (b) that it survived
transfer — I re-run the same census sandbox-side and diff the SHA256.

Design rules it enforces (mirrors pre-staged d21_physical_split.py):
  - FAIL-LOUD: no parsable date column -> exit 2, columns printed, no guess.
  - Date candidates + majority-ratio detection IDENTICAL to d21 so both
    censuses agree by construction.
  - D#24 discipline: the seal protects 2023-25 LABEL STATISTICS, so value
    stats (min/max/mean) are computed on OPEN rows (<=2022-12-31) only.
    Row counts per period + calendar range are reconciliation metadata,
    exactly like d21's split_manifest.
  - Pure stdlib. Streaming. ~3.3M rows OK on a laptop.

Self-test (no real data):  python3 gwl_handback_check.py --selftest
"""
import argparse, csv, datetime, gzip, hashlib, io, json, os, sys, tempfile

TOOL = "gwl_handback_check.py v1.0 (Task 57-KIT, GLM, 2026-09-15)"
CUT = datetime.date(2023, 1, 1)          # first sealed day (D#21)
DATE_CANDIDATES = ["date", "measurement_date", "obs_date", "observation_date",
                   "reading_date", "time", "datetime", "timestamp", "date_of_reading"]
WELL_CANDIDATES = ["well_id", "wellid", "well_code", "wellcode", "well", "site_id",
                   "station_id", "station_code", "wlg_id", "well_no", "id"]
DIST_CANDIDATES = ["district", "district_name", "districtname", "dist_name",
                   "district_code", "lgd_district", "district_lgd"]
VAL_CANDIDATES = ["gwl_m", "gwl", "gwl_m_bgl", "water_level", "water_level_m",
                  "depth_to_water_level", "dtwl", "wl_m", "level_m", "gwl_(m)",
                  "gwl_m_bgl_", "water_level_(m)"]
LAT_CANDIDATES = ["latitude", "lat", "y"]
LON_CANDIDATES = ["longitude", "lon", "long", "x"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%b-%d-%Y", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(s[:len(fmt) + 2], fmt).date()
        except ValueError:
            continue
    return None

def _match(header, candidates, used):
    for j, col in enumerate(header):
        k = col.strip().lower()
        if k in used:
            continue
        if k in candidates:
            return j          # returns INDEX; None if no match
    return None

def open_csv(path, delimiter=None):
    """Encoding-tolerant opener. Returns (filehandle, csv.reader, delimiter)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            fh = open(path, newline="", encoding=enc)
            first = fh.readline()
            fh.seek(0)
            if delimiter is None:
                delimiter = ","
                for d in (";", "\t", "|"):
                    if first.count(d) > first.count(delimiter):
                        delimiter = d
            return fh, csv.reader(fh, delimiter=delimiter), enc, delimiter
        except UnicodeDecodeError:
            continue
    sys.exit("FATAL: cannot decode file as utf-8 or latin-1")

def census(path):
    if not os.path.isfile(path):
        sys.exit(f"FATAL: {path} not found")
    size = os.path.getsize(path)
    fh, rdr, enc, delim = open_csv(path)
    with fh:
        try:
            header = next(rdr)
        except StopIteration:
            sys.exit("FATAL: file has no header row")
        header = [h.strip() for h in header]
        ncol = len(header)

        sample = []
        for i, row in enumerate(rdr):
            if i >= 500:
                break
            sample.append(row)

        # date column: same majority logic as d21 (never silent-guess)
        best_col, best_ratio = None, 0.0
        for j, col in enumerate(header):
            k = col.strip().lower()
            if not any(k == c or k.endswith(c) for c in DATE_CANDIDATES):
                continue
            vals = [r[j] if j < len(r) else "" for r in sample]
            if not vals:
                continue
            ratio = sum(1 for v in vals if parse_date(v)) / len(vals)
            if ratio > best_ratio:
                best_col, best_ratio = j, ratio
        if best_col is None or best_ratio < 0.5:
            sys.exit(f"FATAL: no parsable date column (best ratio {best_ratio:.2f}). "
                     f"header={header}. Register the schema first (AM-5: no silent fixes).")

        used = {header[best_col]}
        wcol = _match(header, WELL_CANDIDATES, used)
        if wcol is not None: used.add(header[wcol])
        dcol = _match(header, DIST_CANDIDATES, used)
        if dcol is not None: used.add(header[dcol])
        vcol = _match(header, VAL_CANDIDATES, used)
        if vcol is not None: used.add(header[vcol])
        latcol = _match(header, LAT_CANDIDATES, used)
        if latcol is not None: used.add(header[latcol])
        loncol = _match(header, LON_CANDIDATES, used)

        # streaming pass
        n = n_open = n_sealed = n_bad = 0
        nulls = [0] * ncol
        wells, districts = set(), {}
        seen_pairs, dupes = set(), 0
        dmin = dmax = None
        v_cnt = v_bad = 0; v_sum = 0.0; v_min = v_max = None   # OPEN rows only (D#24)
        la_min = la_max = lo_min = lo_max = None
        check_dupes = True

        fh.seek(0); rdr = csv.reader(fh, delimiter=delim); next(rdr)
        for row in rdr:
            n += 1
            if n % 1000000 == 0:
                print(f"  ... {n:,} rows", file=sys.stderr)
            if n > 5_000_000 and check_dupes:
                check_dupes = False   # memory guard; noted in receipt
            for j in range(ncol):
                if j >= len(row) or not (row[j] or "").strip():
                    nulls[j] += 1
            g = lambda j: (row[j].strip() if j is not None and j < len(row) else "")
            if wcol is not None:
                wv = g(wcol)
                if wv: wells.add(wv)
            if dcol is not None:
                dv = g(dcol)
                if dv: districts[dv] = districts.get(dv, 0) + 1
            if latcol is not None:
                try:
                    x = float(g(latcol))
                    la_min = x if la_min is None else min(la_min, x)
                    la_max = x if la_max is None else max(la_max, x)
                except ValueError:
                    pass
            if loncol is not None:
                try:
                    x = float(g(loncol))
                    lo_min = x if lo_min is None else min(lo_min, x)
                    lo_max = x if lo_max is None else max(lo_max, x)
                except ValueError:
                    pass
            d = parse_date(g(best_col))
            if d is None:
                n_bad += 1
            else:
                dmin = d if dmin is None or d < dmin else dmin
                dmax = d if dmax is None or d > dmax else dmax
                if d < CUT:
                    n_open += 1
                    if vcol is not None:                    # open-period label stats only
                        try:
                            x = float(g(vcol))
                            v_cnt += 1; v_sum += x
                            v_min = x if v_min is None else min(v_min, x)
                            v_max = x if v_max is None else max(v_max, x)
                        except ValueError:
                            v_bad += 1
                else:
                    n_sealed += 1
                if check_dupes and wcol is not None:
                    key = (g(wcol), d.isoformat())
                    if key in seen_pairs:
                        dupes += 1
                    else:
                        seen_pairs.add(key)

    if n_open + n_sealed + n_bad != n:
        sys.exit("FATAL: internal reconciliation failed — file may have changed during read")

    receipt = {
        "program": "AGRI-TWS-IND", "tool": TOOL,
        "purpose": "gwl_data.csv handback receipt (founder-side census)",
        "file": {
            "name": os.path.basename(path), "bytes": size,
            "sha256": sha256_file(path), "encoding": enc, "delimiter": delim,
        },
        "generated_at": datetime.datetime.now().astimezone().isoformat(),
        "schema": {
            "columns": header, "n_columns": ncol,
            "detected": {
                "date_column": header[best_col], "date_parse_ratio_sample": round(best_ratio, 3),
                "well_column": header[wcol] if wcol is not None else None,
                "district_column": header[dcol] if dcol is not None else None,
                "value_column": header[vcol] if vcol is not None else None,
                "lat_column": header[latcol] if latcol is not None else None,
                "lon_column": header[loncol] if loncol is not None else None,
            },
            "empty_or_null_per_column": {header[j]: nulls[j] for j in range(ncol)},
        },
        "census": {
            "rows_total": n,
            "split_preview_d21": {          # counts only — mirrors split_manifest fields
                "rows_open_le_2022": n_open,
                "rows_sealed_ge_2023": n_sealed,
                "rows_unparsable_date": n_bad,
                "cut": str(CUT), "reconciliation": "PASS",
            },
            "date_min": str(dmin) if dmin else None,
            "date_max": str(dmax) if dmax else None,
            "unique_wells": len(wells) if wcol is not None else None,
            "unique_districts": len(districts) if dcol is not None else None,
            "district_row_counts": dict(sorted(districts.items(), key=lambda kv: -kv[1]))
                                   if dcol is not None else None,
            "duplicate_well_date_rows": dupes if check_dupes else "SKIPPED_GT_5M_ROWS",
            "coord_bbox": {"lat": [la_min, la_max], "lon": [lo_min, lo_max]}
                          if (la_min is not None and lo_min is not None) else None,
            "value_stats_OPEN_ROWS_ONLY": {   # D#24: 2023-25 label stats are sealed
                "column": header[vcol] if vcol is not None else None,
                "n_numeric": v_cnt, "n_nonnumeric": v_bad,
                "min": v_min, "max": v_max,
                "mean": round(v_sum / v_cnt, 4) if v_cnt else None,
                "note": "computed on rows dated <= 2022-12-31 only; sealed-period label values are NOT summarized here",
            },
        },
        "how_to_send": "Attach gwl_data.csv (or the .gz from --gzip) in chat, plus this receipt JSON. "
                       "GLM re-runs the census sandbox-side and must match sha256 + row counts exactly.",
        "warnings": [],
    }
    if n_bad:
        receipt["warnings"].append(f"{n_bad} rows have unparsable dates — d21 will quarantine them "
                                   "from BOTH extracts (fail-loud) pending a ruling")
    if dcol is not None and len(districts) not in (0, 59):
        receipt["warnings"].append(f"{len(districts)} unique district labels found — D#20 basis is 59 "
                                   "(AP26+TG33). Not fatal (naming variants possible) but will be reconciled.")
    if not check_dupes:
        receipt["warnings"].append("duplicate check skipped: >5M rows (memory guard)")
    return receipt, n, n_open, n_sealed, n_bad, dupes, len(wells), len(districts), \
           (v_cnt, v_min, v_max, v_sum), (dmin, dmax)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="path to gwl_data.csv")
    ap.add_argument("--gzip", action="store_true",
                    help="also write gwl_data.csv.gz next to it (for size-capped upload)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 2)
    if not a.path:
        ap.print_help(); sys.exit(2)

    receipt = run(a.path)
    if a.gzip:
        gz = a.path + ".gz"
        with open(a.path, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
            for chunk in iter(lambda: fi.read(1 << 20), b""):
                fo.write(chunk)
        receipt["file"]["gzip"] = {"name": os.path.basename(gz),
                                   "bytes": os.path.getsize(gz),
                                   "sha256": sha256_file(gz)}
        print(f"\ngzip written: {gz} ({os.path.getsize(gz):,} bytes)")

    out = os.path.splitext(a.path)[0] + ".receipt.json"
    with open(out, "w") as fh:
        json.dump(receipt, fh, indent=2)
    c = receipt["census"]
    print(json.dumps({
        "sha256": receipt["file"]["sha256"], "bytes": receipt["file"]["bytes"],
        "rows": c["rows_total"],
        "open/sealed/bad": [c["split_preview_d21"]["rows_open_le_2022"],
                            c["split_preview_d21"]["rows_sealed_ge_2023"],
                            c["split_preview_d21"]["rows_unparsable_date"]],
        "wells": c["unique_wells"], "districts": c["unique_districts"],
        "date_range": [c["date_min"], c["date_max"]],
        "warnings": receipt["warnings"],
    }, indent=2))
    print(f"\nRECEIPT -> {out}\nSend that JSON together with the file upload.")

def run(path):
    r = census(path)
    return r[0]

def selftest():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "mock.csv")
        rows = [
            ["well_id", "latitude", "longitude", "measurement_date", "district", "gwl_m"],
            ["W1", "17.1", "78.9", "2015-03-14", "Karimnagar", "12.4"],
            ["W2", "17.2", "78.8", "2019-11-30", "Karimnagar", "15.1"],
            ["W1", "17.1", "78.9", "2016-03-14", "Karimnagar", "12.9"],
            ["W3", "17.3", "78.7", "2022-12-31", "Nizamabad", "13.9"],
            ["W4", "17.4", "78.6", "2023-01-01", "Nizamabad", "14.0"],
            ["W5", "17.5", "78.5", "2025-06-02", "Nizamabad", "16.2"],
            ["W6", "17.6", "78.4", "not-a-date", "Adilabad", "99.9"],
            ["W7", "", "", "2016-06-01", "Adilabad", ""],
            ["W2", "17.2", "78.8", "2019-11-30", "Karimnagar", "15.1"],
            ["W8", "17.7", "78.3", "2024-02-29", "Adilabad", "10.0"],
        ]
        with open(p, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        rec, n, n_open, n_sealed, n_bad, dupes, nw, nd, vs, dr = census(p)
        # note: the duplicate row (row 9, open period) is still a physical row —
        # it counts in open split preview AND open value stats, exactly like
        # d21 splits rows (not unique observations).
        ok = (n == 10 and n_open == 6 and n_sealed == 3 and n_bad == 1
              and dupes == 1 and nw == 8 and nd == 3)
        ok = ok and dr == (datetime.date(2015, 3, 14), datetime.date(2025, 6, 2))
        ok = ok and vs[0] == 5 and vs[1] == 12.4 and vs[2] == 15.1 and abs(vs[3] - 69.4) < 1e-6
              # open rows only: 12.4+15.1+12.9+13.9+15.1(dup) — excludes 14.0/16.2/10.0/99.9
        ok = ok and rec["schema"]["detected"]["date_column"] == "measurement_date"
        ok = ok and rec["schema"]["detected"]["value_column"] == "gwl_m"
        ok = ok and rec["census"]["district_row_counts"]["Karimnagar"] == 4
        print(f"SELFTEST {'PASS' if ok else 'FAIL'} "
              f"(rows={n} open={n_open} sealed={n_sealed} bad={n_bad} dup={dupes} "
              f"wells={nw} dist={nd} valstats={vs})")
        return ok

if __name__ == "__main__":
    main()
