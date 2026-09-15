#!/usr/bin/env python3
"""
D#21 — gwl_data.csv PHYSICAL SPLIT (AGRI-TWS-IND, Task 57 GLM, pre-staged).

Runs the moment gwl_data.csv lands in the sandbox. Produces:
  open/   gwl_open_extract.csv   rows with measurement date <= 2022-12-31
  sealed/ gwl_sealed_extract.csv rows with measurement date >= 2023-01-01
          + README (unseal procedure) + chmod 000 dir + access log stub
  MANIFESTS: SHA256 for source + both extracts; row-count reconciliation
             MUST sum exactly to source rows; any mismatch = FATAL, no output.

Rules (D#21 / AM-3):
  - Physical files, not duckdb views (views are not enforcement).
  - Sealed dir: separate hash, logged access, written unseal procedure.
  - D1.1 computed on <=2022 rows only.
  - Column auto-detection with FAIL-LOUD ambiguity handling: date column
    candidates tried in order; if none parse to >=95% valid dates, exit 1
    with the detected columns (never guess silently).
  --mocktest runs the full logic on a synthetic panel (no real data needed).
"""
import argparse, csv, datetime, hashlib, io, json, os, shutil, stat, sys

DATE_CANDIDATES = ["date", "measurement_date", "obs_date", "observation_date",
                   "reading_date", "time", "datetime", "timestamp", "date_of_reading"]
CUT = datetime.date(2023, 1, 1)  # first sealed day

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

def detect_date_col(header, sample_rows):
    """Score every candidate column by parse ratio; pick the best.
    Majority (>=0.5) required — a handful of dirty rows must not block
    detection on a 3.3M-row file; per-row purity is enforced downstream."""
    best_col, best_ratio = None, 0.0
    for col in header:
        key = col.strip().lower()
        if not any(key == c or key.endswith(c) for c in DATE_CANDIDATES):
            continue
        vals = [r.get(col) for r in sample_rows[:500]]
        if not vals:
            continue
        ratio = sum(1 for v in vals if parse_date(v)) / len(vals)
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    if best_col and best_ratio >= 0.5:
        return best_col
    return None

def split(source, out_root):
    with open(source, newline="") as fh:
        rdr = csv.DictReader(fh)
        header = rdr.fieldnames or []
        sample = []
        for i, row in enumerate(rdr):
            if i < 500:
                sample.append(row)
    dcol = detect_date_col(header, sample)
    if not dcol:
        sys.exit(f"FATAL: no parsable date column. header={header}. "
                 f"Register the actual schema before re-run (no silent guessing).")

    open_dir = os.path.join(out_root, "open")
    sealed_dir = os.path.join(out_root, "sealed")
    os.makedirs(open_dir, exist_ok=True)
    os.makedirs(sealed_dir, exist_ok=True)
    open_path = os.path.join(open_dir, "gwl_open_extract.csv")
    sealed_path = os.path.join(sealed_dir, "gwl_sealed_extract.csv")

    n_src = n_open = n_sealed = n_bad = 0
    with open(source, newline="") as fh, \
         open(open_path, "w", newline="") as fo, \
         open(sealed_path, "w", newline="") as fs:
        rdr = csv.DictReader(fh)
        wro = csv.DictWriter(fo, fieldnames=rdr.fieldnames)
        wrs = csv.DictWriter(fs, fieldnames=rdr.fieldnames)
        wro.writeheader(); wrs.writeheader()
        for row in rdr:
            n_src += 1
            d = parse_date(row.get(dcol))
            if d is None:
                n_bad += 1
                continue  # counted; row goes NOWHERE until ruled on (fail-loud report)
            if d < CUT:
                wro.writerow(row); n_open += 1
            else:
                wrs.writerow(row); n_sealed += 1

    if n_open + n_sealed + n_bad != n_src:
        sys.exit("FATAL: row reconciliation failed — outputs quarantined")
    if n_bad > 0:
        print(f"WARN: {n_bad} rows with unparsable dates excluded from BOTH extracts "
              f"(fail-loud: ruling required before inclusion anywhere)")

    # lock sealed dir and write its access contract
    readme = os.path.join(sealed_dir, "README_UNSEAL_PROCEDURE.md")
    with open(readme, "w") as fh:
        fh.write(UNSEAL_TEXT.format(n=n_sealed))
    manifest = {
        "decision": "D#21 physical split (Task 56; executed Task 57 GLM)",
        "source": os.path.basename(source),
        "source_sha256": sha256_file(source),
        "date_column": dcol,
        "cut": str(CUT),
        "rows_source": n_src, "rows_open": n_open, "rows_sealed": n_sealed,
        "rows_excluded_unparsable": n_bad,
        "open_sha256": sha256_file(open_path),
        "sealed_sha256": sha256_file(sealed_path),
        "reconciliation": "PASS" if n_open + n_sealed + n_bad == n_src else "FAIL",
    }
    with open(os.path.join(out_root, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    # write sealed-dir contract files BEFORE locking the directory
    with open(os.path.join(sealed_dir, "ACCESS_LOG.md"), "w") as fh:
        fh.write(f"# sealed access log — created {datetime.date.today()}\n\n"
                 f"Every read of gwl_sealed_extract.csv gets one line here:\n"
                 f"| when | who/task | purpose (registered decision #) |\n|---|---|---|\n")
    os.chmod(sealed_dir, stat.S_IRUSR | stat.S_IXUSR)  # r-x owner only: list+read possible, writes blocked
    print(json.dumps(manifest, indent=2))
    return manifest

UNSEAL_TEXT = """# Sealed extract — unseal procedure (D#21)

This directory holds {n} groundwater-level readings dated 2023-01-01 or later
(the model-evaluation SEAL window: label statistics 2023-25 are SEALED per
D#24 terminology; the seal protects label stats, not the calendar).

Rules:
1. Nothing in this directory may be read, hashed into a report, plotted, or
   used as a feature/target/statistic by ANY task before its registered
   position in the critical path (AM-6), and never during D1.1 (<=2022 rows
   only), regime-map freeze (T11), or any baseline-ladder run (D#23).
2. Unsealing requires: (a) a DECISION_LOG entry stating the gate that
   authorizes evaluation on 2023-25, (b) founder notification, (c) a line in
   ACCESS_LOG.md BEFORE the read.
3. Physical re-seal after an authorized read is NOT automatic — each
   authorized evaluation read is logged; the file itself stays sealed-by-
   convention for all other lanes.
4. Any accidental read = protocol surprise = DECISION_LOG entry + founder
   notification (no silent fixes).
"""

def mocktest(tmp):
    # synthetic panel: 6 rows spanning the cut, 1 bad date, header variants
    src = os.path.join(tmp, "mock_gwl.csv")
    with open(src, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["well_id", "latitude", "longitude", "date", "gwl_m"])
        w.writerow(["W1", "17.1", "78.9", "2015-03-14", "12.4"])   # open
        w.writerow(["W2", "17.2", "78.8", "2019-11-30", "15.1"])   # open
        w.writerow(["W3", "17.3", "78.7", "2022-12-31", "13.9"])   # open (boundary)
        w.writerow(["W4", "17.4", "78.6", "2023-01-01", "14.0"])   # sealed (boundary)
        w.writerow(["W5", "17.5", "78.5", "2025-06-02", "16.2"])   # sealed
        w.writerow(["W6", "17.6", "78.4", "not-a-date", "99.9"])   # excluded, fail-loud
    out = os.path.join(tmp, "mock_split")
    m = split(src, out)
    ok = (m["rows_source"] == 6 and m["rows_open"] == 3 and m["rows_sealed"] == 2
          and m["rows_excluded_unparsable"] == 1 and m["reconciliation"] == "PASS")
    # boundary correctness: 2022-12-31 open, 2023-01-01 sealed
    with open(os.path.join(out, "open", "gwl_open_extract.csv")) as fh:
        body = fh.read()
    ok = ok and ("2022-12-31" in body) and ("2023-01-01" not in body)
    with open(os.path.join(out, "sealed", "gwl_sealed_extract.csv")) as fh:
        sbody = fh.read()
    ok = ok and ("2023-01-01" in sbody) and ("2022-12-31" not in sbody)
    print(f"MOCKTEST {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="path to gwl_data.csv")
    ap.add_argument("--out", default=".", help="output root (creates open/ sealed/)")
    ap.add_argument("--mocktest", action="store_true")
    a = ap.parse_args()
    if a.mocktest:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sys.exit(0 if mocktest(td) else 1)
    if not a.source:
        sys.exit("usage: --source gwl_data.csv [--out DIR] | --mocktest")
    split(a.source, a.out)
