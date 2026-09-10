#!/usr/bin/env python3
"""
Rescue submission builder — hardens v4a into a maximally Zindi-safe CSV.

Why: user reports Zindi error "There was a problem while processing your
submission". On-disk v4a/b/c pass all structural checks (280,961 rows,
IDs byte-identical to SampleSubmission/Test order, 2 columns, no NaN).
Residual risk factors found in v4a: 33 values in scientific notation
(e.g. 1e-05) and up to 12 decimal places. While pandas-based scorers
tolerate these, a hardened fixed-point format eliminates the entire
class of float-parsing failures.

Hardening spec:
  - Header exactly "ID,Target"
  - Fixed-point values, 8 decimals, no scientific notation ever
  - Unix line endings (\n), no BOM, UTF-8 plain
  - ID column copied BYTE-FOR-BYTE from SampleSubmission (authoritative)
  - Value order from v4a predictions (never re-sorted)
"""
import csv

SRC = 'download/submission_v4a.csv'
SAMPLE = 'data/SampleSubmission (4).csv'
OUT = 'download/submission_RESCUE_v4a_hardened.csv'

# Load authoritative ID order
with open(SAMPLE, 'r', encoding='utf-8') as f:
    r = csv.reader(f)
    header = next(r)
    sample_rows = list(r)
sample_ids = [row[0] for row in sample_rows]

# Load v4a predictions
with open(SRC, 'r', encoding='utf-8') as f:
    r = csv.reader(f)
    src_header = next(r)
    src_rows = list(r)

assert src_header == ['ID', 'Target'], f"bad source header: {src_header}"
assert len(src_rows) == len(sample_rows), f"row count {len(src_rows)} != {len(sample_rows)}"

# Verify IDs align, then write hardened output
n_checked = 0
with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['ID', 'Target'])
    for src, sid in zip(src_rows, sample_ids):
        assert src[0] == sid, f"ID misalignment at row {n_checked}: {src[0]} vs {sid}"
        val = float(src[1])
        # Hardened fixed-point, 8 decimals, never scientific
        w.writerow([sid, f"{val:.8f}"])
        n_checked += 1

print(f"Written: {OUT} ({n_checked} rows)")

# ---- POST-WRITE ADVERSARIAL VERIFICATION ----
with open(OUT, 'rb') as f:
    raw = f.read()
assert raw[:3] != b'\xef\xbb\xbf', "BOM present!"
assert b'\r' not in raw, "CR bytes present!"

with open(OUT, 'r', encoding='utf-8') as f:
    r = csv.reader(f)
    h = next(r)
    out_rows = list(r)

assert h == ['ID', 'Target'], f"header: {h}"
assert len(out_rows) == 280961, f"rows: {len(out_rows)}"
assert [x[0] for x in out_rows] == sample_ids, "ID order mismatch!"

sci = sum(1 for x in out_rows if 'e' in x[1].lower())
assert sci == 0, f"scientific notation leaked: {sci}"

max_dp = max(len(x[1].split('.')[1]) for x in out_rows)
mins = min(float(x[1]) for x in out_rows)
maxs = max(float(x[1]) for x in out_rows)
# Correlation with source (round-trip integrity)
import math
diffs = [abs(float(a[1]) - float(b[1])) for a, b in zip(out_rows, src_rows)]
max_diff = max(diffs)

print(f"VERIFY: header={h}, rows={len(out_rows)}, sci-notation={sci}, "
      f"max_decimals={max_dp}, range=[{mins:.4f}, {maxs:.4f}]")
print(f"Round-trip max |diff| vs v4a: {max_diff:.2e} (bound: 1e-8 = half-ULP of 8-decimal rounding)")
assert max_diff <= 1.01e-8, "round-trip corruption beyond 8-decimal rounding!"
print("ALL HARDENING CHECKS PASS — file is structurally bulletproof")
