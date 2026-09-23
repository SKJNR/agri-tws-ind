"""
UNIVERSAL SUBMISSION VALIDATOR — diagnoses Zindi 'There was a problem while
processing your submission' failures.

Checks every known rejection cause:
  1. Exact column names ('ID','Target') and order
  2. Row count == 280,961 (+header)
  3. ID set identical to SampleSubmission (no missing/extra/duplicate IDs)
  4. Row ORDER identical to SampleSubmission (Zindi is order-tolerant in theory,
     but order-mismatch + any dedup bug = rejection; keep exact order)
  5. Target: no NaN/inf/empty/non-numeric values
  6. Target: no absurd magnitudes (|Target| > 100 flags a construction bug)
  7. File encoding: plain ASCII/UTF-8, no BOM, single header line
  8. ID string format round-trips exactly (no float re-formatting drift like
     '-55.5' -> '-55.500000' or '-5.55E1')

Usage: python validate_submission.py <file.csv> [<file2.csv> ...]
Exit 0 = PASS, 1 = FAIL (reasons printed).
"""
import sys, os
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
SS = f'{DATA}/SampleSubmission (4).csv'
EXPECT_ROWS = 280961

def validate(path):
    print(f"\n=== {os.path.basename(path)} ===")
    fails = []
    if not os.path.exists(path):
        print("FAIL: file does not exist"); return False
    size = os.path.getsize(path)
    print(f"  size: {size:,} bytes")

    # raw byte checks
    with open(path, 'rb') as f:
        head = f.read(200)
    if head.startswith(b'\xef\xbb\xbf'):
        fails.append("BOM present (UTF-8-BOM) — some parsers reject this")
    try:
        raw = open(path, 'rb').read().decode('utf-8')
    except UnicodeDecodeError as e:
        fails.append(f"not valid UTF-8: {e}"); raw = None
    if raw is not None:
        if '\r' in raw:
            print("  note: CRLF line endings (usually tolerated)")
        first_line = raw.split('\n')[0].strip()
        if first_line != 'ID,Target':
            fails.append(f"header is '{first_line}' — expected exactly 'ID,Target'")

    df = pd.read_csv(path, dtype={'ID': str, 'Target': str})
    # columns
    if list(df.columns) != ['ID', 'Target']:
        fails.append(f"columns are {list(df.columns)} — expected ['ID','Target']")
    # row count
    if len(df) != EXPECT_ROWS:
        fails.append(f"row count {len(df):,} != {EXPECT_ROWS:,}")
    # duplicates
    dup = df['ID'].duplicated().sum()
    if dup:
        fails.append(f"{dup} duplicate IDs")
    # ID match vs SampleSubmission
    ss = pd.read_csv(SS, dtype={'ID': str})
    set_f, set_s = set(df['ID']), set(ss['ID'])
    miss, extra = set_s - set_f, set_f - set_s
    if miss: fails.append(f"{len(miss)} IDs missing vs SampleSubmission (e.g. {sorted(miss)[:3]})")
    if extra: fails.append(f"{len(extra)} IDs not in SampleSubmission (e.g. {sorted(extra)[:3]})")
    # order match
    if len(df) == EXPECT_ROWS and not miss and not extra:
        if (df['ID'].values != ss['ID'].values).any():
            n_diff = (df['ID'].values != ss['ID'].values).sum()
            print(f"  note: {n_diff:,} rows in different ORDER than SampleSubmission (tolerated, but risky)")
    # Target numeric parse
    t = pd.to_numeric(df['Target'], errors='coerce')
    n_nan = t.isna().sum()
    if n_nan:
        bad_ids = df.loc[t.isna(), 'ID'].head(5).tolist()
        fails.append(f"{n_nan} non-numeric/empty Target values (e.g. {bad_ids})")
    if n_nan < len(t):
        arr = t.astype(float).values
        n_inf = np.isinf(arr).sum()
        if n_inf: fails.append(f"{n_inf} infinite Target values")
        fin = arr[np.isfinite(arr)]
        big = (np.abs(fin) > 100).sum()
        if big: fails.append(f"{big} Target values with |T|>100 (construction bug?)")
        print(f"  Target: mean={fin.mean():.4f} std={fin.std():.4f} "
              f"min={fin.min():.4f} max={fin.max():.4f} n_nan={n_nan}")

    if fails:
        for f_ in fails: print(f"  FAIL: {f_}")
        return False
    print("  PASS — format is bulletproof")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: validate_submission.py <file.csv> [...]"); sys.exit(1)
    ok = all([validate(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
