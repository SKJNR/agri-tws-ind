"""
AUDIT 15-c part 5: FLOAT32 AUDIT (item 7)
submission_v8.py casts TWS_t/target/covs to float32 on load. Train TWS has 6-9 decimals.
Quantify truncation error and its RMSE-equivalent contribution.
"""
import numpy as np, pandas as pd, csv

DATA = '/home/z/my-project/data'

def audit_col(path, col, nmax=None):
    vals64, vals32, ndec = [], [], []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            s = row[col]
            if s == '' or s.lower() == 'na':
                vals64.append(np.nan); vals32.append(np.nan); continue
            v = float(s)
            vals64.append(v); vals32.append(np.float32(v))
            if '.' in s:
                ndec.append(len(s.split('.')[1]))
            if nmax and i+1 >= nmax: break
    v64 = np.array(vals64); v32 = np.array(vals32, dtype=np.float64)
    err = np.abs(v64 - v32)
    ok = np.isfinite(err)
    dec = np.array(ndec) if ndec else np.array([0])
    print(f"{path.split('/')[-1]} :: {col}")
    print(f"  n={len(v64):,}  decimals: min={dec.min()} max={dec.max()} mode={np.bincount(dec).argmax()}")
    print(f"  max|err|={np.nanmax(err):.3e}  mean|err|={np.nanmean(err):.3e}  "
          f"rmse={np.sqrt(np.nanmean(err**2)):.3e}  (vs LB scale 0.7: {np.sqrt(np.nanmean(err**2))/0.7*100:.5f}%)")
    return err

print("=== float32 truncation, Train TWS_t (full file) ===")
audit_col(f'{DATA}/Train (1).csv', 'TWS_t')
print("\n=== Train target (full) ===")
audit_col(f'{DATA}/Train (1).csv', 'target')
print("\n=== Train SOIL_MOISTURE_t (sample 300k) ===")
audit_col(f'{DATA}/Train (1).csv', 'SOIL_MOISTURE_t', nmax=300000)
print("\n=== Train SPEI_01_t (sample 300k) ===")
audit_col(f'{DATA}/Train (1).csv', 'SPEI_01_t', nmax=300000)

# cumulative effect if truncation hit the TARGET side of an RMSE: contribution to a 0.707 RMSE
e = audit_col(f'{DATA}/Train (1).csv', 'TWS_t')
contrib = np.sqrt(np.nanmean(e**2))
print(f"\nRMSE-equivalent of float32 truncation: {contrib:.2e}")
print(f"If BOTH prediction and truth were float32-rounded: worst-case combined ~{contrib*np.sqrt(2):.2e}")
print(f"LB granularity (last meaningful digit at 0.707): 1e-6 -> truncation is ~{1e-6/contrib:.0f}x SMALLER")
print("DONE part 5")
