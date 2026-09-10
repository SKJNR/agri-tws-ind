"""Diagnose submission_v15.csv vs SampleSubmission + a known-good submission (v14)."""
import pandas as pd
import numpy as np

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

ss = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
v15 = pd.read_csv(f'{DL}/submission_v15.csv')
v14 = pd.read_csv(f'{DL}/submission_v14.csv')

print("=== shape / columns ===")
print(f"sample : {ss.shape}  cols={list(ss.columns)}  dtypes={dict(ss.dtypes.astype(str))}")
print(f"v15    : {v15.shape}  cols={list(v15.columns)}  dtypes={dict(v15.dtypes.astype(str))}")
print(f"v14    : {v14.shape}  cols={list(v14.columns)}")

print("\n=== ID checks ===")
print(f"same length: {len(ss) == len(v15)}")
print(f"IDs identical (order included): {(ss['ID'].values == v15['ID'].values).all()}")
print(f"IDs as sets equal: {set(ss['ID']) == set(v15['ID'])}")
print(f"dup IDs in v15: {int(v15['ID'].duplicated().sum())}")

print("\n=== value checks ===")
print(f"NaN count: {int(v15['Target'].isna().sum())}")
print(f"inf count: {int(np.isinf(v15['Target'].values).sum())}")
print(f"range: [{v15['Target'].min():.4f}, {v15['Target'].max():.4f}]")
print(f"dtype: {v15['Target'].dtype}")

print("\n=== byte/format comparison vs v14 (known-good) ===")
with open(f'{DL}/submission_v15.csv', 'rb') as f: b15 = f.read()
with open(f'{DL}/submission_v14.csv', 'rb') as f: b14 = f.read()
print(f"v15 bytes: {len(b15):,} | v14 bytes: {len(b14):,}")
print(f"v15 header: {b15.split(b'\n')[0][:80]!r}")
print(f"v14 header: {b14.split(b'\n')[0][:80]!r}")
print(f"v15 first line: {b15.split(b'\n')[1][:60]!r}")
print(f"v14 first line: {b14.split(b'\n')[1][:60]!r}")
print(f"v15 last line : {b15.split(b'\n')[-2][:60]!r}  tail_newline={b15.endswith(chr(10).encode())}")
print(f"v14 last line : {b14.split(b'\n')[-2][:60]!r}  tail_newline={b14.endswith(chr(10).encode())}")
print(f"non-ascii bytes in v15: {sum(1 for c in b15 if c > 127)}")
print(f"n lines v15: {b15.count(chr(10).encode()):,} | v14: {b14.count(chr(10).encode()):,}")

# diff value distribution (in case a weird value snuck in)
d = v15['Target'].values.astype(np.float64) - v14['Target'].values.astype(np.float64)
print(f"\n=== v15 vs v14 value diff ===")
print(f"rows changed: {int((np.abs(d) > 1e-9).sum()):,} (expect 77,850 private h>=4 only)")
print(f"max|d| = {np.abs(d).max():.6f}")

# check for absurd outliers vs v14
ratio = np.abs(v15['Target'].values) / np.maximum(np.abs(v14['Target'].values), 1e-6)
print(f"max |v15|/|v14| where |v14|>0.01: {ratio[np.abs(v14['Target'].values) > 0.01].max():.2f}")
print("\nDIAGNOSIS COMPLETE")
