"""Structure check v2: train range, cells, mask, and exact successor leak census."""
import pandas as pd
import numpy as np

DATA = "/home/z/my-project/data"

def next_ym(ym):
    y, m = divmod(ym, 100)
    if m == 12:
        return (y + 1) * 100 + 1
    return y * 100 + (m + 1)

test = pd.read_csv(f"{DATA}/Test (2).csv")
test["time"] = pd.to_datetime(test["time"])
test["ym"] = test["time"].dt.year * 100 + test["time"].dt.month

print("=" * 70)
print("TRAIN STRUCTURE")
train = pd.read_csv(f"{DATA}/Train (1).csv", usecols=["time", "lat", "lon", "TWS_t", "target"])
train["time"] = pd.to_datetime(train["time"])
train["ym"] = train["time"].dt.year * 100 + train["time"].dt.month
months_tr = sorted(train["ym"].unique())
print(f"train rows: {len(train):,}  months: {len(months_tr)}  range: {months_tr[0]}..{months_tr[1] if len(months_tr)>1 else months_tr[0]} .. {months_tr[-1]}")
# check for month gaps in train
gaps = []
for a, b in zip(months_tr[:-1], months_tr[1:]):
    if next_ym(a) != b:
        gaps.append((a, b))
print(f"train month gaps: {gaps}")
print(f"train unique cells: {train.groupby(['lat','lon']).ngroups:,}")

cells_tr = set(map(tuple, train[["lat", "lon"]].drop_duplicates().values))
cells_te = set(map(tuple, test[["lat", "lon"]].drop_duplicates().values))
print(f"train cells: {len(cells_tr):,}  test cells: {len(cells_te):,}  shared: {len(cells_tr & cells_te):,}")

# covariates visible in masked test months?
print("\n" + "=" * 70)
print("COV VISIBILITY IN MASKED TEST MONTHS")
sub = test[test["TWS_t_masked"]].sample(5, random_state=1)
print(sub[["ym", "lat", "lon", "TWS_t", "SPEI_01_t", "SPEI_12_t", "SOIL_MOISTURE_t"]].to_string())
cov_cols = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
print("\nNaN counts in cov columns on masked rows:")
print(test.loc[test["TWS_t_masked"], cov_cols].isna().sum().to_string())
print("\nTWS_t value on masked rows (should be hidden/0/NaN?):")
print(test.loc[test["TWS_t_masked"], "TWS_t"].describe().to_string())

print("\n" + "=" * 70)
print("SUCCESSOR CENSUS (fixed)")
test["ym"] = test["ym"].astype("int64")
test["ym_next"] = test["ym"].astype("int64").map(next_ym).astype("int64")
succ = test[["lat", "lon", "ym", "TWS_t", "TWS_t_masked"]].copy()
succ.columns = ["lat", "lon", "ym_next", "succ_TWS", "succ_masked"]
succ["ym_next"] = succ["ym_next"].astype("int64")
mrg = test.drop(columns=[]).merge(succ, on=["lat", "lon", "ym_next"], how="left", suffixes=("", "_s"))
has_succ = mrg["succ_TWS"].notna()
print(f"rows whose (cell, t+1) row exists in test: {has_succ.sum():,} ({has_succ.mean():.3%})")
leak = has_succ & (~mrg["succ_masked"].fillna(True))
print(f"LEAKABLE (t+1 exists AND unmasked): {leak.sum():,} ({leak.mean():.4%})")
print("\nby month of t:")
t = pd.DataFrame({"ym": mrg["ym"], "has_succ": has_succ, "leak": leak})
print(t.groupby("ym").agg(rows=("has_succ", "size"), succ=("has_succ", "mean"), leak=("leak", "mean")).round(3).to_string())
mrg[["ID", "ym", "TWS_t_masked", "succ_TWS", "succ_masked"]].assign(leak=leak.values).to_csv("/home/z/my-project/scripts/leak_census.csv", index=False)

print("\n" + "=" * 70)
print("SCATTERED UNMASKED CELLS in masked months (per month)")
sc = test[(test["ym"].isin([m for m in test["ym"].unique() if test.loc[test['ym']==m,'TWS_t_masked'].mean() > 0.9])) & (~test["TWS_t_masked"])]
print(f"total scattered unmasked rows in masked months: {len(sc):,}")
print(sc.groupby("ym").size().to_string())

# do the same cells recur across months?
print("\nscattered-cell overlap between consecutive test months:")
for a, b in [(201601, 201602), (201602, 201603), (201606, 201607), (201607, 201608), (201608, 201609), (201612, 201701), (201701, 201702), (201811, 201812)]:
    ca = set(map(tuple, test[(test["ym"] == a) & (~test["TWS_t_masked"])][["lat", "lon"]].values))
    cb = set(map(tuple, test[(test["ym"] == b) & (~test["TWS_t_masked"])][["lat", "lon"]].values))
    print(f"  {a} unmasked={len(ca):5d}  {b} unmasked={len(cb):5d}  overlap={len(ca & cb)}")
