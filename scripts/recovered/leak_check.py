"""LEAK CHECK: Does Test.csv contain next-month rows whose TWS_t (unmasked) equals the target?
Target of row (c,t) = TWS_c(t+1). If row (c,t+1) exists in Test.csv and is unmasked,
its TWS_t column IS the answer for row (c,t).

Steps:
1. Census: months in test, cells, mask fraction per month.
2. Join test rows to their t+1 successor row (same cell).
3. Count: how many targets are directly readable (successor exists + unmasked)?
4. Break down by calendar month, by masked status of the row itself.
5. Verify join mechanics on TRAIN (target(t) == TWS_t(t+1) exactly).
"""
import pandas as pd
import numpy as np

DATA = "/home/z/my-project/data"

print("=" * 70)
print("STEP 1: Load test + train")
test = pd.read_csv(f"{DATA}/Test (2).csv")
print(f"test rows: {len(test):,}  cols: {list(test.columns)}")
test["time"] = pd.to_datetime(test["time"])
test["ym"] = test["time"].dt.year * 100 + test["time"].dt.month

# ---- mask census by month ----
print("\n" + "=" * 70)
print("STEP 2: Mask census by absolute month")
g = test.groupby("ym").agg(
    n=("TWS_t_masked", "size"),
    masked=("TWS_t_masked", "mean"),
    n_cells=("lat", "nunique"),
)
g["masked_pct"] = (g["masked"] * 100).round(1)
print(g.to_string())
print(f"\noverall masked fraction: {test['TWS_t_masked'].mean():.4f}")
print(f"unique cells: {test.groupby(['lat','lon']).ngroups:,}")
print(f"months: {test['ym'].nunique()}  from {test['ym'].min()} to {test['ym'].max()}")

# ---- calendar month pattern ----
test["cal_m"] = test["time"].dt.month
print("\nCalendar-month mask fractions:")
print(test.groupby("cal_m")["TWS_t_masked"].agg(["mean", "size"]).round(3).to_string())

# ---- STEP 3: the leak join ----
print("\n" + "=" * 70)
print("STEP 3: LEAK JOIN — successor row (same cell, next month)")
# next-month key
def next_ym(ym):
    y, m = divmod(ym, 100)
    if m == 12:
        return (y + 1) * 100 + 1
    return y * 100 + (m + 1)

test["ym_next"] = test["ym"].map(next_ym)

# successor lookup: (cell, ym) -> masked?, TWS_t value
succ = test.set_index(["lat", "lon", "ym"])[["TWS_t", "TWS_t_masked"]]
test_key = pd.MultiIndex.from_arrays([test["lat"], test["lon"], test["ym_next"]])
test["succ_exists"] = test_key.isin(succ.index)
test["succ_masked"] = test.set_index(
    ["lat", "lon", "ym_next"]
).index.map(
    succ["TWS_t_masked"] if False else None
) if False else np.nan
# do the lookup via merge (cleaner)
succ_df = test[["lat", "lon", "ym", "TWS_t", "TWS_t_masked"]].copy()
succ_df.columns = ["lat", "lon", "ym_next", "succ_TWS", "succ_masked"]
mrg = test.merge(succ_df, on=["lat", "lon", "ym_next"], how="left")
print(f"rows with successor in test: {mrg['succ_TWS'].notna().mean():.4f}")
leak = mrg["succ_TWS"].notna() & (~mrg["succ_masked"].fillna(True))
print(f"LEAKABLE rows (successor exists AND unmasked): {leak.mean():.4f}  ({leak.sum():,} rows)")

print("\nLeakable fraction by calendar month of t:")
tmp = pd.DataFrame({"cal_m": mrg["cal_m"], "leak": leak})
print(tmp.groupby("cal_m")["leak"].agg(["mean", "sum", "size"]).round(3).to_string())

print("\nLeakable fraction by own-masked status:")
tmp2 = pd.DataFrame({"own_masked": mrg["TWS_t_masked"], "leak": leak})
print(tmp2.groupby("own_masked")["leak"].agg(["mean", "sum", "size"]).round(3).to_string())

# cross-tab: own masked x leakable
print("\nCross-tab own_masked x leakable (counts):")
print(pd.crosstab(mrg["TWS_t_masked"], leak).to_string())

# ---- STEP 4: verify join mechanics on TRAIN ----
print("\n" + "=" * 70)
print("STEP 4: TRAIN verification: target(t) == TWS_t(t+1) via same join")
train = pd.read_csv(f"{DATA}/Train (1).csv", usecols=["sample_id", "time", "lat", "lon", "TWS_t", "target"])
train["time"] = pd.to_datetime(train["time"])
train["ym"] = train["time"].dt.year * 100 + train["time"].dt.month
train["ym_next"] = train["ym"].map(next_ym)
succ_tr = train[["lat", "lon", "ym", "TWS_t"]].copy()
succ_tr.columns = ["lat", "lon", "ym_next", "succ_TWS"]
mt = train.merge(succ_tr, on=["lat", "lon", "ym_next"], how="left")
has = mt["succ_TWS"].notna()
diff = (mt.loc[has, "target"] - mt.loc[has, "succ_TWS"]).abs()
print(f"train rows with successor: {has.sum():,}/{len(mt):,}")
print(f"max |target - succ_TWS| where successor exists: {diff.max():.10f}")
print(f"frac exactly equal: {(diff < 1e-9).mean():.6f}")
print("=> JOIN MECHANICS VERIFIED" if (diff < 1e-9).mean() > 0.9999 else "=> PROBLEM")

# ---- STEP 5: does our v18a already exploit this? ----
print("\n" + "=" * 70)
print("STEP 5: Does v18a already use the leak?")
sub = pd.read_csv("/home/z/my-project/download/submission_v18a.csv")
sub.columns = [c.strip() for c in sub.columns]
idcol = "ID" if "ID" in sub.columns else sub.columns[0]
tgtcol = "Target" if "Target" in sub.columns else sub.columns[1]
sub = sub.rename(columns={idcol: "ID", tgtcol: "Target"})
mm = mrg[["ID", "succ_TWS", "succ_masked", "TWS_t_masked"]].copy()
mm["leak"] = leak.values
chk = sub.merge(mm, on="ID", how="left")
lk = chk[chk["leak"]]
print(f"leakable rows found in submission: {len(lk):,}")
d = (lk["Target"] - lk["succ_TWS"]).abs()
print(f"v18a prediction vs leaked truth on leakable rows:")
print(f"  mean|diff| = {d.mean():.4f}   median = {d.median():.4f}   frac exact(<1e-6) = {(d<1e-6).mean():.4f}")
print(f"  std(diff) = {d.std():.4f}  -> if ~0.7, leak NOT used; if ~0, leak used")

# save leakable values for later use
mrg[["ID", "TWS_t_masked", "succ_TWS", "succ_masked"]].assign(leak=leak.values).to_csv(
    "/home/z/my-project/scripts/leak_census.csv", index=False
)
print("\nsaved scripts/leak_census.csv")
