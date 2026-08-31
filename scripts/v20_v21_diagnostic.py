"""
Task 15: Interpret v21a/v21b LB results (Aug 31, 2026), decompose segment RMSEs,
and (when the user uploads the CSVs) run the v20-vs-v21 difference diagnostic.

LB scores from user's message (public LB, RMSE, lower = better):
  v21b 0.689902088 (Aug 31 ~03:14)   v21a 0.687374005 (Aug 30 ~14:00)
  v20c 0.631662555 (Aug 30)          v20a 0.633113636 (Aug 30)
  v18a 0.6937 (documented, reference)

Part 1 (runs now): exact segment fractions from Test.csv + constrained
  RMSE decomposition + model-free pairwise constraints.
Part 2 (auto-runs when CSVs are present in download/): where do v20 and v21
  differ — by segment (k=0 vs masked), by month, by cell; distance from
  persistence; disambiguates Scenario A (k=0 breakthrough) vs Scenario B
  (masked-row improvement).
"""

import math
import os

import numpy as np
import pandas as pd

DOWNLOAD = "/home/z/my-project/download"
TEST_PATH = "/home/z/my-project/data/Test (2).csv"

# --- LB scores (user message, Aug 31 2026) ---
LB = {
    "v20a": 0.633113636,
    "v20c": 0.631662555,
    "v21a": 0.687374005,
    "v21b": 0.689902088,
    "v18a": 0.693738722,  # recovered from manifest (clean-line team best before v21)
    "v17b": 0.704955918,  # denoiser-isolation reference
}

SUB_COLS = ["v20a", "v20c", "v21a", "v21b", "v18a", "v17b"]


def fmt(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "  -   "


# ============================================================
# PART 1 — exact fractions + constrained decomposition
# ============================================================
print("=" * 78)
print("PART 1: SEGMENT FRACTIONS + CONSTRAINED RMSE DECOMPOSITION")
print("=" * 78)

test = pd.read_csv(TEST_PATH)
test.columns = [c.strip() for c in test.columns]

# Identify the TWS_t column (unmasked values where available)
tw_col = "TWS_t" if "TWS_t" in test.columns else "TWS_t_masked"
n = len(test)
n_k0 = int(test[tw_col].notna().sum())
n_m = n - n_k0
f_k0 = n_k0 / n
f_m = n_m / n
print(f"\nTest rows: {n:,}   k=0 rows (TWS_t known): {n_k0:,} ({f_k0:.4%})"
      f"   masked rows: {n_m:,} ({f_m:.4%})")
print(f"(Worklog documented 186,913 masked / 94,048 unmasked — checking match)")

# --- Scenario A: masked-row RMSE held at m (grid) -> implied k=0 RMSE ---
print("\n--- SCENARIO A: masked rows fixed at m  =>  implied k=0-row RMSE ---")
print(f"{'assumed m':>10} | " + " | ".join(f"{v:>7}" for v in SUB_COLS + ["v18a"]))
for m_assumed in [0.73, 0.72, 0.71, 0.70]:
    row = []
    for v in SUB_COLS + ["v18a"]:
        val = (LB[v] ** 2 - f_m * m_assumed ** 2) / f_k0
        row.append(math.sqrt(val) if val > 0 else float("nan"))
    print(f"{m_assumed:>10} | " + " | ".join(fmt(r, 3).rjust(7) for r in row))
print("  (audit assumption m=0.72 = v18a-level masked model; the 0.6377 val")
print("   figure = measured competition-data k=0 ceiling, 2013-15 window)")

# --- Scenario B: k=0 rows fixed at k (grid) -> implied masked RMSE ---
print("\n--- SCENARIO B: k=0 rows fixed at k  =>  implied masked-row RMSE ---")
print(f"{'assumed k':>10} | " + " | ".join(f"{v:>7}" for v in SUB_COLS + ["v18a"]))
for k_assumed in [0.64, 0.62, 0.60]:
    row = []
    for v in SUB_COLS + ["v18a"]:
        val = (LB[v] ** 2 - f_k0 * k_assumed ** 2) / f_m
        row.append(math.sqrt(val) if val > 0 else float("nan"))
    print(f"{k_assumed:>10} | " + " | ".join(fmt(r, 3).rjust(7) for r in row))

# --- Model-free pairwise constraints (no assumption needed) ---
print("\n--- MODEL-FREE CONSTRAINTS: Δ(k²) = (LB1² − LB2²)/f_k0  (if masked model shared) ---")
pairs = [("v21a", "v20c"), ("v21b", "v21a"), ("v20a", "v20c"), ("v21a", "v18a"), ("v20c", "v18a")]
for a, b in pairs:
    dk2 = (LB[a] ** 2 - LB[b] ** 2) / f_k0
    print(f"  k²({a}) − k²({b}) = {dk2:+.4f}"
          f"   [e.g. k=0.62 vs 0.40 gives Δ=+0.225]")

# --- Sanity: which (m, k) pairs satisfy each score exactly ---
print("\n--- EXACT FIT CHECK (m=0.72): implied k=0 RMSE per submission ---")
for v in SUB_COLS + ["v18a"]:
    val = (LB[v] ** 2 - f_m * 0.72 ** 2) / f_k0
    k = math.sqrt(val) if val > 0 else float("nan")
    print(f"  {v}: LB {LB[v]:.6f}  ->  k=0 rows @ {fmt(k, 3)}   (masked @ 0.720)")

print("\nREADING:")
print("  * v21a/v21b sit at v18a level (0.694) — the documented reproducible stack")
print("    tops out ~0.687 public. The v20 edge (−0.056) is NOT reproduced by v21.")
print("  * Scenario A (audit's): v20 k=0 ≈ 0.40 vs v21 k=0 ≈ 0.62 (competition-data")
print("    ceiling)  =>  v20's edge lives in the k=0 rows, needs external information.")
print("  * Scenario B: v21 improved masked rows to ~0.71 with k=0 unchanged.")
print("    Totals alone cannot distinguish A vs B — Part 2 (CSV diff) can.")


# ============================================================
# PART 2 — v20 vs v21 difference diagnostic (runs when CSVs present)
# ============================================================
print("\n" + "=" * 78)
print("PART 2: CSV DIFFERENCE DIAGNOSTIC (v20/v21 submissions)")
print("=" * 78)

paths = {v: os.path.join(DOWNLOAD, f"submission_{v}.csv") for v in SUB_COLS}
present = {v: p for v, p in paths.items() if os.path.exists(p)}
missing = [v for v in SUB_COLS if v not in present]

if len(present) < 2:
    print(f"\nNot enough submission CSVs in {DOWNLOAD} yet.")
    print(f"  found   : {sorted(present)}")
    print(f"  missing : {missing}")
    print("\nACTION: upload submission_v20a.csv, submission_v20c.csv,")
    print("        submission_v21a.csv, submission_v21b.csv to download/,")
    print("        then re-run this script — Part 2 will execute automatically.")
else:
    print(f"\nFound: {sorted(present)}  (missing: {missing})")

    # Load all present submissions (robust ID/target column detection)
    subs = {}
    for v, p in present.items():
        d = pd.read_csv(p)
        d.columns = [c.strip() for c in d.columns]
        idc = next(c for c in d.columns if c.lower() in ("id", "sample_id"))
        tgc = next(c for c in d.columns if c.lower() in ("target", "prediction", "tws", "tws_t+1"))
        subs[v] = d[[idc, tgc]].rename(columns={idc: "ID", tgc: v})
        print(f"  loaded {v}: {len(subs[v]):,} rows")

    # Merge with test segments
    seg = test.copy()
    idc = next(c for c in seg.columns if c.lower() in ("sample_id", "id"))
    seg = seg.rename(columns={idc: "ID"})
    seg["k0"] = seg[tw_col].notna()
    seg["month"] = seg["time"].astype(str).str.slice(0, 7)
    merged = seg[["ID", "k0", "month", "lat", "lon", tw_col]].copy()
    for v, d in subs.items():
        merged = merged.merge(d, on="ID", how="inner")
    print(f"\nMerged rows: {len(merged):,}")

    have = list(present)
    ref_pair = [v for v in ("v20c", "v21a") if v in have][:2] or have[:2]

    # --- Per-segment pairwise differences ---
    print("\n--- Pairwise prediction differences by segment (mean |Δ|) ---")
    hdr = " | ".join(f"{a}−{b}" for i, a in enumerate(have) for b in have[i + 1:]) or "-"
    print(f"{'segment':>8} | {hdr}")
    for seg_name, mask in [("k=0", merged["k0"]), ("masked", ~merged["k0"])]:
        cells = []
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                cells.append(f"{(merged.loc[mask, a] - merged.loc[mask, b]).abs().mean():.4f}")
        print(f"{seg_name:>8} | " + " | ".join(cells))

    # --- Distance from persistence on k=0 rows ---
    if any(merged["k0"]):
        print("\n--- k=0 rows: distance of predictions from persistence (TWS_t) ---")
        print("  (RMSE of pred vs TWS_t; NOT model skill — shows how far each model")
        print("   moves away from persistence; val-window persistence skill = 0.59)")
        k0rows = merged[merged["k0"]]
        for v in have:
            rmse = math.sqrt(((k0rows[v] - k0rows[tw_col]) ** 2).mean())
            moved = (k0rows[v] - k0rows[tw_col]).abs()
            print(f"  {v}: RMSE vs persistence {rmse:.4f}   mean|Δ| {moved.mean():.4f}   "
                  f"share |Δ|>0.5: {(moved > 0.5).mean():.1%}")

    # --- Month-level disagreement profile (public-overfit signature check) ---
    print(f"\n--- Month-level disagreement: {ref_pair[0]} vs {ref_pair[1]} ---")
    a, b = ref_pair
    g = merged.groupby(["month", "k0"]).apply(
        lambda x: pd.Series({
            "n": len(x),
            f"mean|{a}−{b}|": (x[a] - x[b]).abs().mean(),
            f"mean({a}−{b})": (x[a] - x[b]).mean(),
        }), include_groups=False)
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    print("  (A legit external-cov advantage should be spread across ALL months;")
    print("   a public-row-specific artifact would concentrate the disagreement.)")

    # --- Cell-level disagreement vs known structures ---
    print(f"\n--- Where does {a} deviate most from {b}? (cell-level, top/bottom 5) ---")
    dev = merged.groupby(["lat", "lon"]).apply(
        lambda x: (x[a] - x[b]).abs().mean(), include_groups=False).sort_values()
    print("  LOWEST disagreement cells:"); print(dev.head(5).to_string())
    print("  HIGHEST disagreement cells:"); print(dev.tail(5).to_string())

    # --- Disambiguation verdict A vs B ---
    if "v20c" in have and "v21a" in have:
        d_k0 = (merged.loc[merged["k0"], "v20c"] - merged.loc[merged["k0"], "v21a"]).abs().mean()
        d_m = (merged.loc[~merged["k0"], "v20c"] - merged.loc[~merged["k0"], "v21a"]).abs().mean()
        print("\n--- SCENARIO A vs B DISAMBIGUATION ---")
        print(f"  mean|v20c−v21a| on k=0 rows  : {d_k0:.4f}")
        print(f"  mean|v20c−v21a| on masked rows: {d_m:.4f}")
        ratio = d_k0 / d_m if d_m > 0 else float("inf")
        print(f"  ratio k0/masked: {ratio:.2f}  (>2 => Scenario A: k=0 breakthrough;")
        print("   ~1 => Scenario B: models differ broadly, incl. masked rows)")

print("\nDone.")
print("\nNOTE (recovered manifest): v21a/v21b masked block == v18a bit-exact (2.3e-07);")
print("v21 gain over v18a (0.6937->0.6874) is ENTIRELY the k=0 a15-blend upgrade.")
print("v20a/b/c = prohibited external-GRACE lane (GDO/GravIS/COSTG/CSR) — never select.")
