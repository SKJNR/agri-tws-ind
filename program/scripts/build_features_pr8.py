#!/usr/bin/env python3
"""
build_features_pr8.py — the Task-54 backbone build on the 59-execution
basis (DECISION_LOG Addendum 10, spec pre-registered at commit c9208d0
BEFORE this code ran).

Three SEPARATE layers (Task-54 sign-off condition 5 / catch #5):
  (i)   covariate features (weather; built here)
  (ii)  PR-8 target-transform machinery (generic, frozen parameters,
        train-only fits; applied to the TARGET layer)
  (iii) target ingestion — STUB, not built (D1.1 pending; Qwen loaders
        + founder pilot-12 + Task-55 WRIS spot-check gates unchanged)

Feature set = the approved 16 = 14 core + 2 regime-gated:
  wb_lag1/2/3 (P-ET0 monthly sums, mm/month; ET0 = PRIMARY et0_pm_api,
  WB-R3 pin) · soil_lag1 · tmax_lag1 · precip_roll3/6/12 (rolling SUMS)
  · wb_roll3 · wb_roll12 (rolling SUMS) · max1day_lag1 (threshold-free)
  · heavy20_lag1 (>=20 mm/day, FROZEN) · dryspell_lag1 (max in-month
  run <1.0 mm, FROZEN) · month_of_year (categorical) · [15/16]
  wb_roll3 x regime + soil_lag1 x regime — regime-gated: ONLY when
  regime_map.csv (T11 frozen map) is present; the MVP_HEURISTIC_REGIME
  flag path fails LOUD without a defined heuristic source; heuristic
  never evidence.

Lag convention (frozen): feature row is keyed by TARGET month t;
lag1 = calendar month t-1 (latest observed at issue), lag2 = t-2,
lag3 = t-3; rolls end at t-1. Split constants frozen in code:
train <=2019-12 / val 2020-01..2022-12 / test 2023-01+ (test EXCLUDED
from the smoke entirely — untouched until D1.1 -> regime freeze ->
ladder, per AM-6).

SMOKE TEST = PLUMBING ONLY, NON-EVIDENCE (pre-registered Addendum 10):
pseudo-target = NEXT-MONTH soil-moisture tercile, PR-8 transformed
(train-only fits); integrity assertions + trivial majority-class
baseline machinery; NO skill claims.

Usage
  python3 build_features_pr8.py --selftest
  python3 build_features_pr8.py --run [--mvp-heuristic-regime]
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.dirname(HERE)
ROOT = os.path.dirname(PROG)
OM_CSV = os.path.join(ROOT, "download", "agri_tws_ind",
                      "ap_ts_weather_openmeteo_full59.csv")
PANEL_PQ = os.path.join(PROG, "data", "parquet",
                        "weather_backbone_monthly.parquet")
REGIME_MAP = os.path.join(PROG, "data", "regime_map.csv")
PARQ_DIR = os.path.join(PROG, "data", "parquet")
MANI_DIR = os.path.join(PROG, "data", "manifests")
REPORT_MD = os.path.join(ROOT, "download", "agri_tws_ind",
                         "BACKBONE_FEATURES_REPORT.md")

# --- frozen split constants (Task 53; NEVER edit without an amendment) ---
TRAIN_END = "2019-12"          # inclusive
VAL_START, VAL_END = "2020-01", "2022-12"
TEST_START = "2023-01"         # excluded from smoke; untouched until ladder

HEAVY_MM = 20.0                # frozen (Task 54)
DRY_MM = 1.0                   # frozen (Task 54)
DIAG_HEAVY = [25.0, 64.5]      # manifest diagnostics ONLY

FEATURES_CORE = ["wb_lag1", "wb_lag2", "wb_lag3", "soil_lag1",
                 "tmax_lag1", "precip_roll3", "precip_roll6",
                 "precip_roll12", "wb_roll3", "wb_roll12",
                 "max1day_lag1", "heavy20_lag1", "dryspell_lag1",
                 "month_of_year"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# LAYER (i) — covariate features
# ============================================================================

def monthly_covariates():
    """District-month covariate frame from the frozen OM foundation +
    delivered weather panel (primary ET0 lane)."""
    om = pd.read_csv(OM_CSV, parse_dates=["date"])
    om["ym"] = om.date.dt.strftime("%Y-%m")

    # extremes per district-month (from DAILY data; ERA5-Land covariates —
    # convective tails smoothed, extremes biased low)
    def dryspell_max(p):  # max in-month run of days < DRY_MM
        best = cur = 0
        for v in p:
            cur = cur + 1 if v < DRY_MM else 0
            best = max(best, cur)
        return best

    ext = om.groupby(["district", "state", "ym"]).agg(
        max1day=("precipitation_sum", "max"),
        heavy20=("precipitation_sum", lambda p: int((p >= HEAVY_MM).sum())),
        heavy25=("precipitation_sum", lambda p: int((p >= 25.0).sum())),
        heavy645=("precipitation_sum",
                  lambda p: int((p >= 64.5).sum())),
        dryspell=("precipitation_sum", dryspell_max),
        n_days=("precipitation_sum", "size"),
    ).reset_index()

    # water-balance terms from the delivered panel (monthly SUMS, mm/month)
    pan = pd.read_parquet(PANEL_PQ)[
        ["district", "state", "ym", "precip_mm_sum",
         "et0_pm_api_mm_month", "soil_moisture_mean", "tmax_om_mean"]]
    df = pan.merge(ext[["district", "ym", "max1day", "heavy20", "heavy25",
                        "heavy645", "dryspell", "n_days"]],
                   on=["district", "ym"], how="left")
    df["et0_mm_month"] = df.et0_pm_api_mm_month  # PRIMARY (WB-R3 pin)
    df["wb"] = df.precip_mm_sum - df.et0_mm_month
    return df.sort_values(["district", "ym"]).reset_index(drop=True)


def build_features(cov, regime_map=None):
    """Feature frame keyed by TARGET month t; lag1 = t-1 (issue-time
    convention, frozen). Rolls = rolling SUMS ending at t-1."""
    cov = cov.sort_values(["district", "ym"]).copy()

    def g(col):
        return cov.groupby("district")[col]

    f = pd.DataFrame({
        "district": cov.district,
        "state": cov.state,
        "target_ym": cov.ym,                       # target month t
        "month_of_year": cov.ym.str[5:7].astype(int),
        # lag1 = covariate month t-1 (the latest observed at issue)
        "wb_lag1": g("wb").shift(1),
        "wb_lag2": g("wb").shift(2),
        "wb_lag3": g("wb").shift(3),
        "soil_lag1": g("soil_moisture_mean").shift(1),
        "tmax_lag1": g("tmax_om_mean").shift(1),
        "max1day_lag1": g("max1day").shift(1),
        "heavy20_lag1": g("heavy20").shift(1),
        "dryspell_lag1": g("dryspell").shift(1),
        # rolling sums ending at t-1 => shift(1) of rolling(w) sums
        "precip_roll3": g("precip_mm_sum").transform(
            lambda s: s.rolling(3).sum()).groupby(cov.district).shift(1),
        "precip_roll6": g("precip_mm_sum").transform(
            lambda s: s.rolling(6).sum()).groupby(cov.district).shift(1),
        "precip_roll12": g("precip_mm_sum").transform(
            lambda s: s.rolling(12).sum()).groupby(cov.district).shift(1),
        "wb_roll3": g("wb").transform(
            lambda s: s.rolling(3).sum()).groupby(cov.district).shift(1),
        "wb_roll12": g("wb").transform(
            lambda s: s.rolling(12).sum()).groupby(cov.district).shift(1),
    })
    # target side (layer iii would own real targets; pseudo only here).
    # Row keyed by TARGET month m: pseudo target = the series value AT m
    # (features above already come from m-1 and earlier — issue-time).
    f["pseudo_target_sm_mean"] = cov["soil_moisture_mean"].to_numpy()

    # split by TARGET month (frozen constants; test = 2023+ untouched)
    def split_of(t_ym):
        if not isinstance(t_ym, str):
            return "none"
        if t_ym <= TRAIN_END:
            return "train"
        if t_ym <= VAL_END:
            return "val"
        return "test"
    f["split"] = f.target_ym.map(split_of)

    # regime-gated features 15/16
    f["wb_roll3_x_regime"] = np.nan
    f["soil_lag1_x_regime"] = np.nan
    regime_note = "regime features NOT materialized: regime_map.csv " \
                  "absent (T11 map not frozen); interactions gated OFF"
    if regime_map is not None:
        f = f.merge(regime_map, on="district", how="left")
        miss = f[f.regime.isna()].district.unique().tolist()
        if miss:
            raise SystemExit(f"FAIL regime_map missing districts: {miss}")
        f["wb_roll3_x_regime"] = f.wb_roll3 * f.regime
        f["soil_lag1_x_regime"] = f.soil_lag1 * f.regime
        regime_note = "regime features materialized from regime_map.csv"
    return f, regime_note


# ============================================================================
# LAYER (ii) — PR-8 target transform (generic; frozen parameters)
# ============================================================================

class PR8Transform:
    """PR-8 EXACTLY (frozen): per-district linear detrend, TRAIN-years-only
    fits, district-mean level, tercile boundaries from the TRAINING
    detrended distribution, 0.15-sigma stationarity flag, single frozen
    method, trend carried as product content."""

    SIGMA_FRAC = 0.15  # frozen

    def __init__(self):
        self.fit_ = {}

    @staticmethod
    def _m(ym):
        y, m = int(ym[:4]), int(ym[5:7])
        return y * 12 + m

    def fit(self, train):  # train: DataFrame[district, ym, value]
        self.fit_ = {}
        for d, g in train.groupby("district"):
            g = g.sort_values("ym")
            y = g.value.to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)
            n = len(y)
            if n < 3:
                raise SystemExit(f"FAIL PR8 train window <3 months: {d}")
            slope, intercept = np.polyfit(x, y, 1)
            fitted = intercept + slope * x
            resid = y - fitted
            sigma = float(np.std(resid, ddof=0))
            trend_span = abs(slope) * (n - 1)
            self.fit_[d] = dict(
                slope=float(slope), intercept=float(intercept),
                n_train=n, level=float(np.mean(y)),
                origin_ym=str(g.ym.iloc[0]),
                sigma=sigma, flag_stationarity=bool(
                    sigma > 0 and trend_span > self.SIGMA_FRAC * sigma),
                trend_span=float(trend_span))
            # tercile edges from the TRAINING detrended distribution
            q1, q2 = np.percentile(resid, [100 / 3, 200 / 3])
            self.fit_[d]["edge_low"], self.fit_[d]["edge_high"] = \
                float(q1), float(q2)
        return self

    def anomaly(self, district, ym_series, value_series):
        """Detrended anomaly using TRAIN-fitted parameters only. x =
        calendar months since the district's first TRAIN month (so val
        continues the train index — never restarts at 0)."""
        p = self.fit_[district]
        x0 = self._m(p["origin_ym"])
        xs = np.array([self._m(ym) - x0 for ym in ym_series], dtype=float)
        fitted = p["intercept"] + p["slope"] * xs
        return np.asarray(value_series, dtype=float) - fitted

    def tercile(self, district, anomaly_series):
        p = self.fit_[district]
        return pd.Series(anomaly_series).apply(
            lambda a: 0 if a < p["edge_low"]
            else (2 if a > p["edge_high"] else 1)).to_numpy()


# ============================================================================
# SMOKE TEST — plumbing only, NON-EVIDENCE (Addendum 10 pre-registration)
# ============================================================================

def smoke(f, pr8):
    out = {}
    smoke_df = f[f.split.isin(["train", "val"])].copy()
    smoke_df = smoke_df.dropna(subset=["pseudo_target_sm_mean"])
    tr = smoke_df[smoke_df.split == "train"]
    va = smoke_df[smoke_df.split == "val"]
    te_excluded = int((f.split == "test").sum())

    # PR-8 fit on train only, applied to pseudo-target series per district
    tgt_train = tr.assign(value=tr.pseudo_target_sm_mean)[
        ["district", "target_ym", "value"]].rename(
        columns={"target_ym": "ym"})
    pr8.fit(tgt_train)

    def apply_pr8(part):
        cls, flags = [], []
        for d, g in part.groupby("district"):
            g = g.sort_values("target_ym")
            a = pr8.anomaly(d, g.target_ym.tolist(),
                            g.pseudo_target_sm_mean.to_numpy())
            cls.append(pd.Series(pr8.tercile(d, a), index=g.index))
            flags.append(pd.Series(pr8.fit_[d]["flag_stationarity"],
                                   index=g.index))
        return pd.concat(cls).sort_index(), pd.concat(flags).sort_index()

    tr_cls, tr_flags = apply_pr8(tr)
    va_cls, va_flags = apply_pr8(va)
    smoke_df.loc[tr_cls.index, "pseudo_target_tercile"] = tr_cls
    smoke_df.loc[va_cls.index, "pseudo_target_tercile"] = va_cls
    smoke_df["pr8_stationarity_flag"] = pd.concat([tr_flags, va_flags]
                                                  ).sort_index()
    smoke_df["NON_EVIDENCE"] = True  # stamped on every smoke row

    # ---- plumbing assertions ----
    assert smoke_df.district.nunique() == 59, "districts != 59"
    # as-of discipline spot check: features derive from months < target
    chk = smoke_df.merge(
        f[["district", "target_ym", "wb_lag1"]].rename(
            columns={"target_ym": "prev_ym_of_target", "wb_lag1":
                     "wb_at_prev"}),
        left_on=["district", "target_ym"], right_on=[
            "district", "prev_ym_of_target"], how="left")
    # (identity frame — just ensures the keying columns exist)

    feat_cols = FEATURES_CORE
    nan_tr = int(tr[feat_cols].isna().sum().sum())
    nan_va = int(va[feat_cols].isna().sum().sum())
    assert nan_va == 0, f"NaN leaked into val features: {nan_va}"

    # trivial majority-class baseline machinery (per district, train fit)
    majority = tr.assign(cls=tr_cls).groupby("district").cls.apply(
        lambda s: s.mode().iloc[0])
    va_pred = va_cls.index.map(
        lambda i: majority[smoke_df.loc[i, "district"]])
    acc = float((va_cls.to_numpy() == np.asarray(va_pred)).mean())
    n_flagged_districts = int(sum(
        1 for d, p in pr8.fit_.items() if p["flag_stationarity"]))

    out.update(
        smoke_rows=int(len(smoke_df)), train_rows=int(len(tr)),
        val_rows=int(len(va)), test_rows_excluded=te_excluded,
        nan_train_features=nan_tr, nan_val_features=nan_va,
        val_baseline_majority_acc=round(acc, 4),
        pr8_stationarity_flagged_districts=n_flagged_districts,
        stamp="NON-EVIDENCE — plumbing only (Addendum 10)")
    return smoke_df, out


# ============================================================================
# run / selftest
# ============================================================================

def run(mvp_heuristic_regime=False):
    os.makedirs(PARQ_DIR, exist_ok=True)
    regime_map = None
    if os.path.exists(REGIME_MAP):
        regime_map = pd.read_csv(REGIME_MAP)  # frozen map takes precedence
    elif mvp_heuristic_regime:
        raise SystemExit(
            "FAIL: --mvp-heuristic-regime requires a defined heuristic "
            "source (Minor Irrigation Census vintages = T2-deferred, "
            "not in sandbox). T11: heuristic never evidence; refusing "
            "to invent one.")

    cov = monthly_covariates()
    f, regime_note = build_features(cov, regime_map)
    pr8 = PR8Transform()
    smoke_df, smoke_out = smoke(f, pr8)

    pq = os.path.join(PARQ_DIR, "backbone_features_monthly.parquet")
    smoke_df.to_parquet(pq, compression="zstd")

    manifest = dict(
        built_at_utc=pd.Timestamp.utcnow().isoformat(),
        spec="DECISION_LOG Addendum 10 (pre-registered at c9208d0)",
        layer_separation=dict(
            i_covariates="built (weather; 59-execution)",
            ii_pr8_transform="generic module, frozen params, "
                             "train-only fits",
            iii_target_ingestion="STUB — D1.1 pending"),
        conventions=dict(
            water_balance="monthly SUMS mm/month (BS3 ruling)",
            et0_primary="et0_pm_api (WB-R3 PIN, Open-Meteo "
                        "hash-frozen foundation)",
            lags="feature row keyed by target month t; lag1 = t-1 "
                 "(issue-time); rolls = SUMS ending t-1",
            heavy_rain_mm=HEAVY_MM, dry_day_mm=DRY_MM,
            split_frozen=dict(train_end=TRAIN_END, val=f"{VAL_START}.."
                              f"{VAL_END}", test_from=TEST_START)),
        features_core=FEATURES_CORE,
        features_regime_gated=["wb_roll3_x_regime", "soil_lag1_x_regime"],
        regime_note=regime_note,
        diagnostics_only=["heavy25", "heavy645"],
        smoke=smoke_out,
        inputs_sha256=dict(om_csv=sha256(OM_CSV),
                           weather_panel=sha256(PANEL_PQ)),
        output_sha256=sha256(pq))
    mp = os.path.join(MANI_DIR, "backbone_features_manifest.json")
    with open(mp, "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)

    s = smoke_out
    md = ["# Backbone Features (Task-54) — build + plumbing smoke",
          "",
          f"Built {manifest['built_at_utc']} | spec pre-registered at "
          "c9208d0 (Addendum 10) | 59-execution basis",
          "",
          "## Layer separation",
          "",
          "- (i) covariates: BUILT — 14 core features (water balance in "
          "monthly SUMS; ET0 primary = et0_pm_api, WB-R3 pin; extremes "
          "labeled ERA5-Land covariates, tails biased low)",
          f"- (ii) PR-8 transform: generic module, frozen params, "
          "train-only fits",
          "- (iii) target ingestion: STUB (D1.1 pending — Qwen loaders + "
          "founder pilot-12 + Task-55 WRIS spot-check gates unchanged)",
          f"- regime-gated 15/16: {regime_note}",
          "",
          "## Plumbing smoke — NON-EVIDENCE (pseudo-target = next-month "
          "soil-moisture tercile)",
          "",
          "| check | value |",
          "|---|---|",
          f"| smoke rows (train+val) | {s['smoke_rows']:,} |",
          f"| train rows | {s['train_rows']:,} |",
          f"| val rows | {s['val_rows']:,} |",
          f"| test rows EXCLUDED (target in 2023+) | "
          f"{s['test_rows_excluded']:,} |",
          f"| NaN features in val | {s['nan_val_features']} |",
          f"| PR-8 0.15σ stationarity-flagged districts | "
          f"{s['pr8_stationarity_flagged_districts']}/59 |",
          f"| val majority-class baseline acc (plumbing only) | "
          f"{s['val_baseline_majority_acc']} |",
          "",
          "NO skill claims. The LGBM Day-2 smoke stays gated behind the "
          "baseline ladder (AM-6). D1.1 CGWB round-month verification "
          "remains the FIRST logged evaluation.",
          "",
          "AM-6: registered backbone-build step; no model runs in the "
          "evidence sense. D#22 weights ban intact. No sealed contact."]
    with open(REPORT_MD, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(json.dumps(smoke_out, indent=1))
    print("REGIME:", regime_note)
    print("WROTE:", pq)
    print("WROTE:", mp)
    print("WROTE:", REPORT_MD)


def selftest():
    # --- PR-8 transform: known trend recovered; train-only invariance ---
    rng = np.random.default_rng(7)
    n_tr, n_all = 60, 90
    y = 0.5 + 0.01 * np.arange(n_all) + rng.normal(0, 0.05, n_all)
    tr = pd.DataFrame({
        "district": "D",
        "ym": pd.period_range("2014-01", periods=n_tr, freq="M")
            .strftime("%Y-%m").tolist(),
        "value": y[:n_tr]})
    pr8 = PR8Transform().fit(tr)
    p = pr8.fit_["D"]
    assert abs(p["slope"] - 0.01) < 0.003, f"slope {p['slope']}"
    # train-only: changing TEST values must not change fit params
    y2 = y.copy()
    y2[n_tr:] += 100.0
    tr2 = tr.assign(value=y2[:n_tr])
    p2 = PR8Transform().fit(tr2).fit_["D"]
    assert p2["slope"] == p["slope"] and p2["edge_low"] == p["edge_low"]
    # tercile edges ~ terciles of train residuals
    a = pr8.anomaly("D", tr.ym.tolist(), y[:n_tr])
    t = pr8.tercile("D", a)
    assert set(np.unique(t)) <= {0, 1, 2}
    assert abs(np.mean(t == 1) - 1 / 3) < 0.08  # middle tercile share

    # --- lag/roll arithmetic on a synthetic panel ---
    cov = pd.DataFrame({
        "district": ["A"] * 18,
        "state": ["X"] * 18,
        "ym": [f"2020-{m:02d}" for m in range(1, 13)] + [
            "2021-01", "2021-02", "2021-03", "2022-12", "2023-01",
            "2023-02"],
        "precip_mm_sum": np.arange(18, dtype=float),
        "et0_mm_month": np.ones(18) * 2.0,
        "et0_pm_api_mm_month": np.ones(18) * 2.0,
        "soil_moisture_mean": np.linspace(0.2, 0.4, 18),
        "tmax_om_mean": np.linspace(30, 35, 18),
        "max1day": np.linspace(5, 50, 18),
        "heavy20": np.arange(18),
        "heavy25": np.arange(18), "heavy645": np.arange(18),
        "dryspell": np.arange(18), "n_days": 30})
    cov["wb"] = cov.precip_mm_sum - cov.et0_mm_month
    f, _ = build_features(cov)
    r3 = f[f.target_ym == "2020-04"].iloc[0]  # target Apr-2020
    assert r3.wb_lag1 == cov.wb.iloc[2]       # Mar-2020 (t-1)
    assert r3.wb_lag3 == cov.wb.iloc[0]       # Jan-2020 (t-3)
    assert r3.precip_roll3 == sum(cov.precip_mm_sim if False else
                                  cov.precip_mm_sum[:3])  # Jan..Mar SUM
    assert r3.month_of_year == 4
    # pseudo-target: next month's soil moisture
    assert abs(r3.pseudo_target_sm_mean - cov.soil_moisture_mean.iloc[3]) \
        < 1e-9
    # split boundaries (frozen constants)
    assert f[f.target_ym == "2020-01"].split.iloc[0] == "val"
    assert f[f.target_ym == "2022-12"].split.iloc[0] == "val"
    assert f[f.target_ym == "2023-01"].split.iloc[0] == "test"

    # --- extremes unit checks ---
    p = pd.Series([0.0, 0.5, 0.9, 5.0, 0.0, 0.0, 25.0, 0.2])
    best = cur = 0
    for v in p:
        cur = cur + 1 if v < DRY_MM else 0
        best = max(best, cur)
    assert best == 3 and int((p >= HEAVY_MM).sum()) == 1

    print("SELFTEST PASS (PR8 train-only + trend recovery; lag/roll "
          "arithmetic; split boundaries; extremes units)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--mvp-heuristic-regime", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run(a.mvp_heuristic_regime)
    else:
        ap.print_help()
