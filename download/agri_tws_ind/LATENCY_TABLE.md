# LATENCY_TABLE.md — v1.1 (AGRI-TWS-IND-v1)

Status: **SPEC** (v1.0 delivered with R5-GLM; v1.1 adds L13 for the
Open-Meteo Archive API, 2026-09-15, Day-1 sprint — T9 rule: new input
source gets a row + version bump BEFORE first use). Empirical audit runs
at Move 1 T+0 as **D1.1-Empirical**; pass condition pre-stated in
DECISION_LOG (0 round-month mismatches, CGWB × {kharif, rabi} × 12 pilot
districts). All "typical lag" values below are priors to be REPLACED by
measured values at D1.1-Empirical — measured values win, one re-freeze
allowed, logged before any skill measurement (T7).

LICENSE FLAG (v1.1): Open-Meteo's free API tier is **non-commercial use
only, attribution required**; the underlying ERA5/ERA5-Land (Copernicus)
permits commercial use with attribution. Prototyping/demo on the free
tier = fine; production commercial = direct CDS retrieval or paid
Open-Meteo tier. LICENSES.md row required; logged BEFORE it becomes a
dependency.

Enforcement: `test_loader_mock_clock.py` (same directory) freezes the
clock at each pilot issue date, runs every loader, and asserts that every
returned artifact's round/release month equals this table's rule. It also
raises `ImportBeyondIssueTime` on any artifact whose release date exceeds
the frozen issue date (Arm-O EVAL_ONLY guard — CI-enforced, not
promise-enforced). Table version bumps require: commit hash + CI re-run
(per the R2 G6 strengthening).

## Columns

- **Vintage rule (as-of)**: the ONLY artifact version a loader may return
  for a given ADVISORY_ISSUE date. ADVISORY_ISSUE = 5th of month (G7).
- **EVAL_ONLY**: artifact class is import-impossible in operational paths
  (oracle/evaluation lanes only).
- **Gates depending on it**: gates that break if this row lies.

## Rows

| # | Input | Native cadence | Typical publication lag (prior) | Vintage rule (as-of) | Loader entrypoint | EVAL_ONLY? | Gates depending on it |
|---|-------|----------------|----------------------------------|----------------------|-------------------|-----------|------------------------|
| L1 | GRACE / GRACE-FO mascons (JPL, CSR, GSFC) | monthly | ~60–85 d after month end | latest mascon whose center-month + 85 d ≤ ADVISORY_ISSUE | `loaders.load_mascons()` | later-released mascons = Arm O oracle, import-impossible in operational paths | G6, G7, E-NC (PR-6) |
| L2 | CGWB quarterly water-level rounds | quarterly (assumed Jan / May / Aug / Nov — VERIFY at D1.1) | +1–2 months release after round quarter | latest round whose RELEASE date ≤ ADVISORY_ISSUE (round month ≠ release month; loader reads release calendar, not round month) | `loaders.load_cgwb()` | future rounds import-impossible | PR-7 strata, rabi ladder (PR-15), D#0 wheat row |
| L3 | CHIRPS prelim (daily/pentad) | rolling | ~2–5 d | latest prelim with end date ≤ ADVISORY_ISSUE − 3 d, flagged `PRELIM` | `loaders.load_chirps()` | — | E-NC error model |
| L4 | CHIRPS final (monthly) | monthly | ~1–2 months | latest final month whose release ≤ ADVISORY_ISSUE − 60 d | `loaders.load_chirps(final=True)` | — | slow features, tercile-climatology baselines |
| L5 | ERA5 / ERA5-Land monthly | monthly | ERA5T prelim ~5–10 d; consolidated ~2–3 months | latest month whose end + 5 d ≤ ADVISORY_ISSUE → `PRELIM` (ERA5T); consolidated only if release ≤ ADVISORY_ISSUE | `loaders.load_era5()` | — | G11 t2m feature, GDD/xclim indices |
| L6 | IMD gridded / station | daily | ~1–3 d | latest day ≤ ADVISORY_ISSUE − 1 d | `loaders.load_imd()` | — | SPI-3 severe class (G3), heatwave outlook anchoring |
| L7 | IMD extended-range outlook (1–2 wk) | weekly-ish | operational product | valid-at-issue bulletin only | `loaders.load_imd_ext()` | — | G11 v1 fallback protective actions |
| L8 | Open-Meteo short-range (≤16 d) | on call | real time | called at ADVISORY_ISSUE (fresh), response cached with call timestamp | `loaders.load_openmeteo()` | — | short-range protective irrigation timing |
| L9 | Farmer photo (vision model) | event | zero (on-device) | photo timestamp == decision time; TFLite on-device inference | `vision.on_device_infer()` | — | G8, T9 |
| L10 | SMAP L4 surface / root-zone | 3-hourly, consolidated | ~2–4 d | latest 24-h product with granule date ≤ ADVISORY_ISSUE − 2 d | `loaders.load_smap()` | — | EnKF assimilation (PR-6 state estimation) |
| L11 | AKB static tables (CROP_PROFILES.csv + D#0 calendar) | versioned release | n/a | version hash pinned at ADVISORY_ISSUE; frozen for the whole issue cycle | `loaders.load_akb()` | — | G10, D#12–D#14 |
| L12 | Minor Irrigation Census / regime map inputs | quinquennial / static | n/a | static pre-2020 inputs only; commit hash before first skill measurement | `loaders.load_regime_map()` | — | T11, PR-14 riders, per-regime tiering |
| L13 | Open-Meteo **Archive** API (ERA5/ERA5-Land reanalysis; distinct from row L8 short-range) | on call (historical) | ERA5T prelim ~5–7 d; consolidated ~2–3 months (mirrors L5) | bulk historical fetch for TRAINING features; for as-of replay the L5-style rule applies (prelim flagged, consolidated only if release ≤ issue) | `fetch_data.py` (Day-1 script) → `loaders.load_openmeteo_archive()` | — | G11 t2m, E-NC forcing, D#0 demo content |

## Notes

1. **L2 is the row that kills projects.** The R3 §3 vintage arithmetic
   stands: at a Jun-5 kharif issue the freshest PUBLISHED round is
   January (May round dark until ~July). The loader must key on the
   RELEASE calendar, never the round month — D1.1-Empirical measures the
   real release dates for the 12 pilot districts and replaces the prior.
2. **L1 is the row that kills credibility** (the parent competition's
   leak was exactly a future-mascon import). The ImportBeyondIssueTime
   guard is therefore CI-enforced on every row, not just L1.
3. Rows L3/L4/L5 keep prelim-vs-final distinction because CHIRPS/ERA5
   vintage mixing was adopted as a data-engineering rule in R2
   (prelim-vs-final pinning).
4. Any new input source entering the feature stack requires a new row +
  version bump + CI re-run BEFORE its first use (T9 generalization:
  applies to data rows, not only models). **L13 is the first exercise of
  this rule in production: row added + v1.1 bump + CI extended and re-run
  green, all before the Day-1 fetch results enter any feature pipeline.**

Version history: v1 (2026-09-14, R5-GLM) — initial 12 rows, priors from
published product documentation; empirical values pending D1.1.
v1.1 (2026-09-15, Task 51) — adds L13 (Open-Meteo Archive, Day-1 sprint)
+ non-commercial license flag; mocked-clock CI extended with the L13
rule, self-check green.
