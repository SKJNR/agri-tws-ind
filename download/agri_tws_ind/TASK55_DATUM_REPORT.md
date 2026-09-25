# TASK-55 TRACK A — gwl_value DATUM-SIGNATURE REPORT (2026-09-24)

**Pre-registered**: Addendum 11 @ commit `07c76f7` (decision rules frozen
BEFORE the run; DECISION_LOG.md). **Report-only (AM-3)**: nothing dropped,
imputed, or reconciled. **Open rows only** (date ≤ 2022-12-31; D#24 sealed
discipline; sealed rows untouched).

**Chain of custody**: HF zip re-supplied (SHA256 `1e9d0cf6…` bit-perfect vs
arrival manifest) → `gwl_data.csv` SHA256 `c1a2e1b3…` (matches receipt +
split manifest). Guardrails PASSED exactly: all-India open rows 992,053;
AP+TG 365,742; wells 3,223; row-level negatives 114,062 (Addendum-4
finding reproduced exactly); range −1147.66..+971.29 m.

## VERDICT (pre-registered rules): **MBGL-FRAME** (depths below ground level)

masl is **physically excluded**: DEM cross-check (20-well Open-Meteo /
Copernicus GLO-90 sample) puts median well terrain at **226.5 m** (range
174–708 m) while 44.8% of Telangana rows are negative — below sea level on
a plateau is impossible. QC-failure as the dominant explanation is also
excluded: the negatives are highly structured (below), and only **93 rows
≤ −50 m** fit the QC-failure class (frozen rule).

## The real structure of the negatives (the significant finding)

| Slice | Rows | Neg share | Median gwl |
|---|---|---|---|
| Andhra Pradesh (all years) | 111,374 | **0.04%** | +7.53 m |
| Telangana ≤ 2019 | ~33,000 | ~0% | +6..+14 m |
| Telangana 2021-02..07 | ~41,600 | 4–10% | **+29.6 m** (anomalous block) |
| Telangana ≥ 2021-08 | ~68,600 | **93–100%** | **−0.3..−7.3 m** |

1. **TG sign flip (vintage artifact)**: monthly series shows onset
   2020-08/09 (transitional, small volume), mixed through 2021-07, then
   **fully flipped from 2021-08** (neg share 93–100% every month through
   2022-12; medians −0.25..−7.34 m). Magnitudes are depth-scale.
2. **Sign mixing is mostly BETWEEN vintages, not within wells**: 289 of
   1,881 TG wells span both conventions (mixed-sign); only 18 wells are
   all-negative; 1,574 all-positive.
3. **TG 2021-02..07 anomalous block**: six months at median +29.5..29.7 m
   — roughly double the clean-era median. Distinct campaign or convention;
   flagged for review (report-only).
4. **The file's `elevation` column is a placeholder** — 10 distinct values,
   constant 5.0 at the 10th–90th percentile, sentinel −9999 present; DEM
   median abs diff 221 m → **void for any datum/terrain use**. Nothing in
   the current pipeline consumes it (verified: backbone features are
   weather-side; PR-8 transform operates on the target; D#20 assignment is
   label-based). The `depth` column is NULL for all AP+TG open rows.
5. **Flipped-era magnitudes are shallower** (−2..−7 m) than clean-era
   (+12..+14 m). Two candidate readings: (a) genuine TG water-table rise
   2021–22 (documented recharge years) under a pure sign flip, or
   (b) the flipped subseries encodes a different quantity. **Not resolvable
   from inside the file** — this is precisely what the Track-B WRIS
   lookups resolve.

## D1.1 implications (report-only; no decisions consumed here)

- gwl_value may be used as depth (mbgl) once sign reconciliation is ruled.
- **TG rows from 2020-08 onward must not be used as-is** (target, feature,
  or tercile edges) until the sign convention is reconciled — a ruling for
  Qwen/founder, pre-staged here with evidence; not executed (AM-3).
- The sealed 2023+ partition presumably CONTINUES the flipped convention —
  must be checked when unsealed for D1.1+ (added to the sealed-unseal
  checklist).
- The 2021-02..07 +29.6 m block needs the same review class as the flip.

## Track-B founder WRIS lookups (the 3 stations; ~5 min)

| # | Station | District | Date | File value | Question |
|---|---|---|---|---|---|
| 1 | **TSGWD_1590** | RANGA REDDY | 2022-12-31 | **−7.11 m** | Does WRIS show +7.11 bgl? (→ pure sign flip) or something else? |
| 2 | **TSGWD_1469** | NALGONDA | 2020-01-22 | +6.86 m | Clean-era control: expect ~6.9 bgl |
| 3 | **TSGWD_1714** | SIDDIPET | 2022-12-31 | +0.20 m | Post-flip positive residual: sanity check |

indiawris.gov.in → Ground Water Level → search station code → compare the
displayed value+units for the same date. Paste the 3 results back — I file
the confirmation and close the datum question.

## Artifacts

- `program/data/manifests/task55_datum_report.json` (full numbers incl.
  monthly table + per-year negative shares + DEM sample)
- `program/scripts/task55_datum_signature.py` (main analysis; guardrails
  fail-loud; run log in worklog)
- `program/scripts/task55_trackA_supplement.py` (distribution shape,
  sign-mixing, vintage onset, DEM cross-check, Track-B sample)
