# FINAL COMPLIANCE VERIFICATION — Selected Files Audit

**Date:** 2026-09-13 (final week, before close)
**Trigger:** Independent external re-review requested verification of five
specific items before the submission lock: `SPEI_01(t+1)`, `fast_resid`,
`nino34.ascii.txt`, ERA5 isolation, and coordinate-derived features.
**Scope:** the exact build chains of the two SELECTED files —
`submission_v28_deccorr.csv` (slot-1, public 0.674859467, md5 `0ebede07887d…`)
and `submission_v21a.csv` (slot-2, public 0.687374005, md5 `6b6e3e41c253…`).
**Method:** direct code + data re-audit from disk (no memory), including fresh
measurements on the competition CSVs.

> **REPRODUCTION PROOF (2026-09-13, this audit):** both selected files were
> REBUILT from the three organizer CSVs using only the packaged scripts —
> every step byte-identical: v21a `6b6e3e41c253…`, v24 `f462e9da7110…`,
> v25 `bad77bbadbb40…`, v26 `273bb70d8945…`, v27 `fefcd0c6075a…`,
> v28 `0ebede07887d…` (chain wall-time ≈ 8 min; LightGBM deterministic
> at seed 0 in this environment; all in-script bit-audits PASS).

---

## Verdict table

| # | Item | Verdict | One-line basis |
|---|------|---------|----------------|
| 1 | `SPEI_01(t+1)` | **LEGAL — intended input** | Target-month covariate row as provided by organizers in Test.csv; model never uses post-target months |
| 2 | `fast_resid(t)` | **CLEAN** | Observed TWS_t minus climatology minus per-cell trend; no LB feedback, no future TWS |
| 3 | `nino34.ascii.txt` | **Decision-log only — never in any submission build** | ENSO prior cross-check printout in the v28 adjudication; constant itself from pre-registered public-LB pair probe |
| 4 | ERA5 | **Excluded from selected files — verified by input audit** | Selected chains read only Train/Test/SampleSubmission CSVs + own prior submissions; ERA5 failed the pre-registered inclusion bar (Phase B) |
| 5 | Coordinate-derived features | **None in final models (post 19-Aug ruling)** | Raw lat/lon dropped in v24 rebuild; retained per-cell train statistics are target-derived, not coordinate encodings |
| 6 | Bonus: v28 correction chain | **Disclosed post-model calibration** | Six month-level additive constants on k0 rows only, from public-LB probe pair readouts, byte-audited |

---

## 1. `SPEI_01(t+1)` — target-month covariate, organizer-provided

**What the feature actually is.** The competition's task structure (measured
directly on the data, this session): each row at month `t` carries observed
`TWS_t`, covariates `*_t`, and the training label `target`. We verified
**`target(t) = TWS at month t+1` exactly** — median |target − TWS(t+1)| = 0.0
over 2,138,306 comparable rows. The task is one-step-ahead TWS forecasting.

The k0 models (both the v24/v28 lineage and the v21a lineage) build their
feature matrix by merging, for each row at `t`, the covariate values of the
row at `t+1` (the month being predicted) — renamed `_nxt`:

- `build_v24_compliant_k0.py` lines 150–152 (train) and 192–194 (test);
- `build_v21_phaseC.py` `build_linear_final()` lines 305–357 (same structure,
  lookup within Test);
- Kalman covariate update at the target month: `kf.cov_update(vz[a+1])`
  (build_v21_phaseC.py line 298).

**Therefore `SPEI_01(t+1)` = the SPEI_01 value of the month being predicted,
as provided by the organizers on that month's own Test.csv row.** It is not
shifted, computed, or imported by us; it is the primary intended input of the
challenge (fast-publishing drought indicators → delayed GRACE TWS). Its SHAP
strength (rank 2, mean |SHAP| 0.074) reflects exactly this intended signal.

**The boundary we did NOT cross.** Using covariates from the month *after*
the target (`covs(t+2)`) — genuine future information — was identified as a
separate lane ("the exploit") in `scripts/cov_lag_exploit.py`, measured, and
**not adopted** in any selected file. Consistent with this discipline, when
the target month's row does not exist (last test month → target 2019-01),
the models fall back to a **reduced model without target-month covariates**
(`colsR`/`bst_r` in build_v24; `predR` in build_v21_phaseC) rather than
fabricating future values.

## 2. `fast_resid(t)` — observed-state input

Definition (task44/build_v24, `Xlin[:,0]`):
`fast_resid(t) = TWS_t − mu_c − trend(t)` — the current-month observed TWS
anomaly after removing per-cell climatology (`mu_c`) and per-cell linear trend.
At inference its only sources are (a) observed train TWS and (b) **unmasked**
Test.csv TWS at anchor months. It is the one-step persistence input (linear
coefficient ≈ 0.596 — physically the AR(1) of the fast state). It involves no
test labels, no masked-row reconstruction, no leaderboard feedback, and no
future TWS. The name means "residual of the fast component" (slow trend
removed) — it does not refer to any leaderboard residual.

## 3. `nino34.ascii.txt` — ENSO prior cross-check in the decision log only

Where it appears: `scripts/task38_m201612_pair_v28.py` §5
"ENSO PRIOR CROSS-CHECK", a console printout comparing the probe-derived
December-2016 bias `e` against a magnitude prior implied by the ONI index
(|ONI| 0.45 → prior 0.05–0.15). The measured `e` came from the
**pre-registered m201612 ±3.0 pair probe on the public leaderboard**
(actuals 1.036776186 / 0.956643248 vs base 0.683791578). The prior
under-called the magnitude ~1.7×; direction agreed.

**No submission file's build reads nino34 or any ENSO data.** The report
(§5.3) discloses the month constants as exactly what they are: post-model
calibration against public leaderboard feedback, estimated under a
pre-registered shrinkage rule (0.8× measured bias, arming gate |e| > 0.05,
two-sided verification gate ±0.0005). For provenance if asked: the file is
the NOAA Climate Prediction Center Niño 3.4 index (public, monthly;
`nino34.ascii.txt`), used solely as a physical plausibility cross-check on
the December 2016 correction decision — never as a feature, never as a
correction source.

## 4. ERA5 — excluded from the selected files (input audit)

**Fresh input audit of the complete selected chains (2026-09-13):**
every file read by `build_v24_compliant_k0.py`, `build_v21_phaseC.py`, and
their shared dependency `a15_common.py` is one of: `Train (1).csv`,
`Test (2).csv`, `SampleSubmission (4).csv`, the team's own prior submission
CSVs (v21a/v18a/v10b — internal lineage), and own code modules. **No ERA5,
GRACE, GDO, COST-G, CSR, GLDAS, or GravIS file is read anywhere in the
selected chains.**

History: ERA5 channels were evaluated in the v21 Phase-B mirror harness and
**failed the pre-registered inclusion bar** (`build_v21_phaseC.py` header:
"ERA5 channels FAILED the bar -> excluded"; mirrors M1/M2 measured).

**Naming warning for reviewers (documented to avoid confusion):** several
identifiers in the final code contain the substring "era" —
`dhat_era`, `Dtil_era`, `era_cov`, `era_rows`. These are **epoch/era
terminology** (anchor-era exponential interpolation of observed test TWS
fields; "era months" = the test window), NOT ERA5 data. All inputs to those
functions are organizer-provided CSV values (verified line-by-line this
session). The ERA5 experiment scripts and external climate binaries
(`*.nc`) are additionally **excluded from the 72h code package** and remain
only in the full repository history for audit completeness.

## 5. Coordinate-derived features — none in the final models

The organizer ruling of 19 Aug (forum thread 34450) prohibited raw
coordinate features. Enforcement in the selected lineage:

- `build_v24_compliant_k0.py` — the compliance fix is flagged in-source
  (">>> THE COMPLIANCE FIX: XLB without lat_arr/lon_arr raw coordinate
  columns <<<"); the final k0 LightGBM uses exactly the 17 features listed
  in `task44_shap_codecarbon.py` (`FEATS`): fast_resid, trend(t+1), the ten
  covariate anomalies (t and t+1), has_covs(t+1), sin/cos(month), beta_c,
  mu_c. **No lat, lon, cell-ID, bin, cluster, or region feature.**
- Retained per-cell quantities `mu_c` (train TWS climatology) and `beta_c`
  (train TWS trend slope) are **statistics of the target variable computed
  from train rows**, indexed by cell — not coordinate encodings. This was
  the adjudicated boundary at the v12b→v24 rebuild.
- Spatial Gaussian smoothers (GAU10 in v24; sigma=2.0° cKDTree smoother in
  v21a) use **grid adjacency** to average neighboring predictions — a
  spatial averaging operator, not a coordinate-valued feature entering the
  model. No target encoding by cell, no kNN features, no region labels.
- lat/lon and the parsed ID are used only for row identity, cell indexing,
  and grid alignment — never as model inputs.

## 6. The v28 correction chain (bonus check)

v28 = v24 + six month-level additive constants, each applied to **k0 rows
only** (201601, 201509, 201606, 201807, 201811, 201612; shrink 0.8× the
probe-measured public bias; 201704 measured below the arming gate and
documented as skipped). Each build step bit-audits that untouched rows are
byte-identical (manifest ledger, e.g. v28: exactly 15,618 of 280,961 lines
differ, all in 201612, all k0). The masked block (186,913 rows) is shared
byte-identically across v18a→v21a→v24→v25→v26→v27→v28 (max |d| = 0) — chain
upgrades never touch masked rows. **The all-private month 201812 carries no
correction by design**: the zero-month probe (7th exact LB hit) proved it
has zero public rows, so no public-feedback constant exists for it, and we
do not apply unverifiable corrections.

---

## Reproducibility pointers

**VERIFIED END-TO-END 2026-09-13** (see the reproduction proof above): the
chain `build_v24_compliant_k0.py → build_v25_and_k0_probes.py →
task29_readout_and_v26.py → task32_readout_v27.py →
task37_v26close_m201704.py → task38_m201612_pair_v28.py` regenerates
slot-1 byte-identically, and `build_v21_phaseC.py` regenerates slot-2
byte-identically (~16 s, deterministic, no LightGBM). One-command runner:
`run_chain.sh` inside `download/72h_code_package.zip`.

- SHAP + CO2e evidence: `scripts/task44_shap_codecarbon.py` →
  `report_shap_carbon.{txt,json}`, `emissions.csv`
- Every claim above is checkable against the files cited; the worklog
  (Tasks 1–47) records the decision path that produced them.
