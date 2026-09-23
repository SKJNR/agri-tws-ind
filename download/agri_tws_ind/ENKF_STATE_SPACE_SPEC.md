# ENKF_STATE_SPACE_SPEC.md — v1.0 (AGRI-TWS-IND-v1)

Status: **SPEC** (delivered with R7-GLM, 2026-09-14). Pre-committed now so
that the E-NC tier decision (build / don't-build the state-estimation
stage) is mechanical when the measurement lands — no design freedom
remains to be exercised after unblinding. Build trigger: PR-6 tiering
(gap(h=2) > 8 pts → MANDATORY build; 4–8 pts → scored experiment; < 4 pts
→ documented immunity, this spec is never executed). All evaluation of
this stage is logged in DECISION_LOG before it runs (T7).

Author assignment of record: GLM (R3 §2; reaffirmed R5 §7, R6 §7).
Harness/ Instrumentation: Qwen. Shared repo.

---

## 1. Purpose and position in the ladder

The ladder forecasts district-level TWS terciles at horizons h ∈ {1, 2, 3}
months from ADVISORY_ISSUE. Its h=1 term is dominated by the *state* at
issue (R3-B analysis: nowcast error is front-loaded), and the state is
exactly what latency corrupts — the freshest released GRACE mascon at a
5th-of-month issue centers ~3 months back (LATENCY_TABLE row L1, ~85 d
release lag). This stage reconstructs the *issue-time* state from lag-legal
observations only:

- CHIRPS precipitation anomalies (L3 prelim, 2–5 d lag),
- SMAP L4 surface + root-zone soil-moisture anomalies (L10, 2–4 d lag),
- released GRACE mascons (L1), assimilated **only on their release date**.

Every input passes `test_loader_mock_clock.py` vintage rules; any artifact
whose release date exceeds the frozen issue date raises
`ImportBeyondIssueTime` (Arm-O guard, CI-enforced). The state this stage
produces feeds Arm A of E-NC and, if adopted, the operational ladder.

## 2. State definition (per district d)

Slow–fast decomposition matching the ladder's architecture (D#3):

```
x_d = [ S_d , Fs_d , Fr_d ]
```

- `S_d` — slow TWS-anomaly component: inter-annual memory (aquifer/
  storage). RAW TWS scale (the state CARRIES the trend; the PR-8
  detrending applies to the evaluation TARGET only — no double-counting:
  features derived from S enter the model, the referee detrends the label).
- `Fs_d` — fast surface-store anomaly: 5–15 d memory.
- `Fr_d` — fast root-zone-store anomaly: 30–60 d memory.

Monthly model step (m → m+1):

```
S_d(m+1)  = S_d(m) + b_d + eta_s          b_d = linear trend, TRAIN-years fit, frozen
Fs_d(m+1) = (1 - ks) Fs_d(m) + cs P_d(m) + eta_fs
Fr_d(m+1) = (1 - kr) Fr_d(m) + cr P_d(m) + eta_fr
```

- `P_d(m)` — CHIRPS monthly precipitation anomaly (the forcing).
- `ks ≈ 0.5` (≈14 d), `kr ≈ 0.15` (≈45 d) — initial values; fitted on
  TRAIN years (2016–2019) by constrained least squares, then frozen.
- `eta_*` — process noise, variances fitted on TRAIN residuals per
  district, frozen before val.

Trend note: `b_d` is fit on TRAIN years only — the same discipline as
PR-8's detrend, applied to the model side. The state may extrapolate the
trend forward; it may never be re-fit after seeing val/test.

## 3. Observations and observation operator

| Obs | Symbol | Cadence | Lag (legal) | Sees |
|---|---|---|---|---|
| SMAP L4 surface SM anomaly | `ys(d,t)` | daily | 2–4 d | `Fs_d` (weakly `S_d`) |
| SMAP L4 root-zone SM anomaly | `yr(d,t)` | daily | 2–4 d | `Fr_d` (weakly `S_d`) |
| CHIRPS pentad precip anomaly | `P(d,t)` | daily/rolling | 2–5 d | forcing (control, not obs) |
| GRACE/GRACE-FO mascon TWS anomaly | `T(d,m)` | monthly | release date per L1 | `S_d + Fs_d + Fr_d` |

Linear observation operator, coefficients fitted on TRAIN years by
regression and frozen before val:

```
ys(d,t) = hs Fs_d + h0s S_d + nu_s
yr(d,t) = hr Fr_d + h0r S_d + nu_r
T(d,m)  = S_d + Fs_d + Fr_d + nu_T
```

`nu_T` variance: official mascon SDS uncertainty combined with the fit
residual on TRAIN years, per district. Precipitation enters as CONTROL
(deterministic forcing from CHIRPS anomalies), not as an assimilated
observation — one forcing, one filter, no double counting.

**Mascon assimilation timing is the whole point:** `T(d,m)` is assimilated
only on its release date (L1 row), never on its center-month date. The
filter therefore runs 2–3 months "blind" on GRACE between releases, which
is precisely the gap this stage exists to fill.

## 4. Filter configuration

- **EnKF, N = 100 members.** District-level state (3 vars × 12 pilot
  districts = 36-dim), so covariance is trivially estimated. Compute:
  the parent competition's full k0 cycle measured 3.9 min / 0.41 Wh
  [FACT-comp]; an EnKF ensemble on 36 dims is noise on that budget.
- **Initialization:** TRAIN-period climatology + fitted spread; seed
  pinned (byte-repro CI rule, parent-program 1-ULP discipline).
- **Multiplicative inflation** `lambda ∈ {1.00, 1.05, 1.10, 1.15, 1.20}` —
  tuned on ONE metric on val 2020–22: as-of replay TWS RMSE at issue
  dates. Selected value frozen before any test-year evaluation. Zero
  tuning on test.
- **Update cadence:** daily when SMAP/CHIRPS obs available (state
  propagated at monthly step, obs update daily on the fast stores);
  mascon assimilation at each release event.
- **Spatial structure:** NO cross-district localization (state is
  per-district); OPTIONAL rank-1 common-mode correction (regional mean
  anomaly shared across districts) — the parent program measured a strong
  common mode in mascon fields [FACT-comp: common_mode_verification.txt];
  rank-1 only, fitted on TRAIN, frozen; NOT per-PC machinery (T8:
  per-PC Kalman is tombstoned — three failures in the parent program).
- **Issue-time state:** posterior mean + ensemble spread at
  ADVISORY_ISSUE − 1 d; spread is logged every issue and is available to
  the conformal layer (D#4) as an uncertainty covariate — optional,
  must be declared before any coverage evaluation if used.

## 5. Outputs and interfaces

1. `state_at_issue(d, issue_date) -> {mean, spread}` for `[S, Fs, Fr]` —
   feeds the ladder's state features (h=1-dominant) and E-NC Arm A.
2. Replay mode: as-of state for ANY historical issue date, using only
   artifacts whose release date ≤ that date (mocked-clock enforced) —
   required by the E-NC harness and the referee protocol.
3. Diagnostics per issue: ensemble spread by store, innovation
   (obs − prior) statistics, mascon assimilation events with dates.

## 6. Validation and kill criteria (pre-registered — PR-6 tiering)

- **Tier > 8 pts (mandatory build):** adopted iff the EnKF recovers
  ≥ half of the h=2 gap (O − A) with lower-CI > 0, evaluated once on
  test 2023–25, logged before running (T7). Fail → the kharif
  consequence table applies (Qwen R4 §3) and the rabi ladder inherits
  the worse branch automatically (PR-15 auto-inheritance).
- **Tier 4–8 pts (scored experiment):** same adoption rule, evaluated on
  the same frozen harness.
- **Tier < 4 pts:** never built; this spec stands as the pre-commitment
  that made the "documented immunity" branch cheap.
- Sanity gate (val years, pre-registered): as-of replay TWS RMSE must
  beat both (a) last-released-mascon persistence and (b) zero-fast-state
  (S-only) baseline. Failing either on val = the filter is mis-specified
  and does not proceed to test at all.
- E-NC report format: regime × h × {O, A, gap, tier} table; per-regime
  gaps mandatory; > 4-pt regime disagreement → per-regime tier
  assignment under the T11 frozen regime map.

## 7. Explicit prohibitions (tombstone compliance)

- T8: no per-PC Kalman in any form; banded/global covariance only.
- T5: no input whose release date exceeds the frozen issue date —
  CI-enforced via `ImportBeyondIssueTime`, not convention.
- T7: every evaluation logged in DECISION_LOG before it runs.
- T2-adjacent: inflation and store coefficients are frozen constants at
  evaluation time — no regime-conditional, no seasonal-adaptive tuning
  post-hoc. Adaptive variants require a NEW pre-registration.

## 8. Implementation notes

- numpy/scipy sufficient; no specialized DA framework dependency.
- Estimated build: filter ~200 LOC + loader integration ~150 LOC +
  replay harness hooks (Qwen). Fitted coefficients shipped as frozen
  JSON artifacts with hashes (byte-repro CI).
- Loader integration is by LATENCY_TABLE rows L1/L3/L10 entrypoints
  (`loaders.load_mascons`, `loaders.load_chirps`, `loaders.load_smap`).

Version history: v1.0 (2026-09-14, R7-GLM) — initial spec, pre-committed
ahead of the E-NC measurement per the pre-registration discipline.
