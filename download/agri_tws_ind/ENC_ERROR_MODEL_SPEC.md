# ENC_ERROR_MODEL_SPEC.md — v1.0 (AGRI-TWS-IND-v1)

Status: **SPEC** (delivered with R7-GLM, 2026-09-14). Locks the full
design of the E-NC experiment (PR-6) so that when Move-1 loaders exist,
the experiment executes with zero remaining design freedom. Pre-stated
rule: everything below is binding; the only permitted post-hoc change is
the single PR-7-style re-freeze if measured CGWB release calendars differ
from priors — logged before the affected experiment runs (T7).

Author assignment of record: GLM (error-model spec; R3 §2, R6 §7).
Replay-harness instrumentation: Qwen. Joint harness, shared repo.

---

## 1. What E-NC measures

The cost of nowcast error: at ADVISORY_ISSUE (5th of month, G7), the
freshest released GRACE mascon centers ~3 months back (LATENCY_TABLE L1,
~85 d release lag). The ladder must therefore *estimate* the state of the
unreleased gap months. E-NC measures how much of the ladder's skill
depends on that estimation — per horizon, per regime — and whether a
state-estimation stage (ENKF_STATE_SPACE_SPEC.md) must be built to
recover it.

## 2. Definitions

- **Issue calendar:** 5th of each month; pilot windows kharif (Jun–Sep)
  + rabi (Oct–Jan) per the referee protocol (PR-5).
- **Gap months** for issue month m: months whose mascons are unreleased
  at issue per the L1 vintage rule — typically m−2 and m−1 (ℓ ∈ {1, 2}).
- **True state** `x_true(d, m')`: the later-released mascon value for
  gap month m′ (Arm-O oracle artifact — EVAL_ONLY).
- **As-of state** `x_asof(d, m')`: the ladder's estimate of month m′
  made from lag-legal data only (Arm A).
- **Nowcast error:** `e(d, m', ℓ) = x_asof(d, m') − x_true(d, m')`.

## 3. Arms

| Arm | State at issue | Role |
|---|---|---|
| **O** (oracle) | released-later mascons for gap months | evaluation device ONLY — EVAL_ONLY flag CI-enforced (ImportBeyondIssueTime on import into any operational path) |
| **A** (as-of) | the ladder's own estimate from lag-legal inputs | the operational reality |
| **A′** (injected) | oracle state PERTURBED by draws from the fitted error model | sensitivity curve: degradation as a controlled function of error magnitude |

Primary quantity: **gap(h) = hit-rate_O(h) − hit-rate_A(h)** per horizon
(tercile hit, trend-adjusted frame primary per PR-8, raw reported
alongside). A′ degradation must bracket A degradation as a sanity check
(A's true errors should sit inside the A′ response surface).

## 4. Error model (fitted on val years 2020–22 ONLY)

T7 discipline: test years 2023–25 are touched once, at scoring time. All
fitting below uses val years only.

1. **Moments, stratified:** per district d, per lag ℓ ∈ {1,2}, per season
   (kharif / rabi / summer): mean bias `mu(d, ℓ, s)` and variance
   `sig2(d, ℓ, s)` of `e`. Seasonal stratification because monsoon
   onset/withdrawal error is a distinct regime from dry-season error.
2. **Temporal persistence:** 2-month seasonal block bootstrap — resample
   consecutive month PAIRS within the same season across val years. This
   preserves the dominant error mode: a missed drought onset biases both
   gap months in the same direction (the R3 §2 objection to iid error
   injection, made structural).
3. **Spatial correlation:** empirical variogram on val residuals,
   `gamma(h)` per lag ℓ; district-pair distances as great-circle km.
   Synthesis via a Gaussian copula with the fitted variogram applied to
   bootstrap scores — marginals preserved exactly (empirical), joint
   structure injected realistically. No parametric field simulation, no
   spectral methods: the copula keeps marginal honesty (the debate's
   thin-slice discipline applied to synthetic data).
4. **Regime stratification:** all parameters additionally stratified by
   the T11 frozen regime map (pumping vs monsoon-fast). Per-regime gap
   reporting is MANDATORY in the report; if regimes disagree by > 4 pts,
   tier assignment applies PER REGIME (product gates are per-regime
   anyway).

## 5. Arm A′ generator (locked pipeline)

```
for b in 1..B (B = 500 draws, seed pinned):
    1. draw temporal trajectories: month-pair block bootstrap per district, per season
    2. de-mean and standardize to unit marginal scale
    3. impose spatial pattern: Gaussian copula with fitted variogram per lag
    4. re-scale to (mu, sig) of the target (d, ℓ, season, regime) stratum
    5. x_A'(d, m') = x_true(d, m') + e_draw(d, m')
    6. run the ladder on x_A' state; record per-h tercile hits
```

Output: degradation curve — hit-rate loss vs error magnitude (in units of
fitted sigma) per h, per regime; A's own operating point marked on the
curve. This is the "sensitivity curve" of R3 §2, now fully specified.

## 6. Sample-size honesty (pre-stated, so nobody discovers it in the report)

- Val years give 36 months × 12 pilot districts × 2 lags = 864
  district-month-lag error draws pooled; per district per lag per season:
  ~6–12 draws. Bias/variance per stratum is therefore NOISY.
- Consequence (mirrors the G3 arithmetic fix): **per-district gaps are
  REPORT-ONLY.** Tier assignment reads the POOLED-PER-REGIME gap. CIs on
  the gap itself from bootstrap over issue months (2-month blocks,
  500 resamples, percentile method).
- If a regime stratum has < 40 gap-month evaluations, its tier is
  REPORT-ONLY with "insufficient evidence" language — same philosophy as
  G4 and the G3 severe-class demotion, applied to ourselves.

## 7. Tier triggers (PR-6, restated verbatim as binding)

- gap(h=2) > 8 pts → state-estimation stage MANDATORY before forecaster
  integration (spec already pre-committed: ENKF_STATE_SPACE_SPEC.md).
- 4–8 pts → scored experiment; adopted iff recovers ≥ half the h=2 gap
  with lower-CI > 0.
- < 4 pts → documented immunity; E-NC enters the report as a negative
  result with numbers.
- Kharif/rabi consequence tables (Qwen R4 §3; GLM R5 §3) apply through
  these tiers automatically — no re-litigation round.

## 8. Error-model validation (before test, on val only)

Leave-one-year-out within val (fit 2 years, predict the third):

- marginal coverage of true errors at the 80% / 90% bands ≥ nominal −2 pts;
- variogram of synthetic draws within ±20% of fitted empirical variogram
  at lags 0–300 km;
- block-persistence: lag-1 autocorrelation of synthetic errors within
  ±0.15 of empirical.

Failing any check → the error model is re-fitted (still val-only) or the
A′ sensitivity curve ships with the failure documented. It never touches
the primary O-vs-A gap, which is assumption-free.

## 9. Report format (frozen now)

```
E-NC REPORT (D1.2)                      run-id: <DECISION_LOG round ID>
  Table 1: regime × h × {hit_O, hit_A, gap, CI, tier}
  Table 2: per-district gaps (REPORT-ONLY)
  Fig   1: A′ degradation curves, per regime, A's operating point marked
  Table 3: error-model validation results (section 8 checks)
  Appendix: mascon release events used, loader vintage log (mocked clock)
```

Logged in DECISION_LOG BEFORE the run (T7); EVAL_ONLY enforcement
evidenced by CI green run of test_loader_mock_clock.py in real-audit
mode.

## 10. Prohibitions

- No fitting on test years, ever (T7). No iid error injection (R3 §2).
- No per-district tier assignment (§6). No regime map changes (T11).
- No re-tuning of the ladder between arms — the SAME frozen ladder
  binary runs O, A, A′; only the state input differs.

Version history: v1.0 (2026-09-14, R7-GLM) — initial spec; pre-committed
with the EnKF spec so the E-NC tier decision is mechanical on arrival.
