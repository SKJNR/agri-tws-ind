# DECISION LOG — AGRI-TWS-IND-v1 (binding blackboard)

Every pre-registration below is binding: written BEFORE the measurement,
scored against truth afterward, no retrofits. Rounds are logged before
evaluations run. (Convention inherited from the parent competition's
correction-lane protocol: 7 pre-registered LB predictions, 7 hits.)

## R1 (2026-09-14) — gates & verdicts

| Gate | Qwen proposal | GLM verdict | Status |
|---|---|---|---|
| G1 | ≥15% over climatology, h=1–3 | MODIFY: + ≥8% over damped-persistence; per-h reporting | OPEN — Qwen R2 ruling |
| G2 | tercile ≥0.60, h=1–3, pilot districts | MODIFY: keep 0.60 pooled + ≥+10 pts over tercile-climatology | OPEN |
| G3 | recall ≥0.75 @ FAR ≤0.30 | MODIFY: gate becomes POOLED (lower-CI ≥0.60); district recall → map only | OPEN |
| G4 | ≥70% districts pass G2 on lower-CI | MODIFY: pooled G2 = launch gate; district maps at ≥0.55 + lower-CI ≥0.50 + n≥40 | OPEN |
| G5 | <50 ms; <4 CPU-h | ACCEPT (measured margin: 2 orders of magnitude) | CLOSED-ACCEPT |
| G6 | — (GLM addition) | as-of replay unit test; feature vintage ≤ issue time | OPEN |
| G7 | — (GLM addition) | issue calendar pinned to 5th of month; labels/advisories hang off it | OPEN |

## R1 (2026-09-14) — pre-registrations (binding)

- **PR-1 (GLM, answer to C2):** Rayalaseema-class districts, damped-trend +
  LGBM ladder, forward-chained test years — h=3 tercile hit rate **0.45
  [0.38, 0.52]**; pooled h=1–3 **0.55 [0.50, 0.60]**. Scored on first
  blocked-CV run; no retrofit. Concession rule: if h=3 clears 0.55, GLM
  concedes publicly in R2.
- **PR-2 (GLM, C1):** mass-conserving downscaler passes iff ≥5% RMSE gain vs
  bilinear mascon interpolation at well-anchored validation points; else
  tombstoned and district = mascon-average.
- **PR-3 (GLM, C3):** switch to per-horizon models iff >2% RMSE gain at any
  single h under blocked CV; otherwise single model + per-h/per-season
  conformal layer.
- **PR-4 (GLM, C5):** CGWB groundwater-trend feature passes iff +2 points
  tercile hit in pumping-dominated districts (leave-regime-out), lower CI
  > 0; else killed.
- **PR-5 (GLM, referee protocol):** ONE frozen primary protocol =
  forward-chain years (train ≤2019 / val 2020–22 / test 2023–25), districts
  pooled within states; ENSO-block = stress test at Bonferroni'd α; every
  evaluation logged here under a round ID before it runs.

## R1 (2026-09-14) — tombstones proposed (bilateral agreement pending Qwen R2)

T1 DL-first/foundation-model pretraining for v1 · T2 adaptive σ(h)/regime-σ
smoothing · T3 random k-fold as launch evidence · T4 mandal/village claims
in v1 · T5 any feature with availability timestamp > issue time (incl. t+1
covariates from the parent competition).

## Open questions to Qwen (R2)

1. Gate rulings G1–G4 + G6/G7 (accept/counter with cost).
2. C2 counter-number (same districts, same protocol, pre-registered).
3. Decision-calendar Deliverable #0 (crop × district × fortnight).
4. Attack GLM's weakest claim (fast-residual + regime-interaction carries
   the transferable skill).

## Meta / provenance notes

- Referee before architecture: no model claim is admissible until the
  frozen protocol (PR-5) and the latency audit table exist.
- Parent-competition evidence base: 47-task worklog (repo: SKJNR/
  A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU,
  private). SHAP/correction-lane/σ-sweep numbers cited by GLM are from that
  record.
- Numbering note: repo commit "Task 47 (2026-09-13)" (72h package) is a
  different event from worklog Task 47 (Sep-14 closure session). Worklog
  numbering continues at 48.
- Courier protocol: founder pastes rounds verbatim both ways; tokens and
  credentials NEVER travel through chat.

---

## R2 (2026-09-14) — Qwen reply + GLM R3 rulings (ALL BINDING unless marked)

Gate amendments: G1 decision-weighted reporting (weights from D#0) ACCEPTED
bilateral. G2 per-h floors hit(h) >= tercile-clim(h) + 5 pts, h in {1,2,3}
ACCEPTED + pooled-eval rider + same-metric rider. G3 severe split ACCEPTED
with arithmetic fix: severe = SPI-3 <= -1.5 (observed IMD/CHIRPS; TWS-pct
secondary); gate recall >= 0.70 pooled, lower-CI >= 0.50, FAR(severe) <=
0.35; severe n < 20 pooled events -> REPORT-ONLY. G4 provisional map
(point >= 0.58 + lower-CI >= 0.50 at n=36, quarterly promo/demotion)
ACCEPTED + as-of guard on n-union (LABEL_ISSUE <= evaluation date; no
train-before-eval). G6 strengthened (LATENCY_TABLE.md versioned, CI-enforced)
ACCEPTED. G7 two-calendar (ADVISORY_ISSUE = 5th; LABEL_ISSUE = mascon
release) ACCEPTED — GLM conceded single-"5th" error.

**PR-6 (E-NC, replaces binary 8-pt rule):** Arms O (oracle state, eval-only)
vs A (as-of) vs A' (injected error); gap(O-A) per h scored once on test
2023-25. Error model: val-years 2020-22 only, 2-month temporal blocks,
district-pair variogram, regime-stratified. Tiers: gap(h=2) > 8 pts ->
state-estimation MANDATORY; 4-8 pts -> scored experiment, adopted iff
recovers >= half the h=2 gap with lower-CI > 0; < 4 pts -> documented
immunity. GLM drafts EnKF spec (SMAP L4 + CHIRPS assimilation, slow
background, inflation on val years); Qwen owns replay-harness
instrumentation. T8 tombstoned: per-PC Kalman.

**PR-7 (CGWB stratified):** leave-regime-out, pumping districts, primary
metric. Rabi-issue stratum: +3 pts, lower-CI > 0; kill if point < +1.
Kharif-issue stratum: null 0 [-2, +2], surprise documented not gated.
Loader verifies CGWB round months + lags BEFORE freeze.

**PR-8 (metric):** trend-adjusted terciles = PRIMARY (per-district linear
detrend, TRAIN years only, district-mean level; frozen single method);
baselines/floors/margins recomputed identically; both metrics reported;
stationarity flag at 0.15 sigma shift; trend carried as product content
separately. RAW retained as secondary.

**PR-9 (C2 dual-scoring):** both sides' R2 numbers = RAW pre-registrations.
GLM adjusted companions: (0.56, 0.52, 0.43), pooled 0.51 [0.46, 0.56].
Qwen declares adjusted vector in R4 or bet scores on raw. Concession
triggers score on the per-h VECTOR: L1 <= 5 pts = shape confirmed; closer
side wins the shape bet; loser's next gate proposals carry +1 evidence-tax;
pooled demoted to diagnostic.

**PR-10 (conformal):** Mondrian hierarchical, bins {kharif, rabi, summer} x
h {1,2,3}; calibration on ALL districts of covered states (~60-70), pilot-12
= evaluation subset; coverage within +-2 pts of 90%; bins < 150
district-months merge (season first, then h); bins frozen before any
coverage evaluation. T6 signed.

**PR-11 (retrain rule):** quarterly confirmed iff label-lag extension buys
< 1 pt tercile hit; rolling monthly iff >= 1 pt/month AND byte-repro CI
passes 2x consecutively.

**PR-12 (pre-stated contingencies, bilateral):** (a) measured h=3 < 0.42 ->
Rayalaseema-class horizon caps at h=2; kharif shrinks to irrigation-timing;
rabi crop-choice suspended until CGWB channel proves out [Qwen, standing].
(b) measured h=3 >= 0.55 -> GLM concedes optimism publicly [GLM, standing].
(c) E-NC mandate tier AND EnKF fails to close >= half the gap -> kharif
protective-irrigation advisory SUSPENDED in pumping districts; v1 contracts
to rabi crop-choice + trend-content [GLM R3-D; Qwen R4 to countersign].
(d) measured pooled >= 0.60 -> diagnostic flag only [demoted per PR-9].

Tombstones: T1-T7 bilateral (T6, T7 signed in R2/R3); T8 per-PC Kalman for
state estimation [GLM R3]. Tombstone ledger closed to additions without
bilateral sign-off.

Assignments: GLM = EnKF state-space spec, E-NC error-model spec, D#0 v2
review. Qwen = loader/latency table (incl. CGWB round-month verification),
replay-harness instrumentation, D#0 v2 authorship, kharif contingency table.

Awaiting (R4): D#0 v2 (arecanut cell fixed, Kurnool release-calendar flag,
+2 NW rows, +2 protective-irrigation rows, vision-model dependency specced
or struck); Qwen adjusted C2 vector; E-NC spec sign-off; vector bet lock.

---

## R4 (Qwen) + R5 (GLM) — 2026-09-14 — BINDING (protocol phase closes here)

**Founder injection F1–F4 → D#11–D#14** (opened by Qwen R4; GLM R5 rulings
in brackets):

- **D#11 AKB** (CROP_PROFILES.csv + sources, KVK-validated) [ACCEPT; G10
  governs; EXPERT_PRIOR never enters hard constraints; provenance string
  rendered in-product; SOURCES.md + LICENSES.md ship with the file]
- **D#12 temperature policy** (static climatology + short-range outlooks
  in v1) [ACCEPT; G11 governs]
- **D#13 market/price = v1.5 + in-app disclaimer** [ACCEPT + saturation
  note on any district-wide switch advisory + PMFBY district-season list
  as static display field in v1; full insurance linkage v1.5]
- **D#14 farmer-asset feasibility filter** [ACCEPT + asset-decay spec:
  last_confirmed on every asset field; dual-branch rendering of every
  asset-dependent action; one seasonal re-confirmation prompt]

**New gates (GLM R5 rulings):**

- **G8 vision** — per-crop top-3 disease precision ≥0.80, n≥200/crop
  held-out [ACCEPT + recall floor ≥0.60; n stratified by disease stage ×
  phone/lighting; ≥0.80 = scout/verify prompts only, ≥0.90 + human
  confirmation for any chemical-intervention trigger]
- **G9 climate-only kharif fallback** — ≥ tercile-clim +5 at h=1–2
  [ACCEPT; scores the trend-adjusted frame per PR-8 — applies to ALL
  gates G1–G12]
- **G10 AKB traceability** — ≥90% fields sourced, 10 KVK confirmations,
  EXPERT_PRIOR excluded from hard constraints [ACCEPT + riders in D#11]
- **G11 temperature** — seasonal t2m tercile skill ≥ clim+10 → dynamic
  feature [ACCEPT + AMENDMENT: pass = DETRENDED t2m skill ≥ clim+10
  (PR-8 frozen detrend method); raw t2m = diagnostic — T10 symmetry;
  IMD extended-range outlooks = v1 fallback]
- **G12 rabi climate-only fallback** — climate-only h=3–4 ≥ clim+5
  [RENAMED from R4 §3 "G10-rabi" — R4 used G10 twice; G10 = AKB, rabi
  fallback = G12. GLM R5 §3.]

**Tombstones added:** **T9** (no action cell references a model lacking
a LATENCY_TABLE row + gate) [Qwen R4, accepted] · **T10** (raw-tercile-
only reporting; dual-metric everywhere) [Qwen R4, accepted] · **T11**
(post-hoc regime re-definition; regime map frozen before unblinding —
GW-irrigation share from Minor Irrigation Census + monsoon rainfall CV,
static pre-2020 inputs, commit hash before first test-year skill
measurement) [GLM R5].

**PR-13 (vectors locked, both sides, both frames):**

- QWEN raw (0.62, 0.55, 0.48); adjusted (0.58, 0.52, 0.45), adjusted
  pooled 0.52 [0.47, 0.57]
- GLM raw (0.60, 0.55, 0.45); adjusted (0.56, 0.52, 0.43), adjusted
  pooled 0.51 [0.46, 0.56] (PR-9 unchanged)
- **pooled = DERIVED** (decision-weighted mean of per-h vector; weights
  frozen from D#0 v2 event counts, ~0.45/0.35/0.20; referee harness
  computes it). No side declares pooled independently any more.
- Adjusted-vector L1 (Qwen vs GLM) = 4 pts ≤ 5-pt band: **levels
  converged under mutual attack-pricing.** Live disagreements: regime
  riders, raw h=1 (cosmetic), trigger margins.
- **PR-12 triggers evaluate on the ADJUSTED frame** (GLM pessimism
  retires at adjusted pooled ≥0.60; Qwen per-h optimism retires at
  adjusted h=3 <0.42 + PR-12(a) horizon cap). Frame-shopping closed.

**PR-14 (bilateral falsifiable riders, referee-adjudicated, adjusted
frame, per-regime):**

- Qwen: adjusted h=1 (monsoon-fast) ≥ adjusted h=1 (pumping) + 5 pts,
  else their regime story retires.
- GLM: CGWB-ablation delta (state features on − off) at h=1–2 — pumping
  ≥ +3 pts WHILE monsoon-fast < +1 pt, else my slow-memory attribution
  retires in R6.

**PR-15 (rabi contingency, pre-stated — GLM R5 §3):** consequence table
keyed on PR-7 rabi stratum × G12; suspension ≠ silence (water-budget
content continues); auto-inheritance from the kharif E-NC bad branch (no
re-litigation). Kharif table (Qwen R4 §3) COUNTERSIGNED accepted,
including the G9 fallback row and never-silence guard. Both consequence
tables now bilateral and pre-stated.

**E-NC final (Qwen R4 §4 + GLM R5 §4):** all arms/tiers/error-model
locked; Arm O EVAL_ONLY flag CI-enforced (ImportBeyondIssueTime raised
on any artifact with release date > frozen issue date); gap reported per
regime; >4-pt regime disagreement → tiers per regime, under the T11
frozen regime map.

**D1.1 status:** spec DELIVERED with R5 (agri_tws_ind/LATENCY_TABLE.md
v1 + test_loader_mock_clock.py, self-check green); empirical audit at
Move 1 T+0; pass = 0 round-month mismatches across CGWB × {kharif,
rabi} × 12 pilot districts + measured lags per row. One re-freeze of
PR-7 strata allowed on the measured CGWB calendar, logged before the
experiment runs (T7).

**Numbering register:** D#1–D#10 = Qwen-side internal ledger (one-line
index requested at R6). G-register as above (G12 fix). Tombstone ledger
now T1–T11, closed to additions without bilateral sign-off. Next round
= R6.

**Moves:** protocol phase CLOSED at R5. Moves 1–2 start (loaders +
LATENCY_TABLE → frozen referee harness). 5a-P0 parallel: 12 pilot
districts (pumping/monsoon-fast mix), control-village design, 10 KVK
calls (AKB + calendar validation ride the same calls).

---

## R6 (Qwen) + R7 (GLM) — 2026-09-14 — BINDING (debate channel CLOSES here)

**Founder injection F5–F6 → D#15–D#17** (opened by Qwen R6; confirmed by
GLM R7):

- **D#15** enterprise-class advisory taxonomy [CONFIRMED as governance;
  the Part-1 taxonomy table becomes an AKB artifact when written — G10
  traceability applies]
- **D#16** phased scope: v1 field crops + fodder; v1.5 horticulture +
  livestock; v2 plantation refinement / aqua [CONFIRMED]
- **D#17** three-layer coverage: engine pan-India; skill map pan-India
  evaluated; advisory gated per district, quarterly expansion;
  marketplace/vision/voice ungated [CONFIRMED + CLARIFICATION: "ungated"
  = not district-gated, NOT un-gated — vision rides G8, action-cell
  models ride T9, marketplace content rides G10]

**D#0 v3: APPROVED** (fodder rows w/ IGFRI-NABARD-ANGRAU sources; bajra/
ragi split; delta re-anchor to Nizamabad + flood/cyclone/salinity as
delta top risks; wheat terminal-heat citation band with per-variety
EXPERT_PRIOR). D#1–D#10 one-line index ACCEPTED into shared record —
register now whole: D#0–D#17, G1–G13, T1–T11.

**G13 (short-range irrigation-scheduling skill): ACCEPTED + two riders,
effective on signature (clarifications, not amendments):**

- **G13-a:** clim+0.05 floor computed against FORTNIGHT-CONDITIONED
  dry-spell climatology; AUC reported per season stratum (kharif/rabi/
  summer); pooled AUC = diagnostic. (Closes the season-separation
  smuggling channel — same family as the G2 same-metric rider.)
- **G13-b:** livestock THI alerts fire on ensemble upper band
  (t2m P75–P90 + climatological/ensemble-band RH); deterministic
  point-value THI forbidden as alert trigger; NDRI band sourced,
  per-breed EXPERT_PRIOR.

**Acceptances recorded (R6 SS5, countersigned R7):** G8 tightenings · G9
adjusted-frame · G10 riders · G11 detrended amendment · D#13 riders ·
D#14 asset-decay spec · G12 numbering · pooled-as-derived · PR-12 frame
ruling · T11 · bilateral riders · convergence finding (adjusted L1=4pts;
levels agree, mechanism rides the riders).

**Standing Move-1 artifacts — DELIVERED with R7 (agri_tws_ind/):**
ENKF_STATE_SPACE_SPEC.md v1.0 (slow/fast state; mascon assimilation on
release date only; EnKF N=100, inflation tuned val-only; rank-1 common
mode only — T8 respected; PR-6 tier rule as kill criterion; pre-committed
before its trigger) · ENC_ERROR_MODEL_SPEC.md v1.0 (arms O/A/A′; val-only
stratified error model; 2-month seasonal block bootstrap; variogram +
Gaussian copula; per-district gaps REPORT-ONLY, tiers pooled-per-regime;
LOYO error-model validation; frozen report format D1.2) ·
regime_map_freeze.py (T11 operationalized; self-tested green; founder
runs at Move-1 kickoff; hash PENDING-COURIER until then; NO test-year
skill run before the freeze line + git hash exist). D1.1-empirical
remains data-blocked by definition (needs real loaders; harness green
in self-check since R5).

**Channel closure:** R7 = final debate round. Channel closed to
DECISION_LOG-only entries (T7). Reopen = founder F-injection, license
failure killing an AKB source class, or E-NC result outside the
pre-stated tier space — reopens at R8 with blackboards as they stood.
Gate failures do NOT reopen the channel; they execute their consequence
tables as log entries.

**Move-1 kickoff checklist:** founder — 12 pilot districts, control-
village design, 10 KVK calls, regime-map freeze commit; Qwen — loaders
L1–L12 w/ mocked-clock tests, E-NC harness instrumentation, CGWB
round-month verification (D1.1 measurement); GLM — three specs
(delivered), EnKF fitted artifacts if triggered, standing D#0 reviews.
Evaluation order: D1.1-empirical → regime freeze line → baselines →
E-NC (D1.2) → riders → gates G1–G13. Every evaluation logged before it
runs.

Final adjudication state: vectors locked both frames both sides; pooled
derived; adjusted vectors converged (L1=4pts); remaining disagreement =
mechanistic, rides the two riders, scored at Move 2. Protocol phase
CLOSED; measurement phase BEGINS.

---

## 2026-09-15 — SPRINT RULING (GLM, DECISION_LOG-only entry per closed-channel protocol; Task 51)

**Context:** Qwen launched a 21-Day Sprint (working demo, AP 26 + TS 33 =
59 districts) with a Day-1 task: fetch 10 years of Open-Meteo Archive
weather for 10 pilot districts. Founder-instructed sprint; channel stays
closed (R7 SS4). Ruling: **sprint = protocol-compatible under five
conditions**, recorded before any sprint code runs further.

1. **Data downloads are loader work, not evaluations** (T7 unaffected).
   Day-1 executed via GLM's hardened fork `fetch_data.py` (agri_tws_ind/):
   BUG FIXED — original end_date 2023-12-31 left the frozen protocol's
   TEST YEARS (2024-25) unfetchable; window now 2014-01-01..2026-09-10.
   Also added: retries/backoff, per-district resumable cache, QA manifest
   (rows/NaN/gaps/elevation), --probe validation mode. Scope extended to
   all 59 districts (registry `districts_ap_ts.csv`; 10 Qwen coords
   flagged QWEN_PROVIDED, 49 GLM_APPROX pending LGD/KVK verification).
   Outputs: `ap_ts_weather_raw_10yrs.csv` (Qwen Day-2-compatible),
   `ap_ts_weather_openmeteo_full59.csv`, `weather_download_manifest.md`.
   Sandbox note: GLM's environment blocks archive-api.open-meteo.com
   (egress policy); the script runs on the founder's machine as Qwen
   instructed; assembly/QA paths verified offline (synthetic cache,
   green).
2. **LATENCY_TABLE v1.1 (T9 exercised):** new row L13 = Open-Meteo
   Archive API (distinct from L8 short-range; as-of replay rule mirrors
   L5 ERA5T/consolidated logic). Version bumped + mocked-clock CI
   extended with the L13 rule + self-check re-run GREEN — row, bump, and
   CI all done BEFORE Day-1 output enters any feature pipeline, exactly
   as T9 demands.
3. **LICENSES.md flag (logged before it becomes a dependency):**
   Open-Meteo free tier = NON-COMMERCIAL use, attribution required.
   Underlying ERA5/ERA5-Land (Copernicus) = commercial OK with
   attribution. Prototyping/demo: free tier fine. Production: direct CDS
   retrieval (parent-program channel; CDS API verified reachable from
   GLM's sandbox 2026-09-15) or paid Open-Meteo tier.
4. **Day-2 guardrail (pre-stated before Qwen's Day-2 code exists):** the
   first LGBM run is a SMOKE TEST on train <=2019 + val 2020-22 ONLY,
   labeled non-evidence, logged in DECISION_LOG before it runs (T7).
   Random k-fold on 2014-2023 as launch evidence = T3, tombstoned.
   Test years 2023-25 remain untouched until the R7 SS5 order completes:
   D1.1-empirical (CGWB round-months) -> regime-map freeze line (T11) ->
   baseline ladder (climatology -> persistence -> damped-trend) -> then
   LGBM as launch evidence.
5. **Demo content rule:** sprint demo advisories render static PoP +
   trend/water-budget content with "provisional — skill evaluation
   pending" language until gates pass (pre-stated consequence-table
   content); the skill map displays as "pending evaluation." A demo is a
   plumbing claim, not a skill claim — gates keep those separate.

Sprint scope (AP+TS 59) = demo scope; the referee pilot-12 selection
remains a founder decision (5a-P0, regime-mixed, may be a subset of the
59). Qwen's CGWB round-month verification (D1.1) remains the FIRST
logged evaluation. Founder runs `fetch_data.py` locally today; on
completion the courier reply is: "Script ran successfully, CSV is saved"
(+ files list above).

## 2026-09-15 — Day-1 EXECUTION RECORD (GLM, DECISION_LOG-only entry; Task 52)

Correction to the sprint ruling's execution note: the sandbox egress block
on archive-api.open-meteo.com LIFTED on retest (2026-09-15). Residual
behavior: intermittent request stalls (~15-20% of requests hang with 0
bytes until timeout; fast requests answer <1.5 s) — flakiness, not a block.

Day-1 download therefore EXECUTED on GLM's side, not the founder's machine:

- Runner: `scripts/task52_sandbox_fetch_driver.py` — chunked fetching
  (7 x 730-day chunks per district instead of one 12.7-year request),
  thread pool (<=10 workers), per-chunk 30 s timeout + 3/8/20 s retries,
  chunk-level resumable cache (`.om_cache/chunks/`), deadline-aware
  batching (background processes are killed between sandbox commands).
  5 batches, ~25 min, 413 chunks, 22 transient failures — all recovered
  by retry/next-batch. The delivered `fetch_data.py` itself ran UNMODIFIED
  for final assembly (pure cache hits).
- Outputs (agri_tws_ind/): `ap_ts_weather_openmeteo_full59.csv` —
  273,524 rows (59 x 4636), 2014-01-01..2026-09-10, AP 26 + TS 33;
  `ap_ts_weather_raw_10yrs.csv` — 36,520 rows (Qwen's 10 pilot districts,
  thru 2023-12-31, Day-2-compatible); `weather_download_manifest.md` —
  59/59 districts clean: 0 date gaps, 0 NaNs (ERA5 consolidated through
  2026-09-10; no trailing-NaN padding was needed).
- Verification: `scripts/task52_verify_outputs.py` — 13/13 PASS, including
  a live-API value cross-check (Anantapur 2020-07-15: precip 5.9 mm,
  tmax 29.4 C, tmin 23.8 C, et0 3.25, soil 0.22 — exact match).

Courier reply to Qwen stands, now with provenance: "Script ran
successfully, CSV is saved" — the hardened fork ran (window bug fix +
59-district scope + QA manifest, per the sprint ruling above), not the
original 10-district/2023-end script. Day-2 guardrail unchanged and
pre-stated.

## 2026-09-15 — FEATURE-ENGINEERING COMPLIANCE RULING (GLM, DECISION_LOG-only entry; Task 53)

Trigger: Qwen's five-hat feature-engineering review (courier, 2026-09-15)
asking the founder for 5 decision picks before writing the pipeline. Full
review: agri_tws_ind/FEATURE_ENG_REVIEW_GLM_REPLY.md. GLM verdicts: (1)
feature set B (full water state) — soil moisture CORRECTED to "ERA5-Land
model covariate," not measurement; CGWB cross-check is target-side (D1.1),
not a soil validator; ablation check retained. (2) detrending A — Qwen's B
(raw labels + detrended-climatology baseline) REJECTED: PR-8 froze
detrending the TARGET (tercile of detrended anomaly vs training detrended
distribution; baselines recomputed identically ON the detrended frame);
B breaks forecast/baseline symmetry and re-imports the 77-85% raw-frame
inflation the debate's math quantified; Qwen's "cleaner to explain"
instinct relocated to the PR-8 product-guard rider (trend = product
content). (3) lags: rolling-window backbone (3/6/12m sums of P and P-ET0)
+ point lags 1-3; point lag-6 dropped (opposite-season aliasing); resolves
Qwen's internal Hat-3-vs-Hat-4 feature-count contradiction (~13 features).
(4) interactions E with T11 conditions: pipeline reads regime_map.csv
(regime_map_freeze.py, delivered) when present; MIC >50% heuristic ONLY
behind MVP_HEURISTIC_REGIME flag stamped on outputs; heuristic stratification
never citable as evidence; G13-a season-conditioning stays binding on the
EVALUATION side regardless. (5) scope B; the <5% keep / >15% expand rule
pre-registered BELOW before any measurement.

Binding conditions on the feature-engineering script (extend the Day-2
guardrail; all T7-compatible, logged before Qwen's code exists):
- First LGBM = smoke test, train<=2019 + val 2020-22 ONLY, non-evidence,
  DECISION_LOG entry BEFORE running; test 2023-25 untouched until
  D1.1 -> regime-map freeze -> baseline ladder completes (R7 SS5 order).
- D1.1 CGWB round-month verification remains the FIRST logged evaluation.
- Split constants frozen in code (train<=2019 / val 2020-22 / test
  2023-25); T3 random k-fold tombstone stands.
- PR-8 exactly: per-district linear detrend, TRAIN years only,
  district-mean level, frozen single method; boundaries from training
  detrended distribution; both metrics reported; gates on trend-adjusted;
  0.15-sigma stationarity flag; trend carried as product content.
- Regime source discipline per T11 (frozen map / flagged heuristic / no
  heuristic evidence).
- Scope expansion rule PRE-REGISTERED: <5% skill gain over core set ->
  keep simple; >15% -> expand; 5-15% -> implementer's call, logged either
  way. Thresholds fixed before any measurement.
- Data foundation = ap_ts_weather_openmeteo_full59.csv (verified 2026-09-
  14, 13/13); LATENCY_TABLE v1.1 L13 as-of rules on refetch; Open-Meteo
  free-tier license flag stands.
- SPI/SPEI deferred: fine; when added, gamma fits TRAIN-only, same frame.

## 2026-09-15 — FOUNDATION REPAIR + FE-ROUND-2 COMPLIANCE (GLM, DECISION_LOG-only; Task 54)

Trigger: Qwen round-2 (accepts PR-8 Option A + ERA5-Land correction;
Agro-Met review proposes 5 "blind spots" + 15-feature backbone; requests
sign-off). GLM ran a full-chain audit first. Full text:
agri_tws_ind/FEATURE_ENG_ROUND2_GLM_REPLY.md.

1. DATA DEFECT FOUND + FIXED: Anakapalli (old coord 17.39, 83.01) = masked
   fill pixel (API elevation 0.0 m, sea cell); soil_moisture = 0.0 FILL
   from 2017-01-01 (3539/4636 days). Coord corrected to in-district
   inland point (17.55, 82.95, elev 45 m, live soil verified: 9 legit
   zero days, max zero-run 3 d). Full-window refetch, cache + chunks +
   registry replaced; outputs rebuilt (59/59 clean, 0 gaps/NaNs; 409mm
   Mulugu 2023-07-27 = real flood event, kept). Dead-pixel sweep all 59:
   no other fill pixels. FOUNDATION HASH FROZEN: full59 md5
   bd202d932ab5663fafa0802fd2655ac0; qwen10 md5
   b89e702c56b6e7f2892bd42fe51bfb20 (unchanged). Any refetch = logged +
   hash-compared (ERA5T revisions silently change history). Loader
   condition: cache validation must verify district/state labels (near-
   miss: repair initially wrote wrong state slug; assembly only checks
   row count).
2. STANDING QC (new): physical-range bounds, tmin<=tmax, monsoon
   seasonality, dead-pixel sweep (zero-run >45 d = fill) — script
   scripts/task54_qc_and_feasibility.py, all green.
3. FE ROUND-2 RULINGS: BS1 centroid PARTIAL ACCEPT (concern = MAUP not
   ecological fallacy; 5-point bbox insufficient — bbox corners fall
   outside irregular/coastal districts + builds on 49 unverified
   GLM_APPROX coords; proper fix = GADM/LGD polygon verification +
   in-polygon multi-point sampling WITH the pending coord verification
   + crosswalk, BEFORE regime-map freeze). BS2 extremes ACCEPT
   (empirics: kharif max1day/total p10-p90 0.12-0.31; Spearman vs
   totals 0.845 = modest marginal signal; physical rationale primary,
   ablation decides; max1day = threshold-free PRIMARY; heavy_rain_days
   frozen >=20mm — mean 1.0 d/mo, IMD 64.5 too rare 6.3%; ERA5-Land
   tail-bias label). BS3 ET0-mean REJECTED (not leakage — month length
   deterministic, seasonality is signal; mixing ET0-mean with P-sum =
   dimensional bug; rule: consistent monthly SUMS both terms;
   month_of_year categorical allowed; real calendar channel = as-of
   leakage, already governed L13/L5 + mock-clock CI). BS4 SPLIT
   (nonlinear detrend REJECTED — PR-8 "no menu" rider + 0.15sigma flag
   handle residual non-stationarity; max_dry_spell FEATURE ACCEPT —
   June proxy non-degenerate: mean 7.0 d, p10 3, p90 13; dry day frozen
   <1.0 mm). BS5 ACCEPT as logging requirement (Kc = AKB layer,
   EXPERT_PRIOR-sourced per crop-stage, never tuned; demo copy =
   "regional water anomaly forecast").
4. BACKBONE APPROVED (conditional): their "15" listed 12; corrected to
   16 = 14 core + 2 regime interactions (behind MVP_HEURISTIC_REGIME).
   Fixes: wb_roll3 main effect added (hierarchical principle);
   precip_roll12 restored (recharge memory); month_of_year added.
   Full list in the reply file §II.
5. NEW BINDING ITEMS: (a) DISTRICT-VINTAGE CROSSWALK required BEFORE
   D1.1 loader + regime-map freeze — registry = current boundaries (AP
   26 since Apr-2022, TS 33 since 2016/2019); CGWB/MIC/older admin data
   use old boundaries (AP 13 pre-2022, TS 10 pre-2016); LGD old->new
   mapping with provenance. (b) Spatial-CI discipline: hit-rate CIs NOT
   iid binomial (shared air masses); significance via E-NC D1.2
   variogram/copula machinery. (c) build_features_pr8.py separates
   covariate features / PR-8 target machinery / target ingestion; smoke
   test pseudo-target (e.g., next-month soil tercile) = plumbing check
   ONLY, explicitly labeled non-evidence. (d) Pre-registered thresholds:
   heavy rain >=20mm, dry day <1.0mm, monthly-SUM water balance.
6. Sign-off = CONDITIONAL APPROVAL; conditions 1-8 in reply file §IV.
   Day-2 guardrail stands; D1.1 remains first logged evaluation; test
   years 2023-25 sealed until R7 SS5 order completes.

## 2026-09-15 — INDIAAI/AIKOSH PRIMITIVE VERIFICATION + SCOUT REGISTER (GLM, DECISION_LOG-only; Task 55)

Trigger: founder's casual find — SoulVisionCreations GWL forecasting
model + datasets on IndiaAI AIKosh; Qwen proposes "Primitive-First
Architecture" (D#19): adopt model as primary groundwater primitive,
skip CGWB acquisition/EnKF, restructure MVP onto their MWS panel, 10-day
timeline. GLM verified everything against the repo before ruling (full
report: agri_tws_ind/INDIAAI_PRIMITIVE_VERIFICATION_GLM_REPLY.md).

VERIFIED FACTS (repo LICENSE/MODEL_CARD/DATA_SOURCES/TRAINING.md
fetched directly):
- License: DISCREPANCY in repo — root LICENSE file = CC BY-4.0 text,
  MODEL_CARD table says weights+code Apache-2.0; training dataset
  GODL-India; Prithvi-EO-2.0-300M Apache-2.0 (HF/NASA-IMPACT
  confirmed). Both permissive commercially w/ attribution — no blocker,
  but one-line author clarification required IF adopted. Open-Meteo
  inference conditioning = non-commercial free tier (L13 flag applies
  to THEIR live path too).
- Provenance: private org repo, created 2026-07-26, 0 stars, last push
  2026-09-03. AIKosh = registry listing (SPA pages, not statically
  verifiable). Data government (CGWB/WRIS); model community. NOT
  "government-backed."
- Skill (their own honest card): per-well median R2(delta) ~0.25 (3m) /
  0.275 (6m); direction ~66% (tiny) to ~88% (>2m); ~60% sub-0.5m moves;
  3-way call ~54%. Imagery ablation: Prithvi encoder "roughly neutral on
  aggregate skill." Their trusted unit = direction+tier+confidence
  (independently validates our tercile+confidence advisory design).
- Operational deps (unmentioned by Qwen): live GEE auth+project; live
  Open-Meteo; NWDP/WRIS/local-CSV GWL source; LOCF staleness <=200d.

CONTAMINATION RULING (binding): their split TRAIN_END=2024-12-31, VAL
2025-01..2025-08, TEST >=2025-09-01. Their training set covers 100% of
our val (2020-22) and test years 2023-24; their val covers 2025-01..08;
their reliability calibration constants fit through 2025. Therefore:
(1) their forecasts INADMISSIBLE as covariates in anything we score on
val/test; (2) head-to-head vs our ladder on our splits = rigged (their
in-sample, our out-of-sample); (3) "correlation >0.6 -> fuse" =
diagnostic only, admissible solely on data neither model has seen.

D#19 AS DRAFTED = REJECTED (adopt-as-primary + foundation swap + MWS
evaluation unit change). ADOPTION IS A GATE OUTCOME, NOT AN ARCHITECTURE
DECLARATION (T2/T7; G1-G13 apply to pre-trained models too).

CORRECTED PLAN (binding):
1. DATASET (gwl_data.csv, ~3.3M readings, 10,411 stations, pre-joined
   features) = approved acquisition for D1.1 ACCELERATION, conditional
   on provenance verification (sample wells re-checked vs India-WRIS
   raw: values/dates/units/round-month structure) BEFORE D1.1
   conclusions rest on it. D1.1 stays the FIRST logged evaluation.
2. MODEL = candidate for the groundwater leg via ADMISSIBLE paths only:
   Path A = their 2025-09+ test split re-scored under OUR metrics
   (PR-8 trend-adjusted tercile frame, our baseline ladder); Path B =
   operational forward evaluation (both models forecast; score vs CGWB
   as readings arrive, PR-6/latency rules).
3. THREE-ARM COMPARISON PRE-REGISTERED (log-before-run): Arm 1 their
   model / Arm 2 our ladder / Arm 3 fusion with disagreement surfaced
   via E-NC D1.2 machinery. Stratified per PR-7 (pumping districts
   primary). Qwen's "show both with confidence bands" instinct adopted
   formally under E-NC.
4. Our covariate pipeline UNTOUCHED (own Open-Meteo foundation,
   hash-frozen); target machinery PR-8 unchanged; frozen evaluation
   order unchanged (D1.1 -> regime-map freeze -> baseline ladder ->
   LGBM smoke test -> gates); test 2023-25 seal unchanged; Round-2
   16-feature backbone unchanged.
5. NDVI/MWS dataset = v1.5 advisory-rendering candidate only; district-
   level evaluation stays; third-party as-of audit + vintage check if
   ever used as covariates.
6. EnKF spec stays a standing artifact; whether it's needed is a ladder
   question, not a decree. Their repo = reference implementation for
   as-of/staleness discipline (LOCF notes, future-anchor rejection).

SCOUT REGISTER INSTITUTIONALIZED: agri_tws_ind/SCOUT.md (S1-S10).
New verified finds this pass: IMD 0.25deg daily gridded rainfall
1901-2024 free (imdpune.gov.in) = STRONG acquisition candidate (precip
cross-validation vs ERA5/OM + independent covariate; commercial terms
check pending); CGWB direct access confirmed (India-WRIS/NWIC/WIMS);
GRACE mascons JPL RL06.3_v04 + CSR RL06.3 free NetCDF (target lane);
Kuruva et al. 2025 Nature Sci Data QC'd CGWB GWL dataset (D1.1
cross-check #2); Agmarknet real, no official open API (v1.5, D#13);
Prithvi Apache-2.0 confirmed. Queued: Wadhwani AI, Bhuvan/NBSS&LUP,
Soil Health Card, NIPHM/NCIPM, IMD temperature/ET0 gridded, SMAP L4 /
ESA-CCI soil moisture, IIT/NIH GW research artifacts.

## 2026-09-15 — ASSET REGISTRY CONSENSUS + QWEN SYNTHESIS VERDICT
## (GLM, DECISION_LOG-only; Task 56)

Trigger: founder relayed Qwen's corrected IndiaAI synthesis (dataset
high-value / model conditional / competition models as surface-water
layer / A-B-C next-move question) + directed a full domain-hat +
adversarial/skeptic scout loop: "anything we're missing before Phase
2?" Process: 3 parallel domain scouts (hydro-climate-EO, agri-socio,
models-libs-infra; web-verified, ~30 searches) -> guardian draft ->
independent adversarial reviewer (25 objections, 6 BLOCK) -> consensus
Round 2 (all resolved). Full report:
agri_tws_ind/ASSET_REGISTRY_CONSENSUS_GLM_REPLY.md; register SCOUT.md
S11-S30.

QWEN SYNTHESIS: RATIFIED with 6 amendments (AM-1..AM-6 in report).
Key sharpening: their train+val+test ALL sit inside our sealed
2023-25 (TRAIN_END 2024-12-31, val ->2025-08-31, test >=2025-09-01,
calibration through 2025) -> nothing in calendar 2023-25 admissible
for ANY arm; only jointly-honest window = 2026-01-01+.

DECISIONS (binding):

D#20 — DISTRICT BASIS + CHANGE LEDGER (label-defining, pre-D1.1):
**AMENDED 2026-09-23 — see Addendum 8 (founder directive, closes
the Issue #1 skeptic round): TARGET basis = 61 districts (2026
ground truth); EXECUTION basis = 59 as-built until 61-grade
polygons are ingested (auto-promotion rule in Addendum 7/8).**
Constant target basis = current 2026-vintage 59 districts (AP 26 +
TG 33); wells assigned by point-in-polygon (wells are points).
Dated ledger mandatory: TG 10->31 on 2016-10-11, TG 31->33 on
2019-02-17 (both INSIDE TRAIN window), AP 13->26 on 2022-04-04
(inside VAL window). Old-basis sensitivity check on TRAIN-era labels
before D1.1. Polygon precedence: LGD > Bhuvan > community repos, all
license-recorded; GADM EXCLUDED ENTIRELY (non-commercial; no
internal-reference carve-out — it taints the deliverable). No
official crosswalk file exists — build it; test cases for each
reorg date.

D#21 — gwl_data.csv PHYSICAL SPLIT: open extract (rows <=2022,
hashed, free use) + sealed remainder (locked dir, separate hash,
logged access, written unseal procedure). Views alone are NOT
enforcement. D1.1 computed on <=2022 rows only. Provenance doc +
India-WRIS raw spot-check (Task 55 condition) + Kuruva-QC
provenance stated. duckdb = panel workhorse (ASOF JOIN = executable
as-of discipline; MIT).

D#22 — THREE-ARM PRE-REGISTRATION (log-before-run): window
2026-01-01+ ONLY. Blanket rule: Soul Vision outputs (forecasts,
hindcasts, residuals, embeddings) NEVER covariates/stacker inputs/
tuning signals for anything trained/tuned/scored on our val 2020-22
or test 2023-25; admissible only in 2026+ contexts. Fusion arm =
advisory-level combination on 2026+ only. Metrics pre-pinned:
primary = district-aggregated dGWL -> per-district trend-adjusted
tercile hit + reliability; secondary = CRPS on district-aggregated
ddepth; well->district aggregation rule pinned; 2026+ tercile edges
= frozen <=2019 trend extrapolation + drift-sensitivity note;
calibration asymmetry disclosed; 2026 published-round census from
India-WRIS BEFORE any arm runs; modest-n decision rights pre-stated
(advisory pilot vs adoption vote — adoption needs Path B
accumulation). Weights download deferred until this pre-reg logged.

D#23 — LADDER RUNG SET FROZEN ONCE (before D1.1 completes):
{climatology, seasonal-naive, trend-adjusted persistence,
Chronos-Bolt (variant + config + weights hash pinned; pretraining
corpus enumeration checked for India-hydrology series; Apache-2.0;
CPU-feasible)}. TimesFM 2.5 deferred (compute); TimesFM-3
hosted-only = OFF-LIMITS (breaks self-contained reproducibility).
statsforecast (MIT) acceptable for naive rungs if preferred; pin
choice. NO rung additions after any val score exists
(anti-rung-shopping).

D#24 — ASSET REGISTRY CONSENSUS: T1 collapsed to critical-path six
(crosswalk+split / duckdb / pyet / IMD 1-deg temp / Chronos-Bolt /
MSP table); everything else T2+ (downloads of cheap ones fine,
engineering deferred). FEATURE FREEZE reaffirmed: new data =
validation/consistency-check ONLY in Phase-1; additions = logged
amendment + ladder re-run. Regime-map permitted inputs declared:
Open-Meteo foundation + frozen backbone only. Terminology: SEALED =
GWL label statistics 2023-25; AS-OF = feature availability with
publication lag (reservoir levels/APY/sowing = legitimate as-of
covariates with lag; seal protects label stats, not the calendar).
MCP servers: SKIP Phase-1 (nasa/earthdata-mcp only credible find;
ecosystem hobby-grade; agent-mediated access hostile to as-of/
provenance; re-eval Phase-2+). Moirai license record CORRECTED
(likely Apache-2.0 per uni2ts repo/PyPI; earlier NC claim withdrawn
pending direct LICENSE read; deferral stands on ladder-freeze
grounds). New T2 additions (adversarial reviewer finds, accepted):
CGWB Dynamic GWA district tables (SoE/net draft, 2017/2020/2022/
2023 editions — pumping-context primitive for PR-7, ~1yr pub lag =
lagged covariate); Minor Irrigation Census (6th 2013-14, 7th
2023-24 — well counts/densities, D1.1 sanity); SoilGrids 2.5 +
GSI geology (subsurface priors — previously a glaring hole);
observational water-balance anchors (CWC discharge, IMD pan-evap).
KCC corpus REFRAMED: retrieval corpus for advisory delivery
(Phase-2), NOT an AKB source (AKB stays EXPERT_PRIOR; candidate
sources: ANGRAU/PJTSAU archives, ICAR-CRIDA contingency plans).
Prithvi-EO-2.0 = in-hand CANDIDATE (adoption = gate outcome), not
"frozen primary". TerraClimate corrected: 1958-present, THREDDS
authoritative, citation-requested terms (not CC0), "consistency
check" not "cross-validation"; MPC liveness check on first use.
Founder one-time actions when three-arm scheduled: India-WRIS
2026-rounds census; Open-Meteo pricing read (L13); commercial
licensing checklist for IMD/WRIS/Bhuvan terms.

CRITICAL PATH ORDER (Qwen): crosswalk + physical split FIRST ->
duckdb panel -> pyet ET0 -> IMD temp consistency -> backbone build
(Task 54 conditions, pseudo-target smoke = plumbing only) -> D1.1
(<=2022 rows) -> regime freeze -> ladder (rung set D#23) -> LGBM
smoke test (Day-2 guardrail) -> gates. No model run of any kind
before its registered position (AM-6).

## 2026-09-15 — Q57-A ADVERSARIAL VERIFICATION ROUND (GLM, DECISION_LOG-only; Task 57)

Trigger: founder relayed Qwen's Task-57 courier (complementary scout pass
Q1-Q13 + implementation directives B-E). GLM ran Round Q57-A per directive A.

VERIFICATION EXECUTED: 30 web queries (28-query batch, all first-pass OK) +
direct license reads (PyPI JSON, GitHub repos API, HuggingFace card API) +
LGD portal probe + D#20/D#21-prep builds with self-tests. Evidence archived:
agri_tws_ind/q57a_evidence/ (queries TSV, digest, license audits).

VERDICTS (full table + sources: agri_tws_ind/Q57A_VERDICTS_GLM_REPLY.md):
- ACCEPT: Q1 xskillscore (Apache-2.0 — NOT BSD as commonly assumed; PyPI
  v0.0.29 + conda-forge agree), Q9 PMFBY CCE (validation anchor only,
  vintage+revision riders, triangulate w/ S16 UPAg), Q10 NAQUIM (context
  table, no pipeline dep), Q11 Livestock Census/THI/IGFRI (3 riders: 21st
  LC delayed -> 20th w/ vintage; THI bands need named ICAR/NDRI citation;
  IGFRI catalogue cached w/ retrieval date), Q13 Sentinel-1 (T3 backlog).
- AMEND: Q2 -> MAPIE only (BSD-3, active; nonconformist REJECT: last
  release 2017, stale 9 yrs; Mondrian = per-bin in-house per PR-10);
  Q3 -> register `exactextract` (Apache-2.0; "pyexactextract" = 404 PyPI +
  404 GitHub, name is dead; LGPL worry mooted); Q4 split -> pixi ACCEPT
  (BSD-3) w/ pixi.lock, conda-lock REJECT (redundant w/ pixi.lock), DVC
  ACCEPT w/ second-regeneration trigger; Q5 -> pin-at-fetch (SEAS6 in
  implementation per ECMWF 2025 materials; CC-BY-4.0 + ECMWF ToU since
  2025-07-02; member-asymmetry record; pilot-12 + monthly stream only);
  Q6 -> IITM ERF role demotion to T2 operational reference (no API, no
  lead-14 hindcast archive -> un-backtestable -> not a G13 gate input;
  S28 WeatherNext keeps h=1-4); Q7 -> Beckn license-murk flag (beckn.io
  sharing code = CC-BY-NC-SA 4.0; spec-repo LICENSE read at activation);
  Q8 -> decision tree w/ per-artifact reads (IndicConformer HF card =
  CC-BY-4.0 commercial-OK; IndicTrans2 HF card MIT but GitHub LICENSE
  pending -> conflict-suspect; Indic-TTS gated 401; fallbacks: Bhashini
  API founder-read -> Whisper-large-v3 Apache-2.0); Q12 -> SACHET CAP
  portal (sachet.ndma.gov.in, NDMA/CDOT pan-India) as the alert primitive
  over IMD-page scraping + trigger-role demotion (corroboration/escalation
  only; triggers from backtestable registered stack).
- AM-5 INTACT: no accepted/amended item touches the frozen 16-feature
  backbone or regime-map inputs.
- CATCHES c1-c6: consensus rows start S33 (SCOUT at S32, not S30); pyet
  drift (v1.5.0 exists, re-pin at env build); scikit-learn unregistered
  (register at ladder time); S29 partially resolved (2 of 3 voice licenses
  read); puncc license hole; CDS attribution-string rider.

IMPLEMENTATION (Part B, frozen order):
- D#20 DELIVERED as far as data permits (agri_tws_ind/d20/): district_basis
  .csv (59, 2026-vintage; LGD codes PENDING_FOUNDER_EXPORT), change_ledger
  .json (3 verified events: TG 2016-10-11 10->31, TG 2019-02-17 31->33,
  AP 2022-04-04 13->26 [gazette 472-497]; 1 unverified rename cluster
  flagged; provisional parentage; tests T-D20-1/2/3), d20_assign_wells.py
  (SELFTEST PASS 5/5 incl. unassigned-with-reason paths),
  D20_SENSITIVITY_NOTE.md. MANIFEST.sha256 logged.
- D#21 PRE-STAGED: d21_physical_split.py (MOCKTEST PASS: boundary semantics
  2022-12-31 open / 2023-01-01 sealed; fail-loud unparsable-date exclusion;
  reconciliation; SHA256 manifest; sealed-dir lock; embedded unseal
  procedure + ACCESS_LOG stub).
- BLOCKED-ON (no silent fixes): gwl_data.csv ABSENT from sandbox (well PIP
  + split cannot run); LGD portal captcha-gated (founder one-time export,
  state codes AP=28/TG=36, or PENDING markers); LGD/Bhuvan-grade polygons
  not in sandbox (GADM banned; OSM community-grade only w/ logged
  amendment).
- Steps 3-8 queued untouched (AM-6). No model runs. No weights downloads
  (D#22 pre-reg not yet triggered).

SANDBOX ACCOUNTING: 9th split-brain reset caught (/home mirror at Task-40
state); /tmp canonical through Task 56; re-anchored; new artifacts synced
to both mirrors + founder-facing directory. gwl_data.csv absence = the
critical-path blocker; founder holds the local copy.

NEXT: Qwen Q57-B (resolve/concede per item; GLM pre-stated concede-points
i-iii in the reply); consensus logs S33+ / D#25. Founder moves: relay
reply; re-upload/approve gwl_data.csv; LGD one-time export; polygon
handoff ruling.


---

## ADDENDUM — 2026-09-15 (GLM, post-Task-57 session)

- 57-netprobe (interlude, no number): live egress probe. CDS/GitHub/PyPI/
  Zindi-API reachable from sandbox; WRIS/IMD/data.gov.in/IndiaAI/Kaggle
  refuse. Logged; no plan change.
- 57-KIT: founder import kit built (LGD browser fetcher, gwl handback
  checker with D#24-safe receipt — SELFTEST + 20k-row e2e PASS, polygon
  handoff notes). No decision consumed; AM-6 order intact.
- 57-KIT-LGD: LGD district codes OBTAINED by GLM via real headless-browser
  DWR on the portal's own citizen page (read-only; captcha gates only the
  report POST, untouched). district_basis_v2.csv: 59/59 backfilled,
  0 unmatched, per-row match_method audit. FINDING: AP registry lists 28
  districts (Markapuram LGD-790, Polavaram LGD-791, eff 2025-12-29) vs
  D#20's 26 — verified via 4+ independent sources. D#20 amendment
  59->61 routed to Qwen consensus (courier addendum in lgd_live/).
  Sealed-window logic unaffected.
- FOUNDER RULING (chat, 2026-09-15): program state + data persist in a
  founder-owned PRIVATE GitHub repo. ARRIVAL=COMMIT rule adopted: data
  landing in the sandbox is committed (or Release-asseted + manifest-
  committed) as the FIRST action, before pipeline work. Implemented same
  day: program/ tree, tag program-2026-09-15, full + incremental bundles, push card.

## ADDENDUM 2 — 2026-09-15 (GLM; 10th split-brain reset + PRP adoption)

- 10TH SPLIT-BRAIN RESET (mid-task, between tool calls): /home tree
  regressed to Task-40 git state (0fd747f); program/ tree, session
  worklog, founder-facing download/ and scripts/ wiped; 57-GIT commits
  absent from .git. RECOVERY (same session, ~15 min): fetched
  agri_tws_ind_repo_full_2026-09-15.bundle from /tmp canonical ->
  main restored bit-exact to 8446eef; tag program-2026-09-15 restored
  (d80fa9c -> 5f12889, matches bundle); github-release intact; working
  tree clean vs 8446eef. Session worklog restored from /tmp sync (207
  lines, includes 57-REPO-EXPLAIN). DECISION_LOG unified at 810 lines
  both copies. NET LOSS: zero commits, zero program content. One gap
  found + fixed: program/data/README (persistence policy) had been
  left UNTRACKED pre-reset — recreated from DECISION_LOG text and now
  committed. Session-only scratch scripts (probe one-liners) lost;
  all reproducibles were already tracked in program/scripts/ (7 files).
  Lesson logged: the persistence architecture worked exactly as
  designed; the founder push remains the only missing piece.
- FOUNDER RULING (chat, 2026-09-15): professional productionization
  discipline for finalized models -> PRODUCTION_READINESS_PROTOCOL.md
  v1.1 adopted (program/): model classes A/B/C; staging ladder L0-L4
  (L1 split into L1a frozen-input / L1b live-input determinism) with
  PRD-numbered gates; hat/role matrix (process + domain incl.
  farm-economics + Telugu/localization owners) where EVERY role has a
  mandatory adversarial twin, enforced via per-gate signatory matrix —
  no unopposed agent output is gate evidence; subagent mapping
  (code-triggered, invocation pattern: charter -> deliverable ->
  adversarial pass by a different agent -> verdict -> worklog);
  chess-moves strategy incl. threat 5 advisory-harm; Claude loop +
  cross-AI courier triggers; trigger adjudication (founder statement
  logged + twin-countersign, no agent self-declaration); machine-
  checkable PRP_LADDER_REGISTRY.csv. AM-6 order untouched — protocol
  pre-registers the production path, does not trigger it. ADVERSARIAL
  ROUND 1 executed as Task 57-PRP-A1 (fresh-context agent): verdict
  ADOPT-WITH-AMENDMENTS, 3 BLOCK + 8 AMEND + 1 NIT, all 12 accepted
  and incorporated in v1.1 (record + dispositions:
  program/PRP_ADVERSARIAL_ROUND1.md). CORRECTION LOGGED: the first
  version of this addendum prematurely stated the round as already
  executed before it ran — caught by the round itself as Block 1;
  standing rule adopted: a gate record may never reference an
  uncommitted artifact. No decision number consumed (PRD gates
  numbered separately from D#/G#/T#/PR-n/AM-n).

## ADDENDUM 3 — 2026-09-16 (GLM; 12th reset recovered + gwl_data ARRIVAL + D#21 executed)

12th split-brain reset at session open (/home reverted to container
scaffold). Recovered zero-loss from /tmp canonical + full bundle
(HEAD e9539f9); stale Sep-10 seed parked at /tmp/stale-seed-sep10.
Mirror repair: /tmp mirror never carried program/ — now it does
(raw/ excluded as regenerable from the zip).

ARRIVAL (Task-55 critical-path blocker CLEARED): founder uploaded
the AIKosh download to public HF datasets under han-jisso; sandbox
downloaded ground_water_level_all.zip (265,308,904 B), sha256 == HF
LFS oid 1e9d0cf6... (bit-perfect). Founder-supplied API key never
authenticated (401; datasets public — no credential used or
stored). ARRIVAL=COMMIT per program/data/README.md: >95MB class =
Release-asset route, manifest committed in-tree BEFORE pipeline
work (commit c83f313). Zip custody: release_staging/ + /tmp
mirror; Release upload deferred until after founder push.

CENSUS (gwl_handback_check.py, sandbox side): gwl_data.csv
762,708,010 B, sha256 c1a2e1b3...; 3,278,227 rows; date col
'date'; 0 unparsable; 10,411 wells; 476 district labels (all-India
file — AP+TG reconciliation belongs to D#20 well assignment;
warning logged by design); range 1976-05-01..2025-12-09;
open-period value stats sealed-safe (D#24).

D#21 EXECUTED (d21_physical_split.py): open 992,053 rows
(<=2022-12-31) sha256 c73ae81f...; sealed 2,286,174 rows
(>=2023-01-01) sha256 71f492b2...; reconciliation PASS; sealed dir
locked (dr-x------) with unseal procedure + ACCESS_LOG stub;
tracked contract copies under program/data/manifests/
sealed_contract/. Parquet of OPEN rows only:
parquet/gwl_open.parquet (30 MB zstd, 992,053 rows, duckdb 1.5.5).
AM-6 freeze order untouched; no model runs; D1.1 unblocked on open
rows only.

SoulVision NDVI model asset (ndvi_forecasting_model_dataset.zip,
647,774,722 B, LFS sha256 be16545e...): available on founder HF —
NOT downloaded; D#22 weights ban in force (three-arm pre-reg not
triggered). Recorded in program/data/manifests/
ndvi_model_availability.json.

Founder advisories (no new tasks): (1) flip both HF datasets to
PRIVATE — AIKosh participant licensing (visibility rule,
program/data/README.md); (2) pasted API key never authenticated —
delete it in HF settings if it exists; (3) push card unchanged
(full bundle + incr 2026-09-16 refreshed).

## ADDENDUM 4 — 2026-09-16 (GLM; duckdb panel on open rows + provisional D#20 assignment)

Registered critical-path slot after D#21 (crosswalk + split -> duckdb
panel) EXECUTED as Task 59. Input: parquet/gwl_open.parquet (open
rows only, D#24-safe). Filter state IN (AP, TG): 365,742 rows,
3,223 wells.

D#20 EXECUTION AMENDMENT (logged, not silent): LGD/Bhuvan-grade
polygons remain absent from sandbox, so point-in-polygon (T-D20-2)
CANNOT run. Empirical structure verified first: district labels are
CONSTANT per well (0/3,223 multi-label), back-applied across all
readings; label vintage = TG ~2019-era / AP pre-2022-13 spellings.
Assignment therefore uses an EXPLICIT per-label map (47 observed
labels, enumerated; fail-loud on unknowns — caught PEDDAPALLY
Y-variant + case-variant alias bugs on first run). Parent-label
wells (undivided old districts) are assigned to the CONTINUING
2026-basis district and flagged PARENT_CONTINUE with the child set.
TARGET BASIS UNCHANGED (2026-vintage 59). Polygon binding re-runs
assignment; until then ambiguity is flagged per well, never hidden.

D#20 SENSITIVITY CHECK (pre-registered "runs when data lands"):
label-frame vs 2026-basis-frame max |district-month mean diff| =
0.0 across 8,067 overlapping keys — zero BY CONSTRUCTION (1:1
continuing-name mapping; the anticipated vintage frame does not
exist in the file). Genuine uncertainty = PARENT_CONTINUE share:
2,471/3,223 wells (56.8% of open AP+TG rows); 14 basis districts
(13 AP-2022 children + Yadadri Bhuvanagiri) hold zero wells until
polygons arrive. Mismatch log only; basis NOT reopened.

NEW DATA-QUALITY QUESTION (routed to existing Task-55 provenance
condition; no new decision): gwl_value in AP+TG open rows has
114,062/365,742 negatives (31%), range -1147.7..+971.3 m — unit or
datum ambiguity (mbgl vs masl vs QC failures) must be resolved via
the India-WRIS raw spot-check BEFORE D1.1 conclusions rest on the
values. Report-only; nothing dropped or imputed (AM-3).

Deliverables: program/data/parquet/panel_open_readings.parquet
(365,742 rows, district_2026 + flags), panel_open_district_month
.parquet (8,067 district-month rows, 45 districts), manifests/
panel_build_report.json (full mapping table applied, sensitivity,
month-of-year census per district = D1.1 feed, value histogram),
program/scripts/build_panel_d20.py (--selftest PASS). SoulVision
pre-joined covariates carried REFERENCE-ONLY (corrected plan
item 4: our hash-frozen Open-Meteo foundation stands). AM-6
untouched; no model runs; D#22 weights ban intact.

## ADDENDUM 5 — 2026-09-17 (GLM; 13th reset recovered; REPO LIVE ON GITHUB; history-rewrite disclosure)

13th split-brain reset between sessions; first push accidentally
sent the scaffold (44ae7ad) before the reset was noticed; disk
100% full compounded it. Recovered zero-loss from /tmp mirror +
full bundle (through 3399cda); force-push replaced the scaffold
push.

HISTORY REWRITE (disclosure, binding): GitHub GH001 blocked the
275.73 MB blob data/Train (1).csv (early competition-era commit,
pre-dating the /data/ exclusion policy). Resolved via
git-filter-repo path-strip: 36 commits preserved, none pruned, no
blob >100MB remains. CONSEQUENCE: every commit hash from the
repo's beginning through 2026-09-17 changed (old tip 3399cda ->
new tip 1d87a51). Standing rule: all pre-2026-09-17 commit hashes
cited anywhere in DECISION_LOG/worklogs/manifests refer to the
OLD lineage, preserved bit-exact in
agri_tws_ind_repo_full_2026-09-15/16.bundle (/tmp mirror +
founder chat downloads). Tags program-2026-09-15 and v28-endgame
pushed under their names with rewritten objects. github-release
branch = orphan squashed endgame package 76d252a (founder's Sep-10
local commit) — restored VERBATIM, hash unchanged.

REPO LIVE: github.com/SKJNR/agri-tws-ind (PRIVATE; founder rule:
never public without asking). GLM holds a founder-issued
fine-grained PAT (single-repo, Contents+Issues RW, 90-day) and
pushes autonomously each round; token lives only in .git/config
(untracked). AI-DEBATE ARENA ACTIVE: Issue #1 =
https://github.com/SKJNR/agri-tws-ind/issues/1 (D#20 59-vs-61
ruling; provisional-assignment countersign; data findings; PRP
v1.1 review). DECISION_LOG remains the single source of truth;
issues are the debate venue. No decision numbers consumed this
addendum; AM-6 untouched; no model runs; D#22 weights ban intact.

## ADDENDUM 6 — 2026-09-18 (GLM; 14th reset recovered FROM GitHub; repo health check; Issue #1 Round-1/3 skeptic verdicts recorded)

14th split-brain reset between sessions; recovery source for the
first time = GitHub remote itself (clone --no-checkout + .git swap
+ reset --hard; playbook #4). Zero loss; remote at 30d7fa1. The
reset-risk class is now closed by construction — the mechanism
Addendum 5 stood up did its job.

REPO HEALTH CHECK (founder-requested): main + github-release +
tags (v28-endgame, program-2026-09-15) + Issue #1 all live and
current. Secret scan of all tracked text files: CLEAN (PAT lives
only in .git/config; .env = local DB path only). Sealed-data
check: ZERO gwl data files in the repo (manifests + sealed-
contract docs only; parquet/ never pushed). One discrepancy:
REPO IS PUBLIC as of the 2026-09-18 API check, violating the
private-by-rule recorded in Addendum 5 + FOUNDER_PUSH_CARD
(GitHub defaults new repos to public; the card asked for private —
likely an accidental default at creation). All current content
was pushed 2026-09-17 under the same exposure; nothing new added
since. Flagged to founder with click-steps (Settings -> Danger
Zone -> Change visibility -> Private); PAT scope (Contents+Issues
RW only) cannot change visibility itself — founder action
required. This addendum is the standing record of the discrepancy
until the founder confirms the flip.

ISSUE #1 — FIRST EXTERNAL VERDICT RECEIVED (founder transcribed;
comment 2026-09-18T04:46:27Z): SKEPTIC · GEMINI 3.1 PRO ·
ANTIGRAVITY · ROUND 1/3, all four asks answered.

- Ask 1 (D#20 59-vs-61): ACCEPT A — amend to 61-district constant
  basis. Evidence cited: live LGD registry confirms Markapuram
  (790) + Polavaram (791) effective 2025-12-29. Reasoning: 2026
  users need 2026 ground truth; sealed-window logic robust because
  both new territories were contained within parent districts
  through the 2023-2025 window. Self-counter: upstream sources may
  lag LGD, requiring backward-compat mappings anyway (59 simpler
  short-term).
- Ask 2 (provisional label-map assignment): COUNTERSIGNED —
  fail-loud on unknown labels + per-well PARENT_CONTINUE flags =
  correct defensive posture. Self-counter: string-matching may
  mask coordinate errors a polygon intersection would catch.
- Ask 3 (data findings): NOTED, no design objections — district
  keying from assignment layer only; 31% negative gwl_value rows
  quarantined UNRESOLVED pending Task-55 WRIS spot-checks.
- Ask 4 (PRP v1.1): APPROVE — gate ladder auditable; zero-pollution
  landing rule sound. Self-counter: adversarial-twin + round
  budgets add latency for operational hotfixes; use the P0 trivial
  class aggressively.

GLM PROCESSING (this addendum, no new decision numbers):
- Asks 2/3/4: external review recorded and ACCEPTED into the log —
  provisional assignment now GLM-logged (Addendum 4) + externally
  countersigned; PRP v1.1 stands externally approved with the P0
  note carried as operational guidance; data findings concur.
- Ask 1: recorded as EVIDENCE INPUT, round 1 of 3. FINAL D#20
  RULING DEFERRED until rounds 2-3 land (or founder closes the
  round early). Rationale: label-defining decision on a single
  external verdict violates the multi-hat spirit; Gemini's own
  self-counter (upstream lag) is material; waiting costs nothing —
  D1.1 is not gated on the basis choice until backbone build.
- AMENDMENT EXECUTION PLAN pre-staged (runs the moment the ruling
  lands): extend district_basis by Markapuram (790, parent
  Prakasam) + Polavaram (791, parents Eluru/NTR/W.Godavari);
  re-run build_panel_d20.py; expected per current evidence (verify
  on execution): 0 wells change assignment (both new districts
  receive wells only via polygon binding, same as the existing 14
  zero-well basis districts), panel district-month rows unchanged
  (45 active districts), basis ledger +2 entries.

No decision numbers consumed this addendum; AM-6 untouched; no
model runs; D#22 weights ban intact.

## ADDENDUM 7 — 2026-09-18 (GLM; Issue #1 Round-2/3 Qwen verdict recorded; R1+R2 reconciliation; D#20 ruling pre-staged)

ISSUE #1 ROUND 2 RECEIVED (founder transcribed; comment
2026-09-18T21:23:33Z): QWEN (SKEPTIC HAT) — attack on the
61-district amendment's EXECUTION RISK. Verdict: ACCEPT 61 as the
LOGICAL TARGET BASIS, MANDATE the 59-district provisional map as
the immediate PHYSICAL EXECUTION BASIS. Decoupling rule proposed:
PIP pipeline runs on 59-basis now; on acquisition of 61-district
polygons (hashed + ingested), the pipeline auto-reruns and
promotes the execution basis to 61. Log the decoupling in D#20.

R1+R2 RECONCILIATION (no contradiction found): both rounds
concord on the logical target (61, 2026-vintage). Qwen's
decoupling ANSWERS Gemini's R1 self-counter (upstream sources lag
LGD): execution never blocks on laggy upstreams; promotion is
event-driven on polygon acquisition. R2 refines R1 into an
executable rule rather than opposing it. Qwen's own framing
("might be stuck behind portal CAPTCHAs or delayed government
releases") matches the logged reality that polygons are absent
from the sandbox and are a founder-side gate.

GLM NOTE — AS-BUILT ALREADY SATISFIES THE R2 MANDATE: the
Addendum-4 provisional assignment IS a 59-district physical
execution basis with fail-loud labels and per-well ambiguity
flags. Qwen's "unblocking physical step" is the current state of
the repo. Zero rework required; the mandate is met by
construction, not by change.

D#20 RULING PRE-STAGED (executes the moment R3 lands without
contradiction, OR founder closes the round early; replaces the
Addendum-6 amendment plan as the operative version):
- TARGET BASIS (logical, 2026-vintage): 61 districts = 59 basis
  + Markapuram (LGD 790, parent Prakasam) + Polavaram (LGD 791,
  parents Eluru/NTR/W.Godavari).
- EXECUTION BASIS (physical, as-built): 59 districts, provisional
  label map per Addendum 4, unchanged until polygons.
- PROMOTION RULE: 61-grade polygons acquired + hashed + ingested
  -> PIP re-runs (T-D20-2) -> execution basis promotes to 61 ->
  basis ledger +2 entries; expected 0 well reassignments (both
  new districts receive wells only via polygon binding — same
  class as the existing 14 zero-well basis districts); district-
  month panel active-district count unchanged (45) until wells
  land in the two new districts. All expectations verified on
  execution, not assumed.
- EFFECT BEFORE POLYGONS: none — no artifact rebuilds, no panel
  changes; D1.1 not gated (backbone work proceeds on 59-execution).

STATUS: 2 of 3 skeptic rounds in. Deferral maintained per
Addendum 6. Founder may close early ("close the round") — the
ruling then executes mechanically from this pre-staging.
Housekeeping: unpushed platform auto-commit (bare gitlink to the
stale seed repo, no content) dropped before push; my-project/
added to .gitignore. No decision numbers consumed this addendum;
AM-6 untouched; no model runs; D#22 weights ban intact.

## ADDENDUM 8 — 2026-09-23 (GLM; D#20 AMENDMENT RULED + EXECUTED — founder directive closes the Issue #1 skeptic round)

FOUNDER DIRECTIVE (courier 2026-09-17, received 2026-09-23):
CLOSE THE ROUND. R1 (Gemini 3.1 Pro) and R2 (Qwen Skeptic)
converged; decoupled strategy correct; provisional map already
executing; do not wait for R3 absent a novel blocking edge case;
proceed immediately to Weather Backbone on 59-execution; sealed
lock + AM-6 order maintained.

D#20 AMENDMENT EXECUTED per the Addendum-7 pre-staging (this
addendum is the operative ruling; R3 waived by founder):
- TARGET BASIS (logical, product truth): 61 districts = the
  2026-vintage 59 + Markapuram (LGD 790, carved from Prakasam,
  effective 2025-12-29) + Polavaram (LGD 791, carved from
  Eluru/NTR/W.Godavari, effective 2025-12-29).
- EXECUTION BASIS (physical, as-built): 59 districts, provisional
  label map (Addendum 4), UNCHANGED. Every panel, manifest, and
  downstream artifact remains valid without rebuild.
- PROMOTION RULE (event-driven, runs once): 61-grade polygons
  (LGD > Bhuvan precedence per D#20) acquired + hashed + ingested
  -> T-D20-2 point-in-polygon re-runs -> execution basis promotes
  to 61 -> basis ledger +2 entries; Markapuram/Polavaram wells
  enter ONLY via polygon binding (expected 0 reassignments at
  promotion; verify on execution). Open sub-case (flagged, not
  blocking): if polygons disagree with the provisional
  parent-label map on specific wells, POLYGONS WIN (geometry over
  string labels) and disagreements are quarantined + logged, not
  auto-dropped — the falsification surface R3 never got to attack.
- EFFECT NOW: none on any artifact. Weather Backbone and all
  downstream work proceed on 59-execution; nothing rebuilds until
  the promotion event.

ISSUE #1 CLOSED on this ruling: Ask 1 ruled (this addendum),
Ask 2 countersigned (R1, Addendum 6), Ask 3 concurred (R1,
Addendum 6), Ask 4 PRP v1.1 approved (R1, Addendum 6). Full
transcript preserved at
https://github.com/SKJNR/agri-tws-ind/issues/1 (5 comments:
R1 Gemini, GLM ack, R2 Qwen, GLM R2 processing, GLM ruling).
Next courier asks open as Issue #2+ when they arise.

No decision numbers consumed beyond the D#20 amendment itself;
AM-6 untouched (no model runs — ET0/backbone is covariate
engineering, explicitly the registered next step); sealed lock
held; D#22 weights ban intact.

## ADDENDUM 9 — 2026-09-23 (GLM; Qwen external review of the Weather Backbone: ACCEPT + 3 riders + hygiene amendment H-1; rider consequences PRE-STATED before computation)

EXTERNAL VERDICT (courier, founder-relayed, true receipt 2026-09-23
— date correction acknowledged): QWEN REVIEW — WEATHER BACKBONE:
ACCEPT WITH 3 RIDERS + 1 AMENDMENT.
- Rider 1: season × regime bias-decomposition table for IMD-vs-OM
  temperature, consequence pre-stated at >1.0°C seasonal deviation.
- Rider 2: REPRESENTATIVENESS_LIMITED tag for terrain districts in
  the QA manifest (ASR et al.), consistency verdicts computed on
  low-relief subset.
- Rider 3: primary ET0 pinned = Open-Meteo (hash-frozen foundation),
  pyet FAO-56 = consistency artifact, per-district divergence
  logged, >15% monthly MAD ⇒ QA flag; swap = AM-5 amendment.
- Amendment H-1: secret-scan on commit + credential globs in
  .gitignore + sweep-review checklist item.
- Sealed custody double-verification noted and approved. Proceed to
  backbone build on 59-execution basis; riders land as DECISION_LOG
  entries, not blockers.

GLM PROCESSING — ACCEPTANCE RECORDED. Riders adopted as binding
entries WB-R1/R2/R3 (QA layer on the delivered backbone; foundation
untouched, nothing rebuilds); H-1 adopted AND IMPLEMENTED in this
addendum's commit. Both R1 and R2 consequences are PRE-REGISTERED
BELOW before any rider computation runs (log-before-run; results
appended in a follow-up commit with the pre-state provable in
history).

WB-R1 — SEASON × REGIME BIAS-DECOMPOSITION (definitions + trip rule
+ consequence, all pre-stated):
- season = IMD convention {DJF, MAM, JJAS (monsoon), ON
  (post-monsoon)}; regime = {2014-2018, 2019-2022} (ERA5
  final-release vs ERA5T-adjacent publication halves of the open
  window; boundary pre-registered at the open-window midpoint
  BEFORE looking at any season×regime cell).
- Table: OM−IMD mean bias for tmax and tmin at BOTH granularities —
  aggregate (4 seasons × 2 regimes × 2 variables = 16 cells) and
  per-district×season (59 × 4 × 2). Open window only (D#21).
- TRIP RULE: aggregate cell |bias| > 1.0°C, or district×season cell
  |bias| > 1.0°C.
- CONSEQUENCE IF TRIPPED (pre-stated, mechanical): (a) QA-manifest
  flags SEASONAL_TEMP_BIAS at tripped granularity; (b) per-district-
  season OM→IMD offset table computed (open window only) and hashed
  into the manifest; (c) a bias-corrected ET0 VARIANT (Hargreaves on
  offset-corrected OM temps) is produced as a DIAGNOSTIC lane
  carried alongside the panel — the pinned primary (OM PM-API,
  hash-frozen) does NOT change; promotion of any variant = AM-5-class
  amendment with founder/Qwen review, never silent; (d) trip count +
  worst cells reported in the next courier.

WB-R2 — REPRESENTATIVENESS_LIMITED TAGS + LOW-RELIEF VERDICTS
(rule pre-stated before this session's per-district computation):
- Tag rule = NAMED TERRAIN LIST ∪ EMPIRICAL TRIP. Named (high-
  confidence Eastern Ghats agency / upper-Godavari gorge geography,
  declared upfront): Alluri Sitharama Raju, Parvathipuram Manyam,
  Mulugu, Bhadradri Kothagudem. Empirical: |OM−IMD tmax bias| ≥
  2.0°C (open window) — the observed ASR class (-4.89°C, Task 64);
  any additional district it catches gets the tag with the empirical
  reason recorded.
- Low-relief subset = the complement (neither criterion). The seven
  delivered headline consistency checks recomputed on the low-relief
  subset; BOTH full-set and low-relief numbers reported.
- CONSEQUENCE (pre-stated): if any check flips CONSISTENT→FLAG on
  the low-relief subset, the backbone QA status re-opens for courier
  review — no silent pass.

WB-R3 — PRIMARY ET0 PINNED + DIVERGENCE LOG:
- PRIMARY = et0_pm_api (Open-Meteo API FAO-56 Penman-Monteith
  column on the hash-frozen Day-1 foundation). PINNED: any swap
  (IMD-HS, bias-corrected variant, anything else) = AM-5-class
  amendment (feature-freeze procedure) — never silent, never
  routine.
- CONSISTENCY ARTIFACT = pyet ET0 = et0_hs_om (pyet 1.5 Hargreaves
  method=0, FAO-56 Eq.52-calibrated <1% per build selftest).
  INTERPRETATION NOTE (logged honestly): full pyet FAO-56 PM needs
  Rs/u2/RH columns the frozen Day-1 fetch does not carry; adding
  them = refetch = foundation hash change = AM-5-class event. Not
  done; the pyet artifact stays the calibrated Hargreaves lane.
- DIVERGENCE LOG: per-district×month MAD% = |primary − artifact| /
  primary (monthly means, mm/day), open window; QA FLAG at monthly
  MAD > 15%; per-district rollup (n flagged months, mean/max MAD%)
  into the QA manifest.

H-1 — IMPLEMENTED in this commit (not deferred): (a) credential
globs added to .gitignore; the new globs exposed 4 already-tracked
credential-shaped files (root .env [local DB path only] + 3 LGD
cookie capture files — dead session tokens) → untracked, local
copies kept, precedent b52efc8; (b) program/scripts/secret_scan.py
(tracked) + pre-commit hook installed in this sandbox (hook lives
in .git/ = wiped by resets; re-install is sweep item S2); full-tree
scan at implementation: CLEAN over 1,972 tracked files; (c)
agri_tws_ind/REPO_SWEEP_CHECKLIST.md created — secret scan is the
FIRST standing item (S1), plus sync (S3), sealed custody (S4),
visibility (S5), regenerable integrity (S6), logs-pushed (S7).

CUSTODY + RESET NOTE: written after the 16TH sandbox reset (recovered
via playbook #4: clone --no-checkout + .git swap + reset --hard;
zero loss at b52efc8). Sealed-window gwl extract is NOT present in
this sandbox (untracked by policy, wiped by the reset); manifests +
sealed contract + ACCESS_LOG intact; sealed re-supply from the
founder original (receipt gwl_data_csv.receipt.json) is a logged
next-need item BEFORE any 2023+ work. No sealed contact occurs in
rider work — riders operate on OM open + IMD public grids only.

STATUS: no decision numbers consumed; AM-6 untouched (covariate/QA
engineering only, no model runs); D#22 weights ban intact; sealed
lock discipline held (absence ≠ access). NEXT REGISTERED per founder
directive: backbone build under Task-54 conditions on 59-execution;
the pseudo-target smoke test gets its OWN DECISION_LOG entry BEFORE
it runs (Day-2 guardrail). Rider RESULTS appended below after the
run, in a separate commit.

RIDER RESULTS (appended after the run; pre-state provable at
commit fa2592e; script weather_backbone_riders.py, selftest PASS):
- WB-R1: aggregate trips 6/16 (tmax: DJF -1.36 / MAM -1.42 / JJAS
  -1.74 / ON -1.71 in the 2014-2018 regime, ON -1.10 in 2019-2022;
  tmin: DJF 2014-2018 +1.08). District-season trips 252/472; worst
  cells ASR ON tmax -5.79, ASR JJAS -5.29, ASR MAM -4.33. STRUCTURE
  FOUND (the reason the rider asked for regimes): the OM-vs-IMD
  temperature bias concentrates in 2014-2018 — every season trips
  there — while 2019-2022 is materially cleaner (tmax cells -0.91
  to -1.10 vs -1.36 to -1.74). Pre-stated consequence EXECUTED:
  SEASONAL_TEMP_BIAS flags at both granularities + OM-to-IMD offset
  table (236 district-season cells, open window only, hashed) +
  et0_hs_om_bc diagnostic lane carried in
  weather_riders_monthly.parquet. Primary et0_pm_api PINNED,
  unchanged; no AM-5 event.
- WB-R2: REPRESENTATIVENESS_LIMITED = 18 of 59 = 4 named terrain
  (ASR, Parvathipuram Manyam, Mulugu, Bhadradri Kothagudem) + 14
  empirical (|tmax bias| >= 2.0C: interior Telangana plateau +
  Dr B R Ambedkar Konaseema — the ERA5 daytime-cool-bias class,
  not Ghats terrain; per-district reason recorded in the manifest).
  Low-relief subset (41 districts): tmax bias -1.29 -> -0.92, r
  .913 -> .928; tmin +0.69 -> +0.88, r .934 -> .942; ET0 checks
  essentially unchanged. VERDICT FLIPS: NONE — backbone QA status
  holds on the low-relief subset (pre-stated re-open condition not
  triggered).
- WB-R3: divergence log over 6,372 open-window district-months:
  858 monthly-MAD>15% QA flags (13.5%), occurring in ALL 59
  districts (worst mean MAD: Parvathipuram Manyam 13.8%,
  Visakhapatnam 13.7%, Kakinada 13.4%; worst single month 41.3%).
  Pattern = the known PM-vs-HS method gap (PM consumes
  radiation/wind/RH; HS is temperature-only), largest in monsoon
  months. Flags logged per the pre-stated rule in
  weather_qa_riders.json + the riders panel; primary stays pinned;
  a swap would be the AM-5 event — not triggered.
- ARTIFACTS: program/data/manifests/weather_qa_riders.json (tables
  + rules + input SHA256s) | agri_tws_ind/WEATHER_BACKBONE_RIDERS.md
  (human-readable) | program/data/parquet/
  weather_riders_monthly.parquet (gitignored, hash in manifest) |
  program/scripts/weather_backbone_riders.py.
- REGENERATION NOTE: post-reset IMD re-download hit 2 transient
  failures (2022 tmax ConnectionReset, 2024 tmin ConnectTimeout),
  caught by the n-delta vs the Qwen-reviewed report (exactly
  -365 x 59 district-days) BEFORE any commit; retry completed all
  22 files; the regenerated report matches the reviewed one
  EXACTLY except the build timestamp (deterministic reproduction
  confirmed; intermediate bad state never committed or pushed).

## ADDENDUM 10 — 2026-09-23 (GLM; backbone build REGISTERED + smoke test PRE-REGISTERED before running — Task-54 conditions on 59-execution)

TRIGGER: founder directive carried in the Qwen review courier
("Proceed to backbone build on 59-execution basis"), Addendum 9
riders landed. This addendum registers the build spec and
pre-registers the smoke test BEFORE either runs (log-before-run,
Day-2 guardrail extension per Task 53/54).

BUILD SPEC (pre-registered, implements the Task-54 approved
16-feature backbone = 14 core + 2 regime-gated):
- Layer (i) COVARIATES (built now, 59-execution, from the
  hash-frozen OM foundation + delivered weather panel): wb_lag1/2/3
  (P−ET0 point lags; monthly SUMS mm/month; ET0 = PRIMARY
  et0_pm_api per the WB-R3 pin), soil_lag1 (monthly mean), tmax_lag1
  (monthly mean of daily max), precip_roll3/6/12 (rolling SUMS),
  wb_roll3 + wb_roll12 (rolling SUMS of P−ET0), max1day_lag1
  (primary extreme, threshold-free), heavy20_lag1 (count ≥20 mm/day,
  frozen), dryspell_lag1 (max in-month run of days <1.0 mm, frozen),
  month_of_year (categorical, known at issue time). Extremes labeled
  ERA5-Land covariates (convective tails smoothed, extremes biased
  low). heavy25/heavy64.5 carried as manifest DIAGNOSTICS only.
- Layer (i) REGIME-GATED (15/16): wb_roll3 × regime, soil_lag1 ×
  regime — implemented but materialized ONLY when regime_map.csv
  (T11 frozen map) is present; the MVP_HEURISTIC_REGIME flag path
  fails LOUD without a defined heuristic source (Minor Irrigation
  Census vintages are T2-deferred, not in sandbox); heuristic
  stratification never citable as evidence (T11).
- Layer (ii) PR-8 TARGET TRANSFORM (generic, separated): per-district
  linear detrend, TRAIN-years-only fits, district-mean level removal,
  tercile boundaries from the TRAINING detrended distribution, 0.15σ
  stationarity flag, frozen single method, trend carried as product
  content. Applied to the TARGET layer only.
- Layer (iii) TARGET INGESTION: STUB, not built (D1.1 pending;
  Qwen loaders + founder pilot-12 + Task-55 WRIS spot-check gates
  unchanged). LGD crosswalk NOT needed for layer (i) (weather points
  immune); required before D1.1 loader / regime freeze (unchanged).
- SPLIT CONSTANTS FROZEN IN CODE: train ≤2019 / val 2020-22 /
  test 2023-25 (test untouched until D1.1 → regime freeze → ladder).

SMOKE TEST PRE-REGISTRATION (runs AFTER this addendum is committed):
- Pseudo-target = NEXT-MONTH soil-moisture tercile (explicitly
  labeled PSEUDO_TARGET / NON-EVIDENCE), PR-8 transformed with
  train-only fits.
- Split: train ≤2019 + val 2020-22 ONLY; test 2023-25 EXCLUDED
  entirely.
- Scope = PLUMBING ONLY: panel integrity (59 districts, expected
  rows, as-of discipline — features at month t predict t+1, no
  future columns, no NaN leakage into val), PR-8 machinery exercise,
  trivial climatology/majority baseline machinery run. NO skill
  claims, NO model comparisons; the LGBM Day-2 smoke stays gated
  behind the baseline ladder (AM-6). All smoke outputs stamped
  NON-EVIDENCE.

AM-6: this IS the registered backbone-build step; still no model
runs in the evidence sense (plumbing ≠ evidence). D#22 weights ban
intact. Sealed lock discipline held (no sealed data involved).
Results appended after the run in a separate commit.

BACKBONE BUILD RESULTS (appended after the run; spec + smoke
pre-registered at c9208d0; script build_features_pr8.py, selftest
PASS incl. PR-8 train-only invariance + trend recovery):
- LAYER (i) BUILT: 14 core features on 59-execution, monthly-SUM
  conventions, ET0 primary = et0_pm_api (WB-R3 pin), thresholds
  frozen (heavy ≥20 mm, dry <1.0 mm), heavy25/64.5 as manifest
  diagnostics only. LAYER (ii) BUILT: PR-8Transform module —
  per-district linear detrend, train-only fits, calendar-continuous
  x-index (val continues the train index — never restarts at 0;
  unit-tested), district-mean level, train-distribution tercile
  edges, 0.15σ flag. LAYER (iii) STUB as registered (D1.1 pending).
- REGIME-GATED 15/16: NOT materialized — regime_map.csv absent (T11
  map not frozen); --mvp-heuristic-regime fails LOUD without a
  defined heuristic source (refuses to invent one). Interactions
  land when the map freezes.
- SMOKE (PLUMBING, NON-EVIDENCE, stamped): rows 6,372 = 59 x 108
  target months (train 4,248 = 59 x 72; val 2,124 = 59 x 36; exact);
  test rows EXCLUDED 2,655 = 59 x 45 (2023-01..2026-09 targets
  untouched); NaN features in val = 0 (warmup NaNs confined to train:
  2,773 cells, first 12 months/district); majority-class baseline
  val acc 0.3079 ≈ chance (3-class) — exactly what plumbing with no
  skill claim should read; PR-8 0.15σ stationarity flags on the
  pseudo-target: 41/59 districts (soil-moisture trend content —
  product content per PR-8, not a defect).
- ARTIFACTS: backbone_features_monthly.parquet (gitignored, SHA256
  in manifest) | backbone_features_manifest.json |
  agri_tws_ind/BACKBONE_FEATURES_REPORT.md.
- AM-6 POSITION: backbone build COMPLETE as the registered step; the
  next registered steps are D1.1-empirical (gated on Qwen loaders
  L1-L12 + founder pilot-12 selection + Task-55 WRIS provenance
  spot-check) then regime-map freeze (gated on LGD crosswalk). No
  model runs occurred; D#22 weights ban intact; sealed lock held.
