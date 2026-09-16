# INDIAAI/AIKOSH PRIMITIVE — VERIFICATION + SCOUT REPORT (Task 55)

From: GLM (via courier) · To: Founder (+ relay to Qwen) · Status: debate
channel CLOSED (R7); DECISION_LOG-only entries. This responds to Qwen's
"Soul Vision stack / Primitive-First Architecture (D#19)" message.

Short version: **the find is real and valuable — Qwen's claims about it
are wrong in four particulars — and one technical fact (their training
window) blocks the proposed adoption path entirely.** Below: what I
verified, what it means, what we do instead, plus the proactive scout
sweep the founder asked me to run.

## Part 1 — Verification results (claims vs. facts)

I fetched the repo, LICENSE, MODEL_CARD.md, DATA_SOURCES.md, TRAINING.md
directly. Qwen's message vs. documented reality:

| Qwen's claim | Verified fact |
|---|---|
| "License: CC BY-4.0" | **Discrepancy in the repo itself**: root LICENSE file = CC BY-4.0 text; MODEL_CARD license table says "weights & code: Apache-2.0"; dataset = GODL-India; Prithvi backbone = Apache-2.0 (confirmed on HuggingFace/NASA-IMPACT). Both permissive for commercial use with attribution — no blocker either way, but the discrepancy needs a one-line clarification from the author IF we adopt. Open-Meteo CC BY-4.0 free tier remains non-commercial (L13 flag). |
| "Government-backed model / IndiaAI published it" | Private repo by org "SoulVisionCreations": **created 2026-07-26, 0 stars, last push 2026-09-03** (~6 weeks old). AIKosh is a registry portal (pages are JS shells — listings not even statically verifiable). The DATA is government (CGWB/WRIS, GODL-India); the MODEL is a community build listed on a government portal. |
| "Solves CGWB data access (months, if ever)" | CGWB access was never the bottleneck: India-WRIS portal downloads + WIMS 6-hourly telemetric exist today (verified). The dataset's REAL value = 3.3M readings, 10,411 stations, CLEANED + PRE-JOINED with CHIRPS/ERA5/SMAP/HLS features. |
| "Solves the EnKF state estimation" | Their model is a well-depth point forecaster. Our EnKF spec targets a DIFFERENT problem (GRACE mascon assimilation into TWS state). Neither replaces the other by decree; the ladder decides. |
| (not mentioned) | **Operational dependencies**: live inference needs Google Earth Engine auth + a GEE project (imagery + weather/static features fetched live) AND Open-Meteo forecasts (non-commercial free tier → same L13 license flag applies to their inference path). |
| (not mentioned) | **Their own skill honesty**: per-well median R²(δ) ≈ 0.25 (3m) / 0.275 (6m); direction right ~66% on tiny moves → ~88% on >2m moves, **~60% on sub-0.5m moves**; 3-way normal/above/below only ~54%. Model card: "the trustworthy unit is the direction + tier + confidence, not the exact number." Their imagery ablation: the Prithvi encoder is "roughly neutral on aggregate skill." |

Credit where due: the repo is unusually honest (self-reported modest
skill, staleness handling, delta-clamp guard, as-of anchor discipline —
future anchor dates are rejected). This is a serious hobbyist/small-team
research artifact, not vaporware. It is also, independently, a
design-philosophy validation of OUR advisory shape (tercile/direction +
confidence bands) — they arrived at the same conclusion from the well
side.

## Part 2 — The decisive technical fact: training-window contamination

From TRAINING.md (their config): `SPLIT_STRATEGY=station_time`,
**TRAIN_END = 2024-12-31**, VAL = 2025-01-01..2025-08-31, TEST ≥ 2025-09-01.

Our frozen protocol: val **2020–22**, test **2023–25**.

Their model's TRAINING SET covers 100% of our val years and 2 of our 3
test years (2023–24); their VAL covers most of the third (2025-01..08).
Their weights — and their calibration constants (the "~7 in 10"
reliability tables were "calibrated on the shipped champion's test
set", i.e., through 2025) — encode CGWB readings from our entire
evaluation window.

Consequences, stated precisely:

1. **Their forecasts are inadmissible as covariates** in any pipeline we
   score on val 2020-22 / test 2023-25. Ingesting them imports their
   target leakage into our model; our evaluation stops being
   out-of-sample.
2. **A head-to-head "their model vs our ladder" on our val/test is
   rigged** — their in-sample vs our out-of-sample. Qwen's proposed rule
   "if the IndiaAI model clearly outperforms on pumping districts →
   adopt as primary," run on our splits, would produce an invalid
   verdict by construction.
3. **"Correlation > 0.6 → fuse" is a diagnostic, not a skill gate** —
   and also only admissible on data neither model has seen.

Qwen's direct question — "does Primitive-First violate PR-8 or T11,
provided we treat their MWS features as covariates and keep the frozen
splits?" — answer: **the frozen splits are exactly what their training
window violates.** PR-8 and T11 themselves are untouched (detrend
method, regime freeze). The binding constraints are T2/T7 (evidence
order: log before measure; nothing enters evaluation unregistered) and
the seal on test 2023-25.

## Part 3 — What the find IS worth (fair accounting)

1. **The DATASET is the prize (accelerates D1.1, our FIRST logged
   evaluation).** gwl_data.csv = cleaned CGWB panel + pre-joined
   features, GODL-India. It removes weeks of scraping/cleaning, not
   "access." Condition: D1.1 is precisely the verification that CGWB
   round-months/lags behave as assumed — so a third-party cleaned panel
   must itself pass a provenance check: sample wells re-checked against
   India-WRIS raw (values, dates, units, round-month structure) before
   D1.1 conclusions rest on it.
2. **The MODEL is a legitimate candidate for the groundwater leg** — to
   be evaluated under OUR gates on ADMISSIBLE splits only (Part 4).
3. **Their code is a reference implementation** for as-of/staleness
   discipline (LOCF ≤ 200d with notes, future-anchor rejection, NWDP
   alternative source) — worth reading before we write ours.
4. **Their model card independently validates two of our rulings**:
   direction+tier is the honest unit (≈ our tercile+confidence design),
   and satellite imagery added ~nothing on aggregate skill (consistent
   with our refusal to bet the MVP on remote-sensing magic).

## Part 4 — Corrected integration plan (replaces Qwen's Day-3)

Admissible evaluation paths for the IndiaAI model:

- **Path A — their test split, our metrics:** re-score their 2025-09+
   forecasts under OUR definitions (trend-adjusted tercile hit, PR-8
   frame, our baseline ladder) on the overlapping wells/districts.
   Like-for-like out-of-sample.
- **Path B — operational forward evaluation:** from today, both models
   issue forecasts; score against CGWB readings as they arrive (PR-6 /
   latency rules; ~1 quarter lag). The honest long-run race.
- **Pre-registered three-arm comparison (log before running, T7):**
   Arm 1 = their model; Arm 2 = our ladder; Arm 3 = fusion (their
   groundwater call + our water-balance call, disagreement surfaced per
   E-NC/D1.2 model-disagreement machinery — Qwen's "show both with
   confidence bands" instinct is right and already has a home).
   Stratified per PR-7 (pumping districts primary). Gates G1–G13 apply
   to ALL arms, including pre-trained ones — "adoption as primary
   primitive" is a GATE OUTCOME, not an architecture declaration.

What changes NOW vs. what doesn't:

- NOW: download the DATASET (provenance verification pass, then D1.1 on
  it); clone the repo for reference; log everything below.
- DOESN'T CHANGE: our covariate pipeline (own Open-Meteo foundation,
  hash-frozen), our target machinery (PR-8), the frozen evaluation
  order (D1.1 → regime-map freeze → baseline ladder → LGBM smoke test
  → gates), test-year seal, the 16-feature backbone from Round 2.
- The MWS/NDVI dataset: good v1.5 advisory-rendering candidate; our
  protocol evaluation stays district-level; if their MWS features ever
  enter as covariates, they carry the same third-party as-of audit +
  vintage check as everything else. Not a foundation swap — the
  foundation is frozen and verified.

Timeline reality: the 21-day sprint's bottleneck was never code — it's
the frozen evaluation order, which no external primitive skips without
reopening the protocol. Real acceleration delivered by this find: D1.1
data prep (weeks → days) and a ready-made groundwater-leg candidate.
That's genuinely good. It is not a 6-weeks-to-10-days transformation.

## Part 5 — GLM scout sweep (the founder's challenge, accepted)

What I verified/hunted this session (sources fetched or searched
directly):

1. **IMD 0.25° daily gridded rainfall, 1901–2024** (imdpune.gov.in,
   free, NetCDF/binary). Covers our whole window through 2024. STRONG
   acquisition candidate: India-standard gauge-based precip — use to
   cross-validate ERA5/Open-Meteo precip (the "house of cards" attack
   from Round 1) and as an eventual independent covariate. License
   terms for commercial use need one check before dependency.
2. **CGWB direct access confirmed** (cgwb.gov.in → India-WRIS; NWIC
   download; WIMS 6-hourly telemetric layer). D1.1 primary source.
3. **GRACE/GRACE-FO mascons** — CSR RL06.3 (RL07 upcoming) + JPL
   RL06.3_v04, free NetCDF. Target pipeline for the EnKF spec, when the
   ladder gets there.
4. **Kuruva et al. 2025 (Nature Sci Data)** — independent QC'd CGWB
   groundwater-level dataset. Second D1.1 cross-check source.
5. **Agmarknet** — real portal + app, daily mandi price/arrivals; no
   official open API surfaced (third-party APIs exist). v1.5 market
   layer per D#13/D#16; scraping/API terms need a check THEN.
6. **Prithvi-EO-2.0-300M** — Apache-2.0 confirmed (HF + NASA-IMPACT).
7. Queued for next scout pass (license + vintage checks each): Wadhwani
   AI pest datasets, ISRO Bhuvan LULC / NBSS&LUP soils, Soil Health Card
   portal, NIPHM/NCIPM advisories, IMD ERA-style temperature gridded
   sets, NWDP well index.

Institutionalizing this: **agri_tws_ind/SCOUT.md** — a standing
register (find, URL, license, vintage, as-of behavior, verification
status, decision). Every task updates it. The founder's casual finds
proved the gap; the register closes it.

## Part 6 — Founder's moves

1. Send BOTH pending courier replies now (Round-2 FE review FIRST — it
   unblocks Qwen's script — then this one).
2. Approve downloading the IndiaAI DATASET (gwl_data.csv, ~700MB) for
   the D1.1 acceleration path. Model weights (1.19GB) only when Path A
   comparison is scheduled.
3. D#19 as drafted ("adopt as primary primitive") = REJECTED; the
   corrected plan above is the DECISION_LOG entry I've logged. If Qwen
   wants the three-arm comparison registered, that's the productive
   version of their instinct — I've pre-stated it in the log.

— GLM, 2026-09-15. Logged as DECISION_LOG Task 55 + worklog.
Verification trail: repo LICENSE/MODEL_CARD/DATA_SOURCES/TRAINING.md
fetches + web searches (see worklog).
