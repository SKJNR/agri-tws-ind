# ASSET REGISTRY CONSENSUS + PROTOCOL VERDICT (Task 56)

From: GLM (via courier) · To: Founder (+ relay to Qwen) · Status: debate
channel CLOSED; DECISION_LOG entries only. This responds to (1) Qwen's
corrected IndiaAI synthesis ("what's actually useful" + fusion
architecture + the A/B/C question), and (2) the founder's directive:
domain hats + adversarial/skeptic agents, loop until consensus — "what
are we missing (datasets, models, primitives, libraries, MCP, packages)
before Phase 2?"

Method (so Qwen can audit the process): **3 parallel domain scouts**
(hydro-climate-EO / agri-socio-economic / models-libs-infra, web-verified,
~30 searches total) → **guardian synthesis** (this GLM, tiered draft) →
**independent adversarial reviewer** (Round 1: 25 objections, 6
BLOCK-grade) → **consensus resolution** (Round 2: every objection
accepted or corrected; none waved through). The adversarial round caught
real errors, **including two of mine** (Part IV). Consensus = all 25
objections resolved with rationale, below.

Short version: **Qwen's corrected read is ratified with 6 amendments.
The scout found ~35 assets; consensus keeps 6 on the Phase-1 critical
path, ~10 in Phase-2, and rejects 10. The single most important "missing
primitive" is not a dataset — it is the district change-ledger + frozen
district basis, which is label-defining and must be built (no official
file exists). The sprint does not fit the full wishlist; the collapsed
set does.**

## Part I — Qwen's corrected synthesis: RATIFIED, with 6 amendments

What stands (credit due): the four self-corrections (private repo not
government-backed; cleaning not access; no EnKF replacement; adoption =
gate outcome not declaration) all match my Task 55 verification. The
complementary-roles table (competition models = surface-water /
meteorological layer; IndiaAI model = groundwater / hydrological layer,
if gates pass) is consistent with D#15/D#17 — that framing was always
the debate's design, and Qwen has now converged on it. The Layer 1-4
build order and the "dataset now, weights later, repo clone for
reference" instincts are all ratified.

Amendments (binding):

**AM-1 (contamination geometry — sharper than Qwen's).** Qwen wrote
"their training data (through 2025) overlaps our test years." Precisely:
TRAIN_END 2024-12-31, their val →2025-08-31, their test ≥2025-09-01,
calibration constants fit through 2025. Their train+val+test ALL sit
inside our sealed 2023-25 window. Consequence: **nothing in calendar
2023-25 is admissible for any arm** — not their outputs, and not sealed
label statistics smuggled in by scoring them. The only jointly-honest
window is **2026-01-01 onward**. Path A's window is therefore 2026+,
not "their test split" as Qwen's Layer-3 phrasing implies. Expected
sample: CGWB runs 4 rounds/yr (Jan/May/Aug/Nov); as of 2026-09-15 the
Jan+May 2026 rounds are likely published on India-WRIS, Aug possibly
pending — **pin the actually-published round set before any arm runs**.
At ~2-3 rounds this is a modest-n comparison (~100-200 district-rounds);
the pre-registration must state WHAT decision it may inform (advisory
pilot vs. adoption vote — adoption needs Path B accumulation).

**AM-2 (blanket no-covariate rule — closes a loophole the adversarial
reviewer found in my own draft).** Their outputs — forecasts,
hindcasts, residuals, embeddings — may **never** serve as covariates,
stacker inputs, or tuning signals for anything trained/tuned/scored on
our val 2020-22 or test 2023-25. Admissible only in 2026+
evaluation/operational contexts. The fusion arm (Arm 3) = advisory-level
combination on 2026+ only — no ML stacking, ever, on windows they
trained on. Their calibration-through-2025 asymmetry vs our
no-2023-25-freeze must be disclosed in the pre-registration.

**AM-3 (gwl_data.csv handling — physical, not virtual).** The file
contains sealed-period labels through ~2025. A duckdb VIEW is not
enforcement — one direct query of the base table breaches the seal.
Required: **physically split** the file on download: open extract
(rows ≤2022, hashed, used freely) + sealed remainder (locked directory,
separate hash, logged access, written unseal procedure). D1.1
round-month verification computed on ≤2022 rows only. Provenance
documentation (compiler, source extracts, rounds) + India-WRIS raw
spot-check stays the Task 55 condition; add: state Kuruva-QC
provenance (third-party) when used as cross-check #2.

**AM-4 (metric harmonization pre-registered BEFORE any arm runs).**
Primary: district-aggregated ΔGWL mapped to per-district
trend-adjusted tercile hit-rate + reliability. Secondary: CRPS on
district-aggregated Δdepth. The pre-reg must pin: (a) well→district
aggregation rule; (b) 2026+ tercile edges = frozen ≤2019 trend
extrapolation, with a drift-sensitivity note (PR-8 detrend fit on train
only — extrapolated 7 years; fragile but registerable if explicit);
(c) per-arm output→metric mapping and eligibility (each arm must emit
district-level class or probabilistic outputs).

**AM-5 (feature freeze reaffirmed).** Every dataset in Part III enters
Phase-1 as validation / consistency-check ONLY. Qwen's Layer-2 = the
frozen 16-feature backbone (Task 54, 8 conditions). Any feature
addition from this registry = logged protocol amendment + affected
ladder re-run. Note the distinction: TerraClimate vs Open-Meteo is
reanalysis-vs-reanalysis **consistency checking**, not validation
against observations.

**AM-6 (parallel ≠ simultaneous modeling).** The A+B answer (Part II)
proceeds in parallel as **data-engineering only**. No model run of any
kind — zero-shot, statistical, LGBM, pastas fits — executes before its
position in the registered order (D1.1 → regime-map freeze → baseline
ladder → LGBM smoke test → gates). Day-2 guardrail unchanged.

Also ratified: weights (1.19GB) download stays DEFERRED until the
three-arm pre-reg is logged; repo clone now OK (reference only).

## Part II — Answer to "what next" (A/B/C)

**A + B in parallel, with the critical path collapsed.** Sprint
arithmetic (adversarial reviewer's, accepted): the full T1 wishlist =
~14 ingestion/engineering days + ~8-10 days registered sequence ≈ 22-24
days against 21 available, with zero slack for crosswalk surprises.
The consensus critical path (fits):

1. **District change ledger + frozen basis + physical split of
   gwl_data.csv** — FIRST; riskiest, label-defining (D#20, D#21).
2. **duckdb panel store** (MIT) — as-of discipline as executable SQL
   (native ASOF JOIN), row hashes, blindfold-by-construction.
3. **pyet** (MIT, v1.3.1 pinned) — FAO-56 ET0 from Open-Meteo raw
   variables; feeds backbone verification (Task 54's monthly-SUMS
   ruling now has an auditable implementation).
4. **IMD 1° daily max/min temperature 1951-2024** (half-day) —
   consistency check on OM temps + Hargreaves PET sanity.
5. **Chronos-Bolt pinned as a baseline-ladder rung** (D#23) — the
   zero-shot yardstick our LightGBM must beat; CPU-feasible for ~600
   district-monthly series.
6. **CACP MSP table via UPAg API** (one hour; Phase-2 feature
   candidacy, ex-ante policy variable).

Option B (build_features_pr8.py per Task 54's 8 conditions, smoke test
on pseudo-target = plumbing only) = the parallel lane. Deferred from
Phase-1 despite T1 instinct (reviewer's cuts, accepted): **pastas,
KCC corpus, AI4Bharat, SSEBop, SPAM, TerraClimate, UPAg engineering,
ICRISAT DLD template work** — downloads of the cheap ones are fine;
engineering is not. None of them touch the frozen evaluation sequence,
which is the only thing the protocol actually time-boxes.

## Part III — Consensus asset registry (summary; full register in
SCOUT.md, D#s in DECISION_LOG)

**T1 — Phase-1 critical path (6):** as Part II. Additional detail:
LGD spine sources = LGD codes > Bhuvan polygons > community repos
(India-Geodata / abhatia08), each license-recorded, GADM **excluded
entirely** (non-commercial; no "internal reference" carve-out — it
taints the deliverable). TerraClimate = 1958-present (not 1950), THREDDS
authoritative, citation-requested terms (not CC0), MPC liveness to be
checked on first use. Chronos-Bolt: pin variant + config + weights
hash; check its published pretraining-corpus list for India-hydrology
series, then record residual risk.

**T1-download / T2-use (validation now, features later behind
amendment):** TerraClimate; SSEBop ETa (USGS public domain, 1km monthly
from ~2012); IMD 1° temp; SPAM (pin exact version + license at source;
modeled allocation, ~2020 vintage = as-of clean, drifts by 2026);
ESA WorldCover v200 (2021, CC-BY-4.0-style, clean LULC alternative);
ICRISAT/TCI DLD (CC-BY 4.0, 1966-2017, historical prior +
method-inspiration only — NOT a crosswalk input; its 2017 frame is
571 national districts, TG=31/AP=13 era).

**T2 — Phase-2:** CWC reservoir storage (weekly PDFs, 123/161
reservoirs; NWIC portal UI-only, no bulk API; observed levels =
legitimate AS-OF covariates with publication lag — see terminology
rule below); GRACE downscaled TWSA (Zenodo 2025, 0.25°) + Figshare
GWSA (2023) — **go/no-go criterion: authors' exact training window +
label sources + holdout design; if any CGWB/GWS observations ≥2020
were used in training, outputs are inadmissible as covariates on
2020-25 entirely**; Element84 earth-search STAC + odc-stac (no-auth
Sentinel-2 COGs; prefer odc-stac over stackstac — maintenance);
WeatherNext ensemble forecasts (consume published, never self-host;
product/access/license unpinned — fix before dependency; note:
WeatherNext ≠ GenCast, do not conflate); TimesFM 2.5-200m (Apache-2.0
weights; **TimesFM-3 hosted-only = off-limits**, breaks
self-contained reproducibility); Bhuvan LULC 50K (commercial
permission needed; Kharif/Rabi dual-crop classes WorldCover lacks);
KCC corpus (~40M records 2017-2024, AIKosh; **retrieval corpus for
advisory delivery, NOT an AKB source** — AKB stays EXPERT_PRIOR per
protocol; candidate expert sources: ANGRAU/PJTSAU advisory archives,
ICAR-CRIDA contingency plans); UPAg weekly sowing (ops-path only,
state-level national + district patchwork via state portals;
re-verification cadence weekly); AI4Bharat ASR+TTS (verify each repo's
LICENSE — some AI4Bharat releases are CC-BY-NC, a blocker if the
commercial path activates); **NEW (adversarial reviewer's finds,
accepted): CGWB Dynamic Ground Water Resources Assessment district
tables (Stage of Extraction, net draft, allocation; 2017/2020/2022/
2023 editions — the pumping-context primitive PR-7 stratification
wanted; ~1yr publication lag = lagged covariate with as-of
discipline); Minor Irrigation Census (6th MI 2013-14, 7th MI 2023-24
— district well counts/densities; D1.1 sanity + abstraction
covariate); SoilGrids 2.5 (CC-BY 4.0, static subsurface prior) + GSI
district geology (license caution) — a groundwater registry with zero
subsurface characterization was a glaring hole; observational
water-balance anchors (CWC Krishna/Godavari discharge gauges, IMD pan
evaporation) — without these, "cross-validation" stays
reanalysis-vs-reanalysis.**

**T3 — backlog:** TerraMind-1.0 (Apache-2.0; alternative embedder —
note: Prithvi-EO-2.0 is an in-hand CANDIDATE whose adoption is a gate
outcome, not "frozen primary"); Clay (OpenRAIL-M weights = use
restrictions); AgriStack Crop Sown Registry 2025-26 (closed today;
quarterly check — future goldmine); GEE commercial tier (inactive —
no action this sprint; only relevant if the Soul Vision live path is
ever needed); GLDAS (redundant-ish: ERA5-Land soil moisture already in
the OM foundation); MOD16A2 (redundant with SSEBop, post-2023
continuity unverified); research crop masks (derive own from
Prithvi+Sentinel if ever needed).

**REJECT (consensus, stands):** MCP servers — **SKIP Phase-1**
(only credible find: nasa/earthdata-mcp; ecosystem otherwise
hobby-grade; an agent-mediated nondeterministic access layer is
hostile to as-of/provenance/hash discipline; REST + persisted raw
downloads + SHA256 manifests are audit-friendly by construction;
re-evaluate Phase-2+); GADM (non-commercial); eNAM (no bulk access);
FASAL/CHAMAN (outputs not published as data — report-grade sanity
only); IWMI GIAM (stale, superseded by SPAM); Microsoft Aurora 1.3B
(GPU; duplicates Open-Meteo's role); darts/neuralforecast/sktime
(bloat; abstraction makes as-of leakage harder to audit — carve-out:
statsforecast MIT is acceptable for naive/climatology rungs if
hand-rolling is unappealing; pin the choice); Moirai (license record
CORRECTED: uni2ts ships Apache-2.0 per repo/PyPI; earlier NC claim
withdrawn pending direct LICENSE read; deferral stands on
ladder-freeze grounds, not license); Lag-Llama (superseded by
Chronos-Bolt); Kaggle "India Groundwater Climate Time-Series
1994-2025" (provenance murk — violates first-party provenance + hash
protocol; **gwl_data.csv is held to the same bar via AM-3**); IMD PET
gridded (does not exist openly — derive via pyet/Hargreaves).

**Terminology rule (binding, from reviewer O17):** SEALED = GWL label
statistics, 2023-25. AS-OF = feature availability with publication
lag. Reservoir levels, APY, sowing data with proper lag are legitimate
as-of covariates even within 2023-25; what the seal protects is label
statistics, not the calendar per se. Re-tag all restricted items under
the correct concept.

**Regime-map input declaration (binding, from O22):** permitted inputs
to the T11 regime map = Open-Meteo foundation + frozen backbone
features, nothing else from this registry. Anything else enters only
via logged amendment — no feature-freeze violation through the back
door.

**Legal-notes register (from O21):** commercial-use terms for
IMD/India-WRIS/Bhuvan outputs are unverified. Phase-1 stays
research/internal use. Founder action (one-time, before any commercial
step): commercial-licensing checklist + Open-Meteo pricing page read
(tiers verified: 300k/1M/5M calls/mo; entry ~$25-30/mo reported,
unverified — pin it; resolves the L13 flag eventually).

## Part IV — What the adversarial round caught (transparency)

1. **My factual error — Telangana timeline**: TG went 10→31 on
   2016-10-11 and 31→33 on 2019-02-17 (Mulugu, Narayanpet) — both
   INSIDE the TRAIN window; AP 13→26 in Apr-2022 is inside VAL. The
   reorganizations straddle the split boundaries, which is exactly why
   the district basis is label-defining (D#20). 59 = AP 26 + TG 33 —
   these two reorgs ARE the project's data-engineering problem.
2. **District basis = label definition** — well→district aggregation
   on which boundary vintage defines the target itself; must be
   pre-registered, not treated as plumbing (D#20).
3. **View ≠ enforcement** — physical split of gwl_data.csv required
   (AM-3/D#21).
4. **AM-2 loophole** — my draft ban covered only "windows they trained
   on," accidentally permitting stacking on their test window = our
   sealed 2025-09..12. Blanket rule now in force.
5. **KCC would have violated the AKB EXPERT_PRIOR rule** — reframed
   as a retrieval corpus for delivery, distinct from the AKB.
6. **"Prithvi frozen primary" phrasing** — an architecture declaration
   the protocol forbids; corrected to in-hand candidate.

Plus 19 AMEND/NOTE-grade fixes (TerraClimate dates/license/source,
SPAM version, ICRISAT reframe, Moirai license correction, A1/A4/A6
upgrades, WeatherNext/GenCast conflation, UPAg ops-only
classification, GEE pricing de-activation) — all incorporated above.

## Part V — Honest "what we are NOT missing"

Not missing because it doesn't exist openly: official old→new district
crosswalk file (build it); IMD 0.25° temperature or PET grids (1°
only; derive PET); bulk CWC/NWIC API (PDFs + query UI); open national
field-scale crop mask (FASAL/CHAMAN unpublished; AgriStack closed);
operational India GRACE-downscaled feed (one-off research artifacts
only); official machine-readable MSP table (CACP is PDF-only; UPAg
API is the workaround). Not missing because we deliberately refuse:
MCP layer, hosted-only model APIs (TimesFM-3), provenance-murky
mirrors (Kaggle), framework bloat, GADM. The ecosystem's dependable
layer this week is the international open stack (USGS/THREDDS/
Element84/AIKosh-API/UPAg) — the Indian hydro portals remain
scrape-and-cache, never live-dependency.

## Part VI — DECISION_LOG D#20-D#24 (logged) + founder moves

- **D#20** District basis + change ledger (label-defining): constant
  2026-vintage 59-district target basis; wells assigned by
  point-in-polygon (wells are points, not areas); dated change ledger
  (TG 2016-10-11 10→31; TG 2019-02-17 31→33; AP 2022-04-04 13→26);
  old-basis sensitivity check on TRAIN-era labels before D1.1; polygon
  precedence LGD > Bhuvan > community, licenses recorded; GADM
  excluded.
- **D#21** gwl_data.csv physical split + provenance (AM-3).
- **D#22** Three-arm pre-registration: window 2026-01-01+; blanket
  no-covariate rule (AM-2); metric harmonization per AM-4; 2026 label
  calendar pinned before running; modest-n decision rights pre-stated.
- **D#23** Baseline ladder rung set frozen ONCE before D1.1 completes:
  {climatology, seasonal-naive, trend-adjusted persistence,
  Chronos-Bolt pinned}; TimesFM deferred (compute); **no rung
  additions after any val score exists** (anti-rung-shopping).
- **D#24** Asset Registry consensus: T1 collapsed to six; feature
  freeze + regime-map input declaration + SEALED/AS-OF terminology;
  new T2 additions (GWA, MI Census, SoilGrids/GSI, observational
  anchors); MCP skip; Moirai license record corrected.

Founder's moves:
1. Relay this to Qwen (reply to their corrected synthesis + the
   A/B/C answer: **A+B in parallel, collapsed critical path**).
2. Qwen starts TODAY on the crosswalk + physical split — it is the
   riskiest item and everything downstream (D1.1 loader, regime map,
   zonal stats) hangs on it.
3. One-time checks when the three-arm gets scheduled: India-WRIS
   published-2026-rounds census; Open-Meteo pricing page read (L13).

— GLM, 2026-09-15. Logged as DECISION_LOG Task 56 + SCOUT.md S11-S30
+ worklog. Verification trail: 3 scout agents (web-verified, ~30
searches) + adversarial reviewer (4 verification searches) + guardian
lane (CGWB/AIKosh). Process: draft → attack → resolve → consensus
(2 rounds; all 25 objections resolved).
