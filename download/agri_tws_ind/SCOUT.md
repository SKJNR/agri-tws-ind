# SCOUT.md — standing open-primitive & data-source register (AGRI-TWS-IND)

Started: 2026-09-15 (Task 55). Rule: every candidate enters with URL,
license, vintage, as-of behavior, verification status, and a decision.
No candidate becomes a dependency without: license verified, vintage
logged, as-of audit passed, provenance logged in DECISION_LOG.
Update cadence: every GLM task appends findings. Founder finds welcome
(this register exists because the founder's casual search beat us to
the IndiaAI stack — good instinct, now institutionalized).

## Register

| # | Find | URL | License | Vintage / coverage | As-of behavior | Verified? | Status / decision |
|---|------|-----|---------|--------------------|----------------|-----------|-------------------|
| S1 | Open-Meteo Archive API (ERA5/ERA5-Land daily weather) | archive-api.open-meteo.com | CC BY-4.0, free tier NON-COMMERCIAL (L13 flag) | 2014-01-01..2026-09-10 fetched; 59 districts | As-of replay rule = L13 (mirrors L5 ERA5T/consolidated) | YES — 13/13 checks, live API cross-check, hash-frozen (full59 md5 bd202d932ab5663fafa0802fd2655ac0) | FOUNDATION (in use) |
| S2 | SoulVisionCreations gwl-forecasting-advisory — MODEL (TFT + Prithvi-EO-2.0-300M, 3m/6m depth-to-water change) | github.com/SoulVisionCreations/gwl-forecasting-advisory | Apache-2.0 (code+weights); Prithvi Apache-2.0 | Repo created 2026-07-26, 0 stars. TRAIN_END 2024-12-31 / val→2025-08-31 / test ≥2025-09-01 → CONTAMINATED vs our val 2020-22 + test 2023-25 | Anchor-date as-of respected (future dates rejected); GEE + Open-Meteo live deps; LOCF ≤200d staleness | YES — LICENSE/MODEL_CARD/DATA_SOURCES/TRAINING.md fetched | CANDIDATE — groundwater leg, gates on admissible splits only (Path A/B, 3-arm). Not covariate on our val/test. |
| S3 | SoulVisionCreations gwl DATASET (gwl_data.csv ~3.3M readings, 10,411 stations, pre-joined CHIRPS/ERA5/SMAP/HLS) | AIKosh dataset listing → ground_water_level_all_v1.zip | GODL-India (training data per repo) | CGWB/India-WRIS wells; panel span needs check on download | Contains readings (LOCF is their inference-time logic, not the panel) | PARTIAL — repo docs verified; panel itself pending download + provenance re-check vs India-WRIS raw | APPROVED for D1.1 acceleration AFTER provenance pass (sample wells vs raw WRIS: values/dates/units/round-months) |
| S4 | SoulVisionCreations NDVI/MWS dataset (fortnightly 14-day, micro-watershed level, MODIS NDVI + SPEI-3, ~5.4M rows) | AIKosh dataset listing | TBD on download (repo says GODL-style for CGWB; NDVI provenance = MODIS via GEE) | TBD | TBD — MODIS latency + fortnight cadence need audit before any use | NO — not yet fetched | v1.5 advisory-rendering candidate; NOT a foundation swap; district-level evaluation unchanged |
| S5 | CGWB / India-WRIS groundwater data (direct) | cgwb.gov.in → India-WRIS portal; NWIC; WIMS 6-hourly | GODL-India (government open data) | CGWB monitoring network, quarterly round-months + 6-hourly telemetric layer | D1.1 exists to verify round-months + lags (first logged evaluation) | YES — portal pages fetched/searched | D1.1 PRIMARY SOURCE |
| S6 | IMD 0.25° daily gridded rainfall | imdpune.gov.in (NetCDF/binary) | Free download; commercial terms need one check | 1901–2024 (124 yr) — covers train/val/test thru 2024 | Published as consolidated gridded product (station-based); as-of rules to log on adoption | YES — product pages found | STRONG acquisition: precip cross-validation vs ERA5/OM + independent covariate candidate |
| S7 | GRACE/GRACE-FO mascon solutions | grace.jpl.nasa.gov (JPL RL06.3_v04); csr.utexas.edu (CSR RL06.3, RL07 upcoming) | NASA/CSR open (NetCDF) | 2002–present, monthly | Mascon release latency rules → LATENCY_TABLE L1 lane (already encoded in EnKF spec: assimilate on RELEASE DATE ONLY) | YES — solution pages found | TARGET pipeline (EnKF spec, when ladder reaches it) |
| S8 | Kuruva et al. 2025, Nature Scientific Data — QC'd CGWB groundwater-level dataset | nature.com (Sci Data) | Paper license TBD (open data record) | CGWB-derived | TBD | PARTIAL — paper found | D1.1 cross-check source #2 |
| S9 | Agmarknet mandi price/arrivals | agmarknet.gov.in (+ Agmarknet 2.0 app) | Government portal; no official open API surfaced (3rd-party APIs exist, e.g. Farmonaut) | Daily, ~3000 mandis | Daily publication | PARTIAL — portal verified | v1.5 market layer (D#13/D#16); API/scrape terms check THEN |
| S10 | Wadhwani AI pest/vision datasets; ISRO Bhuvan LULC + NBSS&LUP soils; Soil Health Card portal; NIPHM/NCIPM advisories | TBD | TBD | TBD | TBD | NO — queued | NEXT SCOUT PASS (license+vintage each); vision/soil/advisory layers, post-MVP scope per D#16 |

## Register (cont.) — Task 56 consensus pass, 2026-09-15

3 domain scouts + adversarial reviewer (2-round consensus; full report:
ASSET_REGISTRY_CONSENSUS_GLM_REPLY.md). Tiers: T1a = Phase-1 critical
path; T1b = download now / use Phase-2 behind amendment; T2 = Phase-2;
T3 = backlog; REJ = rejected.

| # | Find | URL | License | Vintage / coverage | As-of behavior | Verified? | Status / decision |
|---|------|-----|---------|--------------------|----------------|-----------|-------------------|
| S11 | LGD district registry + boundary polygons (Bhuvan, India-Geodata, abhatia08 735-district) | lgdirectory.gov.in; bhuvan.nrsc.gov.in; community repos | LGD open; Bhuvan terms partial; community repos — record each | Current-vintage districts (AP 26 / TG 33) | Static; change ledger dated (TG 2016-10-11, 2019-02-17; AP 2022-04-04) | YES (scout) | T1a — BUILD spine + crosswalk (D#20). GADM EXCLUDED (non-commercial) |
| S12 | TerraClimate monthly 4km (PET/AET/PDSI/soil moisture) | climatologylab.org THREDDS; GEE; MPC | Citation-requested (NOT CC0 — catalog blurb corrected) | 1958-01–present (not 1950) | Monthly; consolidated product | YES (scout+reviewer) | T1b — consistency check only (feature freeze) |
| S13 | IMD 1° daily gridded max/min temperature | imdpune.gov.in (per-year binaries) | Free; commercial terms UNVERIFIED (legal-notes register) | 1951–2024 | Consolidated gridded product | YES | T1a (half-day) — temp consistency + Hargreaves PET sanity. No open 0.25° temp, no open PET grid |
| S14 | SSEBop operational actual-ET 1km monthly | earlywarning.usgs.gov/fews | USGS public domain | ~2012–present (anomaly baseline 2013-22) | Monthly GeoTIFF | YES | T1b — validation cross-check; feature candidacy Phase-2 |
| S15 | ICRISAT/TCI District-Level Database | data.icrisat.org/dld | CC BY-4.0 (verify at source) | ~1966–2017, 560+ districts | Static historical panel | YES | T1b — historical prior + method-inspiration ONLY (not a crosswalk input; 2017 frame ≠ our basis) |
| S16 | UPAg portal + data.upag.gov.in Swagger API + weekly sowing CSVs | upag.gov.in; data.upag.gov.in | Unlisted GoI (flag) | APY 1998→; weekly sowing current season (verified live 2026-09-11) | Weekly publication; APY ~1yr final-est lag | YES | T2 — ops-path + enterprise-API ingestion pattern; re-verify weekly. APY = AS-OF covariate class, not label |
| S17 | Kisan Call Center corpus (~40M records) | AIKosh / data.gov.in; HF benchmark mirror | GODL-India INFERRED — verify on download | 2017–2024, text | Text corpus (no labels) | PARTIAL | T2 — retrieval corpus for advisory delivery; NOT an AKB source (AKB = EXPERT_PRIOR: ANGRAU/PJTSAU/ICAR-CRIDA) |
| S18 | SPAM crop-allocation maps (IFPRI) | mapspam.info; Harvard Dataverse | CC BY-4.0 (pin exact version at source; "v2.2" unconfirmed) | ~2020 vintage, ~9km, 42 crops, irrigated/rainfed split | Static modeled allocation; as-of clean for 2023+ | PARTIAL | T1b — static crop-composition prior; drifts by 2026 |
| S19 | ESA WorldCover 10m v200 | esa-worldcover.org; Zenodo | CC BY-4.0-style (verify product page) | 2020 (v100) / 2021 (v200) | Static | YES | T1b — clean LULC alternative for mask build (v1.5) |
| S20 | Bhuvan/NRSC LULC 50K + 10K SIS-DP | bhuvan.nrsc.gov.in | NRSC: non-commercial/govt free; commercial needs permission | Multiple cycles; 10K 2018-23 (14 states) | Static per cycle | YES | T2 — Kharif/Rabi dual-crop classes (license friction) |
| S21 | CWC reservoir storage bulletins + NWIC portal | cwc.gov.in; nwdp.nwic.gov.in | GoI open data; license text absent (flag) | 123 daily / 161 weekly reservoirs (Apr-2025 portal) | Weekly PDFs; UI query, NO bulk API | YES | T2 — surface-water covariate, AS-OF with lag; scrape+cache never live-dep |
| S22 | GRACE downscaled TWSA 0.25° (Zenodo 2025); Figshare GWSA (2023) | zenodo.org; figshare.com | UNVERIFIED (likely CC BY) | GRACE era | GO/NO-GO: authors' training window + label sources + holdout; if CGWB/GWS obs ≥2020 in training → inadmissible as covariates 2020-25 entirely | PARTIAL | T2 — method audit BEFORE any use |
| S23 | Chronos-Bolt zero-shot TS model | github.com/amazon-science/chronos-forecasting; HF | Apache-2.0 (code + Bolt weights) | Current release | Pretrained on published public TS corpora (enumerate + check for India-hydrology) | YES | T1a — ladder rung, variant+config+hash pinned (D#23) |
| S24 | TimesFM 2.5-200m / TimesFM-3 | github.com/google-research/timesfm | ≤2.5 Apache-2.0; v3 hosted-only | Current | Hosted-only v3 = as-of/provenance violation | YES | TimesFM 2.5 T2 (deferred, compute); TimesFM-3 OFF-LIMITS |
| S25 | duckdb / pyet / pastas | github.com/duckdb, pyet, pastas | MIT × 3 | duckdb current; pyet v1.3.1 (pin); pastas active | Local tools — no as-of surface | YES | duckdb T1a (panel + ASOF JOIN); pyet T1a (ET0, pinned); pastas T2 (timebox 2d/5wells if D1.1 narrative needs) |
| S26 | Moirai (uni2ts) | github.com/SalesforceAIResearch/uni2ts | LIKELY Apache-2.0 per repo/PyPI — earlier NC claim WITHDRAWN pending direct LICENSE read; record on any future use | Current | Pretrained FM | PARTIAL (license conflict resolved toward Apache) | Deferred on ladder-freeze grounds (D#23), not license |
| S27 | TerraMind-1.0 / Clay / Microsoft Aurora | HF ibm-esa-geospatial; Clay-foundation; microsoft.github.io/aurora | Apache-2.0 / OpenRAIL-M (use-restricted) / MIT | 2025 / current / current | Pretrained FMs | YES | T3 / T3 / REJECT (Aurora: GPU + duplicates Open-Meteo role). Prithvi-EO-2.0 = in-hand CANDIDATE (gate outcome, not "frozen primary") |
| S28 | WeatherNext ensemble forecasts (Google DeepMind) | github.com/google-deepmind/weathernext | Apache-2.0 (code+weights); data path unpinned | Current (≠ GenCast — do not conflate) | Precomputed ensemble forecasts; BigQuery/GCS path needs check | PARTIAL | T2 — subseasonal lead-time features; pin product/access/license before dependency |
| S29 | AI4Bharat IndicConformer ASR + IndicParler-TTS | ai4bharat.iitm.ac.in; GitHub/HF | Varies — some releases CC BY-NC (blocker if commercial path activates); verify per-repo | Current | Model weights | PARTIAL | T2 — advisory delivery (Telugu first); per-repo LICENSE check before integration |
| S30 | CGWB Dynamic GWA district tables; Minor Irrigation Census; SoilGrids 2.5 + GSI geology; CWC discharge + IMD pan-evap | cgwb.gov.in (GWA 2017/2020/2022/2023); MI Census (6th 2013-14, 7th 2023-24); ISRIC SoilGrids; gsi.gov.in; India-WRIS gauges | GWA/MI = GoI open (flag); SoilGrids CC BY-4.0; GSI license caution | GWA biennial-ish, ~1yr pub lag; MI census periodic; SoilGrids static | GWA = lagged covariate with as-of discipline; MI = static-ish | PARTIAL | T2 — NEW adds from adversarial round: pumping-context (PR-7), well-density sanity, subsurface priors, observational water-balance anchors |
| S31 | MCP servers (geospatial/EO) | nasa/earthdata-mcp = only credible | varies | n/a | Agent-mediated nondeterministic access = hostile to as-of/provenance/hash | YES | REJ Phase-1 (SKIP; re-eval Phase-2+); REST + persisted raw + SHA256 manifests instead |
| S32 | eNAM; FASAL/CHAMAN outputs; AgriStack Crop Sown Registry; IWMI GIAM; Kaggle India GW Time-Series 1994-2025 | enam.gov.in; ncfc.gov.in; PIB; cgspace; kaggle.com | mixed / unlisted / closed / stale / murky | n/a | n/a | YES | REJ (no bulk access / outputs unpublished / closed today — quarterly watch / stale / provenance murk violates hash protocol). AgriStack = T3 watch |

## Standing scout queries (run each pass)

- site:aikosh.indiaai.gov.in — new groundwater / drought / crop models & datasets
- IMD gridded temperature / ET0 products (imdpune) — DONE 2026-09-15: 1° temp exists, no open PET/0.25° temp
- India-WRIS / NWDP data services and APIs — DONE 2026-09-15: UI+PDF only, no bulk API
- NASA/ESA open EO relevant to TWS/soil (SMAP L4, ESA-CCI soil moisture) — partially covered (S27/S30 class); full check next pass
- IIT/NIH groundwater forecast research artifacts (code+data) — partially covered (S22); keep standing
- Open agri-advisory corpora (KVK advisory text for AKB sourcing) — DONE 2026-09-15: no bulk KVK/mKisan corpus; KCC = retrieval corpus (S17); AKB = EXPERT_PRIOR (ANGRAU/PJTSAU/ICAR-CRIDA)
- Voice/Indic language primitives (AI4Bharat) — already adopted per D#15-class decisions; per-repo license check pending (S29)
- NEW: CGWB GWA next edition + MI Census 7th publication; AgriStack registry opening; Planetary Computer status; Open-Meteo pricing (L13)
