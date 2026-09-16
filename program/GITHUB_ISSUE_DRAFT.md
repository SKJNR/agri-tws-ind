# GITHUB ISSUE DRAFT #1 — paste into a new Issue after the repo is up
**Title:** `D#20 amendment (59 vs 61 districts) + provisional assignment countersign + PRP v1.1 review + data findings`
**Labels (create if missing):** `decision-needed`, `qwen-input`
**Body = the text below, verbatim.**

---

## COURIER 2026-09-16 — four asks (GLM; Tasks 58-59 + pending LGD ruling)

### 1. D#20 amendment ruling: 59 vs 61 districts
Live LGD registry (headless-browser fetch, 2026-09-15, raw payload + SHA256 in repo `program/lgd_live/`): TG 33/33 match; AP has **28 rows, not 26** — Markapuram (LGD 790, from Prakasam) and Polavaram (LGD 791, from Eluru/NTR/West Godavari agency belt), both effective 2025-12-29; verified vs 4 independent sources + markapuram.ap.gov.in.
**Position A (GLM): amend D#20 to 61-district constant basis** — product serves 2026 users; current polygons will carry 28 AP features; sealed-window logic untouched (both new territories sat inside parents through the whole sealed window; no AM-2/AM-3 implications). Position B: keep 59, mark 61 as future vintage (mislabels two real 2026 districts; polygon acceptance checks would need rewording).
Reply: **ACCEPT A / ACCEPT B / CONTEST (state your case)**.

### 2. Countersign — D#20 execution amendment (provisional label-map assignment)
Polygons are not yet in sandbox, so point-in-polygon (T-D20-2) cannot run. Interim: explicit per-label map (47 observed AP+TG labels enumerated, validated against the 59-name basis, fail-loud on unknowns; caught 2 spelling-variant bugs on first run). Parent-label wells assigned to the CONTINUING district, flagged PARENT_CONTINUE with child set: 2,471/3,223 wells = 56.8% of open rows; 14 basis districts hold zero wells until polygon binding. Target basis UNCHANGED; ambiguity flagged per well; T-D20-2 rerun queued on founder polygon export. If Ask 1 = A, the map gains Markapuram/Polavaram parentage in the same rerun.
Reply: **COUNTERSIGNED / OBJECT (reason)**.

### 3. Data findings (affect loader L1-L12 + D1.1 design)
- District labels in gwl_data.csv are CONSTANT per well and BACK-APPLIED (post-2016 TG names on 1999 readings) — there is no as-of-reading vintage frame. Loaders must key district from our assignment layer (or polygons), never the raw column.
- gwl_value: 114,062/365,742 AP+TG open rows negative (31%), range -1147.7..+971.3 — unit/datum question (mbgl vs masl vs QC failures) routed to the Task-55 India-WRIS spot-check BEFORE D1.1 conclusions rest on values. D1.1 harness should treat units as UNRESOLVED until then.
- Month census (D1.1 feed): all-India open peaks Nov/Aug/May (CGWB round structure visible); per-district census in `program/data/manifests/panel_build_report.json`.
- State: gwl_data arrived (HF route, bit-perfect); D#21 split executed (open 992,053 / sealed 2,286,174 / reconciliation PASS, sealed dir locked); duckdb panel built (365,742 AP+TG rows, 3,223 wells all assigned; SoulVision covariates carried REFERENCE-ONLY — our Open-Meteo foundation stands). NDVI model zip NOT downloaded (D#22 weights ban).
Reply: **NOTED + any design objections**.

### 4. PRP v1.1 review
Founder adopted the Production Readiness Protocol v1.1 (model classes A/B/C; ladder L0-L4 with L1a/L1b; PRD-n gates; mandatory adversarial twins via signatory matrix; agent mapping; chess-move strategy incl. advisory-harm; Claude loop + courier triggers; trigger adjudication = founder statement + twin countersign). Self-reviewed adversarially pre-adoption: ADOPT-WITH-AMENDMENTS, 3 BLOCK + 8 AMEND + 1 NIT, all accepted (standing rule born of Block 1: gate records never reference uncommitted artifacts).
Docs in repo: `program/PRODUCTION_READINESS_PROTOCOL.md`, `program/PRP_ADVERSARIAL_ROUND1.md`, `program/PRP_LADDER_REGISTRY.csv`. AM-6 untouched — protocol pre-registers the production path, does not trigger it.
Reply: **APPROVE / AMEND (list)**.

---

**Qwen homework unchanged (now unblocked):** loaders L1-L12 w/ mocked-clock tests (spec: LATENCY_TABLE.md v1); E-NC harness instrumentation; CGWB round-month verification = D1.1 (first logged evaluation; waits for founder pilot-12).
**Unchanged:** AM-6 order; D#22 window 2026-01-01+ and weights ban; sealed 2023-25; no model runs.
