# Q57-A — GLM ADVERSARIAL VERIFICATION OF QWEN'S COMPLEMENTARY SCOUT PASS
## (Q1–Q13) + implementation status · Loop: consensus required

> **FROM GLM · RE: Task 57 courier (Q57 add-list attack + implementation
> directives) · 2026-09-15 · Evidence: 30 web queries + direct API license
> reads (GitHub / PyPI / HuggingFace) + LGD portal probe, archived at
> `agri_tws_ind/q57a_evidence/`**

## 0. Honest accounting (mine, symmetric to yours)

- **What I ran:** 30 web-verified queries (28-query batch, all first-pass
  OK), direct license reads via PyPI JSON + GitHub repos API + HuggingFace
  model-card API (machine-grade citations where possible, not snippets), an
  LGD portal probe (live; findings below), and the D#20/D#21-prep builds
  with self-tests (all PASS, hashes below).
- **What I did NOT run:** a second 3-scout multi-agent fan-out. This round
  is a single adversarial lane over YOUR 13 items — API-grade verification
  per item instead of breadth. If you want the fan-out re-run on my
  amendments before consensus, say so and it runs.
- **Sandbox surprises (logged, no silent fixes):** (a) the /home mirror had
  regressed to the Task-40 snapshot — 9th split-brain reset; /tmp mirror is
  canonical through Task 56 and I re-anchored from it; all new artifacts are
  synced to BOTH mirrors plus a founder-facing copy. (b) **gwl_data.csv is
  NOT in the sandbox** — D#20 step 3 (10,411-well PIP) and the D#21 split
  are physically blocked on file arrival (founder was advised to keep a
  local copy — that copy is now the critical-path item). (c) LGD portal is
  live but captcha-gated; DWR programmatic calls rejected — details in Q-D20
  status.

## 1. Verdict table (Q57-A)

| # | Asset (as you proposed it) | Verdict | License (verified today) | Key findings & amendments |
|---|---|---|---|---|
| Q1 | xskillscore | **ACCEPT** (T1-impl) | **Apache-2.0** (PyPI v0.0.29, 2026-02-18; conda-forge; xarray-contrib) | Not redundant: S25 has duckdb/pyet/pastas, no verification-metrics lib. climpred carve-out AGREED (xskillscore only). CRPS via `crps_ensemble`; ROC-AUC stays with scikit-learn (see catch c3); reliability diagrams = bespoke plotting on xskillscore arithmetic. Pin at pixi env build. |
| Q2 | MAPIE **or** nonconformist | **AMEND → MAPIE only; nonconformist REJECT** | MAPIE **BSD-3-Clause**, scikit-learn-contrib, v1.5.0 (2026-08-05), pushed 2026-09-08, 1.6k stars. nonconformist: MIT but **last PyPI release 2017-06-20, last push 2021-03-20** — 9+ yrs stale | Mondrian per PR-10 = per-bin calibration loop in-house (auditable, no library-semantics lock-in); MAPIE provides the split-conformal core. Alternatives noted, NOT registered (pick-ONE rule): puncc (deel-ai, active, but **no machine-readable license** — flag), crepes (BSD-3, active). |
| Q3 | pyexactextract | **AMEND (name is dead) → register `exactextract`** | **Apache-2.0** (PyPI v0.3.0 2025-12-01; isciences/exactextract pushed 2026-02-24; conda-forge) | `pyexactextract` returns **404 on PyPI and GitHub** — the current installable is `exactextract` (C++/Rust core with Python bindings). Your LGPL worry is mooted (Apache-2.0). rasterio+geopandas slow path stays an in-repo fallback, unregistered. |
| Q4 | pixi/conda-lock + DVC | **AMEND (split the bundle)** | pixi **BSD-3-Clause** (pushed today 2026-09-15); DVC **Apache-2.0** (active, 15.9k stars); conda-lock MIT | pixi ACCEPT with committed `pixi.lock` as THE env pin (1-ULP incident insurance). **conda-lock REJECT — redundant with pixi.lock** (one-locker rule, same carve-out logic as statsforecast). DVC ACCEPT **with trigger**: adopt on the panel store's SECOND regeneration; SHA256 manifests (D#21) stay the primary hash protocol. |
| Q5 | ECMWF SEAS5 hindcasts+forecasts (CDS) | **AMEND (pin-at-fetch)** | **CC-BY-4.0 + ECMWF Terms of Use** (ECMWF retains IP; CDS licence regime moved to CC-BY on 2025-07-02 per ECMWF forum notice) | **SEAS6 is in implementation** (ECMWF materials: "once SEAS6 is the operational seasonal forecast system…"; SEAS4→SEAS5 parallel-run precedent). Registering "SEAS5" today risks building on a transitioning system: live CDS check MANDATORY at download to pin system + stream + hindcast depth; record member-count asymmetry (SEAS5: 25 hindcast vs 51 forecast members) for CRPS calibration. Pilot-12 + monthly-statistics stream only (compute cap agreed). Attribution string into LICENSES register at fetch. Fills the real gap vs S28 (subseasonal). |
| Q6 | IITM Extended Range Forecast | **AMEND (role demotion)** | GoI operational product (no explicit license; attribution; scrape-and-cache) | Verified real + current (IITM ERPS/erpas; MoES ERPS messaging Apr-2026; IMD 2-week ERF bulletins with issue timestamps). **But: no API (PDF bulletins) and no gridded lead-14 hindcast archive → un-backtestable → cannot be a G13 gate input under our own gate discipline.** S28 WeatherNext (Apache-2.0, active — re-verified today) keeps the h=1–4 gate-eligible role. IITM ERF = T2 operational reference: pilot face-validity, alert-verb cross-check, escalation corroboration. |
| Q7 | ONDC (Beckn protocol specs) | **AMEND (license-murk flag)** | **beckn.io "code of sharing" states CC-BY-NC-SA 4.0**; specifications-repo license must be read directly at activation | "Open protocol" ≠ license-cleared. ONDC network participation = commercial/ops question → stays on the founder's checklist (your section E). T2-market, Phase-2, no engineering this sprint — agreed. |
| Q8 | Bhashini ULCA production API | **AMEND (decision tree, per-artifact reads)** | Bhashini production APIs = **commercial pay-as-you-go** (bhashini.ai pricing page); ULCA platform code open (bhashini-dibd/ulca) | AI4Bharat licenses are heterogeneous AND conflict-prone (SoulVision precedent: card vs LICENSE file can disagree): **IndicConformer HF card = CC-BY-4.0** (commercial-OK — positive, partially resolves S29's NC worry); **IndicTrans2 HF card = MIT but GitHub LICENSE read pending** (API rate-limited) → conflict-suspect until direct read; Indic-TTS repo = gated (401). Fallback ladder: per-repo read → if NC/gated → Bhashini API (founder commercial read) → Whisper-large-v3 (**Apache-2.0**, verified today) + best-permissive TTS. No engineering this sprint. |
| Q9 | PMFBY CCE yield tables | **ACCEPT** (T2-validate) | GoI portal data (NCIP upload mandate verified in Operational Guidelines: crop-wise/area-wise historical + CCE yield data on National Crop Insurance Portal) | Validation anchor ONLY — agreed, never gate input, never training target. Riders: log vintage + revision state (advance vs final estimates); triangulate with UPAg APY (S16, already registered) + MoA "Agricultural Statistics at a Glance" — never single-source a validation claim. |
| Q10 | CGWB NAQUIM aquifer maps | **ACCEPT** (T2-context) | GoI publications repository (no explicit license; attribution + no-misrepresentation note) | Verified: cgwb.gov.in/cgwbpnm hosts district-level aquifer management plans. PDF/report grade → cache + manual extraction into a district-keyed context table; NO pipeline dependency. Complements S30 subsurface priors (SoilGrids/GSI). |
| Q11 | Livestock Census 2019 + NDRI THI + IGFRI fodder | **ACCEPT with 3 riders** (T2-AKB) | 20th LC (2019): DAHD district tables verified (99.4MB PDF + state mirrors); **GODL-India = commercial reuse OK with attribution** (verified) | Riders: (a) **21st LC launched 2024-10-25, fieldwork complete Feb 2025, results DELAYED** (ToI, May-2026) → standing query; use 20th with vintage logged until release. (b) THI bands must cite a NAMED ICAR/NDRI publication with formula + band table — today's check surfaced only the international canon (Armstrong 1994: <72 comfort / 72–79 mild / 80–89 moderate / ≥90 severe); Indian crossbred-cow bands get pinned at AKB-build time, not "NDRI says". (c) IGFRI catalogue verified (>300 forage varieties; Bajra-Napier hybrids IGFRI-7, DHN-6) — cache with retrieval date. |
| Q12 | IMD district warnings / CAP | **AMEND (primitive upgrade + role demotion)** | GoI operational alerts | **Better primitive found: SACHET (sachet.ndma.gov.in) — NDMA's CAP-based national alert portal + app** (WMO Oct-2025 confirms; CDOT CAP-Integrated Alert System operational pan-India, 36 states/UTs). Structured CAP beats scraping IMD district pages; IMD warnings (mausam.imd.gov.in + internal.imd.gov.in bulletins, issue-timestamped) = fallback. **Role: corroboration + escalation input in pilot, NOT primary trigger** — triggers must come from the registered, backtestable forecast stack; an alert path keyed to scraped pages is un-backtestable and fragile. As-of = alert issue time. CAP ingestion = Phase-2 engineering. |
| Q13 | Sentinel-1 SAR soil-moisture proxies | **ACCEPT** (T3-backlog) | Copernicus free/open | Agreed, mention-only. Literature active (Rahmati et al. 2026 review; HESS 1km products) — it's a methodology, not a data plug. Revisit trigger: SMAP/ERA5-Land validation showing systematic soil-moisture bias in NAMED districts. |

**AM-5 status: intact.** Nothing above enters the frozen 16-feature backbone
or the regime-map inputs. Q1–Q4 are tooling (T1-impl); Q5–Q12 are gated
non-feature assets; Q13 parked.

## 2. Catches your self-attack missed (c1–c6)

- **c1 — register numbering:** SCOUT.md already ends at **S32** (Task 56
  extension), so consensus entries start at **S33**, not S31+. D#25 as the
  next decision number is correct.
- **c2 — pyet version drift:** S25 pins v1.3.1; **v1.5.0 exists**
  (2026-05-26). Re-pin at pixi env build; one line, logged.
- **c3 — scikit-learn is not in SCOUT.md** (implied by the LGBM smoke step
  but never registered). Register at ladder-build time — the registry
  should name every importable.
- **c4 — S29 partial resolution:** today's reads resolve two of the three
  voice-lane licenses (IndicConformer CC-BY-4.0; Indic-TTS gated). Log at
  consensus.
- **c5 — puncc license hole:** no machine-readable LICENSE in deel-ai/puncc.
  If it ever becomes the conformal pick, direct LICENSE read first.
- **c6 — CDS attribution regime:** ECMWF datasets carry CC-BY-4.0 **plus
  ECMWF Terms of Use** (IP retained). The LICENSES register needs the exact
  attribution string at fetch — a one-liner that is easy to lose.

## 3. Implementation status (Part B, frozen order — AM-6 respected)

**Step 1 (D#20) — DELIVERED as far as data permits** (`agri_tws_ind/d20/`,
manifest `MANIFEST.sha256`):

| Artifact | sha256 (prefix) | State |
|---|---|---|
| district_basis.csv (59, 2026-vintage, LGD-style names + local names + formation/parentage) | d3b751f8… | DONE; `lgd_district_code` = PENDING_FOUNDER_EXPORT |
| change_ledger.json (4 events — 3 VERIFIED via multiple independent sources + 1 UNVERIFIED rename cluster; provisional parentage maps; 3 built-in tests T-D20-1/2/3; blocked-on list) | 4fd910b7… | DONE |
| d20_assign_wells.py (PIP pipeline; deterministic smallest-area tie-break; unassigned-with-reason codes OUTSIDE_LAYER/DEGENERATE_COORD/POLYGON_GAP/NO_POLYGONS) | 39cbd3ce… | SELFTEST PASS 5/5 (incl. overlap + gap + degenerate + empty-layer paths) |
| D20_SENSITIVITY_NOTE.md (old-basis TRAIN-era analysis + pre-registered quantitative check) | 508bbc9e… | DONE |
| d21_physical_split.py (pre-staged step 2) | 97f71407… | MOCKTEST PASS (boundary 2022-12-31 open / 2023-01-01 sealed; fail-loud unparsable-date exclusion; row reconciliation; SHA256 manifest; sealed-dir lock + embedded unseal procedure + ACCESS_LOG stub) |

**Blocked-on (precise):**
1. **gwl_data.csv absent** → well-level assignment (T-D20-2) and the D#21
   split cannot run. Founder: re-upload, or approve the AIKosh download —
   Task-55 condition applies on arrival (provenance spot-check vs
   India-WRIS raw: values/dates/units/round-months on sample wells).
2. **LGD codes**: portal live; `districtWiseDetailReport.do` captcha-gated;
   DWR plain calls rejected. Founder one-time manual export (state codes
   AP=28, TG=36) OR PENDING markers stand until polygon binding.
3. **Polygons (LGD/Bhuvan-grade)** not in sandbox. GADM banned (D#20); OSM
   = community-grade fallback only with a logged amendment. Bhuvan Bhoonidhi
   likely needs portal interaction — founder assist or a documented
   alternative ruling needed.

**Steps 3–8: queued, untouched (correct posture).** duckdb panel store /
pyet ET0 / IMD 1° consistency / D1.1 / regime freeze / ladder / LGBM smoke
all wait on 1–2 per the collapsed critical path. Env pinning will use pixi
(Q4-AMEND) if ratified at consensus.

## 4. Q57-B handoff

Resolve or concede per item; unresolved after round 2 → founder tie-break.
My pre-stated concede-points, to save us a round: (i) DVC — if you insist
on adopting it immediately rather than on second-regeneration trigger, I
concede (cheap either way); (ii) IITM ERF — the demotion stands UNLESS you
can produce a documented gridded ERF hindcast at lead-14, in which case its
gate-eligibility re-opens with an as-of audit; (iii) SACHET — if the CAP
feed turns out non-public, IMD-page scrape-and-cache becomes the fallback,
but the trigger-role demotion stands either way. Everything else I defend
as amended. On your acceptance, consensus logs as **S33+ / D#25**, and I
will append the register rows + LICENSES entries + D#20/D#21 completion in
the registered order the moment the blockers clear.

— GLM, 2026-09-15.
