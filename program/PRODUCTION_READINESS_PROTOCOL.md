# PRODUCTION READINESS PROTOCOL (PRP) v1.1 — finalized models → production use

**Adopted: 2026-09-15 (founder ruling, chat).** Extends `HAT_PROTOCOL.md`
(2026-09-07) from the competition era into the program + production era.
Governs ANY move that takes a finalized model toward "production ready to
use." Any session/agent resuming this project MUST follow both files.

**v1.1 = post-adversarial revision.** Round 1 (Task 57-PRP-A1, fresh
context) returned 3 BLOCK + 8 AMEND + 1 NIT; all 12 accepted and
incorporated. Review + dispositions: `PRP_ADVERSARIAL_ROUND1.md`.

## 0. Prime directive

Productionization is a **GATE OUTCOME, never a declaration** (D#19
precedent). Nothing in this file pulls work forward: the AM-6 frozen
order, feature freeze, sealed windows (D#21), and the D#22 three-arm
pre-registration are **untouched**. This protocol pre-registers HOW
production happens when triggered — it does not trigger it.

**Trigger conditions (any one):** (a) founder production demand AFTER the
advisory engine shell exists; (b) three-arm verdict (D#22) favoring an
adoption vote; (c) a Phase-2 product milestone requiring live model
output. Until a trigger fires, production work is OUT OF SCOPE and must
not consume engineer time, compute, or decision numbers.

**Trigger adjudication (v1.1, from Round-1 obj. 10):** a trigger fires
ONLY via an explicit founder statement logged in DECISION_LOG as a
trigger event (verbatim quote + date + interpretation), countersigned by
an adversarial twin confirming the conditions are met. No agent may
self-declare a trigger, and no casual remark may be read as one.

## 1. Model classes (packaging differs — never treat them alike)

| Class | Members | Production artifact | Hard requirements |
|---|---|---|---|
| **A — deterministic pipelines** | v21a, v12b, v28 chain | pinned code tag + input contracts + rebuild hashes | bit-exact rebuild (md5), fixed seeds, no framework drift |
| **B — pre-trained externals** | Chronos-Bolt (D#23 rung), Prithvi candidate, SoulVision (conditional) | weights + config + license record, all SHA256'd | weights hash pinned; license gate covering **transitive inference-time dependencies** (SoulVision live path → Open-Meteo non-commercial free tier L13 + live GEE auth+project — adopting it encumbers the whole live pipeline); contamination gates (AM-1/AM-2: outputs admissible ONLY in 2026+ live contexts, never covariates for val/test-scored artifacts) |
| **C — our future trained models** | ladder rungs, LGBM smoke | training card + data snapshot + model file | seeds, data-snapshot hash, full training repro card written BEFORE first production use |

Class A additionally carries **Zindi participant/solution terms** on the
submission artifacts themselves (distinct from the data's CC-BY-SA) —
resolved at the L2 license gate before any commercial use (obj. 9).

## 2. Staging ladder L0–L4 (no level skipping; no level declared without evidence)

- **L0 FROZEN ARTIFACT** — pinned code tag, input manifest, SHA256s,
  license provenance recorded. *(Competition era: DONE — submissions
  manifest.)*
- **L1a FROZEN-INPUT REBUILD (v1.1 split, obj. 6)** — bit-exact rebuild
  from raw frozen inputs. *(v21a: DONE — md5
  `6b6e3e41c25317a68089e6b9ca707c05`, under the then-current environment
  standard; re-verify under pixi.lock once the pinned env exists, per
  Q57-A Q4.)* **L1b LIVE-INPUT DETERMINISM** — determinism + as-of
  contract on live inputs, verified with mock-clock CI at L2 build time.
  **Class A may not be promoted past L2 on L1a alone.**
- **L2 HARDENED INFERENCE PACKAGE** — self-contained runner; input schema
  validation (fail-loud, never impute silently); output bounds and
  guardrails; as-of discipline at inference; latency budget per
  **LATENCY_TABLE v1.1 (L13, G6/G7)**; rollback artifact attached.
  **v1.1 additions (obj. 2, 3, 9):** per-source **ingestion runbook**
  (primary path, fallback, founder-fetch procedure per the import-kit
  pattern, staleness ceiling derived per source cadence, fail-loud
  behavior when unreachable — the ≤200 d LOCF figure inherited from
  SoulVision's reference implementation is a placeholder to re-derive,
  not a standard); **advisory disclaimer + confidence-language spec**;
  license gate extended to transitive inference-time dependencies of
  externals AND Zindi participant/solution terms; the **D#24 founder
  commercial-terms checklist (IMD/WRIS/Bhuvan, Open-Meteo pricing) is a
  gate INPUT, not a floating to-do.**
- **L3 MONITORED PILOT — two phases (v1.1, obj. 7):**
  **Phase 1 SHADOW** = D#22 Path B operational window (both models
  forecast; advisories generated and scored against arriving CGWB
  readings; delivered to NO ONE; success criteria pre-registered there).
  **Phase 2 farmer-facing pilot** (pilot-12 cohort per PR-10, expansion
  rule to 59/61 districts at L4) only after the shadow gate passes.
  **PRECONDITION (obj. 2):** a named, NON-SANDBOX deployment target with
  continuity and cost plan — the sandbox resets and portals block it;
  production cannot live here.
  **Monitors:** drift + staleness (ceilings per L2 runbooks);
  **reference-decay monitors with thresholds (obj. 8):** monitoring-well
  attrition/orphan rate; district-registry drift as an operational event
  (the pending 59→61 amendment — Markapuram/Polavaram, eff 2025-12-29 —
  will land mid-pilot if unhandled); tercile-edge aging alarm (frozen
  ≤2019 extrapolation) whose re-derivation is itself a PRD gate; upstream
  reanalysis revision watch (ERA5T). **Misadvice incident class (obj. 3)**
  distinct from infra incidents, with severity taxonomy and farmer-facing
  correction/retraction procedure; **kill-switch keyed to calibration
  monitors, not just uptime**; logged incidents; tested kill-switch.
- **L4 PRODUCTION** — multi-district rollout; **product SLA (obj. 12):
  advisory issued by the 5th of each month, ≥95% of months** (the G7
  ADVISORY_ISSUE calendar); monitoring dashboards; incident procedure
  rehearsed; **escalation/human-in-loop for severe calls (obj. 3)**;
  licensing re-verified at deployment scale.

Every level-up is a **PRD gate** (numbered PRD-1, PRD-2, … in
DECISION_LOG): pre-registered numeric criteria + adversarial pass +
sign-offs per the §3 matrix. Skipping a level or declaring one without
evidence = protocol breach, logged as such. **Standing rule (from
Round-1 Block 1): a gate record may never reference an uncommitted
artifact — evidence lands in-tree FIRST.**

## 3. Hat/role matrix — every role has an adversarial twin

**Binding rule (founder, 2026-09-15): for each agent invoked, its
adversarial counterpart is invoked. No unopposed agent output is ever
gate evidence.** Objections are logged, not silenced.

**Enforcement (v1.1, obj. 4):** the twin rule is enforced through a
**per-gate signatory matrix**, not 22 signatures on everything:
- PRD-L2 (Class A): build engineer + packaging attacker + QA + gate
  auditor **mandatory**; others consulted.
- PRD-L3: + SRE/chaos + hydrologist + adversarial hydrologist + agromet
  scientist + counter-expert.
- PRD-L4: + red-team lawyer + deployment skeptic.
- **Sign-off artifact** = the §4 invocation record (charter, inputs read,
  numbered objections, verdict, worklog IDs of BOTH invocations),
  committed in-tree. An unopposed record is invalid gate evidence.

### 3a. Process ("normal") roles

| Role | Owns | Adversarial twin | Twin hunts for |
|---|---|---|---|
| Product/PM | scope, user story, de-scope decisions | Scope skeptic | feature creep, scope drift, shipping without a user |
| Build engineer | packaging, runner, deps | Packaging attacker | unpinned deps, supply chain, "works on my machine" |
| QA/validation | gate evidence, test design | Gate auditor | evidence-vs-claim gaps, tests that can't fail |
| DevOps/SRE | deploy, rollback, monitoring | Chaos engineer | unrehearsed rollback, silent monitor death, dependency outage |
| Security/licensing | licenses, privacy, terms | Red-team lawyer | commercial-term breaches incl. TRANSITIVE deps (Open-Meteo L13, GEE, Zindi participant terms, CC-BY-SA share-alike, GODL attribution, AIKosh terms, Beckn CC-BY-NC-SA pin) and advisory-liability exposure |
| Release manager | level-up decisions | Deployment skeptic | premature promotion, survivorship bias in pilot reads |

### 3b. Domain-expertise roles

| Role | Owns | Adversarial twin | Twin hunts for |
|---|---|---|---|
| Hydrologist | physical plausibility, regime drift | Adversarial hydrologist | unphysical shortcuts, leakage across seals, spurious skill |
| Agromet scientist | advisory semantics, tercile frames, AKB | Counter-expert | misleading confidence, wrong-stage advisories, tercile edges that moved |
| **Farm-economics/agronomy owner (v1.1, obj. 11)** | MSP table, crop-choice economics, cost-benefit framing | Market-reality skeptic | MSP distortion, basis risk, incentive attacks on crop mix |
| **Localization/accessibility owner (v1.1, obj. 11)** | Telugu surfaces, low-literacy rendering, voice-channel delivery | Misinterpretation adversary | tercile-language mistranslation, voice ambiguity, numeracy traps |
| ML engineer | model classes, calibration, blending | Overfitting skeptic | contamination geometry (AM-1/AM-2), calibration asymmetry, rung shopping |
| Data engineer | as-of joins, provenance, vintage | Provenance auditor | staleness, vintage mixing, silent fixes, orphan wells |
| Metrologist | uncertainty, σ propagation | Measurement adversary | how the numbers could lie (clipping, rounding, split drift) |

**Presentation split (v1.1, obj. 11):** styling work (dashboards, decks)
→ founder review is an acceptable adversarial pass. **Advisory-content
rendering — any user-facing advisory text or voice — REQUIRES the agromet
counter-expert + the misinterpretation adversary.** A founder eyeball is
not sufficient coverage where the harm does its damage.

## 4. Subagent mapping (real machinery, code-triggered)

When actual code or artifact work exists, hats map to concrete agent
invocations:

| Hat pair | Invocation |
|---|---|
| Architect + skeptic | `Plan` agent with architecture charter → independent `general-purpose` adversarial charter on the resulting plan |
| Engineer + attacker | `full-stack-developer` for inference/API/dashboard code → `general-purpose` robustness/security review of the diff |
| Forensics/auditor | `Explore` agent (codebase archaeology, license reads) → `general-purpose` verification pass on its claims |
| Presentation (styling only — see §3 split) | `frontend-styling-expert` / `ppt-expert` → founder review |
| Advisory-content rendering | agromet counter-expert + misinterpretation adversary charters (mandatory twins) |
| Research/scout rounds | `general-purpose` with scoped charters, always in pairs (scout + skeptic) |

**Mandatory invocation pattern:** charter (role, scope, fail-loud rules,
exact inputs) → deliverable → **adversarial pass by a DIFFERENT agent**
(or a fresh-context rerun with a skeptic charter) → verdict
ACCEPT/AMEND/REJECT → worklog entry naming BOTH invocations.

**Status (2026-09-15):** no production code exists yet, so no
engineering agent is warranted today. The one legitimate invocation was
the adversarial review of THIS protocol — executed as Task 57-PRP-A1
(§8).

## 5. Chess-moves strategy (production edition, house style)

- **Opening (already played):** provenance hashes, sealed splits, frozen
  backbone, ARRIVAL=COMMIT — a development advantage we carry forward.
  Do not trade it away for speed.
- **Midgame:** ladder + three-arm = piece coordination. D#23 rung freeze
  = never move the same piece twice (anti-rung-shopping). Pre-registration
  = tempo: the reply is on the board before the opponent moves.
- **Threat ladder at production** (counter → gate):
  1. Licensing breach (Open-Meteo commercial use on free tier; Zindi
     participant terms; CC-BY-SA obligations; transitive deps of
     externals) → license re-verification at L2 and again at L4.
  2. Silent data drift (vintage mixing, pipeline staleness, reference
     decay — wells, districts, tercile edges, ERA5T) → L3 monitors with
     thresholds; fail-loud schemas at L2.
  3. Overconfident advisory (calibration asymmetry) → calibration
     monitors + disclosure on every user-facing surface.
  4. Infra failure / input unavailability (portal blocks, resets) →
     ingestion runbooks with fallbacks + founder-fetch; rollback
     rehearsed BEFORE L4.
  5. **Advisory harm (v1.1, obj. 3)** — a farmer loses a rabi crop on a
     bad tercile call → L2 disclaimer/confidence spec; L3 misadvice
     incident class + calibration-keyed suspension; L4 human-in-loop
     escalation for severe calls.
- **Pins:** unresolved license terms pin the whole lane (Beckn precedent;
  SoulVision's GEE/Open-Meteo transitive pin). Do not move a pinned
  piece — lift the pin (read the terms, price the tier) first.
- **Forks:** an artifact serving both evaluation and production must
  fail-loud in BOTH roles, or be split into two artifacts.
- **Sacrifices:** de-scope v1.5 features to ship the MVP; sacrifice
  polish, never seal integrity.
- **Endgame:** deployment = promotion; only promote when the passed
  pawns (monitors, rollback, runbooks) already exist. Zugzwang avoidance:
  at L3+, standing still without monitoring is itself a losing move.
  Retreat with tempo: a rehearsed rollback loses nothing but time.
- **Every level-up writes the opponent's three best replies** (failure
  scenarios) into the gate record before the move is made.

## 6. The Claude loop + cross-AI courier (when "if needed" fires)

**Standing loop (HAT_PROTOCOL, unchanged):** build → self-review (hat
stack) → adversarial review (fresh context when stakes are high) → GATE
(pre-registered numeric criteria) → deploy → verify (exact prediction
where possible) → worklog entry.

**Cross-AI courier rounds** (Qwen precedent; Claude or any counterpart
the founder relays, same mechanics): GLM draft → courier JSON (claim
list + evidence hashes) → counterpart reviews with FRESH context (no
shared history = independence) → numbered objections → GLM verdicts
ACCEPT/AMEND/REJECT with pre-stated concede-points → consensus →
DECISION_LOG entry.

**Mandatory fires:** (1) before any L3→L4 promotion; (2) any licensing
interpretation with commercial stakes; (3) any AM-1/AM-2 contamination
edge case arising at production time; (4) founder demand.

## 7. Ladder registry (v1.1, obj. 5 — generated from `PRP_LADDER_REGISTRY.csv`)

| Model | L0 | L1a | L1b | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| v21a (Class A) | DONE | DONE (frozen-input, then-standard; pixi re-verify queued) | not started | not started | — | — |
| v12b (Class A) | DONE | PARTIAL (builder exists; bit-exact re-verification pending) | not started | not started | — | — |
| v28 (Class A) | DONE | PARTIAL (builder exists; bit-exact re-verification pending) | not started | not started | — | — |
| Chronos-Bolt (Class B) | PENDING (weights deferred per D#22 pre-reg) | — | — | — | — | — |
| SoulVision (Class B) | CANDIDATE (Class B gates + contamination geometry stand) | — | — | — | — | — |
| Program models (Class C) | NOT_TRAINED (pre-D1.1) | — | — | — | — | — |

Registry is the source of truth (`program/PRP_LADDER_REGISTRY.csv`,
committed); this table must never disagree with it. Evidence links +
hashes live in the registry, not in prose.

The halt at L2 is CORRECT: the advisory engine shell does not exist yet,
and the ratified architecture embeds the competition models as the
surface-water/meteorological layer of that engine — not as standalone
services. Production work before the shell exists would be building a
roof before walls.

## 8. Adversarial round on this protocol (Round 1 — executed 2026-09-15)

**Executed as Task 57-PRP-A1** (fresh-context `general-purpose` agent;
charter and reads logged in the session worklog). Verdict: **ADOPT WITH
AMENDMENTS** — 3 BLOCK, 8 AMEND, 1 NIT; all 12 ACCEPTED and incorporated
into this v1.1 (the draft's original §8 claimed the round prematurely —
Block 1; that claim is hereby corrected, and the standing rule "gate
records never reference uncommitted artifacts" adopted from it). Full
review + per-objection dispositions: `PRP_ADVERSARIAL_ROUND1.md`.

## 9. Guards and registration

- AM-6 frozen order untouched; no code, model, or data operations were
  performed in adopting this protocol.
- No decision number consumed: logged as a founder-ruling addendum in
  DECISION_LOG. Optional Qwen ratification rides the next courier.
- Register collision check: gates numbered **PRD-n** (distinct from
  D#, G#, T#, PR-n principles, AM-n amendments). Latency discipline
  cites LATENCY_TABLE v1.1 (L13, G6/G7) — not PR-6, which is the E-NC
  arms principle (corrected per Round-1 obj. 12).
