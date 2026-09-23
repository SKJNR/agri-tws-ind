# PRP ADVERSARIAL ROUND 1 — review + dispositions

**Date:** 2026-09-15 · **Reviewer:** fresh-context `general-purpose`
agent, Task ID 57-PRP-A1 (charter: skeptic; reads: PRP draft v1.0,
HAT_PROTOCOL, program/data/README, DECISION_LOG tail). **Draft under
review:** PRODUCTION_READINESS_PROTOCOL.md v1.0.

**Verdict: ADOPT WITH AMENDMENTS** — 3 BLOCK, 8 AMEND, 1 NIT.
**GLM disposition: all 12 ACCEPTED; incorporated in v1.1.**

| # | Sev | Objection (condensed) | Disposition |
|---|---|---|---|
| 1 | BLOCK | §8/DECISION_LOG declared the adversarial round "executed" with a file reference before the file existed — evidence-claim at birth violates the protocol's own prime directive | ACCEPTED. This file IS the real Round 1 record; §8 rewritten honestly; DECISION_LOG Addendum 2 corrected; standing rule adopted: gate records never reference uncommitted artifacts |
| 2 | BLOCK | No input-availability or deployment-continuity requirements; portals block the sandbox; no deployment target named; rollback can't restore data a portal won't serve | ACCEPTED. L2 += per-source ingestion runbooks (primary/fallback/founder-fetch/staleness ceiling/fail-loud); L3 precondition = named non-sandbox deployment target with continuity + cost plan |
| 3 | BLOCK | Advisory-harm axis entirely missing (farmer loses crop on bad tercile call) | ACCEPTED. Threat 5 added; L2 disclaimer/confidence spec; L3 misadvice incident class + calibration-keyed kill-switch; L4 human-in-loop escalation |
| 4 | AMEND | Twin rule unenforceable as written (22 signatories/gate, undefined "signature") | ACCEPTED. Per-gate signatory matrix (mandatory core vs consulted); sign-off artifact = §4 invocation record committed in-tree |
| 5 | AMEND | Ladder state unauditable; v12b/v28 "L1: manifest+md5" is L0 evidence; v21a L1 claim predates the pixi standard | ACCEPTED. PRP_LADDER_REGISTRY.csv created (machine-checkable, evidence links + hashes); §7 generated from it; v12b/v28 relabeled L1a PARTIAL; v21a L1a DONE with pixi re-verify queued |
| 6 | AMEND | Class A "L1 DONE" certifies frozen-input reproducibility, not production reproducibility | ACCEPTED. L1 split: L1a frozen-input rebuild / L1b live-input determinism + as-of contract (mock-clock CI); Class A cannot pass L2 on L1a alone |
| 7 | AMEND | Shadow mode missing between L2 and farmer-facing pilot — D#22 Path B already IS shadow mode | ACCEPTED. L3 phase 1 = SHADOW (= D#22 Path B window, criteria pre-registered there); phase 2 = farmer-facing pilot (pilot-12 per PR-10) only after shadow gate; expansion rule to 59/61 at L4 |
| 8 | AMEND | L3 monitors inputs but not references — well attrition, district-registry drift (59→61 pending), tercile-edge aging, ERA5T revision | ACCEPTED. Reference-decay monitors enumerated with thresholds; tercile re-derivation = PRD gate |
| 9 | AMEND | Licensing gaps: transitive inference-time deps of externals (SoulVision → Open-Meteo non-commercial + GEE); Zindi participant/solution terms on submissions; D#24 founder checklist floats outside the ladder | ACCEPTED. Class B hard requirements extended; Class A Zindi-terms gate at L2; D#24 checklist wired as L2 gate INPUT |
| 10 | AMEND | Trigger adjudication unenforceable; spurious-fire risk with overwhelm-prone founder | ACCEPTED. Triggers fire only via explicit founder statement logged as trigger event (quote + date + interpretation), twin-countersigned; no agent self-declaration |
| 11 | AMEND | Role holes: no farm-economics owner (MSP table orphaned), no Telugu/localization role; presentation-row exception carves the twin rule out of advisory content | ACCEPTED. Both roles + twins added; presentation split: styling (founder pass OK) vs advisory-content rendering (counter-expert + misinterpretation adversary MANDATORY) |
| 12 | NIT | PR-6 is the wrong reference for latency; LOCF ≤200 d is SoulVision's constant adopted as "discipline"; SLA undefined | ACCEPTED. Ref corrected to LATENCY_TABLE v1.1 (L13, G6/G7); LOCF marked placeholder-to-re-derive per source cadence; product SLA defined (advisory by the 5th, ≥95% of months) |

## Reviewer's three most important fixes (as stated)

1. Correct §8 and DECISION_LOG Addendum 2 — evidence integrity at birth.
2. Ingestion runbooks (L2) + named non-sandbox deployment target (L3
   precondition) — the actual feasibility wall.
3. Advisory-harm threat — disclaimer spec, misadvice incidents,
   calibration-keyed suspension, shadow mode before any farmer-facing
   pilot.

All three incorporated in v1.1.

## Full review text

Archived verbatim in the session worklog entry for Task 57-PRP-A1
(agent-cff33a64-771b-49f2-82c3-2d8891239fce) and reproduced in the
courier packet for Qwen when the next consensus round travels.
