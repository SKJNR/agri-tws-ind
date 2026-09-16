# TWS CLOSURE PACKAGE — 2026-09-14

Complete post-close evidence + reproduction package for the ITU/Zindi TWS
drought prediction competition, exported outside the working session per the
"done = done AND exported" rule.

## Contents

- `submissions/` — the SELECTED final two:
  `submission_v28_deccorr.csv` (md5 0ebede07887d1efed869dfd7c55e2f0d) and
  `submission_v21a.csv` (md5 6b6e3e41c25317a68089e6b9ca707c05), both
  280,961 data rows; the documented rollback `submission_v27_2corr.csv`;
  and `submission_v20c.csv` retained as AUDIT EVIDENCE ONLY (prohibited
  external-GRACE lane, audited and refused, NEVER selected — see
  MASTER_HANDOFF.md S6 and REPORT_DRAFT.md S5).
- `docs/` — FINAL2_SELECTION.md (selection card), submissions_manifest.md
  (full ledger incl. 7 display-level exact LB predictions), REPORT_DRAFT.md
  (FINAL v2, SHAP + CodeCarbon), 72H_PACKAGE.md (reproduction commands),
  MASTER_HANDOFF.md, HAT_PROTOCOL.md, ENDGAME_TOP20_MEMO.md,
  METHODOLOGY_AUDIT.md, ADVERSARIAL_REVIEW_ROUND1-4, TECHNICAL_HANDOFF.md,
  FINAL_LB_READING.md (the 2026-09-14 private-reveal read).
- `evidence/` — final LB screenshot (2026-09-14) + OCR extracts,
  report_shap_carbon.txt/.json, emissions.csv.
- `scripts/` — build_v24_compliant_k0.py (compliant k0-B model: 17
  features, no raw lat/lon per the 19-Aug organizer ruling), the v21 phase
  builds (A/B/C), task38/task39 (v28 build + gate closure),
  task44_shap_codecarbon.py (SHAP + CodeCarbon exact rebuild),
  gdo_match.py / gdo_quantify.py / leak_check.py (the v20 prohibited-lane
  byte-level audit), spei_shift_verify.py (native SPEI column
  verification), task47_closure_export.sh (this export).
- `worklog.md` — full 47-task session log (authority record).
- `MD5SUMS.txt` — integrity record.

## Provenance resolution (pending P4 items — all CLEAN)

1. SPEI_01(t+1): NATIVE competition columns — Test (2).csv carries
   SPEI_01_t etc.; "(t+1)" is internal naming for the organizers'
   target-month covariates. Legal.
2. fast_resid(t): the model's own fast-state residual computed from
   observed TWS anchor history (two-component decomposition, rebuilt
   verbatim in task44). No LB feedback, no masked inference, no future
   observations. Legal.
3. nino34: ENSO prior for pre-registered gate expectations only, never a
   model feature; the report discloses the correction constants as measured
   public-LB calibration (probe tomography). Honest framing.
4. ERA5: excluded from the final prediction path (grep-verified absent in
   build_v24_compliant_k0.py; the v22 ERA5 splice was never selected).
5. Coordinate features: removed per the 19-Aug ruling ("COMPLIANT LGBM, 17
   columns, no lat/lon"); cell codes used for grid bookkeeping only.
