# AGRI-TWS-IND — program phase (Tasks 41+)

Post-competition program: 16-feature frozen trunk, physical open/sealed
split, district basis, adversarial Qwen<->GLM consensus loop.

## Layout
- d20/            district basis + change ledger + well-assignment pipeline
- lgd_live/       live LGD registry export (2026-09-15), codes backfilled,
                  AP=28 finding, verification evidence + probe artifacts
- q57a_evidence/  Q57-A scout/license verification evidence
- import_kit/     founder import kit (LGD fetcher [now fallback], gwl
                  handback checker, polygon notes)
- scripts/        reproducible build scripts for the above
- data/           data persistence policy (ARRIVAL=COMMIT; see its README)
- worklog.md      shared multi-agent worklog (Tasks 41+)
- Q57A_VERDICTS_GLM_REPLY.md   adversarial round Q57-A

Canonical decision log: ../download/DECISION_LOG.md (competition era +
program phase). Master handoff: ../MASTER_HANDOFF.md.

## Sandbox restore procedure (after a session reset)
1. clone this repo (or fetch the latest daily bundle)
2. verify: git tag --list 'program-*'; sha256sum against MANIFEST files
3. read worklog.md tail -> last Task ID; resume there (never renumber)
4. data/: re-download Release assets per committed manifests if needed

## State at tag program-2026-09-15
D#20 built (59-basis; amendment 59->61 pending Qwen consensus after AP
26->28 reorg discovery); D#21 pre-staged (mocktest PASS); Q57-A round
delivered; LGD codes backfilled 59/59; import kit shipped; gwl_data.csv
handback + polygons await founder moves.
