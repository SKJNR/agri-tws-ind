# 72H CODE + REPORT PACKAGE — pre-staged 2026-09-10 (Task 44)

**Trigger:** if we finish top-10 on the PRIVATE leaderboard after close
(13 Sep 21:29), organizers allow 72 hours to submit model code + report.
Everything below is ALREADY assembled — the 72h window then costs zero
assembly time.

## 1. Package contents (all in ONE place: the git repo)

| Item | Location in repo | Notes |
|---|---|---|
| Full code lineage | `scripts/` (386+ files, v1→v28: every builder, probe, gate, audit) | worklog maps every file to its decision |
| Trustworthiness report | `download/REPORT_DRAFT.md` | FINAL v2 — SHAP + CodeCarbon attached |
| SHAP + CO2 evidence | `download/report_shap_carbon.{txt,json}`, `scripts/task44_shap_codecarbon.py` | reproducible: python3 scripts/task44_shap_codecarbon.py |
| Submission manifest | `download/submissions_manifest.md` | every upload + pre-registration + actual + residual |
| Selection card | `download/FINAL2_SELECTION.md` | picks, md5s, rollback, DQ-trap warnings |
| Compliance audits | `download/ADVERSARIAL_REVIEW_ROUND{1..4}.md`, `audit{A,B,C}_report.md`, `METHODOLOGY_AUDIT.md`, `MASTER_HANDOFF.md` | incl. the v20 audit-and-refusal trail |
| Full narrative | `worklog.md` (Tasks 1–44) | every success AND failure, with numbers |
| LB evidence | `leaderboard.json`, `lb_now.html`, `upload/` screenshots | |

## 2. Reproducing the two selected files

```
# slot-1: submission_v28_deccorr.csv  (public 0.674859467)
python3 scripts/build_v24_compliant_k0.py      # k0-LGBM base (compliant, no lat/lon)
#   -> then chain v25->v28 correction builds (build_v25_and_k0_probes.py etc.)
#   -> md5 0ebede07887d... (full chain + md5s in submissions_manifest.md)
# slot-2: submission_v21a.csv  (public 0.687374005)
python3 scripts/build_v21_phaseC.py            # deterministic, no LightGBM; 16 s
#   -> md5 6b6e3e41c253...
```
Data: competition package from Zindi (`data/Train (1).csv`, `Test (2).csv`) —
NOT in the repo (rules); everything else regenerates from these two files only.

## 3. Durability map (survived 10 workspace resets so far)

| Copy | Location |
|---|---|
| Canonical | GitHub (private): SKJNR/A-Step-Ahead-of-Drought-...-ITU — main 87f0918+ |
| Full tar | /home/sync/repo.tar (774M) + nested_workspace_pre_task42 |
| Git bundle | /home/sync/git_repo_2026-09-10.bundle (pre-filter history) |
| File mirror | /tmp/my-project (rsync target of every session) |

## 4. If the 72h clock starts

1. Clone/fetch the GitHub repo (PAT in chat log; re-issue if expired).
2. Submit `download/REPORT_DRAFT.md` as the report (4 rubric sections ≤100 words
   each + Innovation, all numbers evidence-linked).
3. Point organizers at `scripts/` + `worklog.md` + `submissions_manifest.md` as
   the code + audit trail. Nothing to assemble — this file is the map.
