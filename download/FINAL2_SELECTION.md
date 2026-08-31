# FINAL-2 SELECTION CARD (the only action that must not fail)

**Deadline: select on Zindi by Sep 12 · hard window 13 Sep 21:29 → 21:59 (selection
close 21:59, reveal 22:15).**

## The picks (adversarial review R1, 2026-08-31)

| Slot | File | Public LB | Why |
|---|---|---|---|
| 1 | `submission_v21a.csv` (md5 `6b6e3e41c25317a68089e6b9ca707c05`) | 0.687374005 | Clean best. Best k0 (a15 blend, LB-confirmed) + best-validated masked block. G6: rebuild from raw CSVs is bit-exact. |
| 2 | `submission_v12b.csv` (md5 `f804f70cc1579d37ebfbd8674b5824a6`) | 0.695357171 | Different masked lineage = the private-era hedge (72.6% of private rows are masked; v21a/v18a share one masked block). |

**Upgrade path:** if the Zindi-downloaded `submission_v13b.csv` bit-matches the local
file (md5 check), swap slot-2 → `submission_v13b.csv`: same public content + era
treatment on the 77,850 private h≥3 rows (62.4% of private masked) = strictly better
private carrier. **Fallback:** `submission_v18a.csv` (only if both above fail
verification).

## ⛔ The two ways to lose everything

1. **Missing the manual selection** → Zindi auto-picks your 2 best PUBLIC scores =
   **v20c + v20a — the prohibited external-GRACE files** → code review → DQ +
   6-month ban + 2000 points. Set two calendar alarms (Sep 12 + Sep 13 20:00) and
   have a second person verify.
2. **Selecting any v20 file** "because it's the best score" — see
   `MASTER_HANDOFF.md` §6 for the line-by-line audit. Never.

## Before Sep 12 (free checklist)

- [ ] Paste the full Zindi "My submissions" table into the chat (resolves
      v13b/v14/v15/v17a/v18b/v19/v20b statuses + exact submission count)
- [ ] Download the submitted v12b + v13b files from Zindi → `md5sum` them →
      compare against the table above (closes the C2 anomaly)
- [ ] Refresh the live leaderboard (top-10 cutoff was 0.6671 on Aug 29)
- [ ] Report draft complete (SHAP + CodeCarbon) — pre-stage the 72h package
