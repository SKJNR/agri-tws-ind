# ADVERSARIAL REVIEW — ROUND 1 (and verdict processing)

**Date:** 2026-08-31 · **Reviewer of record:** independent skeptic agent (fresh context, no stake)
+ main-agent verification pass (code-level, evidence-level)
**Subject under review:** the endgame strategy — final-2 selection, v20 prohibition,
E1 conclusion, report-first priority.
**Protocol (standing, per user instruction):** every material decision flows through
an adversarial/skeptic review BEFORE it is executed; a decision is only final when
it survives attack. This is Round 1 of that loop on the endgame decisions.

---

## 1. The decision stack as submitted to review

| # | Decision (as stood before this round) | Origin |
|---|---|---|
| D1 | Final-2 = v21a + v18a | pre-registered rule (v21b ≥ v21a+0.002) |
| D2 | NEVER select v20a/b/c (prohibited lane) | Task 13/16 audits |
| D3 | E1 (D-evolution) = last clean lever; run it, build v22 if R≥0.5-0.6 | MASTER_HANDOFF §8.2 |
| D4 | Clean lane at ~0.687 ceiling; effort pivots to report (30%+20%) | Task 16 |

## 2. Skeptic verdicts (A–F)

- **A. v20 prohibition — UPHELD (strengthened).** Independent code read of
  `build_v20.py`: L117-121 fills masked TWS_t with GDO(t−1) (bit-exact = the masked
  quantity itself, per the build's own L97 verification); L81-82 feed GravIS/COST-G/CSR
  at the row month — since target(t)=GDO(t), these are satellite measurements of the
  LABEL; L196-214 calibrate the blend against `TARGET=0.95×GDO(t)` — the actual public
  truth reverse-engineered from 9 LB points. No "available at prediction time" defense
  exists. v20's GDO archive even excludes 2017/2018 files, so its private-era edge was
  untested anyway. **Main-agent note: verified independently, identical conclusion.**
- **B. Final-2 = v21a + v18a — OVERTURNED.** v21a ≡ v18a bit-exact on 100% of the
  124,615 private masked rows (72.6% of private); they differ only on k0 where v21a is
  measured better (strict-CV 0.6166 → LB 0.617, 1:1 transfer). v18a is a dominated
  pick — it only wins if the confirmed k0 upgrade specifically fails on private anchors.
  Zindi judges best-of-2 on private → slot-2 is free insurance; wasting it on a
  dominated file insures nothing. The pre-registered rule answered "do LOO weights
  transfer to the public easy era?" — a different question than "which portfolio hedges
  the private hard era?"
- **C. Clean ceiling ~0.687 — UPHELD**, with a distinction: giving up LB
  *improvement* is right (E1 now measured dead); giving up LB *hedging/eligibility*
  actions with ~167 slots unused would be wrong.
- **D. M1/M2 validation strength — UNCERTAIN, leaning acceptable.** M1⊂M2 (100%
  nested, cross-prediction corr 0.976 — one validation, not two); single calmer
  2013-15 window; gap +0.045 at top of band. Correct mitigation = the diversified
  slot-2 the team had declined.
- **E. v21a/v18a compliance — UPHELD.** Active corruption probes returned exactly
  0.0; no lookup leak (zero unmasked next-month rows); covs(t+1)/anchors are
  organizer-provided in-file rows.
- **F. Report-first priority — UPHED.** P(top-10 private) ≈ 15-25%; report is the
  conditional gate on 50% of final score; 1.5 days of work is +EV.

## 3. New evidence produced during this review (post-skeptic, main-agent runs)

1. **E1 executed (all 3 tests): FAILED.** Test A pooled Ridge LOO-by-pair R = 0.271
   (bar 0.5); SPEI_12 pooled corr −0.038; Test B R = 0.189; Test C (bug fixed this
   round, now runs) corr +0.195 on 24,000 train pairs. The D-evolution lever is dead
   on every axis. No v22. `download/e1_d_evolution.txt`.
2. **G6 reproducibility gate for v21a: PASS, bit-exact.** Rerun
   `build_v21_phaseC.py` from raw CSVs → md5 `6b6e3e41c25317a68089e6b9ca707c05`
   identical to the submitted file. v18a already covered by gate_review §B
   (end-to-end 1.2e-07). The #1 pick is fully reproducible.
3. **C2 anomaly resolved on the local side:** `c2_v13b_anomaly_check.py` proves the
   local `submission_v13b.csv` is bit-identical to v12b on ALL 109,222 decoded-public
   rows (0 changed), with the era treatment exactly on private h≥3 months
   (2016-09, 2017-03..06; 77,850 rows = 62.4% of private masked). Therefore the
   recorded public score 0.697421091 **cannot have come from this file** → either a
   file mix-up at submission or a score-relay error. Resolution requires the user's
   Zindi submission-history pull (free).

## 4. REVISED DECISIONS (accepted by main agent; supersedes pre-registered final-2)

- **D1-REV — Final-2 selection (execute on Zindi by Sep 12; hard deadline
  13 Sep 21:59):**
  - **Slot 1: v21a** (0.6874, clean, G6 bit-exact, best k0 + best-validated masked).
  - **Slot 2 (default): v12b** (0.6954, submitted & known, DIVERSE masked lineage —
    the private-era insurance the shared masked block lacks).
  - **Slot 2 (upgrade): v13b** — IF the user's Zindi history pull shows the submitted
    v13b file bit-matches the local one (then it is v12b + era on the private hard
    block: strictly better private carrier). If the submitted v13b is a different
    file and the v13b content is wanted, resubmit the local file (1 slot,
    deterministic, zero-risk) and select it.
  - **Emergency fallback:** v18a (only if v12b/v13b fail verification).
  - ⛔ **NEVER rely on Zindi's default selection** (no manual pick → auto-selects the
    2 best PUBLIC = v20c + v20a → prohibited files get judged → DQ). The 21:29→21:59
    window on Sep 13 is a single point of catastrophic failure: two-person check,
    calendar alarms.
- **D2 — UNCHANGED (v20 prohibition).** Additionally: one-line disclosure of the
  v20 audit-and-refusal in the final report (governance trail = trustworthiness
  evidence; nondisclosure + discovery is strictly worse).
- **D3 — CLOSED.** E1 measured dead; no v22; no further score-chasing submissions.
  Remaining slots stay in reserve.
- **D4 — AMENDED.** Report work proceeds (target: draft complete + SHAP + CodeCarbon
  by Sep 5; 72h package pre-staged by Sep 10). Free fog-closers FIRST (all zero-cost,
  all user-side): (a) Zindi submission-history table paste (resolves v17a/v14/v15/
  v18b/v19/v20b statuses + exact count + the v13b score relay), (b) download the
  submitted v12b + v13b files for bit-diff, (c) live LB refresh (top-10 cutoff).

## 5. Residual risk register (ranked)

| risk | P | damage | mitigation |
|---|---|---|---|
| Shared masked block underperforms on private hard era (both v21a and any v18a-like pick sink) | ~30-40% | top-10 → ~0 | D1-REV slot-2 diversity (v12b/v13b lineage) |
| Missing the manual selection window → auto-pick v20c+v20a → DQ | 2-5% if unplanned | catastrophic (DQ + 6mo ban + 2000 pts) | calendar Sep 12, two-person check, never default |
| v12b/v13b submitted-file identity fog (C2 unresolved) | ~40% some input wrong | mis-selection | user history pull + bit-diff before Sep 12 |
| Report incomplete when 72h clock fires | moderate if top-10 | loss of 50% phase-2 | pre-stage by Sep 10 |

## 6. Round-2 status

The revision (D1-REV) inherits the skeptic's own portfolio logic and the new C2/G6/E1
evidence. Round-2 review is scheduled after the user's Zindi history pull closes the
C2 fog — its single question: *does the slot-2 file we can actually select match the
content we think it has?* G6-style md5 verification of the Zindi-downloaded files is
the pass condition. Until then, D1-REV stands as the working decision.
