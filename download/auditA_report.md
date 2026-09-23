# AUDIT A — DATA & PROTOCOL SKEPTIC REPORT
**Auditor:** A (forensic data/protocol) · **Date:** 2026-08-30 · **Scope:** A1–A5 attack list
**Scripts (all new, nothing existing modified):** `scripts/auditA_test_census.py`, `scripts/auditA_val_realism.py`, `scripts/auditA_mirror.py`, `scripts/auditA_checks.py`
**Raw outputs:** `download/auditA_test_census.txt`, `download/auditA_val_realism.txt`, `download/auditA_mirror.txt`, `download/auditA_checks.txt`

---

## 1. FINDINGS TABLE

| # | Claim / item | Verdict | Evidence (number + source) |
|---|---|---|---|
| A1 | Test has ~50%-masked calendar months (1/6/7/9/12) | **REFUTED** | **0 partial months exist.** Every one of the 18 present t_abs months is either ≥99.6% masked (12 months) or 0% masked (6 anchors). The "≈50%" is a pooling artifact across *years*: e.g. calendar month 1 = Jan-2016 (anchor, 0%) + Jan-2017 (99.8% masked) → pooled 0.4985. (auditA_test_census.txt §A1.1–A1.2) |
| A1 | Anchor months [24188, 24192, 24197, 24203, 24222, 24226] | CONFIRMED | = Sep-2015, Jan-2016, Jun-2016, Dec-2016, Jul-2018, Nov-2018; gaps **[4, 5, 6, 19, 4]** (A1.1) |
| A1 | Month 10 absent from test | CONFIRMED | Oct never present; 22 of the 40-month span (Sep15–Dec18) is absent entirely (A1.1) |
| A1 | Test era "~2016+" | REFUTED (minor) | Test **starts 2015-09**, exactly **1 month after train end (2015-08)** (A1.1) |
| A1 | Masked rows at horizons 1–7 from last visible TWS | CONFIRMED (sharpened) | h_back (t − last visible) = 1:62,576 / 2:46,777 / 3:31,076 / 4:15,560 / 5:15,479 / 6:15,445 — **1–6**, never 7; in target-month units 2–7 (A1.4) |
| A1 | Masking is cell-deterministic in 50%-months (mirrorable per cell) | MOOT / REFUTED as framed | Masking is a pure function of the **month instance**, not the cell. Only exceptions: **417 rows (0.22%), 337 cells, unmasked inside full-mask months**; that pool never appears in two consecutive months (overlap=0 for all 12 adjacent pairs) and 330/337 also appear at anchors (A1.5, auditA_checks.txt) |
| A1 | Test obs density ~6.0/cell | CONFIRMED | 94,048 unmasked / 15,715 cells = **5.98**; 15,438 cells have exactly 6 anchor obs (A1.3) |
| A2 | Parent: val gives ~2.5× more unmasked obs/cell (14.9 vs 6.0) | **PARTIALLY CONFIRMED** | Correct number: val 218,534/15,715 = **13.91 vs test 5.98 = 2.32×**. The 14.9 came from dividing by the 14,656 *full-history* cells (218,534/14,656=14.91), not all 15,715 (auditA_val_realism.txt) |
| A2 | Density mismatch explains the +0.020 CV→LB drift | **REFUTED in mechanism, CONFIRMED in effect (different channel)** | D-hat density alone (6→11 anchor fields) moves RMSE only **−0.003** (M1d vs M1, auditA_mirror.txt). The drift is caused by **geometry**: the val protocol scores rows at backward distances that barely exist on test. Measured bwd-vs-fwd advantage: P0 **+0.046** → mirror M1 **+0.022** → mirror M2 **+0.010** → test-mix extrapolation **+0.003…+0.005**. The P0 honest protocol overstates the backward pass's value by **≈0.040** — enough to fully explain the +0.020–0.027 drift |
| A2 | Other structural mismatches | **NEW** | (a) Val anchor gaps [5,1,4,1,1,5,3,1,1,1,1,5,1] — **7 of 13 gaps = 1 month**; test's minimum gap is 4. (b) Val D-hat = 14 anchor fields over 30 months vs test 6 fields over 38 months. (c) Val honest rows: 83% at bwd≤3, **33% at bwd=1 (impossible on test, min bwd=2)**; test: 58% at bwd≥12 or none, val: 0% at bwd≥4. (d) Era: train 2015 volatility spike — anomaly persistence RMSE 0.52 (2013) / 0.53 (2014) / **0.90 (2015)**, std_anom 0.83/0.90/**1.07**. (e) k0 covs(t+1) availability: val 78.4% vs test 66.5% |
| A3 | Honest filter seals the direct-copy shortcut in VAL | CONFIRMED | shortcut = (t+1 ∈ val anchor months) exactly covers the only months whose TWS the backward pass could read at distance 0; month-level masking means no partial cases exist (code read + replay: honest=93,829 reproduces v17b log) |
| A3 | Direct-copy route on TEST sealed | CONFIRMED | Rows with (cell,t+1) visible in test: **0** (all 280,961 rows checked, not just masked rows). No cell is visible in two consecutive months (12/12 adjacent-month visible-set overlaps = 0) (auditA_checks.txt) |
| A3 | k0 features leak | REFUTED (clean) | Features = TWS_t(t) (visible rows only), covs(t), covs(t+1) via intra-test merge, mu_c/clim from train. covs never masked (0 NaN in 186,913 masked rows). covs(t+1) present for 62,576/94,048 = 66.5% k0 rows — matches worklog. Train k0 rows use targets ≤2012-12 only |
| A3 | Other val leakage paths | **NEW (not leakage, but optimism)** | bwd-distance-1 rows (Apr-2014, Apr-2015: next anchor = target+1) read TWS one month *after* the target with one decay step — legal, but **test has zero bwd-1 rows**; this is the single biggest protocol distortion (see A2) |
| A4 | submission_v17a/b format | **PASS** | 280,961 rows; ID set & order **bit-exact** vs SampleSubmission; 0 NaN/inf; 0 duplicate IDs; ranges [−2.64, +2.69] (v17a) / [−2.58, +2.66] (v17b); Sample Target all-zero |
| A4 | target(t) = TWS_t(t+1) on train | CONFIRMED | 1000 random rows: 922 exact matches (<1e-6), 78 with no (cell,t+1) row in train, **0 mismatches** (auditA_checks.txt) |
| A5 | Mirrored protocol feasible | **NEW** | Two concrete protocols (M1, M2) implemented and run — see §4; they reproduce test obs-density (5.96 vs 5.98 obs/cell) and run-type mix (short/long/fwd-only = 42/50/8%) |
| — | NEW: train itself is a gapped panel | **NEW** | Train has **22 absent months** (138 present of 160): 2002-06/07/08, 2003-06/07, 2011-01/02/06/07, 2012-05/06/10/11, 2013-03/04/08/09/10, 2014-02/03/07/08. **13 absences fall inside the val window** — the current val protocol's anchor structure is an accident of train's own gaps, not a design |
| — | NEW: test panel is fuller than train | **NEW** | Present test months have 15,520–15,681 rows (≈99% of 15,715 cells); val-era train months 15,510–15,681 too, but only 23 of 36 months exist |

---

## 2. TEST CENSUS (A1) — from Test (2).csv alone

### 2.1 Per-month census (18 present months, span 2015-09 .. 2018-12)

| t_abs | date | rows | masked | frac | role |
|---|---|---|---|---|---|
| 24188 | 2015-09 | 15,552 | 0 | 0.000 | ANCHOR |
| 24192 | 2016-01 | 15,647 | 0 | 0.000 | ANCHOR |
| 24193 | 2016-02 | 15,663 | 15,625 | 0.998 | full-mask |
| 24194 | 2016-03 | 15,665 | 15,648 | 0.999 | full-mask |
| 24197 | 2016-06 | 15,584 | 0 | 0.000 | ANCHOR |
| 24198 | 2016-07 | 15,591 | 15,526 | 0.996 | full-mask |
| 24199 | 2016-08 | 15,520 | 15,484 | 0.998 | full-mask |
| 24200 | 2016-09 | 15,529 | 15,479 | 0.997 | full-mask |
| 24203 | 2016-12 | 15,618 | 0 | 0.000 | ANCHOR |
| 24204 | 2017-01 | 15,610 | 15,581 | 0.998 | full-mask |
| 24205 | 2017-02 | 15,642 | 15,591 | 0.997 | full-mask |
| 24206 | 2017-03 | 15,677 | 15,636 | 0.997 | full-mask |
| 24207 | 2017-04 | 15,638 | 15,617 | 0.999 | full-mask |
| 24208 | 2017-05 | 15,572 | 15,568 | 1.000 | full-mask |
| 24209 | 2017-06 | 15,584 | 15,550 | 0.998 | full-mask |
| 24222 | 2018-07 | 15,584 | 0 | 0.000 | ANCHOR |
| 24226 | 2018-11 | 15,646 | 0 | 0.000 | ANCHOR |
| 24227 | 2018-12 | 15,639 | 15,608 | 0.998 | full-mask |

Absent months (22): 2015-10,11,12 · 2016-04,05,10,11 · 2017-07..12 · 2018-01..06,08,09,10. Calendar month 10 never appears.
Totals: 280,961 rows; 186,913 masked (66.53%); 94,048 unmasked; masked ⟺ TWS_t NaN (100% agreement); covariates have **0 NaN** anywhere.

### 2.2 Masked-run geometry (the structure the model actually faces)

| run | months | rows | fwd (anchor→target) | bwd (next anchor→target) |
|---|---|---|---|---|
| 1 | Feb16, Mar16 | 31,273 | 2, 3 | 3, 2 |
| 2 | Jul16, Aug16, Sep16 | 46,489 | 2, 3, 4 | 4, 3, 2 |
| 3 ("2017 block") | Jan17..Jun17 | 93,343 | 2..7 | **17,16,15,14,13,12** |
| 4 | Dec18 | 15,608 | 2 | **none** (fwd-only) |

### 2.3 Per-cell structure
- 15,715 cells; rows/cell = 17.88 mean (p5–p95 all 18; 5 cells have 1 row).
- Unmasked obs/cell: mean **5.98**; 15,438 cells have exactly 6 (= the 6 anchors); 64 cells 7 (anchor + exception row); 213 cells <6.
- Gaps between consecutive unmasked obs: p50=5, p90=19, max=24; modal gaps 4 (31,105), 5 (15,683), 6 (15,586), 19 (15,461).
- Masked-row h_back (t − last visible): {1: 33.4%, 2: 25.0%, 3: 16.6%, 4: 8.3%, 5: 8.3%, 6: 8.3%}; h_fwd (next visible − t): {3: 16.6%, 4: 16.6%, 5: 8.3%, 13–18: 50.0%, none: 8.4%}.

---

## 3. VAL-vs-TEST STRUCTURAL MISMATCH (A2)

Current honest-CV val protocol (rebuild_v17b.py): fit ≤2012; val 2013-15; month masked iff pooled test calendar fraction >0.5 (⇒ Feb/Mar/Apr/May/Aug masked; Jan/Jun/Jul/Sep/Oct/Nov/Dec unmasked); anchors = present val months with frac<0.01.

| dimension | val (P0 honest) | test | mismatch |
|---|---|---|---|
| present months | 23 of 36 (train gaps!) | 18 of 40 | — |
| anchor (TWS-readable) months | **14** | **6** | 2.3× denser |
| unmasked obs/cell | **13.91** | **5.98** | **2.32×** (parent's 2.5× used wrong denominator) |
| anchor gaps | [5,1,4,1,1,5,3,1,1,1,1,5,1] | [4,5,6,19,4] | val has 1-month gaps; test min 4, max 19 |
| D-hat input | 14 fields / 30 months | 6 fields / 38 months | val D-hat much less noisy (−0.003 RMSE measured) |
| scored (honest) rows | 93,829 in 6 months: Feb13, Apr14, Feb15, Mar15, Apr15, Aug15 | 186,913 in 12 months | val can't see the 2017-block geometry |
| bwd distance mix | **bwd=1: 33.4%, bwd=2: 16.7%, bwd=3: 33.3%, none: 16.6%; bwd≥4: 0%** | bwd=2: 16.7%, 3: 16.7%, 4: 8.3%, **12–17: 50%, none: 8.3%; bwd=1: 0%** | **the core distortion** |
| h_back mix | 1: 50%, 2: 17%, 3: 33%; 4–6: ~0% | 1: 33%, 2: 25%, 3: 17%, 4/5/6: 8% each | val blind to h=4–6 |
| k0 covs(t+1) availability | 78.4% | 66.5% | mild |
| era | 2013-14 calm (persistence 0.52), 2015 spike 0.90 | 2015-09..2018-12, D-dominant regime | level gap persists in all protocols |

### Decisive experiment (auditA_mirror.py; identical pipeline, fit ≤2012, config phi=.80/lam=.84/bwd/no-dn/Dtil .70/.45/.073)

| protocol | bwd CV | fwd-only CV | bwd advantage |
|---|---|---|---|
| **P0** current honest (93,829 rows) | **0.6535** (reproduces v17b) | 0.6991 (phi .80) / 0.6919 (phi .74) | **+0.046** |
| **M1** mirror, anchor gaps [4,6,6,11,3] | 0.6319 | 0.6537 | +0.022 |
| **M2** mirror, anchor gaps [4,6,6,14] | 0.6905 | 0.7001 | **+0.010** |
| test-mix extrapolation from M2 gain(d) curve | — | — | **+0.003 … +0.005** |

Measured gain(bwd distance) from M2: d=3: +0.026, d=4: +0.019, d=6: +0.025, d=8: +0.012, d=9: +0.004, **d=12: −0.017**, none: 0. On test, 50% of masked rows sit at d=12–17 → backward pass contributes ≈ nothing there.

**Verdict:** the +0.020 drift is real and is a **protocol-geometry artifact**, not primarily an observation-density (D-hat noise) effect. The P0 honest protocol overstates bwd value by ~0.040; the observed v17b-vs-v12b masked regression (≈0.015, with ±0.006 uncertainty from the unknown k0 LB value) is consistent with bwd's true test value ≈ +0.005. Additional robust results: the PC-denoiser "hurts" verdict **survives** the protocol fix (M1: 0.6462 vs 0.6319); the phi ranking **flips** (P0 prefers 0.80; M1 prefers 0.85–0.90 for bwd, 0.70 for fwd).

Residual CV→LB level gap even under M2: 0.7355 − 0.6905 = **+0.045** (era + D-drift span + mirror compression). Only config *differentials* are trustworthy, not absolute levels.

---

## 4. MIRRORED-PROTOCOL SPEC (A5) — implementable without decisions

Both use the existing pipeline **verbatim** (no code change needed beyond constructing the eval DataFrame): pass an `ev_df` containing only the listed months with `masked` flags as specified; anchors auto-derive (frac<0.01); score all masked rows (all are honest by construction). Fit era unchanged (≤2012) so infra is reusable and results comparable.

### M1 (primary selection protocol) — 18 months, obs/cell 5.96 (test 5.98), 187,302 scored rows (test 186,913)
- **Anchors (TWS readable), t_abs:** 24156 (2013-01), 24160 (2013-05), 24166 (2013-11), 24172 (2014-05), 24183 (2015-04), 24186 (2015-07) — gaps [4, 6, 6, 11, 3].
- **Scored masked runs:** run1 {Feb13}; run2 {Jun13, Jul13, Dec13, Jan14}; run3 {Jun14, Sep14, Oct14, Nov14, Dec14, Jan15}; run4 {Aug15}.
- **Exclude (treat as absent):** Mar13, Apr13, Aug13, Sep13, Oct13, Apr14, Feb15, Mar15, May15, Jun15 and all val months not listed. (Feb15/Mar15 excluded because bwd 1/0; May15 bwd 1; Jun15 shortcut.)
- Row-mix vs test: short-gap runs 5/12 months (test 5/12), long-bwd run 6/12 (test 6/12), fwd-only 1/12 (test 1/12). bwd distances covered: 2–9 (test: 2–4, 12–17, none).
- k0 eval (optional but recommended): unmasked rows of the 6 mirror anchors only (≈93.7k rows; covs(t+1) available 5/6 anchors = 83% vs test 66.5%).

### M2 (stress variant; closest to the 2017 block) — 20 months, obs/cell 4.96 (conservative), 234,287 scored rows
- **Anchors:** t_abs [24156 (Jan13), 24160 (May13), 24166 (Nov13), 24172 (May14), 24186 (Jul15)] — gaps [4,6,6,**14**].
- **Scored masked runs:** run1 {Feb13}; run2 {Jun13, Jul13, Dec13, Jan14}; run3 {Jun14, Sep14, Oct14, Nov14, Dec14, Jan15, Feb15, Mar15, Apr15} (bwd distances 12,9,8,7,6,5,4,3,2); run4 {Aug15}.
- Excludes May15 (bwd 1) and Jun15 (shortcut). Includes the volatile Feb15.

**Selection rule:** a masked-row config change is accepted only if it wins on M1 **and** M2 (not on P0 alone). Report both; treat P0-honest as a third, biased view.

**Cannot be matched (residual risks):** (1) test's 19-month anchor gap / bwd 13–17 — train's largest placeable gap is 14, so even M2 slightly overstates bwd value; (2) test era 2015-09..2018-12 with D-drift over 38 months vs mirror 30 months and calmer 2013-14 climate; (3) calendar-month identity of anchors (seasonality); (4) the 417 exception rows (0.22%); (5) k0 covs(t+1) 83% vs 66.5% — optionally reweight k0 CV by the 66.5/33.5 full/reduced split.

---

## 5. TOP 3 RISKS (ranked)

1. **Backward-pass architecture bet is mostly CV artifact.** Measured test value of the bwd pass ≈ +0.003…+0.010 RMSE (M2 + extrapolated gain curve), vs +0.046 claimed by the current honest-CV protocol. 50% of test masked rows (the 2017 block, bwd 12–17) get ≈ nothing from it (measured −0.017 at d=12). Any further tuning of bwd/blend parameters on the P0 protocol is optimizing a phantom. **Action:** switch selection to M1+M2; reconsider phi (mirror says 0.85–0.90 for bwd configs, not 0.80); attack the 2017-block + Dec18 rows (fwd-heavy, 58% of test error mass) instead.
2. **Absolute CV levels remain optimistic by ≈+0.045 even under the best mirror** (M2 0.6905 vs implied LB masked ≈0.731–0.742, central 0.7355). Era volatility (2015 persistence 0.90 vs 2013-14 0.52), D-drift span, and the unknown k0 LB value (masked RMSE only pinned to ±0.006 for k0 ∈ [0.625, 0.65]) mean projected LB totals carry ≥±0.01 uncertainty. **Action:** trust only differentials; submit 1 config-per-hypothesis to keep LB feedback clean.
3. **The val protocol's structure is an accident of train's own missing months** (13 of 36 val months absent; anchor gaps of 1 month; bwd-1 rows). If future work "fixes" val by imputing missing months or re-deriving anchors from calendar rules, it will silently change the geometry again. **Action:** hard-code the mirror month lists (§4) in the harness; never re-derive anchors from `cal_mask_frac` thresholds.

---

## Appendix: reproduction fingerprints
- P0 honest replication: 0.6535 (v17b log: 0.6535); honest rows 93,829 (log: 93,829); fwd-only phi=.74 = 0.6919 (Task 13: 0.6919).
- k0 covs(t+1): 62,576/94,048 (worklog: 62,576).
- LB split check: 186,913/280,961 = 0.66522, 94,048/280,961 = 0.33478 ✓.
- v17b implied masked RMSE = sqrt((0.704955918² − 0.3348·k0²)/0.6652) = 0.7355 for k0=0.640 (range 0.7312–0.7417 for k0 0.65–0.625).
