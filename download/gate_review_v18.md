# GATE REVIEW — v18 submission (submission_v18a.csv / submission_v18b.csv)

**Reviewer:** Gate (independent adversarial reviewer, no stake in shipping) · **Date:** 2026-08-30
**Artifacts reviewed:** `scripts/build_v18.py`, `scripts/build_v18b.py`, `scripts/build_v18_final.py` (authoritative), `scripts/build_v18_final.log`, `scripts/build_v18_grid.log`, `scripts/build_v18_grid2.log`, `download/submission_v18a.csv`, `download/submission_v18b.csv`, `download/submission_v17b.csv`, `download/auditA_report.md`, `download/auditC_report.md`, `data/Train (1).csv`, `data/Test (2).csv`, `data/SampleSubmission (4).csv`.
**Gate scripts (new, nothing else modified):** `scripts/gate_repro.py` (reproduction + leakage probes + mirror audit + bootstrap + end-to-end file reproduction + ACF), `scripts/gate_files.py` (file/format/projection), `scripts/gate_extra.py` (mirror realism + protocol non-independence). Raw outputs: `scripts/gate_repro_out.txt`, `scripts/gate_files_out.txt`, `scripts/gate_extra_out.txt`.
**Method note:** the pipeline functions (`build_infra`, `get_smoother`, `run_masked_v18`) were executed **verbatim by source-extraction from build_v18_final.py** (not re-typed), so the reproduction tests the actual shipped code.

---

## A. LEAKAGE HUNT — CLEAN (all probes pass)

**A1. Spatial smoothing of the fast state.** `Sm = get_smoother(σ)` is a 15,715 × 15,715 sparse gaussian kernel built **only from `cell_xy`, which is derived from Train.csv cell coordinates** (build_v18_final.py L34). Test cells are assert-mapped into train cc codes (`assert (test['cc'] >= 0).all()`), i.e. the test grid IS the train grid; the kernel touches no test data, no TWS, no targets. Row sums = 1.000000 exactly (probe). Application site: `x = Sm @ x` (L182-183) smoothes **only the final fast-state field of the target month** — a same-month, cross-cell average of model predictions. No target values, no future info, no test-month TWS beyond the visible anchors. ✅
**A2. Era-weighted D-hat.** `Dhat_era(t)` (L142-146) weights the anchor fields by `exp(−|A_arr − t|/τ)` — a pure function of anchor/target **time geometry** — over `A_stack` = anchor-month TWS anomalies minus train-era `mu_c`. Masked TWS never enters (`AF` is built only from `~msk` rows of months with mask-fraction < 0.01, which on test are exactly the 6 fully-visible anchors; the 417 exception rows in full-mask months are ignored — conservative). ✅
**A3. Active corruption probes (executed on the verbatim code, M1 mirror):**
| probe | result |
|---|---|
| set ALL `target` values to 917 (BASE config) | max\|Δpred\| = **0.00e+00** |
| set ALL `target` values to 917 (cfg0) | max\|Δpred\| = **0.00e+00** |
| garbage TWS in masked months (BASE / cfg0) | max\|Δpred\| = **0.00e+00** / **0.00e+00** |
No pathway (including both new mechanisms) reads targets or masked TWS. ✅
**A4. k0 block unchanged.** Diffed v18a and v18b vs submission_v17b.csv on all 94,048 unmasked rows: **max|d| = 3.00e-07, mean|d| = 5.1e-09, 0 rows > 1e-6** — float32 noise only; matches the builder's claim exactly. The `K0_SIGMA` smoothing probe was correctly reverted to `None` (below the pre-registered 0.002 inclusion bar; measured −0.0013 at σ=1.0, +0.0049/+0.0158 at 1.5/2.0 — non-monotone, rightly rejected). ✅
**A5. CV-era separation.** `infra_cv` is built from `fit = train[year ≤ 2012]`; probe: `infra_cv['mu_c']` matches the ≤2012 per-cell mean to max 5.4e-07 and differs from the full-train per-cell mean by 0.118 mean-abs → the ≤2012 restriction is genuinely active. Mirrors use only 2013-15 rows (TWS restored at anchors from train, verified vs original). The final phase legitimately refits `infra_full` on all train. k0 CV train rows use targets ≤ 2012-12 (`t_abs ≤ 2012*12+10`), identical to v17b. Direct-copy route: 0 rows (build log + auditA A3). ✅

**Leakage verdict: no leakage found in either new mechanism, in the CV protocol, or in assembly.**

---

## B. REPRODUCTION OF HEADLINE NUMBERS — EXACT

Re-ran baseline + all 3 final configs + ensemble on M1/M2 with the verbatim extracted code, infra_cv from raw Train.csv (gate_repro.py):

| item | claimed | reproduced | Δ |
|---|---|---|---|
| v17b base M1 / M2 | 0.6319 / 0.6905 | **0.6319 / 0.6905** | < 0.00005 |
| cfg0 (φ.80 λ.80 cap8 τ24/full σ2.0) | 0.6077 / 0.6688 | **0.6077 / 0.6688** | < 0.00005 |
| cfg1 (φ.85 λ.80) | 0.6080 / 0.6703 | **0.6080 / 0.6703** | < 0.00005 |
| cfg2 (φ.80 λ.84) | 0.6081 / 0.6695 | **0.6081 / 0.6695** | < 0.00005 |
| top-3 ensemble | 0.6078 / 0.6694 | **0.6078 / 0.6694** | < 0.00005 |

**End-to-end file reproduction:** rebuilt infra_full from raw train, ran the 3 configs on Test (with assemble's NaN→mu_c fallback for 7 rows), compared to the CSVs: v18a masked rows max|d| = **1.19e-07**, v18b = **1.18e-07** (float32 write noise). The shipped files are exactly what the script computes. Scored-row counts: M1 n=187,297, M2 n=234,269 (5/18 rows dropped where cells have no anchor obs in-mirror → pred NaN; on test 7 such rows get mu_c — a 0.003–0.008% asymmetry, immaterial). **NO-GO threshold not triggered.**

---

## C. MIRROR VALIDITY — EXACT MATCH TO auditA SPEC

- Month lists parsed from build_v18_final.py source and asserted equal to auditA_report.md §4: M1_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186] (Jan13/May13/Nov13/May14/Apr15/Jul15), M1 runs {Feb13; Jun13 Jul13 Dec13 Jan14; Jun14 Sep14 Oct14 Nov14 Dec14 Jan15; Aug15}; M2 drops anchor 24183, adds runs Feb15/Mar15/Apr15. **EXACT.**
- Auto-derived anchors (mask-frac < 0.01) equal the hard-coded lists in both mirrors; anchor TWS restored from original train (verified `==` to 1e-6), run-month TWS NaN'd; all months within 2013-15.
- Scored rows: **M1 = 187,302, M2 = 234,287** — identical to auditA. Obs density: **M1 5.96, M2 4.96 vs test 5.98** obs/cell — identical to auditA.
- Run-mix: test run1/2/3/4 = 16.7/24.9/50.0/8.3% of masked rows; mirror reproduces the same shape (2017-block analog = 50% of M2 scored mass).

**Mirror verdict: implemented exactly per spec.**

---

## D. OVERFIT-TO-PROTOCOL — GAIN IS REAL (z ≈ 45); CAVEAT: M1 ⊂ M2

**Search size:** 26 evals in grid-1 + 24 in grid-2 + 4 in the final confirmation = **54 M1+M2 evaluations, ~51 unique configs** (the "~60" in the claim is a ~15% overcount; immaterial).

**Effect size vs noise (bootstrap, cell clusters, 2000 draws, gate_repro.py):**
| protocol | BASE − cfg0 diff | SE | 95% CI | z |
|---|---|---|---|---|
| M1 | +0.0242 | 0.0005 | [+0.0233, +0.0251] | ≈ 48 |
| M2 | +0.0217 | 0.0005 | [+0.0207, +0.0227] | ≈ 43 |

Leave-one-run-month-out jackknife: M1 diff ∈ [+0.0221, +0.0262], M2 ∈ [+0.0193, +0.0234] — no single month carries the result. Per-month decomposition: **cfg0 beats BASE in 27/27 month×protocol cells** (M1 12/12, M2 15/15; smallest gain −0.0032 on Aug15, largest −0.0545 on Dec14-in-M2).

**Dose-response (not a noise peak):**
- σ sweep (grid2 R1, M1/M2): 1.0 → 0.6188/0.6794; 1.5 → 0.6162/0.6769; **2.0 → 0.6157/0.6764**; 2.5 → 0.6167/0.6775; 3.0 → 0.6189/0.6796 — smooth interior optimum with a **plateau 1.5–2.5 (Δ ≤ 0.001)**.
- τ sweep (M2, under φ.85/cap8): 6 → 0.7115 (bad); 12 → 0.6914; **24 → 0.6867**; τ→∞ (static D-hat limit) = 0.6895 — monotone approach to a **bounded basin**, not a knife-edge. τ=48 untested but the limit is known and worse than 24 by less than 24's margin over 12.
- φ×λ surface with smoothing (R3, M2): 0.6688 / 0.6695 / 0.6703 / 0.6716 / 0.6740 / 0.6764 — flat; winner vs runner-up = 0.0015. **Fine-knob winner's-curse exposure ≤ ~0.002 on M2 even if every interior peak is pure luck — 10× smaller than the mechanism gain (0.022–0.024).** The 3-config ensemble is additional insurance against fine-knob luck.

**CAVEAT (structural, quantified):** M1 and M2 are **not independent**: 100% of M1's 187,302 scored rows are inside M2 (M2 adds only Feb15/Mar15/Apr15 = 46,985 rows); on shared rows the two protocols' predictions correlate 0.974–0.976 (RMSE of difference 0.124). "Wins on BOTH M1 and M2" is a **geometry-perturbation robustness check** (anchor-set change), not a second dataset. All selection also sits in a single 30-month window (2013–15), a calmer era than test (auditA: 2015 persistence 0.90 vs 2013-14 0.52; test D-dominant 2015-09..2018-12). The +0.030..+0.050 CV→LB gap band is the honest acknowledgment of this.

**Minor attribution nit:** "smoothing alone 0.6319→0.6157 / 0.6905→0.6764" was measured under φ=0.85+cap8 (baseline 0.6290/0.6895), so pure smoothing is −0.0133/−0.0131 and the quoted delta includes −0.0029/−0.0010 of φ/cap. The **combined** config numbers are direct measurements and reproduce exactly, so the headline is unaffected.

---

## E. PRIVATE-LB ROBUSTNESS — NO SPLIT DEPENDENCE

- Predictions are **field-based**: for each masked month the model produces one value per cell from (train infra, test covariate fields, test anchor TWS). Covariates are never masked (all rows visible), anchors are fully-visible months; nothing in the pipeline depends on which rows are scored. Same month → same field; rows sample it. The 30/70 public/private random-by-row split therefore cannot interact with any v18 mechanism (smoothing included — the kernel acts on the full cell grid, not on submitted rows).
- Public-LB sampling noise: ~56,074 public masked + ~28,214 public k0 rows → SE(masked) ≈ 0.0021, SE(k0) ≈ 0.0027, **SE(LB) ≈ 0.0017** — the pre-registered decision bands (0.692 / 0.700 / 0.7050) are 4–8 SE apart; band boundaries are resolvable.
- Residual transfer risk is era/geometry (D caveat), not split mechanics.

---

## F. SANITY — PASS

- **Format (G1):** `validate_submission.py` PASS on both files (exit 0). IDs bit-exact vs SampleSubmission in set AND order; 280,961 rows; 0 NaN/inf; no |T|>100.
- **Ranges/std:** v18a Target ∈ [−2.577, +2.573], mean −0.0258, std 0.6914; v18b std 0.6869; v17b was std 0.6764. Masked std 0.6586 (v17b) → 0.6815 (v18a) / 0.6747 (v18b): +3.5%/+2.5% — **no explosion**; consistent with the recency-weighted D-hat carrying more signal variance (equal-weight mean of 6 anchors is lower-variance by construction). Per-run stds: run1 0.706/0.701 vs 0.711; run2 0.680/0.675 vs 0.671; run3 0.660/0.652 vs 0.620; run4 0.744/0.734 vs 0.719 — all within ±0.04 of v17b.
- **Spatial autocorrelation of the prediction anomaly field (pred − mu_c, 1° grid-adjacent pairs, n=29,396 pairs/month):** v18a min/mean across the 12 masked months = **0.9820 / 0.9833** (v18b 0.9819/0.9832) — ≥ 0.95 ✓; v17b by comparison 0.9735/0.9782. 2° ≈ 0.941, 5° ≈ 0.741 — smooth, physically plausible decay.
- **Change vs v17b (masked rows):** corr 0.967, mean|d| = 0.129, frac|d|>0.1 = 48.8%, RMSE-of-change largest on the 2017 block (0.2125) — expected, since cap=8 removes the backward pass there and era-Dhat changes the baseline; not a wholesale field replacement.
- v18a vs v18b differ only on masked rows (unmasked max|d| = 0.00e+00) — the two submissions cleanly share the k0 block.

---

## G. PRE-REGISTRATION CONSISTENCY — RECOMPUTED, ONE MATERIAL CAVEAT

- Gap model reproduces exactly: implied v17b masked RMSE = sqrt((0.704955918² − 0.3348·k0²)/0.6652) = **0.7419 / 0.7355 / 0.7311** for k0 = 0.625/0.640/0.650 → gap vs M2 0.6905 = **+0.0514 / +0.0450 / +0.0406**. Central +0.0450 ✓ (and the k0-uncertainty spread of the gap, +0.041..+0.051, brackets the pre-registered band [+0.030, +0.050] reasonably).
- Projection table recomputed: masked = M2_ens 0.6694 + [0.030, 0.050] = [0.6994, 0.7194]; LB at k0=0.640 = [0.6801, 0.6938] — the printed table (0.6791/0.6860/0.6936 at masked 0.698/0.708/0.719) reproduces to 4 decimals (0.001 lower edge from using ens≈0.668 vs 0.6694; immaterial).
- **Central projection:** gap +0.045, k0 0.640 → masked 0.7144 → **LB ≈ 0.6905**, i.e. beats v12b (0.695357) by ~0.005 — above the 0.004 interpretability floor but not comfortably.
- **Material caveat:** joint worst case (gap +0.050 AND k0 0.650) = **0.6969 > v12b 0.6954** — beating v12b is NOT guaranteed by the pre-registered band; the band's upper tail is also open (gap +0.055 → 0.6974). The decision rules handle this correctly ("LB > 0.700 → projection missed → suspect era-weight/smoothing transfer"), and k0 is inherited unchanged from v17b so any LB movement isolates the masked block — but the team should expect a real chance (~1-in-4 by the band's own width) of landing between 0.692 and 0.700, i.e. "modest confirm" rather than "double down".

---

## GATE SPEC STATUS (auditC G1–G5)

| gate | status |
|---|---|
| G1 FORMAT | **PASS** (validator exit 0 on both files, re-run by reviewer) |
| G2 PROVENANCE | **PARTIAL — ACTION REQUIRED pre-upload:** no submissions_manifest row, no git tag yet. md5s (reviewer-computed): v18a `0951a88e0cd11615a2ada99ae93f4df8`, v18b `50356993363335643081b929db0eec69`, v17b `5fde3e75d20953432af377632e7f9d82` |
| G3 CV REGRESSION | **PASS:** mirrored masked 0.6077/0.6688 vs best 0.6319/0.6905 (−0.024/−0.022); k0 0.6254 = v17b exactly (unchanged block, verified to 3e-07) |
| G4 RED-TEAM SIGN-OFF | this document |
| G5 PRE-REGISTRATION | **PARTIAL — ACTION REQUIRED pre-upload:** projections + decision rules exist in build_v18_final.log (build-time), but the worklog entry (predicted LB band + decision rule, before upload) has not been written; worklog currently ends at Task 13 (v17b) |

---

## FINDINGS SUMMARY

**Blockers found: none.** No leakage (active probes returned exactly 0.0 change); all claimed numbers reproduce to <0.00005; files reproduce end-to-end to 1.2e-07; k0 isolation verified at 3e-07; mirror spec followed exactly; effect is 43–48 SE with monotone dose-response and 27/27 month-level consistency.

**Named risks (quantified):**
1. **Single-window selection, M1 ⊂ M2 non-independence** — 100% of M1's scored rows are inside M2; cross-protocol prediction corr 0.976; all selection on one 2013-15 window. Winner's-curse exposure on fine knobs ≤ ~0.002 vs 10× larger mechanism gain — acceptable, but the "wins on BOTH protocols" rule should not be cited as two independent validations in the writeup.
2. **Gap-model extrapolation** — central LB ≈ 0.6905 beats v12b by only ~0.005; joint worst case (gap +0.050, k0 0.65) = 0.6969 does NOT beat v12b; upper tail open. Probability of "modest confirm" band (0.692–0.700) is material.
3. **Process gaps G2/G5** — manifest row + worklog pre-registration entry must be completed before upload (numbers provided above).
4. Minor: masked-pred std +3.5% vs v17b (era-Dhat variance; if LB disappoints, this is the first symptom to check); "smoothing alone" attribution overstated by ~0.003 (φ/cap bundled); "~60 configs" is actually 54 evals/~51 unique; auditA §2.2 run-3 row count typo (93,343 vs census-sum 93,543 = 50.05% of masked); mirror CV drops 5/18 NaN-pred rows vs test's 7 mu_c-fallback rows (0.01% asymmetry).

---

SIGN-OFF: GO-WITH-RISKS (single-CV-window/M1⊂M2 selection — winner's-curse ≤0.002 vs 10x gain; gap-model tail — joint worst case 0.6969 does not beat v12b 0.6954; G2/G5 paperwork (manifest + worklog entry) incomplete before upload)
