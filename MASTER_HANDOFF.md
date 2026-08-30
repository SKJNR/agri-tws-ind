# MASTER HANDOFF — TWS Competition (Single Source of Truth)

**Last updated:** 2026-08-30, by Super Z (main agent), Task 14
**Purpose:** If any AI session loses context, THIS FILE + `worklog.md` rebuild everything.
**Location:** `/home/z/my-project/MASTER_HANDOFF.md` (master copy, always current)
           `/home/z/my-project/download/MASTER_HANDOFF.md` (downloadable snapshot)

---

## 0. RECOVERY PROTOCOL (read this first if you are a new session)

1. Read `/home/z/my-project/MASTER_HANDOFF.md` (this file) — full state.
2. Read `/home/z/my-project/worklog.md` — task-by-task experiment log (Tasks 1–14).
3. Read `/home/z/my-project/upload/TECHNICAL_HANDOFF.md` — technical detail of the V1–V4 model.
4. Data lives in `/home/z/my-project/data/`, scripts in `/home/z/my-project/scripts/`,
   submission CSVs in `/home/z/my-project/download/`.
5. **Never delete or overwrite this file — only append/update sections.**

---

## 1. WHAT DATA WAS USED (complete inventory, no exceptions)

### Competition data (the ONLY data used by this agent, in all analyses + submissions v1–v4)

| File | Path | Size | Contents |
|---|---|---|---|
| Train (1).csv | `/home/z/my-project/data/Train (1).csv` | 275.7 MB | 2,154,021 rows × 13 cols: sample_id, time, lat, lon, TWS_t, SPEI_01_t, SPEI_03_t, SPEI_06_t, SPEI_12_t, SOIL_MOISTURE_t, month_sin, month_cos, target. Monthly, 2002-01→2015-12, 15,715 cells (1° grid, lat -55.5→83.5, lon -179.5→179.5). Downloaded from the Zindi competition data page. |
| Test (2).csv | `/home/z/my-project/data/Test (2).csv` | 32.8 MB | 280,961 rows × 13 cols (same minus target, plus TWS_t_masked). 18 months: 2015-09, 2016-01..09, 2016-12, 2017-01..06, 2018-07, 2018-11, 2018-12. 186,913 rows masked (66.5%) — TWS_t is NaN there. |
| SampleSubmission (4).csv | `/home/z/my-project/data/SampleSubmission (4).csv` | 5.8 MB | 280,961 rows, ID + Target (all zeros). |
| Trustworthiness_Evaluation (2).pdf | `/home/z/my-project/data/Trustworthiness_Evaluation (2).pdf` | 92 KB | Report rubric (see §5). |
| StarterNotebook (2).ipynb | `/home/z/my-project/data/StarterNotebook (2).ipynb` | 1.2 MB | Zindi starter. |

### Scraped pages (public info only)

| File | Contents |
|---|---|
| `/home/z/my-project/competition_main.json` | Competition overview/rules HTML — evaluation (RMSE 50%, Trustworthiness 30%, Innovation 20%), public/private split 30/70 by rows, deadlines. |
| `/home/z/my-project/competition_data.json` | Data page HTML. |
| `/home/z/my-project/leaderboard.json` | **Aug 23, 2026** leaderboard snapshot, top 50 (see §3). |

### External data — STATUS: NONE used by this agent. ⚠️ ONE OPEN QUESTION

- This agent's models (submissions v1–v4) used **only** the competition CSVs above.
- A photo in `upload/IMG20260826134504.jpg` (Aug 26) shows a **Copernicus CDS API key**
  (key c6cc105f-…) — set up 4 days before the v18→v20 LB jump. The v19/v20 code was
  **NOT created in this environment** (created elsewhere, code not present here).
- **OPEN ITEM #1:** the team must inventory every input used by v19/v20. Rules: external
  Copernicus **drought covariates = ALLOWED** (if prediction-time available + documented);
  any **GRACE/TWS-derived product = PROHIBITED** ("must not directly or indirectly include
  future GRACE/TWS information"). See §6 for the full audit.

---

## 2. COMPLETE SUBMISSION HISTORY (all known LB scores, RMSE, lower = better)

| Submission | Public LB | Date | What it was (as far as documented) |
|---|---|---|---|
| v1 | 0.8059 | Aug 23 | LGBM trees, train-era dynamics |
| v2b | 0.7962 | Aug 23 | 1-comp Kalman φ=0.97 + cov obs |
| v2c | 0.8337 | Aug 23 | Pure decay, no covs (ablation) |
| v3a | 0.7984 | Aug 24 | v2b + pure-AR k=0 blend |
| v1b | 0.7152 | Aug 25 | Two-component Kalman + D-tilde (V1) |
| v1c | 0.7168 | Aug 25 | V1 era-interpolated D-tilde |
| v2b(new) | 0.7137 | Aug 25 | V2 trend-extrapolation in D-tilde |
| v4a | 0.7155 | Aug 26 | V4: PC-denoise + backward pass, φ=0.74 |
| v4c | 0.7144 | Aug 26 | V4, φ=0.80 |
| v5a | 0.7056 | Aug 26 | (v5 generation — code not in this env) |
| v5b | 0.7050 | Aug 26 | (v5 generation — code not in this env) |
| v12a–v14 | ~0.695–0.70 | Aug 29 | v12b = 0.6954 (details not in this env) |
| v16_probe | FAILED | Aug 29 | Failed processing — no LB info returned |
| v18a | 0.6937 | Aug 29/30 | (code not in this env) |
| **v20a** | **0.6331** | Aug 30 | (code not in this env) — Jisoo |
| **v20c** | **0.6317** | Aug 30 | (code not in this env) — Jisoo ← current best |

- Account: **Jisoo**. Submissions used: ~29/200. Daily limit 5 (v20 day: ≥2 used).
- **The v18→v20 jump (-0.060) decomposition (Task 13 audit):** LB 0.6317 ≡ k=0 rows
  @ ~0.40 + masked rows @ ~0.72 (v18a-level). Consistent with a legitimate k=0-row
  breakthrough; masked-row model appears unchanged.

---

## 3. LEADERBOARD SNAPSHOT (Aug 23, 2026 — the latest data this agent has)

| Rank | User | Score | # Subm |
|---|---|---|---|
| 1 | MOHAR | 0.5596 | 38 |
| 2 | Shankar (IIT Madras) | 0.5893 | 22 |
| 3 | GIrum (ASTU) | 0.6234 | 48 |
| 4 | CalebEmelike | 0.6319 | 46 |
| 5 | Emo HedgeHog (team) | 0.6474 | 51 |
| 6 | awxlong | 0.6534 | 5 |
| 7 | Ramjas | 0.6559 | 8 |
| 8 | de-coder (TU Kenya) | 0.6689 | 28 |
| 9 | H2-Oh (team) | 0.6704 | 85 |
| 10 | MosCraciunXXX | 0.6708 | 57 |
| — | Benchmark (all zeros) | 0.8999 | — |

**v20c (0.6317) would sit at ~rank 4** on this snapshot. NOTE: snapshot is 7 days old;
current cutoff may differ. Team Jisoo should re-check the live LB.
Interpretation aid: top-4 implied masked-row RMSE ≈ 0.51–0.57 (near the 0.456 noise floor) —
they resolve the latent state; our masked rows are at ~0.72.

---

## 4. THE SCIENTIFIC MODEL (measured, reproducible — full detail in upload/TECHNICAL_HANDOFF.md)

```
TRAIN:  TWS(c,t) = μ_c + SLOW_c(t) + FAST_c(t) + noise     (2002–2015)
TEST:   TWS(c,t) = μ_c + D(c,t)   + FAST_c(t) + noise      (2015–2018)
```
- target(t) = TWS_t(t+1) EXACTLY (corr 1.000000; Dec→Jan included) — proven twice.
- SLOW: per-cell linear trend, 24.9% of anomaly variance, real-GRACE-anchored
  (Greenland −0.165/yr, 100% cells negative; Alaska −0.115/yr; field neighbor-corr 0.98).
- D: quasi-static test-era offset, std 0.90, ~51% of anchor variance, drifts ~2yr scale.
- FAST: mean-reverting, φ∈[0.70,0.85] (harness optimum 0.74), rank ~100 field,
  tracked by scalar-H Kalman with covariate observations (worth 0.06 RMSE).
- Noise: signal fraction λ≈0.84 in train; test λ unpinned (no k=1 overlaps exist).
- No real seasonality (monthly means span 0.07 vs std 0.91; "Amazon vs Sahara
  amplitude" difference was climatology-estimation noise — generator is synthetic
  with real-GRACE trend maps + uniform noise).
- Data generator is SYNTHETIC ⇒ real external GRACE data CANNOT reproduce targets
  beyond the trend/D channel (relevant to legitimacy: even if someone downloaded
  real GRACE, it would not explain sub-0.6 scores).

### Validated architecture (V4, our last reproducible build — scripts/submission_v4.py)
Two-component Kalman (φ=0.74) + PC-denoise(K=200, init+obs) + backward pass +
D-tilde = 0.70·D-hat + 0.45·S + 0.073·trendex + k=0 linear model with covs(t+1).
Offline anchor-LOO harness: 0.7424 → 0.7185.

### Dead ends (do not retry)
Per-PC Kalman (2 attempts), D-tilde PC projection, RW-drift D interpolation,
partial-month λ mining, trees for h≥4 masked rows, climatology/AR(2) features,
pure decay without covs (0.8337).

### CV protocol (trustworthy — Task 12)
Mask train 2013-15 with the test calendar-month pattern, fit on ≤2012 only.
5 variants: Spearman ρ(CV, LB) = 1.000, CV→LB gap ≈ +0.05 for Kalman models.
**OPEN ITEM #2: run the v20 pipeline through this CV. CV≈0.58 ⇒ real gain
(transfers to private 70%); CV≈0.69 ⇒ test-specific info (danger).**

---

## 5. CODE + REPORT REQUIREMENTS (the actual deliverables for prize/points)

**Two-phase evaluation:** Public LB is NOT the final ranking.
- Phase 1 (50%): **Private** LB (70% of test rows), revealed at close.
- Phase 2 (50%): report rubric — Trustworthiness 30%, Innovation/practicality 20%.

**Key rules (verbatim-checked from competition_main.json):**
- Final ranking = **private** leaderboard (70% of test set).
- **Top 10 on PRIVATE LB** get an email requesting model, code, report — **72 hours** to deliver.
- Before close you must **select 2 submissions** to be judged on. Submission close:
  **13 Sep 2026 21:29** (selection close 21:59, reveal 22:15).
- External Copernicus covariates allowed IF prediction-time available, no future
  GRACE/TWS info (direct or indirect), fully documented.
- Cheating penalty: DQ + 6-month prize ban + 2000 points. Multi-account = DQ.

**Trustworthiness report = 4 sections, 100 words max each** (see the PDF):
1. Data & Model Bias — 2. Model Transparency (LIME/SHAP expected) —
3. Approach Reusability — 4. Sustainability & Efficiency (CodeCarbon etc.).
→ Draft started in `download/REPORT_DRAFT.md`.

---

## 6. LEGITIMACY AUDIT (Task 13, 2026-08-30) — verdict + evidence

**Verified clean:**
- Masked TWS_t = NaN in Test.csv — no file-level leakage.
- v1–v4 (this agent's builds): competition data only, fully documented in worklog.
- v16_probe failed processing ⇒ returned no LB information, does not count vs limit ⇒
  LB-probing not the mechanism of the jump.
- Public 30% is representative: benchmark all-zeros = 0.8999 ≈ global target RMS 0.90.
- 0.6317 decomposes exactly as k=0@0.40 + masked@0.72 ⇒ consistent with legitimate
  k=0 breakthrough (the experiment assigned in the Aug 26 handoff).

**Ceilings measured (why the jump is surprising):**
- covs(t+1)-only models: 0.98–1.00 RMSE (2013-15 val) — covariates alone are weak.
- +TWS_t (k=0 ceiling from competition data): 0.6377 val.
- Anchor-calibrated per-cell cov→TWS (LOO): 0.74 — cannot explain masked < 0.72.
⇒ If v20's masked rows are really ~0.72, everything is legitimate. If someone measured
masked ≈ 0.62 from competition data alone, that exceeds every ceiling we measured —
that would need re-derivation.

**Unverifiable from here (team must check):**
- What external data (if any) entered v19/v20. CDS API key photo dated Aug 26.
- The v19/v20 code itself. **OPEN ITEM #1: upload it here for line-by-line audit.**

---

## 7. SCRIPTS INVENTORY (all in /home/z/my-project/scripts/)

| Script | What it does |
|---|---|
| r1_target_fix.py | Proves target(t)=TWS_t(t+1) exactly (calendar-correct merge) |
| verify_common_mode.py / verify_phi_claim.py | Replicated other-AI's neighbor-corr claims |
| decisive_measurements.py | Generator measurements (ACF, spatial, D-hat, covs) |
| v1_calibration.py | Two-component fit, r(h) profile, k=0 decomposition |
| round4_reconciliation.py | Resolved both cross-AI disagreements |
| k0_diagnostic.py | 2013-15 window difficulty, covs(t+1) value |
| submission_v1.py / v2 / v3 / v4.py | Submission builders (V1–V4) |
| v2_sweep.py / acf_resolution.py | Trend+fast decomposition, φ sweep |
| v3_experiments.py / v3_perpc*.py / v3_dtilde.py / v3_slow_rw.py | V3 structural experiments (incl. failures) |
| partial_month_goldmine.py | Partial-month mining (dead end) |
| cv_lb_correlation.py | The honest CV protocol (Task 12) |
| **legitimacy_audit_covs.py** | Task 13: covs(t+1) ceiling measurement |
| **anchor_cov_audit.py** | Task 13: anchor-calibrated cov→TWS ceiling (0.74) |
| **real_vs_synthetic_check.py / real_vs_synthetic_v2.py** | Task 13: synthetic-vs-real GRACE determination |

Other artifacts: `download/CROSS_AI_BRIEF.md`, `CROSS_AI_REPLY_2/3.md` (cross-AI debate),
diagnostic .txt outputs, `download/cv_lb_correlation.{csv,png}`.

---

## 8. OPEN ITEMS (priority order)

1. **Audit v19/v20 inputs + code** (user uploads → agent audits line by line).
2. **Run v20 pipeline on 2013-15 CV** (decides private-LB robustness, free, minutes).
3. **Choose final 2 submissions** before 13 Sep 21:29 (v20-family + one robust blend).
4. **Finish the report** (draft: download/REPORT_DRAFT.md) — 30% of final score;
   run SHAP + CodeCarbon before submission (rubric asks for them by name).
5. Re-check live leaderboard (Aug 23 snapshot is stale).
