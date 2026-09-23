# AUDIT C — STRATEGY & RED-TEAM REPORT
**Auditor:** C (competition strategist / evidence auditor / adversarial red-teamer / scientific-writer planner)
**Date:** 30 Aug 2026 · **Deadline:** 13 Sep 2026, submissions close **21:29**, selection close **21:59** (30-min window)
**Inputs audited:** worklog.md (Tasks 1–13), Trustworthiness_Evaluation (2).pdf (read in full), competition rules HTML (fresh extraction), leaderboard.json (snapshot ≈23 Aug), Test/Train CSVs, rebuild_v17b.py + logs, cross-AI debate briefs (Rounds 1–5), upload/TECHNICAL_HANDOFF.md.
**Numeric checks:** `scripts/auditC_decomposition.py` (all numbers below are from it unless cited otherwise).

---

## C1. EVIDENCE AUDIT — the definitive LB table

### 1.1 Submission evidence table

| # | Submission | Public LB | Config (as known) | Recipe reproducible? | Reliability |
|---|---|---|---|---|---|
| 1 | old v1 (trees) | 0.8059 | LGBM trees + covs, φ_eff≈0.77 | **LOST** (pre-reset) | Score: OK. Config: medium |
| 2 | old v2b (Kalman) | 0.7962 | 1-comp Kalman φ=0.97 + covs + LGB k0 | **LOST** | Score: OK. Config: medium |
| 3 | old v2c (decay) | 0.8337 | pure decay, no covs | **LOST** | Score: OK — **but identical to v1a's 0.8337 (see 1.4)** |
| 4 | old v3a | 0.7984 | Kalman 0.97, pure-AR k0 | **LOST** | OK |
| 5 | v1a (decay, r(h)) | 0.8337 | empirical r(h) profile | YES (submission_v1.py) | OK |
| 6 | v1b (Kalman glb) | 0.7152 | 2-comp, φ_f=0.80, global D̃ | YES | OK |
| 7 | v1c (Kalman era) | 0.7168 | era-interp D̃, φ_f=0.85 | YES | OK |
| 8 | v2b (trendex) | 0.7137 | D̃=0.650D̂+0.456S+0.073·trendex | YES | OK |
| 9 | v4a/b/c | **UNKNOWN — never recorded** | dn+bwd, φ spread | YES (submission_v4.py) | **GAP — recover from Zindi history** |
| 10 | v5–v11 | **LOST** | unknown | **LOST** | **GAP** |
| 11 | **v12b** | **0.695357171 (TEAM BEST)** | unknown (era lost) | **LOST — file recoverable from Zindi history** | Score: SOLID. Config: LOST |
| 12 | v13b | 0.697421091 | unknown; public-row claim anomalous | **LOST — file recoverable** | **ANOMALOUS (C2)** |
| 13 | v16_probe | REJECTED | format error | n/a | n/a |
| 14 | v17a | **submitted? score unknown** | top3-ens masked, no denoise, k0 blend | YES (rebuild_v17b.py, md5 `58a0ed07…`) | **UNRESOLVED — confirm today** |
| 15 | v17b | 0.704955918 | 50/50 denoise hedge, φ=0.80 | YES (md5 `5fde3e75…`) | SOLID |

**Free recovery action (do today, 0 slots):** the user's Zindi *My submissions* page contains every score and every file back to v1. Paste the full table into the worklog and download v12b/v13b (+ any v5–v16 files). This closes evidence gaps #9, #10 and arms the C2 anomaly test. **Also: leaderboard.json is a ~23 Aug snapshot — refresh it before final-week planning (the 0.67–0.71 band is active).**

### 1.2 SOLID conclusions (survive adversarial audit)

1. **Covariate observations help masked rows (~0.03–0.04 LB).** Triple-supported: v2c 0.8337 vs v2b 0.7962 (Δ0.028, >10× any noise floor); mechanism measured (covs carry D at r=0.50–0.74); Round-4 anchor-field experiment (RMSE 0.842→0.733). *Attack failed.*
2. **Two-component test-era structure (μ_c + D + fast + noise).** Multiple independent measurements (anchor-pair correlation collapse after D̂ removal; cov-anchor coupling; ACF non-decay at long lags), survived a hostile cross-AI review with replications both ways. Foundation of the whole architecture. *Attack failed.*
3. **v12b's score 0.6954 is real** (user-confirmed) → the bar for "beat the best" is legitimate.
4. **The leader gap is structural, not k0-confusion.** Arithmetic bound (auditC script): even if the leader's k0 were an impossibly bad 0.50, their masked RMSE ≤ **0.587**; at a realistic k0 0.55–0.62 it is **0.53–0.565** vs our 0.722. Only assumption: public split preserves the 66.5/33.5 row mix (binomial ±0.5% at n≈84k).
5. **Public/private split = ~30%/70%** (rules text). Random-by-row is unverifiable but standard; the masked/k0 mix argument holds.
6. **k0 floor is real:** linear 0.6287 ≈ LGB 0.6273 ≈ blend 0.6254 (honest-CV, n=218k, 2SE=0.002 → resolvable). No large k0 headroom in the current feature set.

### 1.3 SHAKY conclusions (would not survive a hostile committee unchanged)

1. **"PC-denoiser hurts" (+0.023–0.027 honest-CV).** Single window (2013–15), single protocol, **no LB isolation** (v17b was a 50/50 hedge; v17a's score unknown). Keep denoise OFF but treat as *provisional*.
2. **"φ optimum shifted up to 0.80–0.85."** The CV grid spans 0.6535–0.6565 — differences of 0.001–0.003 vs 2SE = **0.0030**. The surface is FLAT; the honest statement is "φ∈[0.74,0.85] indistinguishable; 0.90+ mildly worse." (Silver lining: flat = private-robust.)
3. **"Backward pass is the big gain (−0.037 CV)."** Direction real for short-gap rows; **transfer overestimated** — see C3.1: 50% of test masked rows need backward propagation over 12–17 months; val never tests gap > 3. The CV→LB gap growing +0.056 (fwd era) → +0.080 (bwd-heavy) is exactly what partial non-transfer looks like.
4. **"k0 LGB blend ≈ neutral" (v3a vs v2b, Δ0.0022).** Below the noise floor — the claim is *unproven*, though consistent with CV (+0.003 for blend).
5. **Every v5–v16 config attribution, including "what made v12b best."** Reconstruction, not evidence.
6. **"CV is trustworthy (Spearman 1.0)."** Validated on the OLD shortcut-inflated protocol, across 5 variants spanning a *wide* quality range (0.66→0.83). Valid for architecture-class ranking; **not yet validated for fine deltas or for the honest protocol.** Re-validate with v17a/v17b/v18 points.
7. **k0_test ≈ 0.64** — an assumption, never measured (see C1.5).

### 1.4 Evidence-integrity flag (new finding)

Old v2c (pure decay 0.95/0.82, no covs) and new v1a (empirical r(h) decay) are both recorded as **exactly 0.8337** — different coefficient profiles giving identical 4-decimal LB scores is a ~0.1% coincidence. More likely one score is misattached. The Zindi submissions-page dump resolves this for free. Until then, treat both entries as one data point.

### 1.5 Honest uncertainty on the masked/k0 decomposition

`LB² = 0.6653·masked² + 0.3347·k0²` (186,913/94,048 rows; mix assumed preserved in the 30% public split).

| k0 assumed | v12b masked | v17b masked | gap (17b−12b) |
|---|---|---|---|
| 0.60 | 0.7387 | 0.7522 | +0.0136 |
| 0.6254 | 0.7280 | 0.7418 | +0.0137 |
| 0.64 | 0.7216 | 0.7355 | +0.0139 |
| 0.66 | 0.7125 | 0.7265 | +0.0140 |

- **If v12b and v17b share the same k0 (any value), the masked gap is 0.0138 exactly** — v12b's masked block really is better, independent of the k0 assumption.
- If their k0s differ by ±0.01, the gap moves ±0.004; ±0.02 → gap ∈ [0.005, 0.023]. v12b's k0 lineage was probably the stable linear+recency model (v4-era k0 reproduced bit-exactly by diff_k0.py), so ±0.005 is realistic → **v12b masked ≈ 0.722 ± 0.004**.
- **Decisions at risk:** (a) setting v18's bar as "masked must beat 0.722" — fine, but ±0.004 means a 0.720 vs 0.724 CV-vs-projection comparison is not a verdict; (b) interpreting v17a's score: its *projected* 0.689–0.691 assumes the +0.080 gap; if the honest gap is +0.056 (fwd-era value), v17a ≈ 0.687; if the protocol still flatters bwd, up to 0.71. **v17a's LB score itself is the measurement that pins the gap** — that's the info value of submitting it.

---

## C2. THE v12b/v13b ANOMALY

**Facts:** v13b scored 0.697421091 vs v12b's 0.695357171 (Δ = +0.00206, +0.30%); the era's records claim v13b's "public rows were bit-identical" to v12b's.

**Arithmetic (auditC script):** on a fixed public row set, identical predictions ⇒ identical score *exactly* (RMSE is deterministic; no rounding at 8 decimals). A ΔLB = 0.0021 requires a total squared-error change of ~242 RMSE² on ~84k public rows — e.g. **~2,000 public rows changed by ±0.35**, or ~577 rows changed by ±1.0. So either the files really differed on public rows, or something process-level broke.

**Surviving hypotheses, with strategy implications:**

| Hypothesis | Likelihood | Implication if true |
|---|---|---|
| **H-A: the "public rows" verification was vacuous.** Nobody outside Zindi knows the 30% public set; the team cannot have verified "public rows" except under an *assumed* split. v13b simply differed from v12b on rows that happen to be public. | **~55%** | Nothing exotic. LB deltas are real; noise rule stands. |
| **H-B: file mix-up at submission** (wrong file uploaded under v13b's name). | ~20% | Some config→score attachments in the v5–v16 era may be **wrong** → the evidence table's middle era is unreliable; recover via Zindi history. |
| **H-C: transcription error** (one of the two scores misread/mistyped). | ~15% | Same as H-B; the submissions page is ground truth. The 0.8337 double-entry (C1.4) makes this more plausible. |
| **H-D: Zindi scoring nondeterminism / split drift.** | ~5% | Public LB carries ±0.002 noise → *all* fine-grained LB reasoning dies; selection must be CV-only; final picks should be ensembles. |
| **H-E: score computed on full test (no real split)** — contradicts rules text ("approximately 30%"). | ~5% | Public ranking ≈ final ranking (no shuffle luck); row-subset probes become fully informative. |

**Discriminating experiments, cheapest first:**
1. **FREE, today:** user downloads v12b + v13b files from Zindi submission history → bit-level diff locally.
   - Files **differ** → H-A (or H-E) is live; H-B/H-C die; the anomaly dissolves; keep noise rule.
   - Files **identical** → H-B/H-C/H-D remain; go to 2.
2. **FREE, today:** paste the full submissions table (also fixes C1.4, recovers v4a/b/c + v5–v11 scores).
3. **1 slot, conditional** (only if the diff shows identical files): resubmit v12b **byte-identical** (md5 recorded before upload). Same score → H-D dead, v13b's score was a process error. Different score → H-D live → strategic pivot: CV-only selection, ensemble final picks, ignore sub-0.004 LB order entirely.

**Does the "<0.003 = noise" rule stand?** Yes, but sharpen it: **"LB deltas < 0.003 are uninterpretable, and no decision may rest on a delta < 0.004"** — with the addendum that the *noise* here is attribution/process noise, not sampling noise (paired public-row comparisons are deterministic). Do not soften this rule even if experiment 3 comes back clean.

---

## C3. KNOWN-vs-UNKNOWN ROBUSTNESS (public AND private)

### 3.1 NEW quantitative finding — the honest-CV protocol is badly mismatched to test geometry

From the audit script (this is the strongest evidence yet on the protocol-realism issue):

| Property | TEST (186,913 masked rows) | honest-CV VAL (93,829 rows) |
|---|---|---|
| Horizon h (months since anchor) | h=1: 33% · h=2: 25% · h=3: 17% · **h≥4: 25%** (max 6) | h=1: 50% · h=2: 17% · h=3: 33% · **h≥4: 0%** (max 3) |
| Backward gap (next anchor − target) | 2: 17% · 3: 17% · 4: 8% · **12–17: 50%** · none: 8% | max **3** |
| Anchor spacing | 4, 5, 6, **19**, 4 months | ~2.3 months |
| h-distribution L1 distance | **0.333** | |

The val protocol never evaluates the regime that contains **half the test rows** (2017 block: anchor 6 months back, next anchor 12–17 months ahead) and never tests h≥4. Consequences: (a) the backward-pass gain (−0.037 CV) is measured only in the short-gap regime where it's strongest — at φ=0.80 a 12–17-month backward propagation decays to φ^12–17 ≈ 0.02–0.07, i.e. the bwd estimate carries almost no fast-state information there (the precision blend correctly downweights it, but then those rows are effectively fwd-only + D̃); (b) D̃ quality and trend extrapolation — not Kalman tuning — dominate the 2017 block; (c) this cleanly explains the CV→LB gap growing +0.056 (fwd era) → +0.080 (bwd-heavy). **The v18 "test-mirrored protocol" must reproduce the test h/gap mix (33/25/17/8/8/8% and gaps 2/3/4/12–17/none) — e.g. by mirroring the test anchor *geometry* (one 12–19-month gap per val window) rather than calendar-month fractions — and must pass a BACKTEST: it must reproduce v17b's (0.7050) and v2b's (0.7137) actual LB scores from their CV values within ±0.004.**

### 3.2 (a) Is any config tuned to public LB feedback?

- v1–v3 era: yes — persistence was pushed on a 0.0097 delta, and v3b/v3c (φ=0.99/0.995) were LB bets (scores lost; likely wasted slots).
- v4–v17 era: configs are CV-selected. D̃ weights (0.70/0.45/0.073) were originally LOO-fitted **at the six test anchors** — that is *transductive use of unlabeled test inputs* (legal, and re-confirmed on honest-CV), but worth stating in the writeup as a deliberate design choice.
- Residual LB-dependence: the k0≈0.64 assumption and the CV→LB gap are calibrated from public LB. Unavoidable; final selection should use CV *ranks* + gap-insensitive margins (see 3.5).
- **Verdict: no live config is LB-tuned within the noise floor. The fishing risk now is behavioral** (chasing v12b's number with sub-0.004 deltas) — the gate in C7 is the control.

### 3.3 (b) Is CV-driven selection robust to the public/private split?

If the split is random-by-row (standard, and the 30/70 rules text supports it), public and private share the distribution → CV selection transfers; the 5-variant Spearman-1.0 result supports rank preservation for *large* quality differences. Caveats: (i) that validation ran on the OLD protocol — re-validate on honest/mirrored protocol once v17a + v18 actuals exist (3–4 points by Sep 2); (ii) single-window CV rankings can be window-specific → final freeze must win on **two windows** (2013–15 + 2010–12 or rolling).

### 3.4 (c) Do ensembles have evidence?

CV evidence is thin: top-3 φ/lam ensemble −0.0001 (noise); linear/LGB 50/50 −0.003 (just above 2SE). Theory (variance reduction) and the failure mode we actually fear (protocol/model error, not variance) both favor ensembling anyway. **Verdict: ensembles are insurance, not score-chasing — justified for the final picks.**

### 3.5 (d) If the private split were adversarial

- *All far-horizon rows (h≥4, 25% of masked):* our r(h) profile is data-calibrated (0.87→0.57) and D̃ dominates; mitigation = era-weighted D̃ + trend term. Moderate exposure.
- *2018-heavy:* largest D drift (corr(D_16, D_18)=0.72, drift std 0.76); Dec-18 rows are already fwd-only by design. Mitigation = 2018-anchor-weighted D̃. Moderate exposure — and testable on CV.
- *Region-specific:* per-cell independence ⇒ uniform degradation; nothing to do.
- *k0-heavy:* our k0 is solid (0.625–0.64); low exposure.

### 3.6 RECOMMENDED FINAL-WEEK SELECTION PROCEDURE (freeze protocol)

1. **Sep 8 — freeze architecture** on the test-mirrored honest-CV; winner must beat the runner-up by **>0.003 on BOTH windows**.
2. **Sep 9 — protocol backtest:** predicted LB (gap calibrated from v17a/v17b/v18 actuals) for the top-3 configs; sanity band ±0.004. If the backtest misses, trust raw CV rank, not projections.
3. **Sep 10 — submit final candidates** (2–3 slots); choose the final 2 by CV + backtest, public LB only as tiebreak for deltas >0.004.
4. **Pick diversity:** final 2 = (1) CV-best ensemble; (2) a deliberately *diversified* hedge (different φ regime / era-D weighting / k0 blend). Zindi scores the better of the 2 on private — diversification is free insurance against protocol error.
5. **Sep 12 — explicitly select both on Zindi.** NEVER rely on the default (2 best public): it could auto-select **v12b, whose recipe is lost** — and top-10 code review requires reproducing the score (rules: "If your code does not reproduce your score… we reserve the right to adjust your rank"; "If your code does not run you will be dropped"). Final picks must be regenerable end-to-end.
6. **Sep 13 — nothing after 21:29; confirm selections before 21:59.**

---

## C4. BUDGET & SCHEDULE

**Budget math:** ~31/200 used; 15 submission days remain (Aug 30–Sep 13) × 5 = 75 slots; 31+75 = 106 ≪ 200. **The 200 cap never binds — the binding constraints are information value and build capacity.** Discipline: ≤3/day, every slot pre-registered. Also refresh the leaderboard snapshot (current one is ~23 Aug).

**Mechanics (from rules text):** submission close **13 Sep 21:29**; selection close **13 Sep 21:59**; if no selection, Zindi uses your 2 best public submissions (avoid — see C3.6.5); top-10 private get a 72-hour code+report request after close → pre-stage the package.

### Day-by-day plan

| Date | Submissions (max) | Purpose / pre-registered question | Build work in parallel |
|---|---|---|---|
| **Aug 30 (today)** | 0–1 | If user available: **submit v17a today** (pull schedule forward). User tasks (free): confirm submission count + v17a status; paste full Zindi submissions table; download v12b/v13b files. | Free v12b/v13b diff (C2); v18 test-mirrored protocol + backtest vs v17b/v2b; writeup asset inventory (C5). |
| **Aug 31** | 2–3 | **Slot 1: v17a** (if not already in) — pins the bwd-heavy CV→LB gap; projection 0.687–0.691 (see bands in 1.5). **Slot 2 (conditional): v12b byte-identical resubmit** — ONLY if the free diff shows the files identical (C2 test 3). **Slot 3 (conditional): v18a** if it passes gates G1–G5 and beats honest-CV 0.6535 by >0.003 on the mirrored protocol. | v18 build; era-weighted D̃ (C6-E2); spatial-smoothing CV test (C6-E3). |
| **Sep 1–2** | 2–3/day | **v18 A/B** (protocol-fixed config vs best v17-lineage). **k0-isolation pair** (same masked block; k0 = full model vs pure persistence): pins k0_test AND the masked decomposition exactly (ΔLB ≈ 0.010, resolvable), and anchors the protocol backtest. | D-evolution discriminating regression (C6-E1); SHAP + CodeCarbon runs (C5). |
| **Sep 3–7** | ≤2/day | Leader-gap experiments that survived CV, in EV order (C6 table). Each: CV gain >0.003 before any slot. λ A/B (1 slot) only if the backtest is clean. Optional λ=0.90 probe pre-registered: "if ΔLB > 0.004, test-era noise is lower → retune obs/init balance." | Figures for writeup; report drafting; final-pipeline reproducibility script. |
| **Sep 8** | 0–1 | **Architecture freeze** (two-window CV). | Report review vs rubric; requirements pinning. |
| **Sep 9** | 0–1 | Protocol backtest final check; assemble final ensemble. | Report v1 frozen. |
| **Sep 10** | 2 | **Final candidates submitted**; pick final 2 by CV+backtest. | G6 reproducibility rerun (md5 match). |
| **Sep 11–12** | 1–2 buffer | Emergencies / one late improvement ONLY if CV gain >0.005. **Sep 12: explicitly select final 2 on Zindi.** | Package code+report for the 72h window. |
| **Sep 13** | emergency only | Nothing after 21:29. **21:29–21:59: verify final selections.** | Ship code+report within 24h if contacted. |

**Tomorrow (31 Aug) specifically — the decision tree:**
- If v17a was already submitted (score in Zindi history): skip slot 1; use slots for **v12b resubmit (conditional)** + **v18a (if gate-passed)**.
- Each new submission must beat v12b's 0.6954 **or** answer a pre-registered isolable question — v17a qualifies on both counts (projected best + gap measurement); the v12b resubmit qualifies on the second; v18a on the first.
- Do NOT submit unregistered "hedges": v17a vs v17b differ by only 0.0018 on honest-CV (below 2SE) — one denoiser datapoint on the LB (v17b) plus one without (v17a) is the full isolation; a third hedge adds nothing.

---

## C5. TRUSTWORTHINESS (30%) + INNOVATION (20%) — currently ZERO work done

### 5.1 Rules facts that change the plan (extracted from the competition page)

- Final score = RMSE 50% + **AI Trustworthiness 30%** (per the rubric PDF) + **Innovation & practicality 20%** ("recognition of creative and practical approaches that demonstrate adaptability and robustness for real-world application").
- **Phase 2 applies to the top 10 on the PRIVATE leaderboard**, contacted at close with a **72-hour** window to submit model, code and report. **Code must reproduce the leaderboard score or the rank can be adjusted / the entry dropped.**
- Implication: the writeup is *due after* close (Sep 13–16) — but must be *finished before*, because 72h also contains code packaging. **The current zero-work status is the team's biggest unpriced liability: ~50% of the final score is unprepared, and the #10 threshold (~0.671 public today) is ~0.025 away — reachable territory, so this scenario is live, not hypothetical.**

### 5.2 Rubric mapping (all four dimensions, ≤100 words each per the PDF)

| Rubric dimension | Evidence required | Assets we HAVE | NEW artifacts needed |
|---|---|---|---|
| **1. Data & Model Bias** | Biases identified; assessment/mitigation methods; unresolved biases; cross-context impact | Temporal bias quantified (regime shift: persistence RMSE 0.507 (2002–09) → 0.664 (2013–15); D-offset exists only in test era); regional/spatial bias (per-latitude variance uniformity anomaly; D̂–μ_c anti-correlation −0.77; synthetic-data proof: GPCP r=0.02, non-GRACE trends); masking selection bias (whole-month masking; ~337-cell partial pool). Mitigation: calendar-mirrored honest-CV, shortcut sealing, multi-window evaluation. | 1 figure: D̂ world map + windowed-persistence time series; 3 sentences on unresolved bias (D drift law; test-era λ unpinned). AI Fairness 360 is inapplicable to physical/synthetic grid data — say so honestly. |
| **2. Model Transparency** | Interpretability tools (LIME/SHAP); influential features; figures; unexpected findings | White-box by construction: pred = μ_c + D̃(0.70·D̂+0.45·S+0.073·trendex) + φ^h·fast + Kalman cov updates (H, R calibrated); feature roles measured (covs carry D at r 0.50–0.74; W_m tracks fast state r 0.45–0.50; covs(t+1) worth +0.02 k0 RMSE). Unexpected: synthetic-data proof; target = TWS_t(t+1) identity; PC1 cov-invisibility (r=0.07). | SHAP summary on the LGBM k0 component (native `lgb` + shap); 1 figure. Frame: "SHAP for the 33% ML component; the state-space core is exactly interpretable." |
| **3. Approach Reusability** | Adaptability to other regions/variables/lead times; flexibility design choice; limitation | Architecture is per-cell state-space + covariate observations: region-agnostic, lead-time-parametric (φ^h), cov regression refit per domain; ran on 15,715 global cells. Limitations: needs unmasked anchor months; cov-quality dependent; assumes quasi-static offset regime. | None (write from existing assets). |
| **4. Sustainability & Efficiency** | Emissions estimate (CodeCarbon/ML CO2/Impact Tracker); optimizations; complexity-vs-sustainability trade-off | Pipeline: full retrain + inference in ~140 s on 2 CPU cores, float32, closed-form solves, no GPU, seeds fixed. Trade-off: deliberately chose interpretable low-compute over heavy ensembles. | Install CodeCarbon, measure the final pipeline, record the number (expect <0.01 kg CO2e — a genuinely strong sustainability story). |
| **Innovation (20%)** | "Creative and practical approaches… adaptability and robustness for real-world application" | Cross-AI adversarial debate record (5 rounds, myths killed with replication both directions); generator reverse-engineering (D discovery via anchor-pair correlation collapse); shortcut-sealed honest-CV protocol; negative-results catalogue; the audit trail itself (this report). | Innovation narrative section (~150 words) + pointer to the debate artifacts. |

### 5.3 Writeup plan

**Document outline** (`download/REPORT/final_report.md` → submitted as the Phase-2 section):
1. *Solution summary* (5 lines: two-component generator model; per-cell state-space; masked/unmasked split).
2. *Bias* (≤100 w) + Fig 1 (D̂ map + windowed predictability).
3. *Transparency* (≤100 w) + Fig 2 (model decomposition diagram; SHAP bars for k0 LGBM).
4. *Reusability* (≤100 w).
5. *Sustainability* (≤100 w) + CodeCarbon number.
6. *Innovation & practicality* (~150 w) + Fig 3 (CV↔LB rank-correlation plot — already exists: `cv_lb_correlation.png`) + Fig 4 (protocol-mismatch/honesty figure — new, from auditC numbers; this is a *differentiator*: most teams cannot show they hunted their own validation shortcut).
7. *Reproducibility* appendix: pinned environment, seeds, single entry-point script, md5 of the final submission.

**Schedule (fitted around C4; total ≈ 1.5 days):**
- **Sep 2:** asset inventory + skeleton; install CodeCarbon; SHAP run. (2 h)
- **Sep 3:** Figs 1–4; CodeCarbon measurement. (3 h)
- **Sep 4:** draft all sections (bullet style per the PDF's "Simplified Guide": few sentences, essentials, honesty). (2 h)
- **Sep 5:** adversarial rubric review (Auditor C gate): every claim cites an artifact; word counts enforced; identify anything the rubric wants that we did NOT do and decide honest-disclosure wording. (1 h)
- **Sep 8–9:** update with the final model; freeze v1.
- **Sep 13–16:** if contacted, ship the package within 24 h (pre-staged: code zip, report PDF, requirements, README).

**Risks flagged:** (i) *Top-10 contingency* — if we miss top 10 private, Phase 2 never runs; cost is only ~1.5 days, and the same material feeds any post-competition writeup — acceptable. (ii) *Rubric names tools we haven't used* — SHAP and CodeCarbon are cheap; do them; skip AI Fairness 360 with an honest one-liner. (iii) *Word limits* — the PDF's simplified guide explicitly wants brevity and honesty; resist padding. (iv) *The innovation rubric document was never provided* — cover it with the same narrative; keep it practical (deployment story: monthly operational refresh, ~minutes of compute, no GPU). (v) *Code-reproducibility* — final picks must regenerate byte-exact from raw CSVs (deterministic LGBM flags already set).

---

## C6. LEADER-GAP STRATEGY — be brutally honest about the target

**Reframe first:** the leader (0.5596) is 0.136 away. But the *prize-relevant* threshold is **#10 ≈ 0.6708 public** (snapshot 23 Aug — refresh), i.e. 0.025 below v12b. From the floor decomposition, that means masked 0.722 → ~0.698 (D 0.42→0.38, fast 0.37→0.26 simultaneously) — **plausible with a protocol fix + 2 modest components**. Chasing #1 is a moonshot; **the real fight is #10.**

Leader bound (C1.2.4): leader masked ≤ 0.587 (even with k0 = 0.50); realistic 0.53–0.565. Our absolute floor with current structure: noise alone 0.36–0.456 → LB 0.46–0.60. The leader sits *inside* our D+fast-reducible band — no law of nature is being violated, but they track D and/or the fast state materially better, or test-era noise is lower than assumed.

**Hypotheses, discriminating experiments, expected gains, ranked by EV:**

| Rank | Hypothesis | Cheapest discriminating experiment | Masked-RMSE gain if true | Build cost | Honest probability |
|---|---|---|---|---|---|
| **1** | **E3: Spatial/joint fast-state estimation.** Fast field is smooth (neighbor corr 0.97/0.84; rank ~100); our Kalman is per-cell independent; prior per-PC failures had *identified calibration bugs*, not a refuted concept. Also: residual PC1 = 9.3% ⇒ "noise" has spatial structure — possibly reducible by prediction smoothing. | **FREE, hours:** on honest-CV, smooth the fast-state estimate (and/or final predictions) with a spatial kernel (1°/2°/5°); also try train-calibrated low-rank (not anchor-calibrated) R. | fast 0.37→0.30 ⇒ masked ~0.69, **LB ~0.673 — top-10 territory alone** | 0.5–1 day | 40% small gain, 10% large |
| **2** | **E1: D-evolution tracking (the 51%-variance component).** D drifts on ~2-yr timescales (corr 0.72, drift std 0.76); our D̃ is static per cell. If drift is cov-predictable (SPEI_12 as slow integrator), time-varying D̃ is the single biggest lever. | **FREE, 1 day:** regress (D̂_2018 − D̂_2015-16) on cumulative cov anomalies across cells, held-out cells for honesty. r ≥ 0.6 ⇒ build it. | D 0.42→0.30 ⇒ masked ~0.66, LB ~0.65; with E3 ⇒ LB ~0.62 | 1–2 days if r passes | 25–35% |
| **3** | **E2: Era-weighted D̃ (cheap cousin of E1).** 2018 rows predicted with 2018-anchor-weighted D̃; 2017 with 201612-weighted. Known V4 weakness (era-blind D̃). | FREE: two-window honest-CV A/B. | D 0.42→0.38–0.40 ⇒ LB −0.004–0.008 | hours | 60% |
| **4** | **E4: Lower test-era noise (λ > 0.84).** λ/φ degenerate in anchor fits; if test λ is higher, observations deserve more weight. | Partial-month mining failed (no k=1 chains); the k0-isolation pair (C4) indirectly informs; 1 pre-registered LB slot for λ=0.90 vs 0.84. | LB −0.005–0.015 | 1 slot + hours | 20–30% |
| **5** | **E5: Per-cell φ heterogeneity.** | FREE: per-band φ on honest-CV. | <0.005 LB (φ surface is flat) | hours | 30% tiny gain |
| **6** | **E6: k0 breakthrough.** | k0-isolation pair first; then feature work on the 31k reduced-model rows. | k0 −0.01 ⇒ LB −0.003 only | 0.5 day | Low (floor is real) |
| **7** | **E7: Leader is public-split lucky.** 30% public; 0.5596 could be +0.01–0.02 fortunate. | Nothing to run. | n/a | n/a | Possible; changes nothing — optimize for private, not for chasing the public #1 |

**Moonshot vs plausible:** E3+E2+E1 together are the only path to 0.60–0.63; E3+E2 alone plausibly reach 0.67–0.68 (top-10 fight). E5/E6 are tuning dust. Schedule accordingly (C4: E3/E1 start Aug 31–Sep 2, CV-gated).

---

## C7. PROCESS RED-TEAM — failures that cost score/time, and the institutional gate

### 7.1 Failure catalogue (Tasks 1–13)

| # | Failure | Cost | Root cause | Fix |
|---|---|---|---|---|
| P1 | **Lost v5–v16 recipes** (env reset; only v1–v4+v17 in git) | Cannot iterate from team-best v12b; **cannot reproduce v12b for code review** | No per-task commits; artifacts only in ephemeral FS | Commit after every task; tag every submission; manifest with md5 |
| P2 | **No submission manifest** — count uncertain (~31), v4–v11 scores lost, v17a status unknown | Evidence table holes; planning on fog | Scores relayed by memory/chat | User pastes the Zindi submissions table verbatim after each session; auditor reconciles with manifest |
| P3 | **LB-driven tuning in the v1–v3 era** (persistence pushed on Δ0.0097; v3b/v3c φ=0.99/0.995 bets) | ~2–3 wasted slots | No noise-floor rule yet | Pre-registration + "<0.004 is uninterpretable" rule |
| P4 | **v16_probe format rejection** | 1 attempt (quota unharmed) | No validator | G1 gate below (validator exists — now mandatory) |
| P5 | **Protocol shortcut undetected for ~2 weeks** (v5–v16 era selections ran on the inflated protocol; honest protocol only at v17) | Possibly mis-selected v12b-era configs; the +0.06 inflation distorted every masked number | No standing protocol audit; single window | Protocol realism is a standing check: geometry mirror + backtest vs known LB + second window (C3.1) |
| P6 | **Trustworthiness/innovation (50% of final score) never started** in 8 weeks | Potentially the entire 50% if top-10 | No owner, no deadline | Scheduled Sep 2–5 (C5.3); owner = main agent; Auditor C reviews Sep 5 |
| P7 | **Single-channel manual score relay** — the v13b anomaly may be a transcription slip; the 0.8337 double-entry supports this | Evidence integrity | Human relay | Verbatim submissions-page paste + reconciliation |
| P8 | **Deadline mechanics not internalized** — selection closes 21:59 Sep 13 (30 min after submissions); default selection = 2 best public (could include irreproducible v12b) | Rank-risk at the very end | Rules not re-read | Calendar alarms; explicit selection Sep 12; C3.6 |
| P9 | **Stale leaderboard snapshot** (23 Aug) used for strategy | Wrong thresholds | One-time fetch | Refresh before Sep 8 freeze |

### 7.2 INSTITUTIONAL GATE — every submission passes 5 stages (spec)

| Stage | Check | Method | PASS criterion |
|---|---|---|---|
| **G1 FORMAT** | File integrity | `python scripts/validate_submission.py <file>` (exit 0) | All checks green ("bulletproof") |
| **G2 PROVENANCE** | Reproducibility trail | Git commit + tag `sub_<name>`; row appended to `submissions_manifest` (file, md5, date, config one-liner, pre-registered question, predicted LB band) | Manifest row exists; md5 matches |
| **G3 CV REGRESSION** | No silent regression | Honest-CV (mirrored protocol) masked AND k0 vs current best | Both ≤ best + 0.002, **OR** pre-registered isolation probe (max 1/day, question + expected Δ stated) |
| **G4 RED-TEAM SIGN-OFF** | Adversarial review | Any auditor answers in worklog: "What distribution shift makes this worse on private? What evidence contradicts the assumption this relies on?" | Explicit `SIGN-OFF: GO` / `GO-WITH-RISKS (named risk)` / `NO-GO` |
| **G5 PRE-REGISTRATION & LOG** | Decision discipline | Worklog entry **before** upload: predicted LB + decision rule per outcome band; actual LB + decision after | Both entries exist |

**Final-picks-only additions:** **G6 REPRODUCIBILITY** — rerun of the committed script from raw CSVs reproduces the file (md5 match); environment pinned (requirements.txt, lightgbm version, seeds, `deterministic=True`). **G7 SELECTION** — both final picks explicitly selected on Zindi before 21:59, 13 Sep; neither pick is a file whose recipe is lost.

**Flow:** build → G1 → G2 → G3 → G4 → submit → record actual → apply G5 decision rule. Estimated overhead: <15 min per submission. The v17b run already demonstrated the value (two assembly bugs caught pre-ship).

---

## BOTTOM LINE (Auditor C's verdict)

1. **The evidence base for architecture is strong; the evidence base for *fine config choices* is noise.** Stop tuning φ/λ/D̃-weights (flat surfaces, 2SE=0.003 CV); spend the 14 days on the two structural levers (E3 spatial fast-state, E1 D-evolution) and on protocol honesty.
2. **The honest-CV protocol is geometrically mismatched to test** (h≥4: 25% of test rows vs 0% of val; bwd gap ≥12: 50% vs 0%) — this, not bad luck, explains the growing CV→LB gap and v17b's loss to v12b. Fix (v18) + backtest against known LB scores before trusting any projection.
3. **The competition's real prize fight is #10 (~0.671), not #1 (0.5596)** — and 50% of the final score (trustworthiness+innovation) has zero hours booked. Fix both this week: E3/E1 start tomorrow; writeup Sep 2–5.
4. **The anomaly (C2) most likely = process error, not Zindi magic** — resolve it for free today via the Zindi-history diff; only spend a slot on the byte-identical v12b resubmit if the files really are identical.
5. **Never let an irreproducible file be a final pick** — the rules allow rank adjustment if code can't reproduce the score, and v12b's recipe is gone.

*Artifacts produced by this audit:* `scripts/auditC_decomposition.py` (all numeric checks), this report. No existing files were modified.
