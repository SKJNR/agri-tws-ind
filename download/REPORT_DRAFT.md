# REPORT DRAFT — AI Trustworthiness Sections
**Status:** Draft v1 (2026-08-30). Based ONLY on documented, reproducible work.
**⚠️ Placeholders marked [V20] must be finalized after the v19/v20 code audit.**
Rubric: 4 sections × 100 words max (per Trustworthiness_Evaluation PDF) + Innovation (20%).
Verified deadlines: submission close 13 Sep 2026 21:29; top-10 private → code+report in 72h.

---

## 1. Data & Model Bias (WG Data, max 100 words)

We identified a **temporal regime shift**: training (2002–2015) and test (2015–2018) eras
differ by a quasi-static spatial offset D (std 0.90, ~51% of test-era anomaly variance),
so models fit on train climatology alone underperform on test — we quantified this and
corrected it with anchor-based calibration. **Regional bias**: trends are physically
structured (Greenland −0.165/yr, Alaska −0.115/yr ice loss) and dominate the slow signal;
we model per-cell trends explicitly instead of letting global averages absorb them.
We validated with an honest 2013–15 window (test-mask pattern applied, earlier years
excluded) — **Spearman ρ(CV, LB) = 1.00 across 5 model variants** — confirming no
validation bias. [V20: add any external-covariate provenance statement here.]

*Word count: ~100. Trim to fit.*

---

## 2. Model Transparency (WG Modelling, max 100 words)

Feature importance (LightGBM): **SOIL_MOISTURE_t and SPEI_12_t are the strongest
covariates**, then location/climatology; SPEI_01 contributes least — physically sensible,
since TWS integrates multi-month storage. Our Kalman observation channel is interpretable
by construction (H ≈ 0.24–0.31 quantifies exactly how much each covariate update moves
the state). Unexpected findings from ablations: (a) current-month covariates enter with
**negative** coefficients (they double-count the observed TWS_t), while next-month
covariates add +0.02 RMSE of genuine signal; (b) the target is exactly next month's TWS
(corr 1.000000), which we proved with calendar-correct merging. [V20: add SHAP values —
run `shap.TreeExplainer` before submission.]

*Word count: ~105. Trim to fit.*

---

## 3. Approach Reusability (WG Modelling, max 100 words)

The architecture is deliberately modular and grid-agnostic: (1) per-cell climatology +
trend, (2) a spatial Kalman filter for the fast state, (3) covariate observation
operators, (4) a separate model for rows with observed current state. Each part works on
any regular lat/lon grid and any geophysical field (soil moisture, SPEI, snow water
equivalent) at any lead time — propagating the Kalman k months ahead handles multi-step
forecasts directly, and the filter naturally absorbs irregular/missing observations.
Main limitation: the slow-offset calibration needs occasional unmasked "anchor"
observations; with none, slow-state skill degrades gracefully to trend extrapolation.

*Word count: ~100.*

---

## 4. Sustainability & Efficiency (Sustainable AI, max 100 words)

Our pipeline is deliberately lightweight: closed-form linear algebra (climatology,
trends, PCA denoising, Kalman recursions) plus one 400-tree LightGBM. A full
train-and-predict cycle runs in **minutes on a single CPU** — no GPUs, no cloud
clusters, no hyperparameter search at scale. Reusing cached per-cell statistics, a
new prediction pass takes seconds, so re-forecasting each month in operations is
near-free. Estimated footprint (CodeCarbon, to be attached [V20]): on the order of
**0.01–0.05 kg CO₂e per full run** — comparable to streaming a few minutes of video.
We favored interpretable low-compute components over deep ensembles precisely because
marginal accuracy gains did not justify 100× compute.

*Word count: ~100. Attach actual CodeCarbon measurement before submission.*

---

## 5. Innovation & Practicality (20% of final score — no word limit stated)

**The measurable story (all reproducible from our logged experiments):**

1. **Generator reverse-engineering.** We decomposed the data into
   μ_c + slow trend + fast mean-reverting field + noise, and PROVED the target
   identity (target(t) = TWS at t+1 exactly) — resolving a dispute where a
   calendar-gap merge artifact had produced a false "noisy target" claim. This
   alone redirected the modeling strategy away from denoising toward state tracking.

2. **Honest validation under masking.** We replayed the competition's whole-month
   masking pattern on 2013–15 training data and demonstrated perfect CV→LB rank
   correlation across 5 architectures — i.e., we can iterate offline without
   leaderboard-fishing (a form of overfitting prevention we can demonstrate).

3. **Cross-AI adversarial collaboration.** Two independent analysis tracks debated
   5 rounds; each side reproduced the other's claims before accepting them,
   cataloguing failures publicly (per-PC Kalman, random-walk interpolation, etc.).
   The submission trail (0.806 → 0.714 → 0.705 → 0.632) maps to specific,
   documented hypotheses, not blind tuning.

4. **Operational design.** The two-component structure mirrors real hydrological
   monitoring (slow storage change vs monthly anomaly), degrades gracefully when
   observations are missing, and runs in CPU-minutes — practical for drought
   early-warning use in the exact near-real-time setting the challenge describes
   (GRACE's 2–3 month latency).

---

## 6. TO-DO BEFORE SUBMISSION (checklist)

- [ ] Audit v19/v20 code + data inputs (OPEN ITEM #1) — write the provenance paragraph.
- [ ] Run SHAP on the final model → insert real values in §2.
- [ ] Run CodeCarbon → insert real estimate in §4.
- [ ] Confirm which 2 submissions are selected for final judging.
- [ ] Verify all numbers against final code (any number in this report must be
      reproducible by a script in /home/z/my-project/scripts/ or the v20 package).
- [ ] Keep every claim ≤ what the logs prove (the EDA trail IS the evidence).
