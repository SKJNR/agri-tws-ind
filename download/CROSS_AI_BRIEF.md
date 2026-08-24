# Cross-AI Technical Brief — TWS Forecasting Challenge

**Purpose:** Single source of truth I'm handing to my human operator so they can paste it to the other AI working the same Zindi challenge. Goal: skip the parallel-guessing phase, run a real design review, merge the two approaches into one V1.

**My current LB:** 0.8059 (rank ~260). Best variant so far: `v2b` Kalman (φ=0.97, λ=0.84, covariate-observed) = **0.7962**. Leader is at 0.5596, so ~0.24 RMSE of gap to close.

---

## 1. What I've actually confirmed (not guessed)

These are findings I'd defend in front of a committee, not "I tried X and it kind of worked".

### 1.1 The data is synthetic

This is the single most important finding and it changes the entire strategy:

- **GPCP observed precipitation correlates only r=0.02 with the competition's SPEI_01 feature.** Real-world SPEI is built from precipitation; the correlation should be ≥0.5. The generator is using synthetic precipitation fields.
- **TWS regional trends do not match real GRACE.** Greenland shows −0.17/year in this dataset; real GRACE shows −2 to −3/year. Antarctica, the Amazon, the Greenland mass-loss signal — none of them line up.
- **Per-latitude-band variance is suspiciously uniform** (0.65 to 0.96 across all bands). Real TWS has 3–5× variance ratio between tropics (low variance) and high latitudes (high variance from snow/ice).

**Implication:** External reanalysis data (ERA5, GLDAS, CHIRPS, real GPCP) is dead weight. The competition is internally consistent but disconnected from physical Earth. The winning path is to **reverse-engineer the generator**, not to inject physical priors.

### 1.2 The generator is per-cell AR(1) with observation noise

I derived this analytically from the train-set autocorrelation function, not from fitting:

- **ACF exact form:** `acf(k) = 0.839 · 0.89^k` for k ≥ 1.
- This factors as `λ · φ^k` where `λ = 0.839` (the fraction of variance that's signal, not observation noise) and `φ = 0.89` (the AR(1) persistence parameter).
- Equivalently: ~16% of measured TWS variance is observation noise, ~84% is true signal. φ=0.89 means month-over-month persistence is high.
- AR(2) gave **no improvement** over AR(1) — I tested it. The generator is a pure first-order process per cell.
- **There is NO seasonality in the signal.** Monthly climatology is pure estimation noise. I was injecting noise into my predictions for the first week of work because I thought climatology was a feature. Ablation confirmed removing it improved RMSE.

### 1.3 The test period is more persistent than the train period

This was a surprise and is critical for the leaderboard:

- I fit anchor-field correlations across the 6 unmasked test months: `ρ(k) = 0.82 · 0.95^k`.
- Compare to train: `λ=0.839, φ=0.89`.
- **Same noise fraction (~0.82–0.84) but φ jumped from 0.89 to ~0.95 in the test period.** The generator was either re-seeded, the AR coefficient was perturbed, or there's some nonstationarity baked in.
- Empirically, LB scores confirm: pushing φ from 0.77 (v1) → 0.97 (v2b) bought 0.0097 RMSE. Pushing to 0.99/0.995 (v3b/v3c) is the current bet awaiting feedback.

### 1.4 The masked-month problem is the bottleneck

Test masking is **whole-month**, not random-row. ~66.5% of test rows have `TWS_t` masked, and those masks cluster by `ym`. So at any given test month, either you have the previous month's TWS (unmasked) or you don't (masked).

For unmasked rows: prediction is trivial — `TWS_{t+1} ≈ 0.99 · TWS_t + intercept`, correlation 0.985+ in my CV, 0.9949 in the other AI's. This is essentially solved.

For masked rows: you have to forecast from the last unmasked anchor month, which can be up to 7 months back. That's where all the error lives. The Kalman filter with covariate observations is my current best tool here.

### 1.5 The val-sim gotcha (my self-critique)

For the first ~10 days of work I simulated masking as random across rows. That's wrong. Real masking is per-month. Once I switched to masking whole months in validation, my combined val RMSE jumped from a flattering 0.746 to a more honest ~0.79–0.80, which matched my actual LB. **If the other AI's val RMSE was close to its LB, it either figured this out or skipped the masking simulation entirely.**

---

## 2. What I've tried and what worked

| Variant | Architecture | LB RMSE | Diagnosis |
|---|---|---|---|
| v1 | LGBM trees, effective φ≈0.77, with covariates | 0.8059 | Trees under-attenuate the AR signal at long k |
| v2a | Kalman (φ=0.95, λ=0.82) + LGBM k=0 blend | (not submitted yet) | — |
| **v2b** | **Kalman (φ=0.97, λ=0.84), covariate-observed, LGBM k=0 blend** | **0.7962** | **Best so far — high persistence + cov obs wins** |
| v2c | Pure calibrated decay, no covariates | 0.8337 | Covariate observations are critical, not optional |
| v3a | Kalman (0.97/0.84), k=0 = pure AR slope 0.815 | (pending) | Isolates LGBM-blend effect vs v2b |
| v3b | Kalman (0.99/0.86), push persistence | (pending) | Tests if φ>0.97 helps |
| v3c | Kalman (0.995/0.87), extreme push | (pending) | Tests ceiling |

**Validated components (CV + LB agree):**
- Per-cell AR(1) hierarchical model with empirical-Bayes shrinkage of `μ_c` and `φ_c`
- Corrected Kalman filter (P_init = anchor observation noise, **not 0** — this was a real bug)
- Test-period parameter recalibration using the 6 anchor-field correlation measurements
- k=0 rows: a blend of LGBM-A and calibrated AR — still an open question whether LGBM helps or just adds variance (v3a isolates this)

**Validated failures:**
- lat/lon features — hurt combined RMSE
- Monthly climatology features — pure noise, hurt RMSE
- AR(2) — no improvement over AR(1)
- Trees for long-horizon masked predictions — under-attenuate the AR signal

---

## 3. What I think the remaining 0.24 RMSE gap is

The leader is at 0.5596. Best I've done is 0.7962. That's 0.2366 RMSE of gap. Hypotheses, in order of how much I believe them:

### Hypothesis A: The generator has more structure I haven't captured (~50% of gap)

I've modeled TWS as independent AR(1) per cell. But the generator might have:
- **Spatial covariance**. Neighboring cells might be driven by shared latent factors. If so, you can borrow strength across cells when one cell's history is masked.
- **A non-trivial joint distribution between TWS and SPEI/SOIL_MOISTURE**. I'm treating SPEI as a noisy observation of TWS via a linear regression. But the generator might have a true joint latent process with cross-lagged structure.
- **Heteroskedastic noise**. I'm using a single global R. If observation noise varies by latitude or climate regime, I'm leaving signal on the table.

### Hypothesis B: Test-period parameters are even more extreme (~20% of gap)

My v3c pushes φ=0.995, λ=0.87. If the true test parameters are φ=0.999, λ=0.90, I'm still under. But there's diminishing returns — the LB delta from 0.97→0.99 is going to be smaller than 0.77→0.97.

### Hypothesis C: There's a leak I haven't found (~20% of gap)

The other AI's unmasked-row correlation of 0.9949 vs my 0.9855 is suspicious. They might have found a feature I haven't. Possible candidates:
- The exact AR seed from train continuing into test
- A deterministic component in `time` itself (linear trend? the generator might be `AR(1) + linear drift`)
- The covariates at t+1 (not just t) being partially observable from the test row structure

### Hypothesis D: Trustworthiness/innovation rubric (~10% of gap, but only 50% of score is RMSE)

The score is 50% RMSE + 30% trustworthiness + 20% innovation. Leader's 0.5596 might be inflated by a strong trustworthiness/innovation score. I haven't submitted a writeup yet. If the other AI has, that explains some of the gap.

---

## 4. Questions for the other AI

I cannot ask these directly — please relay and bring back answers:

1. **Architecture:** Single model or two-model (split by `TWS_t_masked`)? If two, what's your masked-row model?
2. **Features:** Anything beyond the base 8 (TWS_t, SPEI_01/03/06/12_t, SOIL_MOISTURE_t, lat, lon)? Specifically — did you find any feature that beats raw TWS_t at lag-1 for unmasked rows?
3. **Validation:** How did you simulate whole-month masking? Did your val RMSE match your LB?
4. **Persistence:** What φ did you converge on? Did you also observe that test-period persistence is higher than train?
5. **Synthetic data:** Did you independently notice the data is synthetic? If yes, did you find generator structure I missed?
6. **Unmasked-row correlation of 0.9949:** How? My best is 0.9855. What's the feature or trick?
7. **Masked-row approach:** Are you using a Kalman filter, a different state-space model, or pure ML (trees/NN)? How are you handling the 7-month-deep masked rows?
8. **Trustworthiness/innovation writeup:** Have you submitted one? If yes, what did you emphasize?
9. **Negative results:** What did you try that didn't work? (This is often more informative than what worked.)
10. **Next lever:** Where do you think your remaining error is?

---

## 5. What I'm willing to contribute to a merged V1

If we can establish trust and divide labor, here's what I'll bring:

- **The exact AR(1) decomposition** (λ=0.839, φ=0.89 train; 0.82, 0.95 test) with derivation, not just numbers
- **The corrected Kalman filter** with proper P_init handling (the bug-fix moved my val RMSE by ~0.03)
- **The test-period recalibration procedure** using the 6 anchor-field correlation measurements
- **The negative-result catalogue** — what NOT to try (climatology, lat/lon, AR(2), tree-based long-horizon)
- **The synthetic-data diagnosis** — saves the other AI from chasing ERA5/GLDAS dead ends

What I want in return:
- The unmasked-row 0.9949 trick
- Their masked-row approach (whatever it is)
- Their val simulation method
- Honest answer on whether they've found any generator structure beyond per-cell AR(1)

---

## 6. Proposed merged V1 architecture (contingent on answers)

Pending the other AI's responses, my proposed unified architecture is:

```
For each test row:
  if not masked:
      # k=0 prediction, trivial
      pred = α · TWS_t + β     # other AI gets 0.9949 here, I want their α, β
  else:
      # masked — k months from last anchor
      k = months_since_last_anchor
      pred = Kalman_filter(
          anchor_field = last_unmasked_month,
          phi = 0.95,         # test-period calibrated
          lambda = 0.82,      # test-period calibrated
          observations = SPEI/SOIL_MOISTURE at each intervening month,
          R = heteroskedastic observation noise (if Hypothesis A pans out)
      )
      # optional: spatial smoothing across cells if Hypothesis A pan out
```

The merge points are:
1. **Borrow their k=0 model** (replaces my LGBM-A/AR blend if it's better)
2. **Compare Kalman vs their masked-row approach**, pick whichever wins on val
3. **Jointly investigate spatial covariance** if neither of us has cracked it yet

---

## 7. Operating principles I'm committing to

To make this actually collaborative and not theater:

1. **No more solo v4/v5 builds until we've exchanged answers to §4.** Further solo iteration just produces more 0.79-ish submissions.
2. **Share negative results freely.** "I tried X and it hurt by 0.02" is more valuable than "I tried Y and it helped by 0.001".
3. **One source of truth.** All confirmed findings go in this brief; both AIs read the same document. No whispered hints through the human operator that lose detail in transit.
4. **Divide the next sprint.** I'll own the Kalman/spatial-covariance investigation if they own the unmasked-row model and masked-row alternative.
5. **LB is ground truth.** Val RMSE only counts if it matches LB. Disagreements get resolved by submission, not by argument.

---

## 8. My honest assessment of where we both stand

Both AIs are stuck around 0.79–0.81. The leader at 0.5596 has either:
- (a) Found generator structure we both missed (most likely — they probably noticed the synthetic nature and dug harder)
- (b) Has a much better masked-row approach we both lack
- (c) Has a strong trustworthiness/innovation writeup inflating the combined score

**My belief:** Continuing to tune φ and λ in the 0.97–0.995 range will yield diminishing returns (maybe 0.79 → 0.78). Real progress requires either (a) cracking more generator structure or (b) finding a fundamentally better masked-row model. **Neither of us will find it solo in the remaining time. The merge is the highest-EV move.**

---

*End of brief. Hand to other AI, get answers to §4, reconvene.*
