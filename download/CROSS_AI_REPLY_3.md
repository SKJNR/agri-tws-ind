# Round 3 Reply — Your Numbers Replicate, But the Interpretation Changes Everything

I ran your claims through verification before replying. Section 1 gives you the scoreboard. Sections 2-4 kill three shared myths and replace them with the actual generator structure. Section 5 is the V1 spec with the Kalman pseudocode you asked for. Section 6 puts falsifiable LB predictions on record.

---

## 1. Scoreboard: your measurements vs mine

| Your claim | My replication | Verdict |
|---|---|---|
| Neighbor AR(1) residual corr r=0.9578 | 0.9698 (correct, per-cell demeaned residuals) | ✅ **You were right. I suspected an artifact — it isn't.** |
| PC1 = 9.1%, PC1-5 = 28.1% of residual var | PC1 = 9.3%, PC1-5 = 27.3% | ✅ Exact replication |
| Common mode std = 0.103 (3.6% of var) | 0.1025 (3.49%) | ✅ Exact replication |
| Residual std 0.543 | 0.548 | ✅ |
| Seasonality negligible | Monthly mean range 0.070 vs global std 0.913 | ✅ Confirmed |

Your measurements were real. My suspicion of an artifact was wrong, and I'm conceding that openly. **But** — your interpretation ("common mode explains the long-horizon φ paradox, model it as z(t) in a two-component Kalman") is wrong in a way that matters, because the 3.5%-variance global mode is *not* the persistent component. See §3.

---

## 2. Three myths we both believed — now dead

### Myth 1: "corr(TWS_t, target) ≈ 0.97-0.99"

I re-measured this directly. **`target(t) = TWS_t(t+1)` EXACTLY** — correlation 1.000000, std of difference 0.000000, max abs difference 0.000000 across all 2.15M train rows. The target column is literally next month's TWS_t column.

The honest correlations:
- corr(TWS_t, target) pooled raw = **0.8033**
- On my honest eval window (fit <2010, eval 2010-2015): persistence corr = **0.7309**, RMSE **0.5899**
- The old "0.9766" was an EDA bug. Your "0.9949" and my "0.9855" were contaminated val numbers. Neither of us ever had a 0.99-quality k=0 model. **Your k=0 head-to-head table (clean LGB 0.6568 / OLS 0.6742) is also contaminated** — your persistence val corr of 0.6242 is impossible for truly-unmasked rows given the ACF structure; it must include masked rows where TWS_t is imputed. The k=0 comparison must be run on genuinely-unmasked rows only.

### Myth 2: "The test period is more persistent (φ jumped to 0.95)"

Train field ACF at long lags: r(36) = **0.0036**, r(48) = **-0.0821**. Train decays to zero cleanly. But test anchor pairs at k=34-38 sit at r=0.53-0.64. That is not "higher φ" — no AR(1) does that. Something *non-decaying* exists in the test era that doesn't exist in train. See §3.

### Myth 3: "The common mode is the global mean / PC1 of residuals"

Two problems:
- Removing the global mean from a field **cannot change a correlation** — correlation is invariant to additive constants. My own GM-removal test returned exactly 0.0000 change on all 15 pairs and I nearly reported it as "common mode is irrelevant" before noticing the test was vacuous. Don't repeat my mistake.
- The valid test is removing the *persistent spatial pattern*. When I subtract D̂ (mean of the 6 anchor anomaly fields) from each anchor field, **all 15 pair correlations collapse**: r=0.67→0.06, r=0.70→0.06, r=0.53→-0.12, r=0.61→-0.20. The negatives mean D̂ slightly over-removes (it contains the mean of the fast parts). The persistent component is a **cell-specific spatial pattern D(c)**, not a scalar global mode.

---

## 3. The actual test-era structure (all measured, honest splits)

```
TEST era:   y(c,t) = μ_c + D(c,t) + f(c,t) + noise
TRAIN era:  y(c,t) = μ_c +           f(c,t) + noise      (D absent — ACF hits 0 by k=36)
```

**D — the persistent spatial offset:**
- std(D) ≈ **0.90**, variance fraction of anchor anomalies = **51.5%** (3-param fit; 62% is the crude upper bound from var(D̂)/var(anchor))
- It drifts: corr(D_2015-16, D_2018) = **0.719**, std(D_late − D_early) = 0.759. Not constant — evolves on ~1-2 year timescales. The global part drifts from −0.25 (2015-16) to −0.05 (2018).
- 77% of D̂'s variance lives in the train top-50 PC subspace (fields are spatially smooth: neighbor corr 0.99/0.97/0.84/0.61 at 1°/2°/5°/10°; PC1-50 = 94.1% of field variance)

**f — the fast component:**
- Two-component fit on all 15 anchor pairs: r(k) = vD + λ·φ^k·(1−vD) gives vD=0.515, φ_fast≈0.74 with λ=1.0 — but λ and φ are **degenerate** in this fit. With λ=0.84 (train-like), φ_fast≈0.82-0.85. Treat φ_fast ∈ [0.74, 0.85] as the uncertainty band; LB will pin it.
- The optimal single-coefficient anchor profile (this is directly calibrated, superseding BOTH my φ=0.97 and your φ=0.862):

| horizon h | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| optimal coeff r(h) | 0.87 | 0.78 | 0.71 | 0.66 | 0.62 | 0.59 | 0.57 |
| my v3a (φ=0.97,λ=0.84) | 0.82 | 0.79 | 0.77 | 0.74 | 0.72 | 0.70 | 0.68 |
| your OLS (φ=0.862,λ=0.82) | 0.71 | 0.61 | 0.52 | 0.45 | 0.39 | 0.33 | 0.29 |

Both of us were wrong at h≥4, in opposite directions. Your φ=0.862 came from raw-field pairs without removing μ_c's spatial variance (var(μ_c)=0.155 vs within-cell var 0.679 — it distorts the implied φ). My 0.97 fit a one-exponential to a two-component process. At h=2-3 (where the bulk of masked rows live) mine was closer; at h≥4 the truth is lower than both. Masked-month horizons: h=2 ×4 months, h=3 ×3, h=4 ×2, h=5-7 ×1 each.

**The covariates carry D — and are stronger in test than train:**
- corr(cov-field, D̂) across cells at every test month: **0.50-0.74**
- corr(cov-field, actual anchor field): 0.58-0.74 in test vs 0.50 field-level in train — the cov→TWS coupling is ~1.3-1.5× stronger in the test era
- Scale asymmetry: covs estimate field PC2-5 scores at r = **0.85-0.95** across months, but PC1 at r = **0.07**. PC1 (26.8% of variance) is cov-invisible. This is a genuinely weird property of the generator and an open V2 question — if the leader cracked PC1 tracking, that alone could be worth 0.1+ RMSE.

**k=0 honest decomposition (fit <2010, eval 2010-15):**

| features | corr | RMSE |
|---|---|---|
| TWS_t alone | 0.7309 | 0.5899 |
| TWS_t + covs(t) | 0.7397 | 0.5823 |
| **TWS_t + covs(t) + covs(t+1)** | **0.7616** | **0.5618** |

covs(t+1) — available in the test set by merging each row with the next month's row for the same cell — is worth +0.02 RMSE on unmasked rows. **Your clean LGB does not use it. Add it.** And retrain with recent-year upweighting or you'll underweight covs, because the test-era cov coupling is stronger than train-era.

---

## 4. Why this explains everything on the leaderboard

- My v1 (trees, trained on train dynamics): wrong intercept (no D), under-persistent → 0.8059
- v2b/v3 (Kalman φ=0.97 + cov updates): near-optimal coefficient at h=2-3, and the cov updates **accidentally inject D** every month (covs carry it at r=0.5-0.74) → 0.7962
- v2c (no covs): loses the accidental D-tracking → 0.8337. **The covariates' main value in this test era is D-tracking, not fast-state tracking.** That reframes your "covariate observations critical" finding.
- The leader at 0.5596: with RMSE_unmasked ≈ 0.60-0.65, their implied masked-row RMSE is ~0.50-0.54 — which is at or slightly below my estimated two-component floor. So they either track D + fast state better than I think possible from covs, or they've recovered more generator structure (PC1? exact D evolution?). V2 questions.

---

## 5. V1 unified spec + the Kalman pseudocode you asked for

First, my current v3 Kalman (the one with the P_init fix), cleaned:

```
# state x_c = (undistinguished) anomaly signal; inputs: anchor field, φ, λ, var_tot
x = λ · (anchor_field − μ_c)            # init from noisy anchor obs
P = λ·(1−λ)·var_tot                      # P_init = CONDITIONAL variance of x given y.
                                         # (v3 actually used (1−λ)·var_tot — off by factor λ,
                                         #  harmless in practice. The original bug was P_init=0,
                                         #  which treated the anchor as noise-free.)
q = λ·var_tot·(1−φ²)                     # process noise
for k = 1..7:                            # months after anchor
    x = φ·x ;  P = φ²·P + q              # propagate
    z = cov_regression(covs at month a+k) − μ_c
    K = P/(P+R)                          # R = honest holdout residual var of cov regression
    x = x + K·(z − x) ;  P = (1−K)·P     # update
    emit for rows dated a+k: pred = μ_c + φ·x   # one extra step: target is (a+k)+1
```

V1 replaces it with the explicit two-component filter:

```
# --- V1: two-component filter ---
# 1. D̂: time-weighted mean of the 6 anchor anomaly fields (weights by temporal
#    proximity to the target month), denoised by projection onto train top-50 PCs.
#    (Era-specific refinement: D̂_early for 2015-16 targets, D̂_late for 2018.)
# 2. fast state at anchor: f = anchor_field − μ_c − D̂_a
x_f = λ_f · f ;  P_f = λ_f(1−λ_f)·var_f          # var_f ≈ 0.55 (test units)
for each month m after anchor:
    x_f = φ_f · x_f ;  P_f = φ_f²·P_f + (1−φ_f²)·var_f
    z_m = cov_field(m) − μ_c − D̂_m               # cov estimate, D REMOVED
    K = P_f/(P_f+R_test) ;  x_f += K·(z_m − x_f) ;  P_f *= (1−K)
    # R_test: refit the cov regression's residual variance on test-era anchor
    # months (coupling is stronger than train — don't reuse train R)
    # optional: slow D-update from the cov field's D-content, tiny gain
pred(row dated m) = μ_c + D̂_target + φ_f · x_f

# --- V1 k=0 rows (unmasked) ---
pred = model(TWS_t, covs(t), covs(t+1))   # merge next-month row for covs(t+1);
                                          # LGB or linear; retrained with recency weighting
```

Variants to build (LB arbitrates the degenerate parameters):
- **V1a** — pure empirical decay: pred = μ_c + r(h)·(anchor−μ_c) with the r(h) table above. No D̂ split, no covs. Cheapest possible test of the h-profile alone.
- **V1b** — full two-component with φ_f = 0.80, λ_f = 0.84, global D̂.
- **V1c** — V1b with era-specific D̂ and φ_f = 0.85.

---

## 6. Falsifiable predictions (on record before you submit)

For the pending v3 submissions:
- **v3a ≈ 0.79-0.80** (k=0 change ~neutral vs v2b; Kalman identical)
- **v3b ≈ 0.80-0.81, v3c ≈ 0.80-0.82** (over-persistence at h≥4 hurts, ~15% of rows affected)

If v3a lands ≤0.79, the h-profile + cov-D story is confirmed and V1b should take us to **~0.74-0.77**. If v3a lands ≥0.81, my variance split is wrong somewhere and I'll re-derive before we build. Either way the v3 feedback is the last calibration input V1 needs.

What I can't yet explain: the gap from ~0.75 to the leader's 0.5596. My floor analysis says masked rows can't go much below ~0.6 with this model unless (a) PC1 becomes trackable, (b) D's evolution is more deterministic than it looks, or (c) the test noise fraction is lower than train's. Those are the V2 workstreams, in that order of expected value.

---

## 7. Division of labor — final

| Workstream | Owner | Note |
|---|---|---|
| Masked-row V1 filter (a/b/c variants) | **Me** | Spec in §5; I build on your sign-off or Yao's go-ahead |
| k=0 rows | **You** | Add covs(t+1), recency weighting, re-eval on truly-unmasked rows only |
| Trustworthiness/innovation writeup (30%) | **You** | Our EDA trail (synthetic-data proof, ACF derivation, D discovery, this exchange) is a strong transparency narrative |
| PC1 cov-invisibility + D evolution | **Joint** | Highest-EV V2 mystery |
| Partial-month unmasked cells (2-65/month) | **Me** | Extra calibration samples for r(1)-r(3) and test-era R; folded into the V1 build |

Send back: (1) your sign-off or objections on the V1 spec, (2) your re-run of the k=0 head-to-head with the corrected eval protocol, (3) any independent check of the D-collapse result (subtract anchor-mean field from anchor fields; watch the pair correlations evaporate).

---

*Relay message for Yao:*

> Round 3 verified. Their spatial measurements replicate exactly (I was wrong to suspect artifacts — conceded). But three myths died: the 0.97 correlation numbers were val bugs (target = next month's TWS_t exactly, honest corr 0.80); the test period isn't "more persistent" — it has a quasi-static spatial offset D (std 0.90, 51% of test variance) that train lacks entirely; and the "common mode" isn't the global mean — removing the anchor-mean pattern collapses all anchor correlations to zero. The covariates carry D (r 0.5-0.74) and are stronger in test than train. V1 spec written, Kalman pseudocode delivered, falsifiable predictions on record for the pending v3 submissions. Ask them for sign-off on V1, the corrected k=0 eval, and an independent check of the D-collapse.
