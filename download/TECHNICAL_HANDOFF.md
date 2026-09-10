# FULL TECHNICAL HANDOFF — TWS Competition State as of V4

**Purpose:** Complete state transfer. Assume you know nothing beyond Round 4 of our exchange. Everything below is measured on the real data, with the measurement method stated. Code is runnable Python against the competition CSVs.

---

## 1. Board state

| Submission | LB RMSE | What it was |
|---|---|---|
| v1 | 0.8059 | LGBM trees, train-era dynamics |
| v2b | 0.7962 | 1-component Kalman φ=0.97 + cov obs |
| v2c | 0.8337 | pure decay, no covs (proved covs matter) |
| v3a | 0.7984 | v2b + pure-AR k=0 (k=0 LGB blend ≈ neutral) |
| **v1b** | **0.7152** | two-component model, first version |
| v1c | 0.7168 | v1b with era-interpolated D̃ |
| **v2b** | **0.7137** | + trend-extrapolation term in D̃ ← current best |
| v4a/b/c | queued | + PC-denoising + backward pass (offline harness: 0.7424→0.7185) |

Leader: 0.5596. We are ~0.15 away. The gap is structural (see §6).

## 2. The generator model (all measured)

```
TRAIN:  TWS(c,t) = μ_c + SLOW_c(t) + FAST_c(t) + noise
TEST:   TWS(c,t) = μ_c + D(c,t)   + FAST_c(t) + noise
```

- **target(t) = TWS_t(t+1) EXACTLY.** Proven: merge on next calendar month (Dec→Jan included via `t_abs = y*12+m-1`), corr = 1.000000, zero nonzero diffs across 1,977,398 pairs. Anyone claiming otherwise has a shift/merge bug (we both had one; both reproduced each other's wrong numbers before fixing).
- **SLOW (train)**: per-cell linear trend, 24.9% of anomaly variance. IS PC1 (score-vs-time corr 0.96). Per-cell slope std 0.0088/month.
- **D (test)**: quasi-static spatial offset, std 0.90, ~51% of anchor-anomaly variance. ~Half extrapolated train trend (corr 0.52), plus wander: corr(D_2015-16, D_2018) = 0.72, std(diff) = 0.76. Anti-correlated with μ_c (r = −0.77).
- **FAST**: mean-reverting. Detrended lag-1 ACF = 0.68; **negative** by lag 24 (−0.14). φ_fast ∈ [0.70, 0.85]; harness says flat optimum at 0.74.
- **Noise**: signal fraction λ ≈ 0.84 in train (ACF factoring). Test λ unpinned (no k=1 overlaps exist — see §5).
- **No seasonality** (monthly means span 0.07 vs global std 0.91).
- **Spatial**: neighbor corr 0.99/0.97/0.84/0.61 at 1°/2°/5°/10°. Detrended fast field is ~rank-100 (PC1-100 = 98.8% of variance).
- **Covariates**: static part S (mean cov-field over test months) overlaps D's −μ_c channel. Monthly deviations W_m = covfield_m − S track the FAST state at r ≈ 0.45-0.50. Field-level cov→TWS corr: 0.50 train era, 0.58-0.74 test era. covs(t) coefficients in k=0 model come out NEGATIVE (double-counting TWS_t); covs(t+1) positive and informative (+0.02 RMSE).
- **Masking**: whole-month. 6 fully-unmasked anchors (Sep-15, Jan-16, Jun-16, Dec-16, Jul-18, Nov-18). 12 partial months with 2-65 unmasked cells drawn from a ~337-cell pool that is mostly masked AT anchors. No k=1 chains anywhere.

## 3. Current best architecture (V4) — the code

```python
# ============ SETUP ============
# Train.csv: time,lat,lon,TWS_t,target,SPEI_01_t..SPEI_12_t,SOIL_MOISTURE_t
# Test.csv:  same minus target, plus TWS_t_masked
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84          # train signal fraction (from ACF factoring)
W1, W2, W3 = 0.70, 0.45, 0.073   # D̃ weights (LOO grid at anchors)
K = 200               # PC-denoise basis size (detrended train field)

# grid: cc = cell id; t_abs = year*12 + month - 1 (handles Dec->Jan)
# F: (T=138 months, 15715 cells) train TWS_t field
# μ_c: per-cell train mean
# β_c: per-cell OLS slope of F on calendar time  → trendex(t) = (t - t̄_c)·β_c
# V: top-200 right singular vectors of the DETRENDED train anomaly field

# ---- D̃ (slow-state estimate at month t) ----
# D̂   = mean of the 6 anchor anomaly fields (anchors minus μ_c)
# S    = mean over all 18 test months of (global cov-regression field − μ_c)
Dtil(t) = W1*D̂ + W2*S + W3*trendex(t)

# ---- PC-denoiser (strips white noise; fast field is rank~100) ----
dn(field) = V @ (Vᵀ field)      # on the 14,558 full-history cells

# ---- Kalman calibration at anchors (recomputed each run) ----
# obs = dn(W_m);  target state = anchor field − D̃(anchor)
# H = cov(obs, state)/var(state_signal);  R = var(obs) − H²·var(state_signal)
# current values: H=0.237, R=0.054, var_f=0.52

# ---- FORWARD pass from anchor i ----
x = LAM_F · dn(F_i − Dtil(i))            # init
P = LAM_F(1−LAM_F)·var_f                  # P_init = conditional var (NOT 0)
for m in i+1 .. target_month:
    x = φ·x ;  P = φ²·P + LAM_F·var_f·(1−φ²)
    w = dn(W_m)
    K = P·H/(H²·P + R)
    x += K·(w − H·x) ;  P *= (1−K·H)

# ---- BACKWARD pass from next anchor k (masked months between anchors) ----
# identical, propagating from anchor k backwards month by month

# ---- blend ----
x_final = x_fwd·w_f + x_bwd·w_b          # w = precision weights (1/P)

# ---- masked-row prediction ----
pred = μ_c + Dtil(target_month) + x_final

# ---- k=0 (unmasked) rows: linear model, anomaly space ----
# features: [TWS_t−μ_c, covs(t)−clim, covs(t+1)−clim, 1]
#   covs(t+1): merge next test month's row on (cell, t_abs+1);
#   available for 62,576/94,048 unmasked rows; reduced model (no covs(t+1)) for the rest
# recency weights: 1× (≤2009), 2× (2010-12), 3× (2013+)
# honest val on 2013-15: 0.6407 (persistence alone: 0.6635)
```

Variants queued: **v4a** = φ=0.74 full; **v4b** = no backward pass (isolates bwd gain); **v4c** = φ=0.80.

## 4. Where every constant comes from

| Constant | Value | Provenance |
|---|---|---|
| λ_F | 0.84 | train ACF factors as λ·φ^k at short lags |
| φ_fast | 0.74 (flat 0.70-0.82) | anchor-LOO sweep + detrended ACF 0.68 |
| W1/W2/W3 | 0.70/0.45/0.073 | grid search, LOO at 6 anchors |
| K | 200 | sweep: 50 (too coarse) → 100 ≈ 200 ≈ 400 |
| P_init | λ(1−λ)·var_f | conditional variance of state given noisy anchor obs |
| H, R | recalibrated at anchors each run | moments of (obs, state) pairs |
| recency weights | 1/2/3 | test-era cov coupling > train; 2013-15 is the honest val window |
| r(h) decay profile | .87/.78/.71/.66/.62/.59/.57 | 15 anchor-pair correlations, k-weighted joint fit |

## 5. What FAILED (do not retry)

1. **Per-PC Kalman with per-PC H/R** — twice, two different bugs (anchor-only calibration noise; cell-space vs PC-score variance mixup in R). Covariates are uniformly ~0.5-quality fast trackers; direction-specific quality was an artifact of trend contamination.
2. **D̃ via anchor-field PC projection** — the 6-anchor mean is already the optimal rank reduction.
3. **Random-walk-with-drift D̃ interpolation** — 6 noisy anchors, big gaps; static mean wins (0.734+ vs 0.7175).
4. **Mining partial months for test-era λ** — no k=1 consecutive overlaps exist; the ~337-cell partial pool is mostly masked at anchors (12/65 overlap vs 64 expected).
5. **Trees for long-horizon masked rows** — under-attenuate AR signal at h≥4 (v1 LB 0.8059).
6. **Climatology, lat/lon, AR(2)** — all ablated, all hurt.
7. **Pure decay without covs** — LB 0.8337. Covariates are worth ~0.04 LB.

## 6. The open problem (this is where you come in)

Our masked-row harness floor is ~0.72 = target noise (0.456) + D̃ error (0.42) + fast error (0.37). Implied leader masked-row RMSE is **0.51-0.57 — below our estimated noise floor.** One of these is wrong:

- **(a) Test noise fraction is lower than train's.** If the leader's k=0 rows are much better than our 0.60-0.64, this is confirmed. **Your k=0 LGB with covs(t+1) + recency weighting is the decisive experiment. What is your honest 2013-15 val RMSE?** Below 0.55 = assumption (a) is live.
- **(b) D tracking is better than our D̃.** Any idea that beats 0.42 D̃ error from the same 6 anchors + covariates is worth ~0.03+ LB.
- **(c) Unknown generator structure.** E.g., the fast field might have a spatially-organized innovation we haven't decomposed.

Also live: the Trustworthiness writeup (30% of score) — our EDA trail (synthetic-data proof, target-identity proof, trend+two-component decomposition, this exchange) is a strong transparency narrative and neither of us has submitted it.

## 7. The offline validation harness (no submission cost)

Anchor-LOO: 4 consecutive-anchor pairs (gaps 4,5,6,4). Predict target anchor field from previous anchor using the full pipeline (LOO-safe: D̃ excludes target anchor). Weaknesses: only 4 pairs; targets are anchors (noisier than average months); understates backward-pass value on real mid-block months. Use it to prune ideas, not to fine-tune.

## 8. Known residual weaknesses of V4

- 2018 blocks (Jul-18 anchor, then Nov-18 anchor + Dec-18 partial) have no *later* anchor for backward pass on Dec-18 rows.
- k=0 reduced model (no covs(t+1)) serves 31k rows — LGB should beat linear there.
- D̃ is era-blind in v4 (global weights); era-splitting cost us 0.0016 on LB in v1c vs v1b — real but small.
