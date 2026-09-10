# Worklog

---
Task ID: 1
Agent: Super Z (main agent)
Task: Learn/research the Zindi competition "One Step Ahead of Drought: Forecasting Global Water Storage Challenge"

Work Log:
- Fetched leaderboard page: https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/leaderboard
- Fetched main competition info page (overview, rules, evaluation, prizes)
- Fetched competition data page (dataset structure, files)
- Extracted embedded leaderboard table (top 50 + benchmark) from HTML
- Saved raw JSON fetches to /home/z/my-project/leaderboard.json, competition_main.json, competition_data.json

Stage Summary:
- Competition: TWS (Total Water Storage) one-month-ahead global forecasting, organized by ITU/AI for Good with UN partners
- Prize: €2,000 total (1000/600/400); 5,000 Zindi points
- Timeline: 09 Jul 2026 → 13 Sep 2026; 1,070 joined, 414 active
- Data: Train.csv 2,154,021 rows / Test.csv 280,961 rows; features: TWS_t, SPEI (1/3/6/12), SOIL_MOISTURE_t; keyed by (lat, lon, date)
- Target: TWS at t+1; 66.5% of test rows have TWS_t masked → effective horizon 1-7 months
- Metric: RMSE (50%) + Trustworthiness rubric (30%) + Innovation/practicality (20%)
- Rules: 5 subs/day, 200 total, max team 4, no AutoML, open-source only, CC-BY SA 4.0
- Leaderboard top: MOHAR 0.5596, Shankar 0.5893, GIrum 0.6234; benchmark 0.8999

---
Task ID: 2
Agent: Super Z (main agent)
Task: Test whether competition data can be downloaded from Zindi site or Google Drive

Work Log:
- Extracted file download URLs from Zindi data page (api.zindi.world/v1/competitions/.../files/...)
- Tested direct download: HTTP 401 "not authorized" — auth_token=null fails; login required
- Installed gdown 6.1.0 via pip --break-system-packages
- Verified connectivity: drive.usercontent.google.com, docs.google.com, googleapis.com all reachable
- End-to-end test: downloaded gdown's official example file (SPNet model) from Google Drive — successfully pulled 260MB before test timeout stopped it
- Cleaned up test artifacts; confirmed 9.2GB free disk space

Stage Summary:
- Zindi direct download: NOT possible (auth wall, 401)
- Google Drive download: CONFIRMED WORKING (260MB test download succeeded)
- Disk space available: 9.2GB — enough for Train.csv (275MB) + Test.csv (33MB) + workspace
- Recommended workflow: user downloads files from Zindi, uploads to Google Drive with public link, shares link for agent to download

---
Task ID: 7 (session restored after env reset wiped data/scripts)
Agent: Super Z (main agent)
Task: LB-feedback-driven v3 calibration

LB feedback from user's submissions:
- v1 (trees ~0.77 eff, with covs): 0.8059 (rank 260)
- v2b (Kalman 0.97/0.84, with covs, k=0 LGBM blend): 0.7962 <- BEST so far
- v2c (pure decay 0.95/0.82, NO covs): 0.8337 (worse than v1)

Decoded signals:
- Higher persistence helps: v2b (0.97) < v1 (0.77) by 0.0097
- Covariate observations critical: v2c (no covs) > v1 by 0.028
- Combination: high persistence + cov-obs Kalman is the winning direction

v3 plan (rebuild after env reset; uses pure-AR k=0, no LGBM dependency):
- v3a: (0.97, 0.84) Kalman+covs, k=0 pure AR slope 0.815  -> isolates LGBM-blend effect vs v2b
- v3b: (0.99, 0.86) Kalman+covs, k=0 pure AR slope 0.857  -> push persistence
- v3c: (0.995, 0.87) Kalman+covs, k=0 pure AR slope 0.869 -> extreme push

Files: download/submission_v3a.csv, v3b.csv, v3c.csv

---
Task ID: 8
Agent: Super Z (main agent)
Task: Cross-AI Round 3 — verify other AI's common-mode claims, establish corrected generator model

Work Log:
- Ran verify_common_mode.py: neighbor AR(1) residual corr = 0.9698 (their 0.9578 replicates; NOT an artifact — conceded). Residual PCA: PC1 9.3%, PC1-5 27.3%. Global residual-mean std 0.1025. GM-removal test vacuous (correlation is shift-invariant).
- Ran decisive_measurements.py: target(t) = TWS_t(t+1) EXACTLY (corr 1.000000, zero diff) — kills the 0.9766 myth; honest corr(TWS_t,target)=0.803. Train ACF decays to 0.004 by k=36 (no floor in train). Spatial: neighbor corr 0.99/0.97/0.84/0.61 at 1/2/5/10 deg; anomaly-field PCA PC1-50=94.1%. Covs: per-cell R2=0.254; PC2-5 scores cov-estimable at r=0.85-0.95 but PC1 at r=0.07. D6: D-hat (mean of 6 anchor fields) = 62% of anchor variance, std 0.899; removing D-hat collapses all 15 anchor pair corrs (0.53-0.70 -> ~0.06 to -0.45).
- Ran v1_calibration.py: cov fields at test months carry D (corr 0.50-0.74); test-era cov-anchor coupling 0.58-0.74 vs train 0.50. Two-component fit: var(D)=51.5%, phi_fast=0.74 (lambda degenerate; band 0.74-0.85). Optimal anchor coefficient profile r(h): 0.87/0.78/0.71/0.66/0.62/0.59/0.57. k=0 honest decomposition: persistence 0.5899 RMSE; +covs(t) 0.5823; +covs(t+1) 0.5618. D drifts: corr(D_early,D_late)=0.719, std(diff)=0.759.
- Wrote download/CROSS_AI_REPLY_3.md: concessions (their numbers replicate), three myth-kills, corrected generator model (train: mu+AR1+noise; test: mu+D+AR1_fast+noise), V1 spec (two-component filter + k=0 with covs(t+1)), Kalman pseudocode, falsifiable v3 LB predictions (v3a 0.79-0.80, v3b 0.80-0.81, v3c 0.80-0.82), final division of labor.

Stage Summary:
- CORRECTED GENERATOR MODEL: test era = train-like fast process (phi_fast 0.74-0.85) + quasi-static spatial offset D(c) (std 0.90, 51% of test variance, drifts on ~2yr timescales, 77% inside train top-50 PC subspace) + noise. Train era has NO D.
- Covariates' main test-era value = D-tracking (not fast-state tracking). Explains v2b >> v2c on LB.
- V1 variants spec'd: V1a (empirical r(h) decay), V1b (two-component, phi_f=0.80, global D-hat), V1c (era-specific D-hat, phi_f=0.85). Expected 0.74-0.77. Leader's 0.5596 implies additional structure (PC1 tracking? exact D evolution?) — V2 workstreams.
- Pending: v3a/b/c LB feedback from user; other AI's sign-off on V1; corrected k=0 eval from them.
- Artifacts: download/CROSS_AI_REPLY_3.md, common_mode_verification.txt, decisive_measurements.txt, v1_calibration.txt; scripts verify_common_mode.py, decisive_measurements.py, v1_calibration.py

---
Task ID: 9
Agent: Super Z (main agent)
Task: Cross-AI Round 4/5 — resolve two disagreements, build V1

Work Log:
- r1_target_fix.py: airtight merge (t_abs+1, Dec->Jan included) proves target(t) = TWS_t(t+1) EXACTLY: corr=1.000000, 0/1,977,398 nonzero diffs, 187,351 Dec->Jan pairs all zero. Other AI's claim (0.9766/4.90/0.039) reproduced ONLY via sort+shift without month-gap check (7.53% of pairs skip months) -> their claim is a join artifact.
- round4_reconciliation.py R2: (a) LOO corr(cov-field, D-hat_-m) = 0.45-0.71 (my number survives w/o circularity); (c) per-cell-demeaned covs ridge = 0.17-0.30 (EXACTLY reproduces their 0.13-0.34); (e) raw-cov direct ridge also 0.17-0.29; (d) KEY: corr(D-hat, mu_c) = -0.77 (real D structure) and corr(cov-field static S, mu_c) = -0.85 (regression shrinkage) -> my 0.5-0.74 works largely through shared -mu_c channel; (b) monthly deviation W_m tracks (F_m - D-hat) at r=0.37-0.50 (REAL fast-state tracking); (f) DECISIVE: anchor-field prediction RMSE 0.842 -> 0.733 with weights (0.54 D-hat, 0.84 cov-field), stable across all 6 anchors -> cov updates validated.
- submission_v1.py: built V1a (empirical r(h) decay), V1b (two-component Kalman phi_f=0.80, global D-tilde = 0.676*D-hat + 0.490*S), V1c (phi_f=0.85, era-interpolated D-tilde). k=0 rows: linear anomaly-space model with covs(t+1), recency-weighted, full/reduced dual models. Kalman obs: W_m with H=0.305-0.313, R=0.114-0.120. covs(t+1) available for 62,576/94,048 unmasked rows.
- k0_diagnostic.py: 2013-15 window genuinely harder (persistence 0.664 vs 0.507 on 2002-09) — no bug; full model beats persistence ~0.023 RMSE consistently. covs(t) coefs negative (double-count), covs(t+1) positive (new info).
- All 3 CSVs verified: 280,961 rows, IDs match, no NaN.

Stage Summary:
- BOTH Round-4 disagreements resolved: (1) target = next month's TWS_t exactly — their noise claim was a join artifact (their numbers reproduce only via the gapped sort+shift); (2) cov->D coupling: their 0.13-0.34 = demeaned-cov view, my 0.5-0.7 = includes -mu_c channel; both reconciled; operationally cov updates worth 0.11 RMSE on anchor fields (measured).
- V1a/b/c BUILT: download/submission_v1{a,b,c}.csv. Predictions: v1a ~0.82-0.84 (no covs), v1b ~0.76-0.79, v1c ~0.75-0.78.
- New finding: train predictability degrades in 2013-15 (transition to test-era regime) — windows matter for val protocol.
- Pending: v3a/b/c + V1a/b/c LB scores from user; other AI's k=0 LGB with covs(t+1); PC1 cov-invisibility (V2).

---
Task ID: 10
Agent: Super Z (main agent)
Task: Process V1 LB feedback (v1b=0.7152 NEW BEST, v1c=0.7168, v3a=0.7984), resolve ACF contradiction, build V2

Work Log:
- v2_sweep.py: (S1) implied masked RMSE ~0.72-0.77 at u=0.6-0.64; leader needs masked ~0.51-0.57. (S2) anchor-LOO sweep over 4 consecutive-anchor pairs (h=4,5,6,4): best phi_f=0.70, lam insensitive, obs=train-reg W beats anchor-refit W (0.7414 vs 0.7488); cov obs worth 0.06 vs no-obs. (S3) PC1 discovery: score corr with time = 0.9588, ACF 0.91 at lag 24 -> looks like a trend, contradicts old ACF decay.
- acf_resolution.py (brute-force calendar-correct ACF via merge): RAW-value ACF has a floor (r(36)=0.24) that per-cell-anomaly ACF missed. Resolution: train anomaly = per-cell LINEAR TREND (24.9% var, ~ PC1, near-linear/random-walk trajectory) + MEAN-REVERTING fast process (detrended lag-1 = 0.68, NEGATIVE by lag 24). The two nearly cancel in pooled anomaly ACF -> old "AR(1) lambda=0.839 phi=0.89" was an effective composite. Detrended fast phi ~ 0.81 (with lam 0.84), anchor sweep says 0.70-0.80 band.
- Test-era D vs trend: corr(D-hat, trend-extrapolated field) = 0.525, std(trendex D)=0.887 vs std(D-hat)=0.901; 2018 anchors show trend flattening (actual global -0.03..-0.07 vs trendex -0.32..-0.33) -> D = partial trend extrapolation + wander.
- submission_v2.py: D-tilde = w1*D-hat + w2*S + w3*trendex(t), LOO-fitted weights (0.650, 0.456, 0.073 - trendex marginal). LOO RMSE: D-hat only 0.8421, old combo 0.8201, new 0.8180. Variants v2a (phi=0.80), v2b (0.74), v2c (0.70), all with k=0 model unchanged from V1. CSVs verified (280,961 rows, IDs match, no NaN).

Stage Summary:
- GENERATOR MODEL (v3): TWS = mu_c + SLOW_c(t) + FAST_c(t) + noise. SLOW = per-cell near-linear trend (~25% var, extrapolates into test era = ~half of D's variance + wander). FAST = mean-reverting, phi 0.70-0.81, tracked by Kalman with cov observations. Test D = w1*D-hat + w2*S + w3*trendex.
- v1b=0.7152 new best (was 0.7962). V1 architecture validated. v3a=0.7984 confirms k=0 LGB blend was ~neutral (v2b 0.7962).
- V2a/b/c built: trendex D-tilde + phi spread 0.80/0.74/0.70. Predictions: v2a 0.710-0.715, v2b 0.705-0.713, v2c 0.703-0.713.
- Leader gap analysis: they need masked ~0.51-0.57 vs our ~0.72-0.77. Remaining structural unknowns: slow-wander tracking (SPEI_12?), k=0 LGB (other AI's lane), PC1/slow-pattern cov channels.
- Recommendation: submit v2a+v2b today (2 slots left), v2c tomorrow + relay Round 6 brief to other AI.

---
Task ID: 11
Agent: Super Z (main agent)
Task: Post-v2b (LB 0.7137) structural experiments + build V4

Work Log:
- v3_experiments.py: detrended field spectrum PC1-50=92.3%, PC1-100=98.8% (fast field ~rank-100). Anchor-LOO: obs-denoise K=200 = 0.7424->0.7237 (biggest gain); +init-denoise 0.7264; +backward pass 0.7200. phi insensitive 0.70-0.82.
- v3_perpc.py + v3_perpc3.py: per-PC Kalman FAILED twice (0.76-1.16 vs 0.72 scalar). Root causes: anchor-only calibration noise; R-scaling bug (cell-space vs PC-score variance). Conclusion: covariates are ~uniform-quality (r~0.5) fast trackers; the "PC2-5 quality" was for trend-inclusive anomaly, not fast state. Per-PC dead end.
- v3_dtilde.py: D-tilde anchor-PC projection neutral (0.7195 vs 0.7200) - anchor-mean already optimal rank reduction. First version had a centering bug (projected onto deviation PCs). E7 weight grid: (w1,w2)=(0.70,0.45) -> 0.7185.
- v3_slow_rw.py: RW-with-drift smoothed D-tilde FAILED (0.734-0.757 vs 0.7175 static). 6 noisy anchors + big gaps: static mean hard to beat.
- partial_month_goldmine.py: partial months' unmasked cells = ~337-cell pool, mostly MASKED at anchors (12/65 overlap vs 64 expected); no k=1 consecutive overlaps -> test lambda unpinned.
- submission_v4.py: v4a (phi=0.74, denoise K=200, bwd, weights 0.70/0.45/0.073), v4b (no bwd - isolates bwd on real LB), v4c (phi=0.80). k=0 unchanged. CSVs verified.

Stage Summary:
- Validated config: scalar-H Kalman + PC-denoise(init+obs K=200) + backward pass + D-tilde(0.70 D-hat + 0.45 S + 0.073 trendex). Harness 0.7424->0.7185.
- Structural dead ends catalogued: per-PC obs model, D-tilde projection, RW slow interpolation. The 0.72 harness floor = target noise (~0.456) + D-tilde error (~0.42) + fast error (~0.37).
- V4a/b/c built. Predictions: v4a ~0.705-0.710, v4b ~0.708-0.713, v4c ~0.705-0.711.
- Leader gap remains structural: they need masked ~0.51-0.57 vs our ~0.72-0.75. Remaining hypotheses: (a) test noise fraction lower than train, (b) k=0 model much better, (c) unknown generator structure. Tomorrow: submit v4a/v4b/v4c.

---
Task ID: 12
Agent: Super Z (main agent)
Task: Validate CV↔LB rank correlation using 5 reproducible submitted configs

Work Log:
- Built scripts/cv_lb_correlation.py: applies the test set's calendar-month masking pattern to train 2013-2015 (val window), fits all infrastructure (mu_c, clim, cov regression, anchors, D-hat, S, W, beta_c trend) on train 2002-2012 only (no leakage), and reruns 5 reproducible variant configs (v1a_decay, v1b_kalman_glb, v1c_kalman_era, v2b_trendex, v3a_simple) on the val masked rows.
- Calendar-month mask fractions from test: months 2/3/4/5/8 = ~100% masked; months 1/6/7/9/12 = ~50%; month 11 = 0% (anchor). Applied same threshold (>0.5 → masked) to val → 39.1% val rows masked (140,573 of 359,107).
- 14 val anchor months identified (where mask_frac<0.5); LOO D-tilde weights on val: w1(D-hat)=0.786, w2(S)=0.218, w3(trendex)=0.049 (vs 0.650/0.456/0.073 on test — val anchors favor D-hat more since trend is weaker over 3yr than over 11yr).

Results (CV=2013-15 honest window, LB=test):
  v1a_decay       : CV 0.8292 / LB 0.8337 / diff -0.005
  v1b_kalman_glb : CV 0.6619 / LB 0.7152 / diff -0.053
  v1c_kalman_era : CV 0.6653 / LB 0.7168 / diff -0.052
  v2b_trendex    : CV 0.6599 / LB 0.7137 / diff -0.054
  v3a_simple     : CV 0.7559 / LB 0.7984 / diff -0.042

Spearman rho = 1.0000 (p<0.001), top-3 overlap 3/3, no rank inversions.

Stage Summary:
- CV IS TRUSTWORTHY. Perfect rank match across all 5 reproducible variants. From tomorrow onward: iterate on CV only, no LB-fishing.
- CV-LB gap is ~0.05 for Kalman variants, ~0.00 for pure decay. The ~0.05 gap is consistent (val period is slightly easier than test, probably due to lower noise ratio in 2013-15 vs 2024).
- Leader's implied CV = 0.5596 - 0.05 = ~0.51 (if same gap holds). Our best CV = 0.6599. The 0.15 gap to leader is far above our ~0.01 CV noise floor → leader has structurally different model, not just better tuning. Either (a) they cracked lower test-period lambda (some test months are cleaner than val), (b) different k=0 architecture, or (c) unknown generator structure.
- v1b vs v2b_trendex CV gap (0.002) matches LB gap (0.0015) → confirms trendex D-tilde is real but marginal; we should not over-invest in trendex variants.
- Recommendation: ship v4a/b/c tomorrow (CV-validated bets). Reserve 1-2 slots for pseudo-labeling + ensemble (Kalman+LGB) on best CV model.
- Artifacts: scripts/cv_lb_correlation.py, download/cv_lb_correlation.png, download/cv_lb_correlation.csv

---
Task ID: 13
Agent: Super Z (main agent)
Task: Post-reset recovery: diagnose v16_probe failure, rebuild pipeline, produce v17a/v17b

Work Log:
- User reported v16_probe.csv rejected ("There was a problem while processing your submission"); daily quota untouched (0/5, total 29/200). Env reset had wiped ALL v5-v16 artifacts (scripts + CSVs); only v1-v4 + worklog survived in git.
- Built scripts/validate_submission.py — universal format validator (columns, row count, ID set/order, NaN/inf, magnitudes, BOM, header). All past submissions PASS; use it on every future file.
- Rebuilt pipeline (rebuild_v17.py): regenerated V4 infra; ran A/B decomposition of masked-row configs. FOUND PROTOCOL FLAW: the val (2013-15) protocol has a SHORTCUT — masked months May/Aug have targets (Jun/Sep) landing on unmasked val months, so the backward Kalman reads the answer (bwd-only scored 0.5456 inflated). On TEST the designers removed every pre-anchor month (no May-16/Nov-16/Oct-18...), sealing this route. Verified: direct-copy via target=TWS_t(t+1) identity yields exactly 0 rows (visible pool at partial months is disjoint from previous month's row set).
- rebuild_v17b.py — HONEST CV protocol (evaluate only masked val rows whose t+1 is masked/absent, n=93,829, matching test structure). Results: v2b-equiv 0.6919, bwd-only 0.6551, dn+bwd (v4 recipe) 0.6785, dn-only 0.7023 => PC-DENOISER HURTS on test-like rows (+0.023-0.027); backward pass is the real gain (-0.037). Grid: phi=0.80/lam=0.84 best (0.6535); Dtil weights (0.70,0.45,0.073) confirmed; top3 ensemble 0.6534.
- k=0 decisive experiment (handoff §6a): LINEAR 0.6287 vs LGBM(400) 0.6273 vs 50/50 blend 0.6254 on val (honest — features never see TWS(t+1)). Verdict: LGB ≈ linear, small blend gain; the "leader has better k=0" hypothesis is weak; k=0 floor ~0.62-0.63 is real.
- CAUGHT 2 assembly bugs before shipping: (1) k0 predictions missing +mu_c (anomaly vs full value); (2) direct-copy map +1 on wrong side (would have replaced all 94k k0 rows with persistence). Both fixed; reproduced historical v4a.csv k0 block bit-exactly (corr 1.000000, max diff 0) to validate the toolchain.
- LB decomposition corrected: v2b 0.7137 = 0.665*masked(0.752)^2 + 0.335*k0(0.643)^2; masked CV->test gap ~+0.060. v12b 0.6954 => masked ~0.722. Projected v17a ≈ 0.689-0.691 (masked ~0.713, k0 ~0.640) — potential new best.

Stage Summary:
- DELIVERED: download/submission_v17a.csv (top3-ens masked phi{0.80,0.80,0.85}/lam{0.84,0.80,0.8}, bwd-only NO denoise, Dtil 0.70/0.45/0.073, k0 = 50/50 linear+LGB) and download/submission_v17b.csv (same k0; masked = 50/50 denoiser hedge at phi=0.80/lam=0.84). Both format-validated.
- v17a is the CV-best bet (projected ~0.690 vs v12b 0.6954); v17b isolates the denoiser question on the real LB.
- STRATEGIC FINDINGS: (1) honest-CV protocol must exclude t+1-unmasked rows — old protocol inflated all masked numbers by ~0.06 and distorted config selection; (2) PC-denoiser should be OFF (hurts honest rows); (3) phi optimum shifted UP to 0.80-0.85 on honest protocol; (4) direct-copy/target-identity route fully sealed by design (~0 rows); (5) k=0 LGB is a dead end (blend ≈ linear).
- LOST with the reset: exact v5-v16 recipes (v12b's masked config unknown). v13b/v12b +0.0021 public-row anomaly investigation cannot continue without the files — if user uploads v12b/v13b/v14 CSVs, resume bit-level analysis; meanwhile treat <0.003 LB deltas as noise.
- Next: submit v17a+v17b today; if v17a < 0.6954, iterate masked-row ideas on honest-CV (Dtil era-weighting, 2017-block handling); trustworthiness writeup (30% of score) still pending.

---
Task ID: 14
Agent: Super Z (main agent) + independent auditors A/C + gate reviewer (subagents)
Task: Process v17b LB feedback (0.704955918), institutionalize adversarial gate, full-hats audit, build v18

Work Log:
- LB post-mortem: LB^2 = 0.6652*masked^2 + 0.3348*k0^2 decomposition => v17b masked ~0.7355 (k0 0.64 assumed) vs v12b ~0.7216 => v17b WORSE by 0.014 masked. Honest-CV->LB gap drifted +0.060 (fwd era) -> +0.080 (bwd-heavy v17b): protocol realism problem.
- AUDIT A (data/protocol skeptic, subagent): REFUTED the "50%-masked months" belief - test has 6 fully-unmasked anchor months (Sep15/Jan16/Jun16/Dec16/Jul18/Nov18, gaps [4,5,6,19,4]) and 12 fully-masked months, month-instance level. Test = 2015-09..2018-12 (starts 1 month after train end). Masked-run geometry: Feb-Mar16 (31k, bwd 2-3), Jul-Sep16 (46k, bwd 2-4), 2017-block Jan-Jun17 (93k, bwd 12-17!), Dec18 (15.6k, fwd-only). Decisive mirror experiment: bwd-vs-fwd advantage P0 +0.046 -> M1 +0.022 -> M2 +0.010 -> test-mix extrapolation +0.003-0.005. P0 protocol overstated bwd value by ~0.040 = explains v17b regression. Built M1/M2 mirror protocols (hard-coded months; obs/cell 5.96/4.96 vs test 5.98). Direct-copy on test = 0 rows (all 280,961 checked). Leakage: sealed.
- AUDIT C (strategy red-team, subagent): CV 2SE=0.003 => all fine-config claims (phi grid, denoiser, v17a-vs-v17b) below noise. v12b/v13b +0.0021 anomaly ~90% process error (resolve via free Zindi-history file diff). Real prize fight = #10 (~0.671 public), not #1 (0.5596). Leader masked <=0.587 even at k0=0.50 (structural gap). Rules: public ~30%/private 70%; selection closes 13 Sep 21:59 (30 min after subs); DEFAULT PICKS = 2 BEST PUBLIC (could auto-select irreproducible v12b - must explicitly select final 2). Top-10 code review within 72h post-close; code must reproduce score. Trustworthiness rubric = 4 sections x <=100 words + innovation; missing only SHAP + CodeCarbon (~3h). Gate spec G1-G7 + failure catalogue P1-P9 written. E-rank: E3 spatial fast-state > E1 D-evolution > E2 era-Dtil > E4 lambda probe.
- AUDIT B (code/k0 skeptic): attempted twice, failed on service timeouts; its scope was absorbed by the gate reviewer (independent code re-verification + corruption probes) + A/C coverage.
- v18 BUILD (build_v18.py, build_v18b.py, build_v18_final.py): selection switched to M1+M2 (win on BOTH). Round-1: phi 0.85/cap8 small gain; era-Dhat tau=24/full wins; spatial smoothing of fast state BIG WIN (sigma sweep monotone to 2.0). Round-2 (spherical kernel): WINNER phi=0.80/lam=0.80/cap8/tau24-full/sigma2.0 => M1 0.6077, M2 0.6688 (vs v17b base 0.6319/0.6905: -0.024/-0.022, wins on BOTH, 27/27 month-level, dose-response monotone). Pre-smoothing probes (af_smooth, w_smooth) FAILED - output smoothing only. phi optimum dropped back to 0.80 once smoothing reduced noise (high-phi was compensating D-misfit; era-Dhat now does that properly). k0 smoothing +sigma1.0 = -0.0013 (below 0.002 bar) -> REVERTED to v17b k0 bit-exact (clean LB attribution: v18a vs v17b isolates masked block).
- GATE REVIEW (independent subagent, download/gate_review_v18.md): all numbers reproduced to <0.00005; CSVs reproduce end-to-end to 1.2e-07; corruption probes (targets->917, masked-TWS->garbage) change predictions by exactly 0.0 => zero leakage; mirror spec exact; k0 isolation 3e-07; bootstrap SE 0.0005 (z~45); 1-deg ACF of prediction field 0.982. SIGN-OFF: GO-WITH-RISKS (risks: M1 subset-of-M2 single-window selection, winner's-curse <=0.002; gap-model worst case 0.6969 does not beat v12b; G2/G5 paperwork completed after review).
- Manifest created (download/submissions_manifest.md) with md5s: v18a 0951a88e0cd11615a2ada99ae93f4df8, v18b 50356993363335643081b929db0eec69.

Stage Summary:
- DELIVERED: submission_v18a.csv (top3-ens masked: phi{.80,.85,.80}/lam{.80,.80,.84}, bwd-cap8, era-Dhat tau24/full, spatial smooth sigma2.0deg; k0 = v17b bit-exact) and submission_v18b.csv (single cfg0 phi.80/lam.80). Both format-PASS, gate GO-WITH-RISKS.
- Projected LB v18a: 0.679-0.694 (central 0.6905) vs v12b 0.6954 (best) and v17b 0.7050. Worst case 0.6969.
- Root cause of v17b regression established: P0 protocol geometry artifact (bwd overstated 0.040) + denoiser hedge. Fixed via mirror protocols.
- NEW CAPABILITY: spatial smoothing of the fast-state field (sigma 2 deg) = -0.021 mirror-CV; era-weighted D-hat (tau 24) = -0.006; both survive dose-response + both-protocol tests. This is C's E3 lever - confirmed on CV.
- NEXT (per audit C schedule): submit v18a (+v17a if not yet in) tomorrow; free user actions: paste Zindi submissions table + download v12b/v13b for the anomaly diff; E1 D-evolution regression (drift vs cumulative SPEI) if v18a <= 0.692; trustworthiness writeup Sep 2-5 (SHAP + CodeCarbon); freeze architecture Sep 8; explicitly select final 2 on Zindi Sep 12 (NEVER default - could pick irreproducible v12b).
