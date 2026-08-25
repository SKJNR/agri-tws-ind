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
