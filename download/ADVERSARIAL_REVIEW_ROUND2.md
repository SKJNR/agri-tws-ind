# ADVERSARIAL REVIEW — ROUND 2 (endgame, fresh-eyes skeptic)

**Date:** 2026-09-05 (live web data pulled tonight) · **Reviewer:** independent adversarial
subagent, fresh context, no stake in prior decisions · **Task ID:** 20-a
**Subject under review:** the current decision stack D1–D5, especially the NEW D3
(v22 splice) proposed this round.
**Evidence base:** all six context docs, all 50 submission CSVs, Test.csv re-census,
bit-level file-diff algebra (new, tonight), live Zindi pages + API + discussions feed
(new, tonight), v18/v21 build code read.

---

## 0. Executive summary

| # | Decision | Verdict | One-line reason |
|---|---|---|---|
| D1 | Final-2 = v21a + v12b (upgrade v13b, fallback v18a) | **UPHELD** (amended) | Slot-2 logic survives attack; alternates are measured-dominated; add the v23 branch below |
| D2 | Clean-lane public ceiling ~0.687; every lever measured/tombstoned | **UPHELD** | Found two more tombstones (per-cell H/R −0.0024, D-hat smoothing grid R4); no new legal modeling lever exists |
| D3 | v22 splice (v12b-masked + v21a-k0), band 0.685–0.690, fallback "outside band → keep v12b" | **OVERTURNED as specified — replaced by a REVISED splice program (GO)** | Rationale is false, band is unsupported, decision rule is mis-keyed; the *other* diagonal (v23) is the +EV play |
| D4 | Win paths = private reshuffle + DQ cascade + report | **REVISED (downgraded)** | Live LB tonight: field has densified (top-10 cutoff 0.6559, ~34 teams above our clean 0.6874); DQ cascade is a hope, not a plan |
| D5 | Manual selection before 13 Sep 21:59; never default | **UPHELD** | Rules text + API deadlines re-verified tonight; auto-pick = v20c+v20a = DQ trap is real |

**The single most important new finding:** the claim "v21a carries the LB-confirmed
best k0 block" was never established against the *right* opponent. a15 was only ever
raced against v18a's lin/lgb k0 (ΔMSE 0.0205, assumption-free). The v12-era k0
(era-inclusive Dcache carried by **v12b**) was never carried into the v18/v21 era and
was never raced against a15. Bit-level algebra tonight shows the two k0 blocks are
genuinely different files (0.0748 rms on k0 rows between v12b and v10b), and the
v12-era class decode puts the Dcache k0 at **~0.585–0.59 implied public RMSE vs a15's
~0.59–0.64 (point 0.617)** — i.e., the orphaned old k0 may be BETTER than the
"confirmed upgrade". If so, the never-built file **v23 = v18a/v21a masked block +
v12b k0 block** projects to **~0.676 public — a new clean best by ~0.011**. Both
splice diagonals (v22, v23) are now built, validated, md5-recorded locally.

---

## 1. Hat 1 — Competition strategist / game theorist

**Hardest question:** with MOHAR 39 subs @ 0.5596, Shankar 24 @ 0.5893, GIrum 76 @
0.6234, OverfitStorage 66 @ 0.6319, lode4 120 @ 0.6346 — what do the counts imply
about private robustness, and is the "DQ cascade" a strategy or a coping mechanism?

**Evidence checked (live leaderboard pulled tonight, saved to
`download/zindi_lb_live_2026-09-05.txt`):**

- Live top-10: MOHAR 0.5596 (39, last sub ~2 months ago) · Shankar 0.5893 (24) ·
  GIrum 0.6234 (76) · **Jisoo 0.6317 (35, rank 4, carried by prohibited v20c)** ·
  OverfitStorage 0.6319 (66) · lode4 0.6346 (120, active 6 h ago) · Emo HedgeHog
  0.6474 (74) · Ahsan_496 0.6532 (73) · awxlong 0.6534 (**5 subs**) · Ramjas 0.6559
  (**8 subs**). Ranks 11–48: MosCraciunXXX 0.6593 (97), H2-Oh 0.6611 (106), Real
  Fake 0.6655 (79), Gliding Moran 0.6671 (**7**), de-coder 0.6689 (29), then a dense
  pack 0.673–0.695 (supercwk 87, AIMS bois 42, Vladee 62, umut34 55, sdo 19,
  Mutombwa 43, algernon 18, kindi 47, tebogomap 9, Abhiraj-H 14, Berry 18, Diop221 39,
  urielnguefack 37, MasterShittu 19, Ombado-M 6, Donaldmunjanja 10, Z_Abby 22, BigZ 46,
  lionelfragniere 80, RubensSousa 57, nestorarriaga01 8, iam-dante 21, Naila_Abby 62,
  MonaR 15, Guelmbaye 26, eyadhjarray111 31, PatrykW 104, …).
- Our clean best v21a (0.687374) would be **~rank 35–36 today**. The Aug-29 snapshot
  (cutoff 0.6671, top-10 in reach) is stale: the cutoff is now 0.6559 and ~34 teams
  sit above our clean score. 1273 joined / 562 active (was 1070/414).
- Submission-count reading: the *efficient* top teams (MOHAR 39, Shankar 24, awxlong
  5, Ramjas 8, Gliding Moran 7) are NOT LB-overfit profiles — their public scores
  will mostly transfer; MOHAR's early stop + 0.5596 (below the measured external-lane
  floor 0.63 → they model the residual on top of GRACE/GDO state) is the profile of
  a decoded generator, not a lucky overfit. The *grinders* (lode4 120, MosCraciunXXX
  97, H2-Oh 106, PatrykW 104, Ahsan_496 73) are the private-reshuffle candidates —
  classic public-LB optimizers on a 43%-k0/h1-h2 public window.
- DQ cascade, honestly: two independent mechanisms exist tonight — (a) the
  GRACE/TWS-prohibited lane (rules text re-verified: "do not directly or indirectly
  include future GRACE/TWS information"), (b) the organizer's **Aug 16–19 discussion
  ruling** that raw coordinates / cell IDs / coordinate-derived encodings must not be
  model features (many tabular-LGBM teams will fail this). BUT: a competitor
  (Koleshjr, Aug 31 discussion) publicly asserts ~0.70 clean+ERA5 with no coordinate
  features, and the enforcement question ("must non-compliant submissions be
  replaced before close?") asked Aug 18 by Belal_Emad is **still unanswered**.
  Zindi enforcement on a €2k prize is historically inconsistent.
- Realistic probabilities after densification: **P(top-10 private) ≈ 10–20%** (docs
  said 15–25%), **P(top-3 prize) ≈ 3–8%**. The report (30%+20% phase-2) remains the
  highest-EV controllable — but it only pays in the top-10 branch.

**Verdict: PARTIALLY OVERTURNED.** The 2-file portfolio logic is sound; the
win-path *estimates* were stale and optimistic. The DQ cascade is a hope, not a
plan — treat it as free optionality, never as a reason to skip report work or to
relax compliance discipline.

---

## 2. Hat 2 — Validation scientist

**Hardest question:** is the 2013-15 single-window CV trustworthy for the 2017 hard
era, and what specifically could break the v21a masked block on private?

**Evidence (fresh Test.csv census tonight, with the decoded split):**

- Public = 2015-09, 2016-01/02/03/06/07/08 (109,222 rows; 46,939 k0 = 43.0%,
  62,283 masked = 57.0%). Private = 2016-09, 2016-12, 2017-01..06, 2018-07, 2018-11,
  2018-12 (171,739 rows; **47,109 k0 = 27.4%**, **124,630 masked = 72.6%**).
- **Three of the six anchor months are PRIVATE: 2016-12, 2018-07, 2018-11** (full
  15.6k-row anchors). The k0 fight on private includes two anchors 2.5–3 years
  beyond the train era (2018-07, 2018-11) — both k0 candidates (a15, Dcache) share
  this extrapolation risk.
- Private masked h-mix (from the anchor calendar): h1 = 2017-01 + 2018-12 (25.0%),
  h2 = 2017-02 (12.5%), h3 = 2016-09 + 2017-03 (25.0%), h4–h6 = 2017-04/05/06
  (37.5%). **Corrections to MASTER_HANDOFF §4:** "62.5% of private masked rows at
  h4+" is actually **h≥3** (62.4%); h≥4 is 37.5%. There is **no h7** anywhere (max
  h = 6). The "h4–h7 gap block" phrasing is wrong; the real hard block is h3–h6.
- **Structural exposure found:** for the 2017-04/05/06 rows the next anchor
  (2018-07) is 12–13 months beyond the target month > `bwd_max_gap=8` → the v18a
  backward pass is OFF for the entire 2017 block (forward-only from 2016-12), while
  2016-09 (gap 3) keeps it. Public masked rows are h1–h2 only, so **the public LB
  cannot see any of the machinery that matters most on private** — the M1/M2
  hard-era mirror (era-Dhat+smoothing win z≈45) is the only validation of it. That
  mirror is the correct instrument, single-window risk included; the honest gap
  model (+0.045) is carried. Nothing here is new *risk*, but it sharpens what the
  slot-2 hedge is for: a total failure of era-D transfer on the 2017 h3–h6 rows.
- One-way check on the mask: TWS_t is NaN on exactly the 186,913 masked rows (no
  value leaks in the file); no unmasked next-month rows exist (no lookup leak) —
  re-confirmed.

**Verdict: UPHELD, with the doc corrections above.** The validation machinery is
the strongest part of this project. The residual private risks (D-regime shift,
2018-anchor extrapolation, 2017 no-bwd-pass rows) are unhedgeable in the clean lane
at measured ceilings and are exactly what the diversified slot-2 insures.

---

## 3. Hat 3 — Forensic data scientist

**Hardest question:** test truth = 0.95×GDO(t) + residual (proven). Is there ANY
legal input that predicts the residual that hasn't been tried?

**What I checked tonight:**

- **Contemporaneous covs (covs at the target month) for masked rows:** already used.
  `build_v18_final.py:pass_kalman` runs `range(i+1, tm+1)` — the Kalman forward pass
  propagates through the target month and applies the cov observation *at the target
  month* when the next row exists (66.68% of masked rows, per
  `a14a_6_covariates.txt` §G). The biggest "missed lever" hypothesis I had is
  already implemented.
- **Per-cell Kalman H/R** (instead of global scalar H): was tried in the v8 era —
  `audit_15b_loo.log`: obs-dn+percell 0.7369 vs obs-dn only 0.7393 → **−0.0024,
  marginal**. Tombstone exists.
- **Spatial smoothing of D-hat** (the Aug-16 organizer ruling explicitly blesses
  neighbor-TWS aggregation): tested in `build_v18_grid2.log` R4 (af_smooth 1.0/2.0
  probes) — worse than the winning w_smooth configuration. Tombstone exists.
- **The residual's D component:** E1 (R 0.19–0.27), ENSO/ONI (R −0.005/+0.036),
  ERA5 (≤0.0005), GPCP (weaker than in-file SPEI) — all dead per the docs; I
  re-read the artifacts (`e1_d_evolution.txt` shows the Test-A/B numbers and the
  fixed Test-C crash). The D-evolution channel has no driver, in-file or external
  climate. The forum's open GLDAS-2.1-Noah question (Sep 4, unanswered) is moot for
  us: any GLDAS *TWS-like* output is "indirect TWS information" — prohibited — and
  its moisture fields duplicate in-file SOIL/SPEI.
- **Noise floor arithmetic** (0.456² + D-err 0.42² + fast 0.37² ≈ 0.722 masked
  floor): consistent with the implied public masked RMSE 0.736–0.767 of every clean
  file. The top-3's implied masked 0.51–0.57 remains explainable only with TWS
  information — the clean lane is genuinely closed for *modeling*.

**BUT — the combination miss (the real finding of this hat):** the forensic work
was so focused on *mechanisms* that a *portfolio combination* was missed. Bit-exact
file algebra tonight:

- v21a ≡ v18a on 100% of masked rows (max|d| = 0.000000) — re-verified.
- **v12b ≡ v10b on 100% of masked rows (max|d| = 0.000000)** — NEW: v12b's masked
  block is the v6c-lineage, and its k0 block is the era-inclusive Dcache (v12b vs
  v10b differ 0.0748 rms on k0 rows only). v12a ≡ v10b on k0 (bit-exact) — v12a was
  the era-Dtil-masked experiment, v12b the Dcache-k0 experiment.
- Assumption-free public-k0 relations (weights wk = 46,939/109,222 = 0.42968):
  `K_LGB² − K_a15² = 0.020455` (from v18a vs v21a, shared masked);
  `K_v8² − K_dc² = 0.015068` (from v10b vs v12b, shared masked);
  `M_v12a² − M_v12b² = 0.004344` (from v12a vs v10b, shared k0).
- The v12-era class decode (probe-consistent, val-prior-anchored): K_v8 ≈ 0.59–0.60,
  hence **K_dc ≈ 0.585–0.59**; the v18/v21-era implied **K_a15 ≈ 0.59–0.64**
  (point 0.617 if M18 = 0.736). The a15-vs-Dcache head-to-head is UNRESOLVED and
  the evidence leans toward **the orphaned Dcache being better**.
- Consequence: **v23 = v21a/v18a masked + v12b k0** (never built by the team)
  projects to 0.676–0.690; at the decode's central reading **~0.676 = new clean
  best by ~0.011 public AND ~+0.008 private** (k0 is 27.4% of private rows).
  The proposed v22 (v12b masked + a15 k0) is the *same measurement* seen from the
  losing side (v22² − v12b² = wk·(K_a15² − K_dc²) = −(v23² − v21a²)) with no
  upside on slot-1.

**Verdict: UPHELD on the modeling question (no missed legal lever); OVERTURNED on
"every clean combination has been tried" — the best k0 block was never raced against
the best masked block.**

---

## 4. Hat 4 — Rules lawyer (Zindi compliance)

**Hardest question:** is the splice compliant? Is using decoded public/private split
knowledge compliant? Any disclosure obligations?

**Evidence (rules text re-extracted tonight from the live page + API):**

- Rules unchanged since launch (API: `rules.updated_at = 2026-07-07`). Timeline
  API-confirmed: submissions close 2026-09-13 21:29, selection close 21:59, reveal
  22:15. "5 submissions per day, 200 overall", "choose 2 submissions … if you do
  not make a selection your 2 best public leaderboard submissions will be used",
  top-10 private → 72 h code+report, cheating → DQ + 6-month ban + 2000 points.
- External data rule (verbatim): covariates from Copernicus resources incl. EDO/GDO
  and CDS are allowed *provided* available at the stated prediction time, "do not
  directly or indirectly include future GRACE/TWS information, and are fully
  documented". Also: "For predictions at month t, only information available at or
  before t may be used; future observed values must not be used to fill or infer
  masked TWS values."
- **The splice (v22 or v23): COMPLIANT.** It is a row-class ensemble of two clean,
  competition-CSV-only, deterministic files. No external data, no future
  information, no target fitting. Standard stacking — document it in the 72 h
  package as such (one paragraph: which rows come from which model, why — k0 rows
  have observed state, masked rows need state-space forecasting).
- **Decoded split knowledge:** the split was decoded via LB probes (probe_PE/PK,
  predicted PE 1.1198 vs actual 1.1226, random split rejected 3σ). Nothing in the
  written rules prohibits probing; the covariance-decomposition trick
  (E[y·p] = (E[y²]+E[p²]−RMSE²)/2 from the all-zeros submission) is posted *publicly
  in the competition's own discussion* by eugenius — multiple competitors do this.
  Constructing files with row-class treatments (horizon gating, k0/masked lanes) is
  legitimate modeling; the h≥3 row class is 99.7% private by test *design*, not by
  our knowledge of truth. **Obligation: disclose the probing/decode methodology in
  the trustworthiness report** — concealment is the only way this becomes a
  liability. (Note: rules say "approximately 30%" public; the decoded window is
  38.9% of rows — the empirical probe evidence overrides the boilerplate, but say
  so in the report rather than silently assuming.)
- **NEW compliance surface (organizer ruling, Aug 16–19, discussion thread
  "Neighbouring cells' past TWS — permitted or not?"):** neighbor cells' TWS at
  months ≤ t is permitted, including spatial filtering/aggregation and
  neighborhood means; **raw coordinates, cell IDs, or coordinate-derived encodings
  must NOT be supplied as predictive features**. Checked our picks: v21a/v12b/splice
  k0 features = TWS_t + covs(t) + covs(t+1) anomalies only (no coordinates); the
  Kalman/spatial machinery uses coordinates solely to define neighborhoods — the
  explicitly permitted use. v20 additionally used "geo" features (moot — prohibited
  lane anyway). The 72 h package should state coordinate usage explicitly to avoid
  a false-positive DQ at review; several top competitors are exposed to this ruling
  (Belal_Emad's unanswered enforcement question).
- **Open forum questions that could matter:** GLDAS-2.1 (Sep 4, unanswered),
  ERA5-final-as-ERA5T-proxy (Aug 28, unanswered), ERA5-at-prediction-month (Aug 28,
  unanswered). None affect our final picks (competition CSVs only); all add DQ-cascade
  fuel for teams that used those products.

**Verdict: UPHELD — the splice and the strategy are compliant, with disclosure
obligations (probing methodology, coordinate usage, v20 audit-and-refusal) that the
report must honor.**

---

## 5. Hat 5 — Risk manager

**Hardest question:** the selection-window failure modes, the C2 file-identity
anomaly, and the 72 h code-review obligation — what breaks us?

- **C2 anomaly (v13b):** re-verified tonight — local v13b is bit-identical to v12b
  on ALL 109,222 decoded-public rows (rms 0.000000), with the era treatment exactly
  on the private h≥3 months (2016-09, 2017-03..06; 77,850 rows). The recorded score
  0.697421091 mathematically cannot come from this file. **New hypothesis to check
  in the history paste:** the recorded score may belong to v14 or v15 (both have
  UNKNOWN scores in the manifest) — the paste resolves this, the file download
  resolves the rest. Still not done by the user 8 days out — this is now the #1
  free-action overdue item. If the team wants the local v13b content, a fresh
  resubmission is deterministic and would *also* serve as a free bit-exactness test
  (it must score exactly 0.695357).
- **Selection window:** the auto-pick trap is real (rules text above). Jisoo sits at
  public rank 4 *carried by v20c* — high visibility, high temptation. The plan
  (select NOW as default, changeable until deadline; two-person check; alarms Sep 12
  + Sep 13 20:00) is right. **Addition:** submit any new files (v23/v22) EARLY —
  days before Sep 12 — so the selection screen is final and quiet during the last
  48 h; never leave the selection empty overnight.
- **72 h code-review obligation:** pre-stage by Sep 10. REPORT_DRAFT.md is from
  Aug 31; SHAP and CodeCarbon have not been run. With P(top-10 private) ≈ 10–20%
  and the report gating 50% of the final score in that branch, this is the
  highest-EV open item and it is *behind schedule*. If v23 lands a new clean best,
  the code package must also cover the v12-era k0 (Dcache) build — its script
  (submission_v12.py) exists; a G6-style bit-exact rerun for the chosen final files
  is the pass condition (open item #5, only v21a is G6-verified).
- **Budget realism:** "165 slots free" is misleading — the binding constraint is
  5/day with 9 days left (≈45 more usable). The required plan (v23, maybe v22,
  maybe v13b resubmit) needs ≤ 3. No budget risk.
- **New threat (from hat 1):** field densification — the win-path probability
  downgrade. Not actionable, but it recalibrates effort: report > any further
  score-chasing.

**Verdict: UPHELD on mechanics; the schedule risk has moved to the report (and the
user-side fog-closers are still open).**

---

## 6. Hat 6 — Portfolio quant

**Hardest question:** is v21a + v12b (or v21a + splice) the right 2-file portfolio
under private-score uncertainty? Would v13b / v18b / v10b / v6c be better?

**Measured tonight (private-masked disagreement vs v21a, public LB):**

| Candidate | Public LB | priv-masked rms vs v21a | corr | priv-k0 rms vs v21a | Assessment |
|---|---|---|---|---|---|
| v12b | 0.695357 | 0.1877 | 0.9624 | 0.2202 | **right slot-2**: best public carrier of the only genuinely different masked lineage (v6c-base) + a different k0 |
| v13b | (≡v12b on public) | 0.1764 | 0.9655 | 0.2202 | C2-conditional upgrade: v12b + private-only era treatment on 77,850 h≥3 rows (unvalidated on LB, mirror-era logic only) |
| v13 | 0.696325 | 0.1762 | 0.9653 | 0.2202 | same family, worse public, no edge |
| v17b | 0.704956 | 0.1930 | 0.9581 | 0.2053 | more diverse but 0.010 worse public; dominated in EV |
| v10b / v6c | 0.700/0.703 | 0.1877 / 0.1877 | 0.9624 | — | **dominated**: masked bit-identical to v12b's, k0 strictly worse (pre-Dcache), public worse |
| v18b | (unknown) | 0.0161 | 0.9998 | 0.2053 | **useless hedge**: masked is one component of the same top3 ensemble |
| v18a | 0.693739 | 0.0000 | 1.0000 | 0.196 | dominated (Round 1 verdict stands) |

So the D1 default (v21a + v12b) is the correct 2-file portfolio **among existing
files**. The splice question then reduces to pure k0 arithmetic on slot-2 (masked
lineages are IDENTICAL between "v21a+v12b" and "v21a+v22"; the claim that the
splice "maximizes lineage diversity on the masked block" is false — diversity is
unchanged; what changes is slot-2's k0 block):

- Portfolio A (v21a + v12b): private picks = (M18+K_a15) vs (M12+K_dc).
- Portfolio B (v21a + v22): private picks = (M18+K_a15) vs (M12+K_a15).
B beats A only in the joint event {M12 ≥ M18 (P≈0.15–0.3 per the mirror evidence) ∧
K_a15 > K_dc on private anchors}; the gain in that branch is ~+0.001–0.002 RMSE.
B loses to A in the event {M12 ≥ M18 ∧ K_dc > K_a15}: the k0 hedge (0.2202 rms of
real disagreement on 27.4% of private rows) is destroyed. **The v22 splice as
proposed is a near-EV-neutral bet that concentrates k0 risk on the *unproven* side
of an unresolved head-to-head — while its stated public band (0.685–0.690) is not
supported.** Honest projection: v22 = v12b + wk·(K_a15²−K_dc²); with the two
constraint sets above the plausible range is **0.693–0.706, most likely ~0.697** —
i.e. probably *worse than v12b itself* (which also makes the pre-registered
fallback rule "outside 0.685–0.690 → keep v12b" mis-keyed: it would discard a
splice that merely ties, and the band's edges are arbitrary).
- The correct decision variable is **relative**: adopt a splice iff it beats the
  file it would replace (v22 iff < 0.695357; v23 iff < 0.687374). And the correct
  ORDER is v23 first: it measures the same quantity
  (v23²−v21a² = −(v22²−v12b²) = wk·(K_dc²−K_a15²)) from the side that has upside —
  if the Dcache k0 is better, v23 is a new clean best (slot-1 upgrade, ~0.676
  public / ~+0.008 private, unconditionally better on both public and private k0
  rows), and the portfolio becomes {v23, v12b} (masked diverse, k0 concentrated on
  the *measured* winner — acceptable once measured). If a15 is better, v23 lands
  ≥ 0.6874, we keep v21a, and v22 remains an optional marginal bet with the
  corrected rule.

**Verdict: OVERTURNED-REVISED.** v21a+v12b stands; the v22-splice proposal as
written is killed; a v23-first splice program replaces it (GO, rules below).

---

## 7. Web research findings (all fresh, tonight)

1. **Deadlines/timeline (API `api.zindi.world/v1/competitions/...`):** submissions
   close 2026-09-13T21:29Z, selection close 21:59Z, reveal 22:15Z — matches the
   docs; page shows "9 days left". Rules last updated 2026-07-07 — **no rule
   changes**.
2. **Live leaderboard** (JS-rendered, captured to
   `download/zindi_lb_live_2026-09-05.txt`): see hat 1. Jisoo rank 4 = 0.631662555
   (v20c), 35 submissions, last sub 5 days ago. Field: 1273 joined / 562 active.
3. **Discussions feed (23 threads, via API)** — key items:
   - **Organizer ruling (Aug 16–19):** neighbor-TWS ≤ t permitted (incl. spatial
     filtering); **coordinates/cell IDs/coordinate-derived encodings prohibited as
     features**. (Directly relevant: our picks comply; many competitors likely
     don't.)
   - **"Is <0.70 Possible Without GRACE Data + Lat/Lon?" (Aug 31):** Koleshjr:
     clean + documented ERA5 lands just above 0.70 with no coordinate features —
     independent confirmation of our ~0.687 clean ceiling.
   - **"is NASA GLDAS-2.1 Noah Early Product allowed?" (Sep 4): unanswered.**
   - ERA5-timing questions (Aug 24–28): all unanswered.
   - Enforcement question for pre-ruling coordinate-feature submissions (Aug 18,
     Belal_Emad): **unanswered** — the DQ cascade depends on this.
   - Historical: the test-set structural leak (target = shifted TWS_t) was public
     since Jul 7; data was updated Jul 13 after the relaunch; benchmark saga
     (0.65947 old vs 0.8999 current) confirms the data was regenerated.
   - LB-probing with the covariance identity is public practice in this comp.
4. No discussion tab is reachable without login in the new Zindi UI; the API feed
   (titles + last comments) is the accessible surface — captured to
   `download/zindi_discussions_2026-09-05.json`.

---

## 8. Final verdicts on D1–D5

- **D1 — UPHELD (amended).** Final-2 default v21a + v12b; upgrade path v13b (C2-
  conditional); fallback v18a; NEVER v20a/b/c; never the auto-default. Amendment:
  if the v23 measurement (below) lands < 0.687374, slot-1 becomes v23 and slot-2
  stays v12b/v13b.
- **D2 — UPHELD.** Clean-lane modeling is at its measured ceiling; two additional
  tombstones verified tonight (per-cell H/R, D-hat smoothing). No new legal modeling
  lever. The one gap was a *combination* (see D3).
- **D3 — OVERTURNED as specified; GO on the revised splice program.**
  The v22-splice as proposed (rationale "maximizes masked diversity" — false;
  band 0.685–0.690 — unsupported, honest range 0.693–0.706; fallback keyed to an
  absolute band — wrong) is killed. Replaced by:
  1. **Submit v23 = v21a/v18a masked block + v12b k0 block** (built tonight:
     `download/submission_v23_splice.csv`, md5 `a6d996f75bf20a35045bfeabe3bb2a56`,
     validator PASS, v12b row-order/format, masked ≡ v21a and k0 ≡ v12b bit-exact).
     Pre-registered rules (record before upload): **if S23 ≤ 0.687374 → Dcache-k0
     wins the head-to-head; adopt v23 as slot-1 (new clean best); final-2 =
     v23 + v12b (or v13b if C2-cleared). If S23 > 0.687374 → keep v21a; a15
     confirmed; v22 becomes optional.** Interpretation is exact:
     S23² − 0.687374² = 0.42968·(K_dc² − K_a15²).
  2. **v22 only as an optional second move** (built: `download/
     submission_v22_splice.csv`, md5 `aca3c8f15d1c2c8d2e3e215bebbfd0ac`, PASS):
     submit only if v23 showed a15 > Dcache AND you accept k0 concentration;
     adopt iff S22 < 0.695357 (never the 0.685–0.690 band).
  3. Optional free local check first (no slot): run the v12-era k0 (Dcache) build on
     the a15 phaseA strict-CV protocol — if it confirms K_dc < 0.6166, expect v23 to
     win; either way the submission is the decisive measurement.
  4. Both splices are clean-lane row-class ensembles (hat 4) — document in the 72 h
     package; run a G6-style reproducibility note for whichever files get selected.
- **D4 — REVISED (downgraded).** Win paths unchanged in kind, downgraded in
  probability (field densification: cutoff 0.6559, ~34 teams above our clean best).
  The report is the highest-EV controllable and is behind schedule. The DQ cascade
  is optionality, not a plan.
- **D5 — UPHELD.** Select v21a + v12b TODAY as the default (changeable); two-person
  check + alarms for Sep 12/13; API-confirmed hard window 21:29→21:59 on Sep 13.

## 9. New opportunities / threats (net list)

**Opportunities**
1. **v23 diagonal splice** — potential new clean best ~0.676 public / +0.008 private
   if the Dcache-k0 is the real winner; 1 slot; bounded downside. (The round's find.)
2. The unanswered GLDAS/ERA5/enforcement forum threads — free intelligence; re-check
   before Sep 12 (any organizer answer changes the DQ-cascade odds).
3. The organizer's coordinate-feature ruling — an enforcement wave would clear
   several teams above us; our compliance is already airtight.

**Threats**
1. Field densification (win-path probabilities down; report is the difference-maker
   in the top-10 branch and it is late).
2. 2017 h4–h6 rows get no backward pass (gap 12–13 > 8) — the private hard block is
   forward-only; unhedgeable in-lane, insured only by slot-2.
3. C2 fog still open 8 days out (user-side, free, 10 minutes).
4. The k0 head-to-head cuts both ways: if a15 < Dcache, v21a (slot-1) itself is
   carrying a slightly suboptimal k0 — measured and fixed by v23, not by hope.
5. Minor doc corrections to carry into the report/handoff: h≥3 (not h≥4) = 62.4% of
   private masked; no h7 exists; 19 test months; ~45 (not 165) usable slots;
   "approximately 30%" public in rules vs decoded 38.9% (say so in the report).

## 10. Required next actions (owner / deadline)

1. USER (today, free): Zindi "My submissions" table paste + download submitted
   v12b/v13b for md5 bit-diff (closes C2; also tests the v14/v15-score hypothesis).
2. MAIN AGENT (Sep 6, 1 slot): pre-register the v23/v22 rules above in the
   submissions manifest, then submit **v23**; read S23 against 0.687374.
3. MAIN AGENT (Sep 6–10): finish REPORT_DRAFT (SHAP + CodeCarbon + disclosure
   paragraphs: probing methodology, coordinate usage, v20 audit-and-refusal);
   pre-stage the 72 h package incl. whichever final files are chosen + G6 notes.
4. USER (today): select v21a + v12b on Zindi as the standing default; alarms for
   Sep 12 + Sep 13 20:00; never leave the selection empty.
5. MAIN AGENT (Sep 12): final decision card refresh after S23/C2; re-check the
   forum threads; freeze.
