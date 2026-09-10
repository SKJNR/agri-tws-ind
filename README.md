# One Step Ahead of Drought — TWS Forecasting (Zindi)

**Final-stage competition workspace** for the [Zindi challenge *One Step Ahead of
Drought: Forecasting Global Water Storage*](https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge)
(ITU / AI for Good, deadline 13 Sep 2026). One-month-ahead global Total Water Storage
(TWS) forecasting on a 1° grid; metric = RMSE on a 30/70 public/private row split.

- **Clean-lane best (public LB): 0.687374** — `submission_v21a.csv`, fully reproducible
  from the competition CSVs alone (md5-verified end-to-end rebuild).
- **Selected pair for the private round (per adversarial review R1):** `submission_v21a.csv`
  + `submission_v12b.csv` (slot-2 = private-era lineage hedge; see
  `download/ADVERSARIAL_REVIEW_ROUND1.md`).
- The project reverse-engineered the data generator (train = GDO archive bit-exact;
  test = 0.95×GDO + synthetic residual), which settled both the modeling ceiling and a
  rules-compliance audit that **rejected** a prohibited external-GRACE lane scoring 0.6317.

## Results lineage (public LB, lower = better)

| Version | Public RMSE | Note |
|---|---|---|
| v1 | 0.8059 | LGBM baseline |
| v2b | 0.7137 | two-component Kalman + D-tilde |
| v6c | 0.7030 | W-pool smoothing stack |
| v10b | 0.7000 | k0 two-comp blend |
| v12b | 0.6954 | k0 era-inclusive Dcache *(slot-2 pick)* |
| v18a | 0.6937 | top3-ens + era-Dhat + 2.0° smoothing |
| **v21a** | **0.6874** | v18a masked block + a15 k0 blend *(slot-1 pick, G6 bit-exact)* |
| ~~v20c~~ | ~~0.6317~~ | ⛔ prohibited external-GRACE lane — audited, refused, never selected |

Full table incl. every variant: `download/MASTER_HANDOFF.md` §2 · every shipped file
with md5 + pre-registered prediction: `download/submissions_manifest.md`.

## Repository layout

```
├── README.md                     # this file
├── MASTER_HANDOFF.md             # single source of truth (state, rules, audit verdicts)
├── worklog.md                    # full task-by-task history (Tasks 1-17)
├── data/                         # competition CSVs (NOT in the GitHub branch — see below)
├── scripts/                      # all generation + analysis + audit + gate code
│   ├── submission_v1…v15.py      #   submission builders (lineage)
│   ├── build_v17/v18/v19/v20*.py #   v17-v20 builders (+ .log run records)
│   ├── build_v21_phaseA/B/C.py   #   final clean build (phaseC = the shipped v21a/b)
│   ├── a14*/a15*/tb*.py          #   experiment series (spectra, k0 lane, validation bands)
│   ├── gate_*.py, audit*.py      #   G1-G5 submission gates + legitimacy audits
│   ├── cv_lb_correlation.py      #   the honest CV protocol (Spearman ρ=1.0 vs LB)
│   ├── validate_submission.py    #   format validator (run before any upload)
│   └── e1_d_evolution.py         #   E1 lever test (measured dead, kept for the record)
├── download/                     # all outputs: 65 submission CSVs + audit reports + diagnostics
│   ├── ADVERSARIAL_REVIEW_ROUND1.md   # skeptic review that revised the final-2 pick
│   ├── METHODOLOGY_AUDIT.md           # "all hats" coverage matrix
│   ├── gate_review_v18.md, auditA/B/C_report.md
│   ├── REPORT_DRAFT.md           # trustworthiness report draft (30% of final score)
│   └── submissions_manifest.md   # THE provenance ledger (md5 per shipped file)
└── upload/                       # user-provided handoffs/photos (provenance)
```

## Reproducing the selected submission

The scripts assume this exact directory layout (clone the repo to
`/home/z/my-project`, or adjust the `DATA`/`DL` constants at each script head).

1. Place the competition files in `data/`: `Train (1).csv`, `Test (2).csv`,
   `SampleSubmission (4).csv` (downloaded from the Zindi data page).
2. `python scripts/build_v18_final.py` → produces `download/submission_v18a.csv`
   (masked block; verified end-to-end to 1.2e-07 in `gate_review_v18.md`).
3. `python scripts/build_v21_phaseC.py` → reproduces `download/submission_v21a.csv`
   **bit-exact** (md5 `6b6e3e41c25317a68089e6b9ca707c05`, re-verified 2026-08-31).
4. `python scripts/submission_v12.py` → the v12b lineage (slot-2).
5. `python scripts/validate_submission.py <file>` → format gate (280,961 rows,
   ID-order exact, no NaN).

Deterministic: no LightGBM in the final picks; fixed seeds everywhere; every number
in the docs was reproduced by an independent gate pass.

## Data & external products — provenance

- **Clean lane (v1–v18a, v21):** competition CSVs only. ERA5/GPCP were downloaded and
  tested — **measured no marginal value** (`download/era5_value_test.txt`) — and are not
  used in any selected file.
- **Prohibited lane (v19/v20, NOT selected):** external GRACE TWS products
  (GDO/GravIS/COST-G/CSR) used to fill the masked TWS state + a public-LB-calibrated
  truth proxy. Line-by-line audit: `scripts/build_v20.py` + `MASTER_HANDOFF.md` §6.
  Kept in the repo **as evidence of the audit**, never as a candidate.
- Competition data is CC-BY-SA 4.0 (per competition rules); the raw CSVs are excluded
  from the GitHub branch only because of file-size limits — md5s and download
  instructions are in `data/README.md`.

## Pushing to GitHub

The `main` branch is the full durable workspace (includes `data/`, ~900 MB — exceeds
GitHub's 100 MB per-file limit). To publish, push the **`github-release`** branch
(orphan, same content minus `data/*.csv`, every file < 100 MB):

```bash
git remote add origin git@github.com:<you>/tws-drought-forecasting.git
git push github-release   # then on GitHub: Settings → Branches → default = github-release
```

## Governing documents (read order for any new session)

1. `MASTER_HANDOFF.md` → 2. `worklog.md` → 3. `download/submissions_manifest.md` →
4. `download/ADVERSARIAL_REVIEW_ROUND1.md` → 5. `upload/TECHNICAL_HANDOFF.md`
