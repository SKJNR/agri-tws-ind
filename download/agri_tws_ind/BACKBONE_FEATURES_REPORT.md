# Backbone Features (Task-54) — build + plumbing smoke

Built 2026-09-23T10:45:11.382718+00:00 | spec pre-registered at c9208d0 (Addendum 10) | 59-execution basis

## Layer separation

- (i) covariates: BUILT — 14 core features (water balance in monthly SUMS; ET0 primary = et0_pm_api, WB-R3 pin; extremes labeled ERA5-Land covariates, tails biased low)
- (ii) PR-8 transform: generic module, frozen params, train-only fits
- (iii) target ingestion: STUB (D1.1 pending — Qwen loaders + founder pilot-12 + Task-55 WRIS spot-check gates unchanged)
- regime-gated 15/16: regime features NOT materialized: regime_map.csv absent (T11 map not frozen); interactions gated OFF

## Plumbing smoke — NON-EVIDENCE (pseudo-target = next-month soil-moisture tercile)

| check | value |
|---|---|
| smoke rows (train+val) | 6,372 |
| train rows | 4,248 |
| val rows | 2,124 |
| test rows EXCLUDED (target in 2023+) | 2,655 |
| NaN features in val | 0 |
| PR-8 0.15σ stationarity-flagged districts | 41/59 |
| val majority-class baseline acc (plumbing only) | 0.3079 |

NO skill claims. The LGBM Day-2 smoke stays gated behind the baseline ladder (AM-6). D1.1 CGWB round-month verification remains the FIRST logged evaluation.

AM-6: registered backbone-build step; no model runs in the evidence sense. D#22 weights ban intact. No sealed contact.
