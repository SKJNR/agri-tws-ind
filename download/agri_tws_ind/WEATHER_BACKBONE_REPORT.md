# Weather Backbone Report — pyet ET0 + IMD 1-deg consistency

Built 2026-09-23T10:31:29.364179+00:00 | OM rows 273,524 | districts 59 | IMD 2014-2024

## Headline checks (open window <=2022, D#21)

| Check | n | bias | RMSE | Pearson r |
|---|---|---|---|---|
| OM tmax vs IMD tmax (daily, open window) | 193,933 | -1.29 | 2.06 | 0.913 |
| OM tmin vs IMD tmin (daily, open window) | 193,933 | +0.69 | 1.49 | 0.934 |
| OM vs IMD tmax MONSOON | 64,782 | -1.38 | 2.17 | 0.853 |
| OM vs IMD tmax DRY | 129,151 | -1.25 | 2.00 | 0.929 |
| OM-PM vs OM-HS (method check) | 193,933 | +0.25 | 0.76 | 0.882 |
| OM-HS vs IMD-HS (source check) | 193,933 | -0.52 | 0.72 | 0.906 |
| OM-PM vs IMD-HS (worst case) | 193,933 | -0.27 | 0.87 | 0.832 |

## Verdict

- Temperature: OM(ERA5) vs IMD 1-deg daily tmax bias -1.29 C, r 0.913 — CONSISTENT
- ET0 method check: PM(API) vs HS(pyet, OM temps) bias +0.25 mm/day, r 0.882 — CONSISTENT
- ET0 source check: HS(OM) vs HS(IMD) bias -0.52 mm/day, r 0.906 — CONSISTENT
- Worst tmax-bias district: Alluri Sitharama Raju (-4.89 C)

## Notes
- IMD 1-deg grid representativeness: nearest cell to district centroid; coastal/Ghats districts inherit grid-cell bias.
- Open-Meteo = ERA5/ERA5-Land reanalysis (CC-BY-4.0); IMD gridded = station-based 1-deg official product (IMD Pune, free download, academic use; raw .GRD not committed).
- Statistics on open window (<=2022) only per D#21; monthly panel carries full exogenous range with in_open_window flag.
- AM-6 untouched: covariate engineering only, no model runs; D#22 weights ban intact.
