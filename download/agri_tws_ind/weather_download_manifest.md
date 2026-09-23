# weather_download_manifest.md — Day-1 foundation download

- API: https://archive-api.open-meteo.com/v1/archive (Open-Meteo Archive; ERA5/ERA5-Land reanalysis)
- window: 2014-01-01 .. 2026-09-10 | timezone: Asia/Kolkata
- variables: precipitation_sum, temperature_2m_max, temperature_2m_min, et0_fao_evapotranspiration, soil_moisture_0_to_7cm_mean
- fetched: 2026-09-15T04:52:51 | districts: 59
- trailing-day NaNs = ERA5T preliminary availability (normal);
  trailing consolidated months finalize ~2-3 months later (LATENCY_TABLE L5/L13 rule).

| state | district | coord_source | elev_m | rows | gaps | NaN(precip/tmax/tmin/et0/soil) | first | last |
|---|---|---|---|---|---|---|---|---|
| AP | Anantapur | QWEN_PROVIDED | 348.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Kurnool | QWEN_PROVIDED | 279.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Guntur | QWEN_PROVIDED | 38.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Krishna | QWEN_PROVIDED | 23.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Chittoor | QWEN_PROVIDED | 314.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Nizamabad | QWEN_PROVIDED | 388.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Karimnagar | QWEN_PROVIDED | 271.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Warangal | QWEN_PROVIDED | 267.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Khammam | QWEN_PROVIDED | 124.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Mahabubnagar | QWEN_PROVIDED | 485.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Alluri Sitharama Raju | GLM_APPROX | 928.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Anakapalli | GLM_APPROX | 45.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Annamayya | GLM_APPROX | 371.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Bapatla | GLM_APPROX | 9.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Dr B R Ambedkar Konaseema | GLM_APPROX | 6.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | East Godavari | GLM_APPROX | 29.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Eluru | GLM_APPROX | 20.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Kakinada | GLM_APPROX | 6.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Nandyal | GLM_APPROX | 207.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | NTR | GLM_APPROX | 121.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Palnadu | GLM_APPROX | 66.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Parvathipuram Manyam | GLM_APPROX | 119.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Prakasam | GLM_APPROX | 12.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Sri Potti Sriramulu Nellore | GLM_APPROX | 16.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Srikakulam | GLM_APPROX | 17.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Sri Sathya Sai | GLM_APPROX | 355.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Tirupati | GLM_APPROX | 156.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Visakhapatnam | GLM_APPROX | 26.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | Vizianagaram | GLM_APPROX | 63.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | West Godavari | GLM_APPROX | 6.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| AP | YSR Kadapa | GLM_APPROX | 131.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Adilabad | GLM_APPROX | 270.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Bhadradri Kothagudem | GLM_APPROX | 76.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Hanumakonda | GLM_APPROX | 255.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Hyderabad | GLM_APPROX | 502.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Jagtial | GLM_APPROX | 275.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Jangaon | GLM_APPROX | 362.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Jayashankar Bhupalpally | GLM_APPROX | 166.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Jogulamba Gadwal | GLM_APPROX | 326.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Kamareddy | GLM_APPROX | 523.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Kumuram Bheem Asifabad | GLM_APPROX | 214.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Mahabubabad | GLM_APPROX | 203.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Mancherial | GLM_APPROX | 143.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Medak | GLM_APPROX | 471.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Medchal Malikajgiri | GLM_APPROX | 576.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Mulugu | GLM_APPROX | 209.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Nagarkurnool | GLM_APPROX | 446.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Nalgonda | GLM_APPROX | 230.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Narayanpet | GLM_APPROX | 439.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Nirmal | GLM_APPROX | 332.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Peddapalli | GLM_APPROX | 232.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Rajanna Sircilla | GLM_APPROX | 321.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Rangareddy | GLM_APPROX | 533.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Sangareddy | GLM_APPROX | 521.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Siddipet | GLM_APPROX | 481.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Suryapet | GLM_APPROX | 178.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Vikarabad | GLM_APPROX | 639.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Wanaparthy | GLM_APPROX | 394.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |
| TS | Yadadri Bhuvanagiri | GLM_APPROX | 428.0 | 4636 | 0 | 0/0/0/0/0 | 2014-01-01 | 2026-09-10 |

coord_source: QWEN_PROVIDED = Day-1 task coordinates (approx centroids);
GLM_APPROX = approximate centroids pending LGD/GADM verification (loader task).

Sprint ruling (DECISION_LOG, 2026-09-15): this CSV is loader work, not an
evaluation (T7 unaffected). Test years 2023-25 stay untouched until the R7-SS5
order completes: D1.1-empirical -> regime-map freeze -> baseline ladder.
First LGBM run = SMOKE TEST on train<=2019 + val 2020-22 only, logged first.