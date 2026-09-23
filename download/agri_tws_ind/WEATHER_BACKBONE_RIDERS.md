# Weather Backbone Riders — WB-R1/R2/R3 (Addendum 9)

Built 2026-09-23T10:35:47.826939+00:00 | open window <=2022 (D#21) | rules pre-stated in DECISION_LOG Addendum 9 BEFORE this run

## WB-R1 — season x regime bias decomposition (OM - IMD, degC)

| season | regime | var | n | bias | RMSE | trip |
|---|---|---|---|---|---|---|
| DJF | R1418 | tmax | 26,609 | -1.36 | 1.90 | YES |
| DJF | R1418 | tmin | 26,609 | +1.08 | 1.83 | YES |
| DJF | R1922 | tmax | 21,299 | -0.94 | 1.72 |  |
| DJF | R1922 | tmin | 21,299 | +0.53 | 1.45 |  |
| MAM | R1418 | tmax | 27,140 | -1.42 | 2.20 | YES |
| MAM | R1418 | tmin | 27,140 | +0.73 | 1.57 |  |
| MAM | R1922 | tmax | 21,712 | -0.91 | 2.06 |  |
| MAM | R1922 | tmin | 21,712 | +0.49 | 1.45 |  |
| JJAS | R1418 | tmax | 35,990 | -1.74 | 2.38 | YES |
| JJAS | R1418 | tmin | 35,990 | +0.73 | 1.35 |  |
| JJAS | R1922 | tmax | 28,792 | -0.93 | 1.88 |  |
| JJAS | R1922 | tmin | 28,792 | +0.59 | 1.25 |  |
| ON | R1418 | tmax | 17,995 | -1.71 | 2.25 | YES |
| ON | R1418 | tmin | 17,995 | +0.78 | 1.61 |  |
| ON | R1922 | tmax | 14,396 | -1.10 | 1.73 | YES |
| ON | R1922 | tmin | 14,396 | +0.40 | 1.30 |  |

Aggregate trips (>1.0C): **6/16** | district-season trips: **252/472**

## WB-R2 — REPRESENTATIVENESS_LIMITED + low-relief verdicts

- Named terrain: Alluri Sitharama Raju, Parvathipuram Manyam, Mulugu, Bhadradri Kothagudem
- Empirical (|tmax bias| >= 2.0C): Alluri Sitharama Raju, Dr B R Ambedkar Konaseema, Hyderabad, Jangaon, Kamareddy, Mahabubnagar, Medak, Medchal Malikajgiri, Nagarkurnool, Narayanpet, Nirmal, Rangareddy, Sangareddy, Siddipet, Vikarabad
- TAGGED (18): Alluri Sitharama Raju, Bhadradri Kothagudem, Dr B R Ambedkar Konaseema, Hyderabad, Jangaon, Kamareddy, Mahabubnagar, Medak, Medchal Malikajgiri, Mulugu, Nagarkurnool, Narayanpet, Nirmal, Parvathipuram Manyam, Rangareddy, Sangareddy, Siddipet, Vikarabad
- Low-relief subset (41 districts)

| Check | n (full) | bias (full) | r (full) | n (low-relief) | bias (low) | r (low) |
|---|---|---|---|---|---|---|
| tmax_overall | 193,933 | -1.29 | 0.913 | 134,767 | -0.92 | 0.928 |
| tmin_overall | 193,933 | +0.69 | 0.934 | 134,767 | +0.88 | 0.942 |
| tmax_monsoon | 64,782 | -1.38 | 0.853 | 45,018 | -0.97 | 0.871 |
| tmax_dry | 129,151 | -1.25 | 0.929 | 89,749 | -0.90 | 0.943 |
| et0_method | 193,933 | +0.25 | 0.882 | 134,767 | +0.25 | 0.883 |
| et0_source | 193,933 | -0.52 | 0.906 | 134,767 | -0.45 | 0.910 |
| et0_worst | 193,933 | -0.27 | 0.832 | 134,767 | -0.20 | 0.837 |

Verdict flips on low-relief subset: **NONE — QA status holds**

## WB-R3 — ET0 divergence log (primary OM-PM vs pyet artifact HS-OM)

- District-months (open window): 6,372
- QA flags (monthly MAD > 15%): **858**
- Districts with any flag: 59

| district | flagged months | mean MAD% | max MAD% |
|---|---|---|---|
| Adilabad | 7 | 8.2 | 23.0 |
| Alluri Sitharama Raju | 12 | 8.4 | 25.2 |
| Anakapalli | 31 | 11.7 | 24.9 |
| Anantapur | 40 | 12.0 | 22.2 |
| Annamayya | 12 | 8.5 | 19.7 |
| Bapatla | 24 | 9.4 | 24.7 |
| Bhadradri Kothagudem | 8 | 6.0 | 31.7 |
| Chittoor | 6 | 5.0 | 22.2 |
| Dr B R Ambedkar Konaseema | 27 | 9.9 | 30.9 |
| East Godavari | 3 | 5.4 | 21.3 |
| Eluru | 3 | 4.9 | 20.7 |
| Guntur | 2 | 4.3 | 17.7 |
| Hanumakonda | 9 | 7.4 | 20.3 |
| Hyderabad | 14 | 7.7 | 18.1 |
| Jagtial | 10 | 8.3 | 21.6 |
| Jangaon | 8 | 7.6 | 20.0 |
| Jayashankar Bhupalpally | 5 | 5.9 | 24.3 |
| Jogulamba Gadwal | 4 | 7.8 | 16.7 |
| Kakinada | 44 | 13.4 | 33.1 |
| Kamareddy | 25 | 9.8 | 22.7 |
| Karimnagar | 9 | 7.4 | 22.2 |
| Khammam | 3 | 4.7 | 22.5 |
| Krishna | 1 | 3.9 | 17.8 |
| Kumuram Bheem Asifabad | 8 | 6.7 | 27.9 |
| Kurnool | 10 | 8.4 | 18.9 |
| Mahabubabad | 7 | 6.5 | 24.0 |
| Mahabubnagar | 25 | 9.7 | 22.9 |
| Mancherial | 10 | 6.3 | 22.8 |
| Medak | 14 | 8.2 | 21.9 |
| Medchal Malikajgiri | 25 | 9.5 | 19.8 |
| Mulugu | 4 | 6.4 | 23.7 |
| NTR | 2 | 4.0 | 18.6 |
| Nagarkurnool | 9 | 7.9 | 17.1 |
| Nalgonda | 3 | 5.7 | 18.9 |
| Nandyal | 4 | 5.6 | 22.1 |
| Narayanpet | 28 | 10.6 | 21.4 |
| Nirmal | 19 | 9.1 | 19.0 |
| Nizamabad | 25 | 10.0 | 20.9 |
| Palnadu | 4 | 4.4 | 19.2 |
| Parvathipuram Manyam | 47 | 13.8 | 41.3 |
| Peddapalli | 13 | 8.9 | 21.2 |
| Prakasam | 14 | 8.1 | 23.8 |
| Rajanna Sircilla | 10 | 7.0 | 21.2 |
| Rangareddy | 13 | 7.5 | 18.2 |
| Sangareddy | 14 | 8.6 | 18.3 |
| Siddipet | 39 | 11.5 | 22.7 |
| Sri Potti Sriramulu Nellore | 10 | 6.9 | 25.1 |
| Sri Sathya Sai | 11 | 9.5 | 19.4 |
| Srikakulam | 31 | 10.4 | 26.7 |
| Suryapet | 4 | 5.9 | 18.9 |
| Tirupati | 2 | 4.0 | 22.0 |
| Vikarabad | 38 | 12.0 | 25.8 |
| Visakhapatnam | 45 | 13.7 | 28.4 |
| Vizianagaram | 11 | 7.1 | 21.9 |
| Wanaparthy | 16 | 9.3 | 21.2 |
| Warangal | 10 | 7.5 | 20.0 |
| West Godavari | 3 | 6.5 | 18.5 |
| YSR Kadapa | 12 | 8.1 | 21.5 |
| Yadadri Bhuvanagiri | 11 | 8.1 | 19.6 |

Primary ET0 et0_pm_api remains PINNED (OM hash-frozen; swap = AM-5 amendment). et0_hs_om_bc carried as diagnostic lane in weather_riders_monthly.parquet.

AM-6 untouched (QA/covariate engineering, no model runs); D#22 weights ban intact; no sealed contact.
