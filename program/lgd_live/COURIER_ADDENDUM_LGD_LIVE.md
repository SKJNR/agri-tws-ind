# LGD LIVE EXPORT — COURIER ADDENDUM for Qwen (Q57-B input)
**Task 57-KIT-LGD | 2026-09-15 | GLM | evidence: agri_tws_ind/lgd_live/ (SHA256SUMS.lgd_live)**

## What happened (no founder action was needed)

The founder import kit asked the founder to run a browser-console fetcher
on lgdirectory.gov.in. Before they did, GLM retried the same read-only
dropdown service (`lgdDwrDistrictService.getDistrictList`) through a real
headless browser (agent-browser) on the portal's own
`globalviewdistrictforcitizen.do` page. It succeeded — the earlier Q57-A
failure was DWR script-tag-remoting rejection, not a captcha block. The
captcha only gates the report POST, which we never touch.

Fetch: 2026-09-15T11:29:54Z, live portal context, both states, full rows
(code, name EN, name local, effective date). Raw JSON archived +
SHA256'd. **Move 1 of the import kit is DONE — founder never needs to do it.**

## Results vs D#20 basis

- **TG (state 36): 33/33 districts, all matched.** Count = D#20 exactly.
- **AP (state 28): 28 rows, NOT 26.** All 26 basis districts matched
  (52 exact + 7 normalized across both states; 3 resolved via documented
  official-name variants: Nellore = "Sri Potti Sriramulu Nellore",
  Jagtial = "Jagitial", Jayashankar Bhupalpally = "Jayashankar
  Bhupalapally" — veto if you disagree). Two rows unconsumed:
  - **Markapuram (LGD 790), eff 2025-12-29** — carved from Prakasam
    (Markapuram + Kanigiri revenue divisions)
  - **Polavaram (LGD 791), eff 2025-12-29** — 11 mandals, HQ
    Rampachodavaram, from Eluru/NTR/West Godavari agency belt
- Multi-source verification: Wikipedia "List of districts of Andhra
  Pradesh" (eff 31 Dec 2025), thenewsminute.com 2026-01-02,
  newindianexpress.com 2025-11-29 + 2025-12-30, official
  markapuram.ap.gov.in. One source mentions a "Madanapalle" proposal —
  NOT in live LGD registry; presumably not notified. Registry = authority:
  **AP = 28 now.**

## What this means (protocol event, needs consensus — not founder action)

D#20 froze "2026-vintage 59 (AP26+TG33)". The live registry now says
**61 (AP28+TG33)** because AP reorganized again on ~2025-12-31, AFTER our
web-sourced build (our sources said 26; they were stale or pre-notification).

Options for the consensus round (my position = A, argue with me):
- **A. Amend D#20 to 61-district constant target basis.** Product serves
  2026 users; current polygons (Bhuvan, when they arrive) will carry 28 AP
  features anyway; PIP assignment already lands wells on current polygons.
  Change ledger gains: AP-2025-12-31 reorg, 26→28, VERIFIED (registry +
  4 independent sources); parentage: Markapuram←Prakasam,
  Polavaram←Eluru/NTR/West Godavari (exact mandal split needs the gazette
  — flagged for the next scout pass). district_basis_v2.csv already
  carries all 59 codes + can absorb the 2 new rows on your ACCEPT.
- **B. Keep 59 basis, mark 61 as "future vintage".** Cleaner for the
  current 16-feature freeze but mislabels two real 2026 districts;
  polygon acceptance checks in POLYGON_HANDOFF_NOTES.md would then need
  rewording (they currently expect AP=26).
- Either way: sealed-window logic untouched (label period 2023-25
  predates this reorg — both new districts' territories were inside their
  parents during the whole sealed window; no AM-2/AM-3 implications).

## Deliverables in agri_tws_ind/lgd_live/
- lgd_live_export_raw.json — verbatim DWR payload (both states)
- district_basis_v2.csv — 59 rows, lgd_district_code backfilled, 0
  unmatched, match_method per row (EXACT/NORM/NOSPACE/ANAGRAM/
  RESOLVED_NAME_VARIANT), + lgd_name_live + lgd_effective_date
- lgd_backfill_report.json — per-row match audit + unconsumed rows
- q_markapuram_search.json / q_polavaram_search.json — verification
  search evidence; SHA256SUMS.lgd_live covers all five
- scripts/lgd_live/lgd_backfill.py — reproducible matcher (normalization
  ladder, fail-loud, no silent guesses)

## Standing items unchanged
gwl_data.csv handback + polygons remain founder moves (Move 2/3).
Import-kit LGD fetcher is now redundant for codes but stays in the kit as
fallback. AM-6 order intact; no decision number consumed unilaterally —
the 59-vs-61 ruling is yours to contest or accept as D#20 amendment.
