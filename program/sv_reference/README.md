# sv_reference/ — SoulVision repo snapshot (reference material)

Task-55 directed: "clone the repo for reference; log everything." This
is the codeload tarball of github.com/SoulVisionCreations/
gwl-forecasting-advisory @ main, fetched 2026-09-15 (reachable from
sandbox; AIKosh portal is NOT). Originally committed as b8a09f2; lost to
the 11th reset; re-fetched and re-committed — SHA256 verified IDENTICAL
(c2a46a702a9e4a2c7d6ea694c221a962a7805211c00f73393fb9c0945cdc62e6),
proving the upstream repo is unchanged.

Key facts read from this snapshot (data/README.md + docs/DATA_SOURCES.md):
- gwl_data.csv is NOT in the repo. Sole distribution: AIKosh dataset
  page (aikosh.indiaai.gov.in/web/datasets/details/ground_water_level
  _all.html) -> "Download Dataset" -> ground_water_level_all_v1.zip
  -> unzips to a single gwl_data.csv, ~760 MB, 25 cols, ~3.3M rows.
- Model weights best_model.pt ~1.3 GB — NOT fetched (deferred per D#22
  pre-registration; AM-6 order intact).
- Satellite composites not distributed (GEE regeneration only).

Implication for the critical path: the one-time download is a founder
move by necessity (AIKosh blocks this sandbox). Professional handling
on arrival is pre-built: gwl_handback_check.py receipt -> ARRIVAL=COMMIT
-> D#21 physical split -> Parquet working copies -> Release-asset home.
