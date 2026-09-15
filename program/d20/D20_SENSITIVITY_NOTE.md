# D#20 — Old-basis sensitivity note on TRAIN-era labels (AGRI-TWS-IND)

**Status:** NOTE (analysis) · **Decision refs:** D#20 (Task 56), courier Task
57 B.1 · **Author:** GLM, 2026-09-15 · **Blocking:** none (note is a
deliverable); the quantitative sensitivity RUN itself waits for gwl_data.csv.

## Why this note exists

The district basis is label-defining: every GWL reading aggregates to a
district via the 2026-vintage polygons, but the readings themselves were
taken under three different district geographies inside our open window:

| Era | Geography active | Reorgs inside era |
|---|---|---|
| ≤ 2016-10-10 | TG 10 districts, AP 13 | — |
| 2016-10-11 .. 2019-02-16 | TG 31, AP 13 | TG-2016 |
| 2019-02-17 .. 2022-04-03 | TG 33, AP 13 | TG-2019 |
| 2022-04-04 .. 2022-12-31 | TG 33, AP 26 | AP-2022 (inside val years) |
| 2023-01-01+ (SEALED) | TG 33, AP 26 | — |

None of this changes WHERE a well is — wells are points; the 2026-basis
assignment is time-invariant for any well that never moved. What changes is
which OFFICIAL district name a label would have carried at measurement time,
and therefore: (a) any vintage-keyed join against external district tables
(CCE yields, GWA, Livestock Census), and (b) any categorical feature or
stratification keyed to district-as-of-label-time.

## The ruling that keeps this safe

D#20 froze the target basis as CONSTANT 2026-vintage (point-in-polygon on
current polygons). All model district keys are 2026-basis. Vintage names
enter only through:

1. **External table joins** — vintage-keyed sources must be crosswalked to
   the 2026 basis via `change_ledger.json` parentage BEFORE joining; the
   crosswalk is applied to the SOURCE table (moving old rows forward into
   the 2026 basis), never to our labels.
2. **TRAIN-era label reconstruction checks (T-D20-3)** — a 2015 reading
   labeled "Mahabubnagar (2015 vintage)" must land in one of the four
   2026-basis children {Mahabubnagar, Wanaparthy, Nagarkurnool,
   Jogulamba Gadwal} — the polygon test enforces this automatically.

## Quantitative sensitivity check (pre-registered, runs when data lands)

On the ≤2022 (open) rows: re-aggregate district-level monthly statistics
under the vintage geography active at reading time (10/31/33/26-district
eras as above) and compare to the frozen 2026-basis aggregation:

- metric = max |district-month mean dGWL(vintage) − district-month mean
  dGWL(2026)| over the overlapping keys, plus count of keys that exist
  under one basis but not the other;
- expected magnitude: ZERO for well-level features (wells don't move);
  nonzero only for district-aggregated label statistics, driven by
  composition changes at reorg dates (TG-2016 is the big one: 10 → 31);
- reporting rule: frozen-report format, mismatch log only — no re-freeze of
  the basis (the basis is D#20-frozen; this check documents drift, it does
  not reopen the decision).

## Known trap (flagged for the record)

The AP-2022 reorg sits on 2022-04-04 — INSIDE the val window's final year
(open era). District-aggregated val labels for Jan–Mar 2022 use the same
2026-basis polygons as everything else, so no leakage follows; but any
external AP table with 2022 rows (e.g. CCE season 2021-22 published under
the 13-district frame) must be crosswalked forward before joining. The
crosswalk direction is always source → 2026 basis.

## Blocked-on (restated)

- gwl_data.csv (well coordinates + reading dates) — founder re-upload or
  approved AIKosh download (Task-55 provenance condition applies on arrival).
- LGD district codes (portal captcha-gated today) — founder one-time export
  or PENDING markers stand until polygon binding.
- LGD/Bhuvan-grade polygons — not in sandbox; GADM banned (D#20);
  OSM = community-grade fallback requiring a logged amendment.
