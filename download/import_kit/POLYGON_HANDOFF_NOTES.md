# POLYGON HANDOFF NOTES — AGRI-TWS-IND (Task 57-KIT, 2026-09-15, GLM)

Unblocks: `d20_assign_wells.py` (PIP well→district assignment, selftest
already PASS 5/5 but waiting on real polygons). Blocks nothing else.

## Why founder-only

- Sandbox cannot reach NRSC/Bhuvan portals (live probe 2026-09-15: WRIS
  DNS-dead, IMD refused, IndiaAI 403 — same egress class).
- **GADM is BANNED by D#20** (license + non-official derivation risk). Do
  not substitute it "just to test" — that would need a logged amendment.
- OSM / Datameet boundaries = community-grade fallback ONLY, and require a
  logged amendment before use. Official-source-first is the D#20 ruling.

## Preferred source: Bhuvan Bhoonidhi (ISRO/NRSC)

1. Go to **https://bhoonidhi.nrsc.gov.in** (or bhuvan.nrsc.gov.in →
   "Bhoonidhi"). Free registration may be required for downloads.
2. Browse/Download → search the layer catalog for **administrative
   boundaries** at **district** level (names drift between releases:
   "Administrative Boundaries", "District Boundary", "Admin Units" — the
   acceptance checks below matter more than the menu path).
3. Select coverage **Andhra Pradesh** and **Telangana** (two exports are
   fine — keep them separate, do not merge).
4. Export format: **Shapefile ZIP** (keep all sidecar files: .shp/.shx/
   .dbf/.prj) or **GeoJSON**. EPSG:4326 preferred; any CRS is acceptable
   IF you tell me which (it is recorded in .prj — send the file as-is,
   never hand-edit it).
5. Screenshot the layer's metadata/terms page at download time (license
   provenance for the asset registry — T2-context lane).

## Acceptance checks (2 minutes, QGIS or any viewer)

- Open the file (QGIS: Layer → Add Vector Layer).
- Attribute table row count: **AP = 26 features, TG = 33 features**
  (D#20 2026-vintage basis). Fewer = wrong vintage (pre-2019 TG or
  pre-2022 AP reorg) — do not send, search for a newer vintage.
- There is a district-name attribute (ideally also an LGD code attribute —
  if present, it cross-checks Move 1's export for free).
- AP vintage must be **≥ 2022-04-04** (26-district reorg in force).

## What to send back

- The two exports (as downloaded, zipped, untouched).
- The metadata/terms screenshot.
- One line: file names + vintage shown in metadata.

## If Bhoonidhi fights back (fallback ladder, in order)

1. State portals (AP/TG State GIS / e-Governance map services) — official,
   acceptable without amendment.
2. OSM/Datameet extract — **only after** a logged amendment (tell me and
   Qwen first; we will draft it).
3. Nothing works → say so; I file a "documented alternative" ruling
   request with Qwen for the consensus round (assignment could proceed
   provisionally on district-label joins with a logged sensitivity note,
   but that is a decision, not a default).
