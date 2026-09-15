#!/usr/bin/env python3
"""
Task 57 (GLM) — D#20 build: district basis + change ledger + PIP pipeline
+ D#21 pre-staged physical-split script + unseal procedure + manifest.

Per courier Task 57 Part B:
  Step 1 (D#20): district_basis.csv (59, 2026-vintage), change_ledger.json
    (TG 2016-10-11 10->31; TG 2019-02-17 31->33; AP 2022-04-04 13->26),
    point-in-polygon assignment pipeline (blocked on gwl_data.csv arrival —
    mock-tested now), old-basis sensitivity note on TRAIN-era labels.
  Step 2 (D#21): physical split script pre-staged (runs when gwl_data.csv
    lands), unseal procedure written.

Sources verified 2026-09-15 (web): TG-2016 (Wikipedia districts-of-Telangana,
The Hindu / NDTV 2016-10-11 reports, 21 new districts -> 31); TG-2019
(ToI / TheNewsMinute / IndiaTomorrow 2019-02-17, Mulugu + Narayanpet -> 33);
AP-2022 (The Hindu 2022-04-03/04 "13 new districts from Monday April 4",
AP Gazette notifications 472-497 dated 03-04-2022).
LGD codes: portal live; districtWiseDetailReport.do captcha-gated; DWR call
rejected -> lgd_code columns carry PENDING_FOUNDER_EXPORT marker.

All parentage mappings beyond the three verified events are PROVISIONAL
(dominant-parent synthesis from public record; GO-level verification pending
at polygon-binding; built-in union tests encoded in ledger).
"""
import csv, hashlib, io, json, os, sys, datetime

OUT = "/tmp/my-project/download/agri_tws_ind/d20"
os.makedirs(OUT, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def w(path, text):
    with open(path, "w", newline="\n") as fh:
        fh.write(text)

# ---------------------------------------------------------------- basis ----
# 2026-vintage 59 districts. Names normalized to LGD-style spellings;
# district_name_local = spelling used in districts_ap_ts.csv (QWEN_PROVIDED
# centroids, Open-Meteo fetch S1 basis).
AP_OLD = ["Srikakulam", "Vizianagaram", "Visakhapatnam", "East Godavari",
          "West Godavari", "Krishna", "Guntur", "Prakasam", "Nellore",
          "Kurnool", "Ananthapuramu", "Chittoor", "YSR Kadapa"]
TG_OLD = ["Adilabad", "Nizamabad", "Karimnagar", "Medak", "Hyderabad",
          "Rangareddy", "Mahabubnagar", "Nalgonda", "Warangal", "Khammam"]

# (lgd-style name, local csv name, formed_or_renamed, parent_event)
AP_NEW = [
    ("Parvathipuram Manyam", "Parvathipuram Manyam", "2022-04-04", "Vizianagaram"),
    ("Alluri Sitharama Raju", "Alluri Sitharama Raju", "2022-04-04", "Visakhapatnam"),
    ("Anakapalli", "Anakapalli", "2022-04-04", "Visakhapatnam"),
    ("Kakinada", "Kakinada", "2022-04-04", "East Godavari"),
    ("Dr. B.R. Ambedkar Konaseema", "Dr B R Ambedkar Konaseema", "2022-04-04", "East Godavari"),
    ("Eluru", "Eluru", "2022-04-04", "West Godavari"),
    ("NTR", "NTR", "2022-04-04", "Krishna"),
    ("Bapatla", "Bapatla", "2022-04-04", "Guntur"),
    ("Palnadu", "Palnadu", "2022-04-04", "Guntur"),
    ("Nandyal", "Nandyal", "2022-04-04", "Kurnool"),
    ("Sri Sathya Sai", "Sri Sathya Sai", "2022-04-04", "Ananthapuramu"),
    ("Annamayya", "Annamayya", "2022-04-04", "YSR Kadapa"),
    ("Tirupati", "Tirupati", "2022-04-04", "Chittoor"),
]
TG_NEW_2016 = [
    ("Kumuram Bheem Asifabad", "Kumuram Bheem Asifabad", "2016-10-11", "Adilabad"),
    ("Mancherial", "Mancherial", "2016-10-11", "Adilabad"),
    ("Nirmal", "Nirmal", "2016-10-11", "Adilabad"),
    ("Kamareddy", "Kamareddy", "2016-10-11", "Nizamabad"),
    ("Jagtial", "Jagtial", "2016-10-11", "Karimnagar"),
    ("Peddapalli", "Peddapalli", "2016-10-11", "Karimnagar"),
    ("Rajanna Sircilla", "Rajanna Sircilla", "2016-10-11", "Karimnagar"),
    ("Sangareddy", "Sangareddy", "2016-10-11", "Medak"),
    ("Siddipet", "Siddipet", "2016-10-11", "Medak"),
    ("Vikarabad", "Vikarabad", "2016-10-11", "Rangareddy"),
    ("Medchal-Malkajgiri", "Medchal Malikajgiri", "2016-10-11", "Rangareddy"),
    ("Wanaparthy", "Wanaparthy", "2016-10-11", "Mahabubnagar"),
    ("Nagarkurnool", "Nagarkurnool", "2016-10-11", "Mahabubnagar"),
    ("Jogulamba Gadwal", "Jogulamba Gadwal", "2016-10-11", "Mahabubnagar"),
    ("Suryapet", "Suryapet", "2016-10-11", "Nalgonda"),
    ("Yadadri Bhuvanagiri", "Yadadri Bhuvanagiri", "2016-10-11", "Nalgonda"),
    ("Jangaon", "Jangaon", "2016-10-11", "Warangal"),
    ("Jayashankar Bhupalpally", "Jayashankar Bhupalpally", "2016-10-11", "Warangal"),
    ("Mahabubabad", "Mahabubabad", "2016-10-11", "Warangal"),
    ("Bhadradri Kothagudem", "Bhadradri Kothagudem", "2016-10-11", "Khammam"),
]
TG_NEW_2019 = [
    ("Mulugu", "Mulugu", "2019-02-17", "Warangal"),
    ("Narayanpet", "Narayanpet", "2019-02-17", "Mahabubnagar"),
]

def rows():
    for n in AP_OLD:
        yield ("AP", 28, n, n, "PRE-2022-BASIS", "", "CONTINUES_2022 (boundaries changed)")
    for lgdn, locn, d, p in AP_NEW:
        yield ("AP", 28, lgdn, locn, d, p, "NEW_2022")
    for n in TG_OLD:
        if n == "Warangal":
            yield ("TS", 36, "Warangal", "Warangal", "RENAME~2018-21 UNVERIFIED",
                   "Warangal (Rural)", "RENAME_EVENT: ex-Warangal-Rural; GO check pending")
        elif n == "Hyderabad":
            yield ("TS", 36, "Hyderabad", "Hyderabad", "PRE-2016-BASIS", "",
                   "CONTINUES_2016 (minor boundary changes)")
        else:
            yield ("TS", 36, n, n, "PRE-2016-BASIS", "", "CONTINUES_2016 (boundaries changed)")
    yield ("TS", 36, "Hanumakonda", "Hanumakonda", "RENAME~2018-21 UNVERIFIED",
           "Warangal (Urban)", "RENAME_EVENT: ex-Warangal-Urban; GO check pending")
    for lgdn, locn, d, p in TG_NEW_2016 + TG_NEW_2019:
        yield ("TS", 36, lgdn, locn, d, p, "NEW")

buf = io.StringIO()
wtr = csv.writer(buf)
wtr.writerow(["state", "lgd_state_code", "district_name_lgd", "district_name_local",
              "formed_or_renamed", "dominant_parent_at_formation", "basis_note",
              "lgd_district_code"])
n = 0
for r in rows():
    wtr.writerow(list(r) + ["PENDING_FOUNDER_EXPORT"])
    n += 1
w(f"{OUT}/district_basis.csv", buf.getvalue())
assert n == 59, f"expected 59 districts, got {n}"

# --------------------------------------------------------------- ledger ----
ledger = {
  "ledger": "district_change_ledger",
  "program": "AGRI-TWS-IND",
  "decision": "D#20 (Task 56 consensus; Task 57 GLM build)",
  "built": "2026-09-15",
  "constant_target_basis": "2026-vintage 59 districts (AP 26 + TG 33); wells assigned by point-in-polygon on current polygons",
  "verified_sources_2026-09-15": {
    "tg_2016": ["en.wikipedia.org/wiki/List_of_districts_of_Telangana",
                 "thehindu.com 2016-10-11 'Telangana gets 21 new districts'",
                 "ndtv.com 2016-10-11 'Telangana Map Redrawn Adding 21 New Districts'"],
    "tg_2019": ["timesofindia 2019-02-17 'TRS govt creates 2 new districts'",
                 "thenewsminute.com 2019-02-17 'Narayanpet and Mulugu'",
                 "indiatomorrow.net 2019-02-17"],
    "ap_2022": ["thehindu.com 2022-04-03 'A.P. to have 26 districts from today (Apr 4)'",
                 "apegazette.cgg.gov.in notifications 472-497 dated 03-04-2022"]
  },
  "events": [
    {"event_id": "TG-2016-10-11", "state": "Telangana", "effective": "2016-10-11",
     "type": "REORGANISATION", "before": 10, "after": 31,
     "in_window": "TRAIN (pre-2022 open window)",
     "districts_added": 21, "districts_continued": 9,
     "note": "Warangal split into Rural + Urban (continuing names); all 10 old names otherwise continued with changed boundaries",
     "confidence": "VERIFIED (3 independent sources)"},
    {"event_id": "TG-2019-02-17", "state": "Telangana", "effective": "2019-02-17",
     "type": "REORGANISATION", "before": 31, "after": 33,
     "in_window": "TRAIN",
     "districts_added": ["Mulugu (from Warangal Rural)", "Narayanpet (from Mahabubnagar)"],
     "confidence": "VERIFIED (3 independent sources)"},
    {"event_id": "AP-2022-04-04", "state": "Andhra Pradesh", "effective": "2022-04-04",
     "type": "REORGANISATION", "before": 13, "after": 26,
     "in_window": "VAL (2022 is inside val window per D#20 ruling: split is 2020-22 open / 2023-25 sealed; AP reorg lands in the final open year)",
     "districts_added": 13, "districts_continued": 13,
     "gazette": "AP Gazette notifications 472-497, PART I EXTRAORDINARY, 03-04-2022",
     "confidence": "VERIFIED (The Hindu + gazette reference)"},
    {"event_id": "TG-RENAME-UNVERIFIED", "state": "Telangana",
     "type": "RENAME_EVENTS", "detail": [
        "Warangal (Urban) -> Hanumakonda (date unverified, ~2018-2021)",
        "Warangal (Rural) -> Warangal (date unverified, ~2018-2021; after Mulugu carve-out 2019-02-17)"],
     "confidence": "UNVERIFIED — standing query; does not affect 2026-basis assignment (current names are the target basis), affects vintage-label reconstruction only"}
  ],
  "parentage_map": {
    "TG-2016-10-11": {p: [] for p in TG_OLD},
    "TG-2019-02-17": {"Warangal (Rural)": ["Mulugu"], "Mahabubnagar": ["Narayanpet"]},
    "AP-2022-04-04": {p: [] for p in AP_OLD},
    "confidence": "PROVISIONAL — dominant-parent synthesis from public record; exact mandal-level composition pending GO/gazette read at polygon-binding",
    "unverified_shares": ["Alluri Sitharama Raju: Visakhapatnam dominant, Vizianagaram share unverified",
                           "Parvathipuram Manyam: Vizianagaram dominant, Srikakulam share unverified",
                           "Eluru: West Godavari dominant, Krishna share unverified",
                           "Medchal-Malkajgiri: Rangareddy dominant, Hyderabad share unverified",
                           "Siddipet: Medak dominant, Rangareddy/Nizamabad shares unverified"]
  },
  "tests": [
    {"test_id": "T-D20-1", "trigger": "on polygon binding",
     "rule": "each old-district polygon must be approx-union of its children's polygons (area overlap >= 98%); failures -> GO-level investigation, not silent fix"},
    {"test_id": "T-D20-2", "trigger": "on well assignment",
     "rule": "every well assigned to exactly one 2026-basis district or explicitly unassigned-with-reason (outside AP+TG boundary layer / degenerate coords / polygon gap); no defaults"},
    {"test_id": "T-D20-3", "trigger": "on vintage reconstruction",
     "rule": "a well's 2015 label under 10-district TG basis must map to a district whose 2026 polygon contains the well (label lineage check for TRAIN-era sensitivity note)"}
  ],
  "blocked_on": [
    "gwl_data.csv not present in sandbox -> well-level PIP (T-D20-2) and D#21 split cannot run; founder re-upload or approve AIKosh download (Task-55 condition: provenance re-check vs India-WRIS raw on arrival)",
    "LGD district codes: portal captcha-gated (districtWiseDetailReport.do), DWR plain calls rejected -> founder one-time manual export of LGD district list for AP (state code 28) + TG (36), or accept PENDING markers until polygon binding",
    "District polygons LGD/Bhuvan-grade not in sandbox; GADM banned (D#20); OSM = community-grade fallback only with logged amendment"
  ]
}
for lgdn, locn, d, p in TG_NEW_2016:
    ledger["parentage_map"]["TG-2016-10-11"].setdefault(p, []).append(lgdn)
for lgdn, locn, d, p in AP_NEW:
    ledger["parentage_map"]["AP-2022-04-04"].setdefault(p, []).append(lgdn)
w(f"{OUT}/change_ledger.json", json.dumps(ledger, indent=2))
print(f"district_basis.csv: 59 rows OK")
print(f"change_ledger.json: {len(ledger['events'])} events, {len(ledger['tests'])} tests")
print(f"TG-2016 parentage: {sum(len(v) for v in ledger['parentage_map']['TG-2016-10-11'].values())} children (expect 20 new names + Warangal split)")
print(f"AP-2022 parentage: {sum(len(v) for v in ledger['parentage_map']['AP-2022-04-04'].values())} children (expect 13)")
