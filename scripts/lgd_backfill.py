#!/usr/bin/env python3
"""Backfill lgd_district_code into district_basis.csv via normalization ladder.

Match ladder (strictest first; anything not matched EXACTLY or NORM gets a
loud CHECK marker — never a silent guess):
  1. EXACT   raw string equal (case/pad-trimmed)
  2. NORM    lowercase, strip .,-,', collapse spaces
  3. NOSPACE norm with all spaces removed
  4. ANAGRAM sorted-letter key of nospace (catches jangoan/jangaon)
  5. FUZZY   difflib ratio >= 0.90 on norm  -> CHECK: candidate (not auto)
  6. SUFFIX  lgd name endswith basis name (>=6 chars) -> CHECK: candidate
Output: district_basis_v2.csv + lgd_backfill_report.json
"""
import csv, difflib, json, re, sys

RAW = "/tmp/lgd_raw.json"
BASIS = "/home/z/my-project/download/d20/district_basis.csv"
OUT_CSV = "/home/z/my-project/download/agri_tws_ind/lgd_live/district_basis_v2.csv"
OUT_RPT = "/home/z/my-project/download/agri_tws_ind/lgd_live/lgd_backfill_report.json"

# Documented official-name variants (basis spelling -> LGD live spelling).
# Each is a well-known naming variant of the SAME district; resolving these
# here is mechanical normalization, not a modeling decision. Kept explicit
# and loud for the Qwen adversarial pass to veto if it disagrees.
KNOWN_VARIANTS = {
    ("AP", "Nellore"): "Sri Potti Sriramulu Nellore",   # official full name since 2008
    ("TS", "Jagtial"): "Jagitial",                       # common transliteration variant
    ("TS", "Jayashankar Bhupalpally"): "Jayashankar Bhupalapally",  # spelling variant
}

def norm(s):
    s = (s or "").lower().strip()
    s = re.sub(r"[.\-'\u2019()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def nospace(s): return norm(s).replace(" ", "")
def anagram(s): return "".join(sorted(nospace(s)))

raw = json.load(open(RAW))
if isinstance(raw, str):
    raw = json.loads(raw)

lgd = {}   # state -> list of dicts
for st in ("AP", "TG"):
    lgd[st] = [{"code": d["districtCode"], "name": (d["districtNameEnglish"] or "").strip(),
                "local": (d["districtNameLocal"] or "").strip(),
                "effective": (d.get("effectiveDate") or "")[:10]}
               for d in raw[st]["rows"]]

# indexes per state
idx = {}
for st in lgd:
    idx[st] = {
        "exact":  {d["name"].strip().lower(): d for d in lgd[st]},
        "norm":   {norm(d["name"]): d for d in lgd[st]},
        "nospace": {nospace(d["name"]): d for d in lgd[st]},
        "anagram": {anagram(d["name"]): d for d in lgd[st]},
        "normlist": [(norm(d["name"]), d) for d in lgd[st]],
    }

rows = list(csv.DictReader(open(BASIS, newline="")))
report = {"rows": [], "summary": {}}
n_exact = n_norm = n_check = n_none = 0
used_codes = set()

for r in rows:
    st, bn = ("TG" if r["state"].strip().upper() in ("TG", "TS", "TELANGANA") else "AP"), r["district_name_lgd"]
    r["state"] = "AP" if st == "AP" else "TS"   # keep basis's own label in output
    d = None; how = None; cand = None
    I = idx[st]
    variant = KNOWN_VARIANTS.get((r["state"], bn))
    if bn.strip().lower() in I["exact"]:
        d = I["exact"][bn.strip().lower()]; how = "EXACT"
    elif variant and norm(variant) in I["norm"]:
        d = I["norm"][norm(variant)]; how = "RESOLVED_NAME_VARIANT"
    elif norm(bn) in I["norm"]:
        d = I["norm"][norm(bn)]; how = "NORM"
    elif nospace(bn) in I["nospace"]:
        d = I["nospace"][nospace(bn)]; how = "NOSPACE"
    elif anagram(bn) in I["anagram"]:
        d = I["anagram"][anagram(bn)]; how = "ANAGRAM"
    else:
        # fuzzy + suffix -> CHECK candidates (never auto-assign)
        best = difflib.get_close_matches(norm(bn), [x[0] for x in I["normlist"]], n=1, cutoff=0.90)
        if not best:
            for nm, dd in I["normlist"]:
                if len(nm) >= 6 and (nm.endswith(norm(bn)) or norm(bn).endswith(nm)):
                    best = [nm]; break
        if best:
            cand = dict(I["norm"][best[0]]); how = "CHECK"
    if how == "EXACT": n_exact += 1
    elif how == "RESOLVED_NAME_VARIANT": n_norm += 1
    elif how in ("NORM", "NOSPACE", "ANAGRAM"): n_norm += 1
    elif how == "CHECK": n_check += 1
    else: n_none += 1

    if d is not None:
        r["lgd_district_code"] = str(d["code"])
        r["lgd_effective_date"] = d["effective"]
        r["lgd_name_live"] = d["name"]
        r["match_method"] = how
        used_codes.add((st, d["code"]))
    elif cand is not None:
        r["lgd_district_code"] = f"CHECK_CANDIDATE:{cand['code']}:{cand['name']}"
        r["lgd_name_live"] = cand["name"]
        r["match_method"] = "FUZZY_CHECK"
    else:
        r["lgd_district_code"] = "NO_MATCH_IN_LGD"
        r["match_method"] = "NONE"
    report["rows"].append({k: r[k] for k in ("state", "district_name_lgd", "lgd_district_code",
                                             "lgd_name_live", "match_method") if k in r})

# LGD rows not used by any basis row = the "extras" story (new districts etc.)
extras = []
for st in ("AP", "TG"):
    for d in lgd[st]:
        if (st, d["code"]) not in used_codes:
            extras.append({"state": st, "code": d["code"], "name": d["name"],
                           "effective": d["effective"]})

report["summary"] = {
    "basis_rows": len(rows),
    "matched_exact": n_exact, "matched_normalized": n_norm,
    "fuzzy_check_needed": n_check, "no_match": n_none,
    "lgd_rows_not_consumed": extras,
}

import os
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
fn = ["state", "lgd_state_code", "district_name_lgd", "district_name_local",
      "formed_or_renamed", "dominant_parent_at_formation", "basis_note",
      "lgd_district_code", "lgd_name_live", "lgd_effective_date", "match_method"]
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fn, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
json.dump(report, open(OUT_RPT, "w"), indent=2)

print(json.dumps(report["summary"], indent=2))
print("\n-- rows needing human eyes --")
for rr in report["rows"]:
    if rr["match_method"] in ("FUZZY_CHECK", "NONE"):
        print(f"  {rr['state']}  basis='{rr['district_name_lgd']}'  ->  {rr['lgd_district_code']}")
print(f"\nwrote {OUT_CSV}\nwrote {OUT_RPT}")
