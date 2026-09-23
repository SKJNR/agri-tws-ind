#!/usr/bin/env python3
"""Parse live LGD export, reconcile vs D#20 59-district basis, emit backfill."""
import json, csv, sys

raw = json.load(open("/tmp/lgd_raw.json"))
if isinstance(raw, str):          # agent-browser eval returns a quoted JSON string
    raw = json.loads(raw)
if "error" in raw:
    sys.exit(f"FATAL: {raw['error']}")

basis = {}
with open("/home/z/my-project/download/d20/district_basis.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        basis[(r["state"], r["district_name_lgd"].strip().lower())] = r

report = {"fetched_at": raw["fetched_at"], "source": raw["source"], "states": {}}
rows_out = []          # backfill rows for district_basis.csv

for st in ("AP", "TG"):
    data = raw[st]
    dists = sorted(
        [{"code": d["districtCode"], "name": (d["districtNameEnglish"] or "").strip(),
          "local": (d["districtNameLocal"] or "").strip(),
          "effective": (d.get("effectiveDate") or "")[:10]} for d in data["rows"]],
        key=lambda d: d["name"].lower())
    names = {d["name"].strip().lower() for d in dists}

    matched, unmatched = [], []
    for (bs, bn), r in basis.items():
        if bs != st:
            continue
        if bn in names:
            matched.append(bn)
        else:
            unmatched.append(r["district_name_lgd"])

    extras = sorted(names - {b for (bs, b) in basis if bs == st})

    report["states"][st] = {
        "n_rows": data["n_rows"],
        "expected_d20": 26 if st == "AP" else 33,
        "basis_matched": len(matched),
        "basis_missing_in_lgd": unmatched,
        "lgd_extras_not_in_basis": extras,
    }
    for d in dists:
        rows_out.append({"state": st, "lgd_state_code": data["state_code"],
                         "district_name_lgd": d["name"], "district_name_local": d["local"],
                         "lgd_district_code": d["code"], "lgd_effective_date": d["effective"],
                         "in_d20_basis": "yes" if d["name"].lower() in {b for (bs, b) in basis if bs == st} else "NO"})

# de-dup check
codes = [r["lgd_district_code"] for r in rows_out]
dupcodes = [c for c in set(codes) if codes.count(c) > 1]
names = [(r["state"], r["district_name_lgd"].lower()) for r in rows_out]
dupnames = [n for n in set(names) if names.count(n) > 1]

report["integrity"] = {"total_rows": len(rows_out), "duplicate_codes": dupcodes,
                       "duplicate_names": dupnames}

json.dump(report, open("/tmp/lgd_recon.json", "w"), indent=2)

with open("/tmp/lgd_backfill.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print(json.dumps(report, indent=2))
print("\nbackfill rows written: /tmp/lgd_backfill.csv (also print of all names next)")
for st in ("AP", "TG"):
    print(f"\n=== {st} ({report['states'][st]['n_rows']} rows) ===")
    for r in rows_out:
        if r["state"] == st:
            flag = "" if r["in_d20_basis"] == "yes" else "   <-- NOT IN D#20 BASIS"
            print(f"  {r['lgd_district_code']:>5}  {r['district_name_lgd']:<35} eff {r['lgd_effective_date']}{flag}")
