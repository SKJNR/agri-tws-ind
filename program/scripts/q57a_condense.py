#!/usr/bin/env python3
"""Condense q57a_results/*.json into a readable digest for verdict-writing."""
import json, glob, os, re

OUT = "/home/z/my-project/scripts/q57a_results"
files = sorted(glob.glob(os.path.join(OUT, "q*.json")))
lines = []
for f in files:
    qid = os.path.basename(f).replace(".json", "")
    try:
        d = json.load(open(f))
    except Exception as e:
        lines.append(f"## {qid} — PARSE ERROR {e}")
        continue
    lines.append(f"## {qid}")
    for r in (d or [])[:4]:
        name = (r.get("name") or "").strip()[:110]
        url = r.get("url") or ""
        host = r.get("host_name") or ""
        snip = re.sub(r"\s+", " ", (r.get("snippet") or "")).strip()[:320]
        lines.append(f"- [{name}]({url})  ({host})")
        if snip:
            lines.append(f"  {snip}")
    lines.append("")

digest = "\n".join(lines)
with open(os.path.join(OUT, "DIGEST.md"), "w") as fh:
    fh.write(digest)
print(digest[:200])
print(f"\n[full digest: {os.path.join(OUT, 'DIGEST.md')} — {len(files)} queries]")
