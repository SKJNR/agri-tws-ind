#!/usr/bin/env python3
"""Q57-A: authoritative license + maintenance check via GitHub API + PyPI JSON."""
import json, urllib.request, time

REPOS = [
    ("Q1", "xarray-contrib/xskillscore"),
    ("Q1b", "climpred/climpred"),
    ("Q2", "scikit-learn-contrib/MAPIE"),
    ("Q2b", "donlnz/nonconformist"),
    ("Q2c", "deel-ai/puncc"),
    ("Q2d", "henrikbostrom/crepes"),
    ("Q3", "isciences/exactextract"),
    ("Q3b", "isciences/pyexactextract"),
    ("Q4", "prefix-dev/pixi"),
    ("Q4b", "iterative/dvc"),
    ("Q4c", "conda-incubator/conda-lock"),
    ("Q8", "ai4bharat/IndicTrans2"),
    ("Q8b", "ai4bharat/IndicConformer"),
    ("Q8c", "ai4bharat/Indic-TTS"),
    ("Q5", "google-deepmind/weathernext"),
]

PYPI = ["xskillscore", "mapie", "nonconformist", "puncc", "exactextract",
        "pyexactextract", "pyet", "duckdb"]

def get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "q57a-license-audit",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

out = {"repos": {}, "pypi": {}}
for tag, repo in REPOS:
    try:
        d = get(f"https://api.github.com/repos/{repo}")
        lic = d.get("license") or {}
        out["repos"][repo] = {
            "tag": tag,
            "license": lic.get("spdx_id"),
            "pushed_at": d.get("pushed_at"),
            "archived": d.get("archived"),
            "stars": d.get("stargazers_count"),
        }
        print(f"{tag:4} {repo:38} license={lic.get('spdx_id')} pushed={d.get('pushed_at')} stars={d.get('stargazers_count')} archived={d.get('archived')}")
    except Exception as e:
        out["repos"][repo] = {"tag": tag, "error": str(e)}
        print(f"{tag:4} {repo:38} ERROR {e}")
    time.sleep(0.4)

for pkg in PYPI:
    try:
        d = get(f"https://pypi.org/pypi/{pkg}/json")
        info = d.get("info", {})
        rel = (d.get("releases") or {})
        # latest upload date across files of the info version
        files = rel.get(info.get("version", ""), []) or []
        upl = max((f.get("upload_time") or "" for f in files), default="")
        out["pypi"][pkg] = {
            "version": info.get("version"),
            "license": (info.get("license") or "")[:120].replace("\n", " "),
            "requires_python": info.get("requires_python"),
            "upload_time": upl,
        }
        print(f"PYPI {pkg:16} v={info.get('version'):12} lic={out['pypi'][pkg]['license'][:60]:60} last_upload={upl}")
    except Exception as e:
        out["pypi"][pkg] = {"error": str(e)}
        print(f"PYPI {pkg:16} ERROR {e}")
    time.sleep(0.3)

with open("/home/z/my-project/scripts/q57a_results/license_audit.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nsaved -> q57a_results/license_audit.json")
