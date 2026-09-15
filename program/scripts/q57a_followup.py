#!/usr/bin/env python3
"""Q57-A follow-up: AI4Bharat licenses via HF API + SEAS6 status + retries."""
import json, urllib.request, time

def get(url, accept="application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": "q57a-followup", "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

out = {}

print("== HuggingFace model licenses ==")
for mid in ["ai4bharat/indictrans2-indic-en-1B",
            "ai4bharat/IndicConformer",
            "ai4bharat/indic-tts-voice-bureau",
            "openai/whisper-large-v3"]:
    try:
        d = get(f"https://huggingface.co/api/models/{mid}")
        lic = (d.get("cardData") or {}).get("license")
        out[mid] = {"license": lic, "lastModified": d.get("lastModified"), "downloads": d.get("downloads")}
        print(f"{mid:44} license={lic} lastMod={d.get('lastModified')}")
    except Exception as e:
        out[mid] = {"error": str(e)}
        print(f"{mid:44} ERROR {e}")
    time.sleep(0.3)

print("\n== GitHub retries (rate-limit recovery) ==")
for tag, repo in [("Q1", "xarray-contrib/xskillscore"),
                  ("Q3b", "isciences/pyexactextract"),
                  ("Q4", "prefix-dev/pixi"),
                  ("Q8", "ai4bharat/IndicTrans2"),
                  ("Q8b", "ai4bharat/IndicConformer")]:
    try:
        d = get(f"https://api.github.com/repos/{repo}")
        lic = d.get("license") or {}
        out[repo] = {"tag": tag, "license": lic.get("spdx_id"), "pushed_at": d.get("pushed_at")}
        print(f"{tag:4} {repo:38} license={lic.get('spdx_id')} pushed={d.get('pushed_at')}")
    except Exception as e:
        out[repo] = {"tag": tag, "error": str(e)}
        print(f"{tag:4} {repo:38} ERROR {e}")
    time.sleep(1.0)

with open("/home/z/my-project/scripts/q57a_results/followup_audit.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nsaved -> q57a_results/followup_audit.json")
