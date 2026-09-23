#!/usr/bin/env python3
"""Package the TWS competition deliverables into a dated, re-buildable zip.

Output: /home/z/my-project/download/TWS_competition_package_<YYYY-MM-DD>.zip

Includes (user requirement: code + results + submission files, incl. ALL previous
successful versions):
  - root docs        : README.md, MASTER_HANDOFF.md, worklog*.md, competition_*.json,
                       leaderboard.json, lb_now.html, .gitignore
  - scripts/         : every builder / experiment / audit / gate script + build logs
  - download/        : all 65 submission CSVs (v1..v21b incl. v12b/v13b/v20c/v21a),
                       provenance manifest, audit + review reports, diagnostic .txt
  - data/README.md   : data provenance note (the only data file included)

Excludes (deliberate, professional hygiene):
  - .env             (secrets — the Aug-31 morning zip included it by mistake)
  - data/*.csv/.ipynb/.pdf  (Zindi-owned competition data; re-downloadable, huge)
  - upload/          (personal screenshots / photos)
  - .git, skills/, node_modules/, tool-results/, db/, __pycache__
  - heavy binaries   (*.nc, *.npz, *.npy, *.parquet) and any file > 60 MB
  - download/*.zip   (no zip-in-zip recursion)

Re-runnable: edit the policy sets below and re-run. Deterministic content list.
"""
import hashlib
import os
import sys
import zipfile
from datetime import date

ROOT = "/home/z/my-project"
OUT_DIR = os.path.join(ROOT, "download")
STAMP = date.today().isoformat()
OUT_PATH = os.path.join(OUT_DIR, f"TWS_competition_package_{STAMP}.zip")

EXCLUDE_ROOT_FILES = {".env"}
EXCLUDE_DIRS = {".git", "__pycache__", "skills", "node_modules", "tool-results",
                "db", "upload", "recovered"}
HEAVY_EXTS = {".nc", ".npz", ".npy", ".parquet", ".zip"}
MAX_FILE_BYTES = 60 * 1024 * 1024  # 60 MB safety valve
DATA_KEEP = {"README.md"}          # only file kept from data/

# The two selections (plus upgrade/fallback) — verified present + md5-printed.
FINAL2_FILES = ["submission_v21a.csv", "submission_v12b.csv",
                "submission_v13b.csv", "submission_v18a.csv"]


def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_files():
    picked = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        top = "." if rel_dir == "." else rel_dir.split(os.sep)[0]

        if top in EXCLUDE_DIRS:
            dirnames[:] = []
            continue

        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
            if rel == os.path.relpath(OUT_PATH, ROOT):
                continue  # never zip ourselves
            ext = os.path.splitext(name)[1].lower()
            if top == "." and name in EXCLUDE_ROOT_FILES:
                continue
            if ext in HEAVY_EXTS:
                continue
            if top == "data" and name not in DATA_KEEP:
                continue
            if top == "download" and ext == ".zip":
                continue
            full = os.path.join(dirpath, name)
            if os.path.getsize(full) > MAX_FILE_BYTES:
                print(f"  [skip >60MB] {rel}")
                continue
            picked.append((full, rel))
    picked.sort(key=lambda x: x[1])
    return picked


def main() -> int:
    files = collect_files()
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    total_unc = 0
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for full, rel in files:
            zf.write(full, rel)
            total_unc += os.path.getsize(full)

    print(f"PACKAGE  : {OUT_PATH}")
    print(f"CONTENTS : {len(files)} files ({total_unc / 1e6:.1f} MB uncompressed, "
          f"{os.path.getsize(OUT_PATH) / 1e6:.1f} MB zipped)")
    print(f"ZIP MD5  : {md5_of(OUT_PATH)}")
    print("SUBMISSION CSVs INCLUDED:")
    n_csv = 0
    for full, rel in files:
        if rel.startswith("download/submission") and rel.endswith(".csv"):
            n_csv += 1
    print(f"  {n_csv} submission CSV files")
    print("FINAL-2 VERIFICATION (must all be PRESENT):")
    ok = True
    by_rel = {rel: full for full, rel in files}
    for name in FINAL2_FILES:
        rel = f"download/{name}"
        if rel in by_rel:
            print(f"  [OK]   {rel}  md5={md5_of(by_rel[rel])}")
        else:
            ok = False
            print(f"  [MISS] {rel}")
    print("RESULT   : " + ("PASS" if ok else "FAIL — final-2 file missing!"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
