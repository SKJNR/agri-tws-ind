#!/usr/bin/env python3
"""
Task 18: Build the deliverable ZIP package.

Purpose (user instruction): "generate zip file along with code and results
submission file ... for previous successful ones too" — a single professional
archive with the full code lineage, every submission CSV (v1 -> v21, including
all previous successful versions), analysis result artifacts, and the complete
documentation set (master handoff, adversarial review, methodology audit,
final-2 selection card, report draft, recovered worklogs, cross-AI debate).

Layout:
  TWS_Zindi_Endgame_Package/
    PACKAGE_README.md          navigation + key-file md5 table
    01_code/                   all build/experiment/audit scripts + build logs
    02_submissions/            all 50 submission CSVs (full lineage)
    03_analysis_results/       probe CSVs, txt result files, figures
    04_docs/                   handoff / reviews / audits / report / worklogs
    05_manifest/               md5 manifest of every archived file

Output: /home/z/my-project/download/TWS_Zindi_Endgame_Package_2026-09-01.zip
"""
import os
import glob
import zipfile
import hashlib
import datetime

ROOT = "/home/z/my-project"
OUT = os.path.join(ROOT, "download", "TWS_Zindi_Endgame_Package_2026-09-01.zip")
TOP = "TWS_Zindi_Endgame_Package"

KEY_FILES = {
    "02_submissions/submission_v21a.csv": "clean best, FINAL-2 slot 1",
    "02_submissions/submission_v12b.csv": "FINAL-2 slot 2 (default)",
    "02_submissions/submission_v13b.csv": "FINAL-2 slot 2 (upgrade candidate)",
    "02_submissions/submission_v18a.csv": "emergency fallback",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    """Partition the workspace into the four content buckets."""
    code = sorted(
        glob.glob(os.path.join(ROOT, "scripts", "*"))
    )
    code = [p for p in code if os.path.isfile(p)]

    subs = sorted(
        glob.glob(os.path.join(ROOT, "download", "submission_*.csv"))
    )

    docs = []
    for pat in ("download/*.md", "MASTER_HANDOFF.md", "worklog*.md", "README.md"):
        docs.extend(glob.glob(os.path.join(ROOT, pat)))
    docs = [p for p in docs if os.path.isfile(p)]
    # dedupe: download/MASTER_HANDOFF.md is a byte-identical snapshot of the root
    # master copy (md5-verified) -> archive only the root copy.
    docs = [p for p in docs
            if not (p == os.path.join(ROOT, "download", "MASTER_HANDOFF.md"))]
    # disambiguate the two different READMEs: root = project README,
    # download/ = results-folder README.
    doc_arc = {}
    for p in docs:
        name = os.path.basename(p)
        if p == os.path.join(ROOT, "README.md"):
            name = "README_PROJECT.md"
        doc_arc[p] = name

    analysis = []
    skip = set(subs) | set(docs) | {OUT, os.path.join(ROOT, "download", "MASTER_HANDOFF.md")}
    for p in sorted(glob.glob(os.path.join(ROOT, "download", "*"))):
        if not os.path.isfile(p):
            continue
        if p in skip:
            continue
        analysis.append(p)

    return code, subs, docs, analysis, doc_arc


README_TMPL = """# TWS Zindi Endgame Package — {date}

One-step-ahead drought / Total Water Storage forecasting (Zindi, ITU AI-for-Good).
Complete competition workspace archive: code, results, every submission file
(including all previous successful versions), and the full decision-record.

## Contents

| Folder | What | Count |
|---|---|---|
| 01_code/ | All build / experiment / audit scripts (v1 -> v21 lineage) + build logs | {n_code} |
| 02_submissions/ | Every submission CSV ever produced, v1 -> v21b (incl. previous successful versions) | {n_subs} |
| 03_analysis_results/ | Probe files, measurement txt outputs, figures | {n_ana} |
| 04_docs/ | Master handoff, adversarial review, methodology audit, final-2 selection card, report draft, recovered worklogs, cross-AI debate | {n_docs} |
| 05_manifest/ | md5 manifest of every archived file | 1 |

## The files that matter most

| File (inside zip) | Role | md5 |
|---|---|---|
{key_table}

## Reading order for a reviewer / new session

1. 04_docs/MASTER_HANDOFF.md — single source of truth (recovery protocol, LB table, audits)
2. 04_docs/FINAL2_SELECTION.md — the endgame selection card (deadlines, DQ traps)
3. 04_docs/ADVERSARIAL_REVIEW_ROUND1.md — skeptic verdicts on every endgame decision
4. 04_docs/METHODOLOGY_AUDIT.md — the "all hats" audit (11 worn, 6 declined with reasons)
5. 04_docs/worklog.md — full task-by-task history (Tasks 1-17)
6. 01_code/build_v21_phaseC.py — deterministic builder of the FINAL-2 slot-1 pick

## Compliance note

submission_v20a/b/c.csv are archived for AUDIT TRAIL ONLY (prohibited external
GRACE/TWS lane — see MASTER_HANDOFF section 6). They must never be selected on
Zindi. The compliant picks are v21a + v12b/v13b per FINAL2_SELECTION.md.
"""


def main():
    if os.path.exists(OUT):
        os.remove(OUT)

    code, subs, docs, analysis, doc_arc = collect()
    print(f"collected: code={len(code)} subs={len(subs)} "
          f"docs={len(docs)} analysis={len(analysis)}")

    date = datetime.date.today().isoformat()
    key_rows = []
    for rel, why in KEY_FILES.items():
        local = os.path.join(ROOT, "download", os.path.basename(rel))
        key_rows.append(f"| {rel} | {why} | {md5(local) if os.path.exists(local) else 'MISSING'} |")
    readme = README_TMPL.format(
        date=date, n_code=len(code), n_subs=len(subs), n_ana=len(analysis),
        n_docs=len(docs), key_table="\n".join(key_rows),
    )

    manifest_lines = []
    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(f"{TOP}/PACKAGE_README.md", readme)
        n_files += 1

        for src, dest_dir, arc_of in (
            (code, "01_code", None),
            (subs, "02_submissions", None),
            (analysis, "03_analysis_results", None),
            (docs, "04_docs", doc_arc),
        ):
            for p in src:
                base = arc_of[p] if arc_of else os.path.basename(p)
                arc = f"{TOP}/{dest_dir}/{base}"
                z.write(p, arc)
                manifest_lines.append(f"{md5(p)}  {arc}")
                n_files += 1
                total_bytes += os.path.getsize(p)
                if n_files % 50 == 0:
                    print(f"  archived {n_files} files ...")

        z.writestr(f"{TOP}/05_manifest/MD5SUMS.txt", "\n".join(manifest_lines) + "\n")
        n_files += 1

    print()
    print(f"ZIP written: {OUT}")
    print(f"  files archived : {n_files}")
    print(f"  raw bytes      : {total_bytes / 1e6:.1f} MB")
    print(f"  zip size       : {os.path.getsize(OUT) / 1e6:.1f} MB")
    print(f"  zip md5        : {md5(OUT)}")

    # verification pass: open the zip and confirm integrity + key members
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
        assert bad is None, f"corrupt member: {bad}"
        names = set(z.namelist())
        for rel in KEY_FILES:
            assert f"{TOP}/{rel}" in names, f"missing member: {rel}"
        assert f"{TOP}/05_manifest/MD5SUMS.txt" in names
    print("  integrity      : testzip PASS, key members present")


if __name__ == "__main__":
    main()
