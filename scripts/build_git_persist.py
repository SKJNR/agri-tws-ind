#!/usr/bin/env python3
"""Git persistence build (Task 57-GIT, founder ruling 2026-09-15).

Commits the program phase (post-Sep-10: D#20/D#21, Q57-A, LGD live, import
kit, worklog) into the canonical nested repo, tags it, and produces:
  - full bundle (first-time push)
  - incremental bundle (if founder already has the Sep-10 state)
  - program/ zip (no-git 60-second path)
  - FOUNDER_PUSH_CARD.md
Safety: .gitignore extended (*.bundle) BEFORE staging; size guard on staged
files; fail-loud at every step.
"""
import os, shutil, subprocess, sys, datetime

REPO = "/home/z/my-project/my-project"
STAGE = os.path.join(REPO, "program")
DL = "/home/z/my-project/download"          # founder-visible (outside repo)
TODAY = "2026-09-15"
TAG = f"program-{TODAY}"
BASE = "0fd747f"                             # Sep-10 HEAD (incremental base)
IDENT = ["-c", "user.name=GLM sandbox (AGRI-TWS-IND)",
         "-c", "user.email=glm@agri-tws-ind.local"]

def sh(*args, cwd=REPO, check=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"FATAL: {' '.join(args)}\nstdout: {r.stdout}\nstderr: {r.stderr}")
    return r.stdout.strip()

# ---------- 0. guards ----------
branch = sh("git", "branch", "--show-current")
print(f"repo branch: {branch}, HEAD: {sh('git','rev-parse','--short','HEAD')}")

gi = os.path.join(REPO, ".gitignore")
gi_txt = open(gi).read()
add_lines = [l for l in ("download/*.bundle", "download/*.gz") if l not in gi_txt]
if add_lines:
    open(gi, "a").write("\n".join([""] + add_lines) + "\n")
    print(f".gitignore += {add_lines}")

# ---------- 1. commit A: record current working-tree state ----------
sh("git", "add", "-A")
staged = sh("git", "diff", "--cached", "--name-only")
if not staged:
    print("commit A: nothing to record (already committed earlier run) — skip")
else:
    big = []
    for f in staged.splitlines():
        p = os.path.join(REPO, f)
        if os.path.isfile(p) and os.path.getsize(p) > 50 * 2**20:
            big.append((f, os.path.getsize(p)))
    if big:
        sys.exit(f"FATAL: oversized staged files (guard): {big}")
    sh("git", *IDENT, "commit", "-m",
       "state: record post-reset working tree (upload/ files lost to 9th reset; "
       "recoverable from history); .gitignore += *.bundle, *.gz (repo never "
       "absorbs its own backups)")
    print("commit A done")

# ---------- 2. build program/ tree ----------
if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)

def cp_tree(src, dst):
    shutil.copytree(src, os.path.join(STAGE, dst), dirs_exist_ok=True)

Q57A_EV = "/home/z/my-project/download/Q57A_evidence"
if not os.path.isdir(Q57A_EV):
    Q57A_EV = "/home/z/my-project/my-project/download/agri_tws_ind/q57a_evidence"
cp_tree(f"{DL}/d20", "d20")
cp_tree(f"{DL}/agri_tws_ind/lgd_live", "lgd_live")
cp_tree(Q57A_EV, "q57a_evidence")
cp_tree(f"{DL}/import_kit", "import_kit")
shutil.copy(f"{DL}/Q57A_VERDICTS_GLM_REPLY.md", f"{STAGE}/Q57A_VERDICTS_GLM_REPLY.md")
shutil.copy("/home/z/my-project/worklog.md", f"{STAGE}/worklog.md")

os.makedirs(f"{STAGE}/lgd_live/probe_artifacts")
for f in ("lgd_dwr_iface.js", "lgd_district_report.html", "lgd_home.html",
          "lgd_dwr_28.txt", "lgd_cookies.txt"):
    p = f"/home/z/my-project/scripts/q57a_results/{f}"
    if os.path.exists(p):
        shutil.copy(p, f"{STAGE}/lgd_live/probe_artifacts/{f}")

os.makedirs(f"{STAGE}/scripts")
for f in ("build_d20.py", "q57a_condense.py", "q57a_search_batch.sh",
          "q57a_followup.py", "q57a_license_audit.py"):
    shutil.copy(f"/home/z/my-project/scripts/{f}", f"{STAGE}/scripts/{f}")
for f in ("lgd_parse.py", "lgd_backfill.py"):
    shutil.copy(f"/home/z/my-project/scripts/lgd_live/{f}", f"{STAGE}/scripts/{f}")

# ---------- 3. program/data/ + READMEs ----------
os.makedirs(f"{STAGE}/data")
open(f"{STAGE}/data/README.md", "w").write(f"""# data/ — persistence policy (founder ruling 2026-09-15)

**ARRIVAL = COMMIT.** Any data file that lands in the sandbox is committed
here (or attached to a Release, see below) as the FIRST action, before any
pipeline work runs. A session reset after arrival must never again mean
data loss.

## Sizing rules
- file (or .gz of it) <= 95 MB  -> commit directly under data/
- bigger                         -> GitHub **Release asset** (up to 2 GB,
                                   private-repo scoped); the SHA256
                                   manifest + receipt are committed here so
                                   the tree always proves what the asset
                                   must hash to
- gwl_data.csv -> gzip first (CSVs shrink ~70-80%)

## Licensing (hard rule)
This repository MUST stay **PRIVATE**. gwl_data.csv is IndiaAI/AIKosh
competition data under participant license - no public redistribution.
LGD district exports are GODL (open) but live here for simplicity.

## Sealed window (D#21/D#24)
sealed/ extracts may be committed, but the seal is PROCEDURAL and git makes
it auditable: README_UNSEAL_PROCEDURE.md and ACCESS_LOG.md travel with the
data; any authorized read logs a line BEFORE the read (commit the log
change first). An unlogged read is a protocol surprise - DECISION_LOG
entry + founder notification, no silent fixes.

## Naming
data/raw/        as-first-received files (never edited)
data/open/       <=2022 extracts (active pipeline inputs, AM-3)
data/sealed/     >=2023 extracts (procedural seal)
""")

open(f"{STAGE}/README.md", "w").write(f"""# AGRI-TWS-IND — program phase (Tasks 41+)

Post-competition program: 16-feature frozen trunk, physical open/sealed
split, district basis, adversarial Qwen<->GLM consensus loop.

## Layout
- d20/            district basis + change ledger + well-assignment pipeline
- lgd_live/       live LGD registry export (2026-09-15), codes backfilled,
                  AP=28 finding, verification evidence + probe artifacts
- q57a_evidence/  Q57-A scout/license verification evidence
- import_kit/     founder import kit (LGD fetcher [now fallback], gwl
                  handback checker, polygon notes)
- scripts/        reproducible build scripts for the above
- data/           data persistence policy (ARRIVAL=COMMIT; see its README)
- worklog.md      shared multi-agent worklog (Tasks 41+)
- Q57A_VERDICTS_GLM_REPLY.md   adversarial round Q57-A

Canonical decision log: ../download/DECISION_LOG.md (competition era +
program phase). Master handoff: ../MASTER_HANDOFF.md.

## Sandbox restore procedure (after a session reset)
1. clone this repo (or fetch the latest daily bundle)
2. verify: git tag --list 'program-*'; sha256sum against MANIFEST files
3. read worklog.md tail -> last Task ID; resume there (never renumber)
4. data/: re-download Release assets per committed manifests if needed

## State at tag {TAG}
D#20 built (59-basis; amendment 59->61 pending Qwen consensus after AP
26->28 reorg discovery); D#21 pre-staged (mocktest PASS); Q57-A round
delivered; LGD codes backfilled 59/59; import kit shipped; gwl_data.csv
handback + polygons await founder moves.
""")

# ---------- 4. DECISION_LOG addendum ----------
dl_path = os.path.join(REPO, "download", "DECISION_LOG.md")
with open(dl_path, "a") as fh:
    fh.write(f"""

---

## ADDENDUM — {TODAY} (GLM, post-Task-57 session)

- 57-netprobe (interlude, no number): live egress probe. CDS/GitHub/PyPI/
  Zindi-API reachable from sandbox; WRIS/IMD/data.gov.in/IndiaAI/Kaggle
  refuse. Logged; no plan change.
- 57-KIT: founder import kit built (LGD browser fetcher, gwl handback
  checker with D#24-safe receipt — SELFTEST + 20k-row e2e PASS, polygon
  handoff notes). No decision consumed; AM-6 order intact.
- 57-KIT-LGD: LGD district codes OBTAINED by GLM via real headless-browser
  DWR on the portal's own citizen page (read-only; captcha gates only the
  report POST, untouched). district_basis_v2.csv: 59/59 backfilled,
  0 unmatched, per-row match_method audit. FINDING: AP registry lists 28
  districts (Markapuram LGD-790, Polavaram LGD-791, eff 2025-12-29) vs
  D#20's 26 — verified via 4+ independent sources. D#20 amendment
  59->61 routed to Qwen consensus (courier addendum in lgd_live/).
  Sealed-window logic unaffected.
- FOUNDER RULING (chat, {TODAY}): program state + data persist in a
  founder-owned PRIVATE GitHub repo. ARRIVAL=COMMIT rule adopted: data
  landing in the sandbox is committed (or Release-asseted + manifest-
  committed) as the FIRST action, before pipeline work. Implemented same
  day: program/ tree, tag {TAG}, full + incremental bundles, push card.
""")

# ---------- 5. commit B + tag ----------
sh("git", "add", "-A")
staged_b = sh("git", "diff", "--cached", "--name-only")
if staged_b:
    sh("git", *IDENT, "commit", "-m",
       f"program: Tasks 41-57 phase — D#20/D#21 built, Q57-A adversarial round, "
       f"LGD codes backfilled 59/59 (AP=28 reorg finding -> D#20 amendment "
       f"pending), founder import kit, git persistence per founder ruling, "
       f"DECISION_LOG addendum. Tag {TAG}.")
    print("commit B done")
else:
    print("commit B: nothing new — skip")
sh("git", *IDENT, "tag", "-f", "-a", TAG, "-m",
   f"Program state {TODAY}: D#20/D#21, Q57-A, LGD live export, import kit, "
   f"persistence rule ARRIVAL=COMMIT.")
print("commit B + tag done")

# ---------- 6. bundles + zip ----------
full = f"{DL}/agri_tws_ind_repo_full_{TODAY}.bundle"
incr = f"{DL}/agri_tws_ind_repo_incr_{TODAY}.bundle"
sh("git", "bundle", "create", full, "--all")
sh("git", "bundle", "create", incr, f"{BASE}..{TAG}")
r = subprocess.run(["zip", "-rq", f"{DL}/program_{TODAY}.zip", "program"],
                   cwd=REPO, capture_output=True, text=True)
if r.returncode != 0:  # fallback if zip binary missing
    import zipfile
    with zipfile.ZipFile(f"{DL}/program_{TODAY}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(STAGE):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, REPO))

for f in (full, incr, f"{DL}/program_{TODAY}.zip"):
    print(f"{os.path.getsize(f)/2**20:8.1f} MB  {os.path.basename(f)}")

# ---------- 7. push card ----------
open(f"{DL}/FOUNDER_PUSH_CARD.md", "w").write(f"""# FOUNDER PUSH CARD — make the repo permanent ({TODAY})
**One-time setup, ~60 seconds (Path A) or ~5 min (Path B). After this,
session resets can never eat our work again.**

## Path A — no git needed (60 seconds)
1. Go to **github.com** → sign in → top-right **+** → **New repository**
2. Name: `agri-tws-ind` · Visibility: **Private** (required — competition
   data licensing) → **Create** (do NOT add README/license)
3. On the empty-repo page click **"uploading an existing file"**
4. Open `program_{TODAY}.zip` (from this chat's downloads), drag its
   `program` folder contents into the page → **Commit changes**
✅ Done. All program state is now on GitHub.

## Path B — proper git (5 minutes, keeps full history since Task 18)
1. Create the same **private** repo on github.com (no README)
2. Download `agri_tws_ind_repo_full_{TODAY}.bundle`
3. In a terminal, inside the folder where you saved it:
   ```
   git clone agri_tws_ind_repo_full_{TODAY}.bundle agri-tws-ind
   cd agri-tws-ind
   git remote set-url origin https://github.com/<your-username>/agri-tws-ind.git
   git push -u origin {branch} --tags
   ```
   (If you already pushed the Sep-10 bundle earlier, instead fetch the
   small `agri_tws_ind_repo_incr_{TODAY}.bundle`:
   `git fetch agri_tws_ind_repo_incr_{TODAY}.bundle {TAG}` then push.)
✅ Done.

## After today
- Whenever I hand you a `..._incr_....bundle`: `git fetch <bundle> <tag>` + push.
- When gwl_data.csv arrives in chat, I commit its receipt + manifest
  immediately (ARRIVAL=COMMIT); the file itself goes up as a Release
  asset if it's over 95 MB — I'll hand you exact click-steps then.
- Everything stays PRIVATE. One rule for you: never make this repo public
  without asking me first (competition data licensing).
""")

shutil.copy(f"{DL}/FOUNDER_PUSH_CARD.md", f"{STAGE}/FOUNDER_PUSH_CARD.md")
sh("git", "add", "program/FOUNDER_PUSH_CARD.md")
sh("git", *IDENT, "commit", "-m", "program: add founder push card copy")
sh("git", "bundle", "create", full, "--all")     # refresh bundles w/ final commit
sh("git", "bundle", "create", incr, f"{BASE}..HEAD")
print("\nFINAL LOG:")
print(sh("git", "log", "--oneline", "-5"))
print(sh("git", "tag", "-l"))
print("ALL DONE")
