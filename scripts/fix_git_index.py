#!/usr/bin/env python3
"""Repair the git index/object corruption in /home/z/my-project (Task 18).

Symptom: `git add -A` / `git commit` fail with
    error: invalid object 100755 dbfab114... for 'scripts/v3_perpc3.py'
    error: Error building trees

Cause: index entries reference blob objects missing from .git/objects (home/tmp
sync + consolidation interrupted a git write at some point; the earlier
"314 files changed, 0 insertions" phantom was the same disease).

Fix (idempotent): drop .git/index, rebuild it from HEAD (mixed reset, worktree
untouched), then `git add -A` re-hashes EVERY file from the working tree and
writes fresh blobs — which also re-creates any blob whose content on disk still
matches the recorded sha. History stays intact; the new commit records reality.
"""
import os
import subprocess
import sys

ROOT = "/home/z/my-project"


def sh(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


# --- 1. Diagnose: how many index-referenced blobs are missing? ---------------
ls = sh("git", "ls-files", "-s")
if ls.returncode != 0:
    print("FATAL: git ls-files failed:", ls.stderr.strip())
    sys.exit(1)

entries = [l.split() for l in ls.stdout.strip().splitlines() if l]
shas = sorted({e[1] for e in entries})
proc = subprocess.Popen(["git", "cat-file", "--batch-check"], cwd=ROOT,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = proc.communicate("\n".join(shas) + "\n")
missing = [l for l in out.splitlines() if "missing" in l]
print(f"index entries: {len(entries)} | unique blobs: {len(shas)} | "
      f"MISSING objects: {len(missing)}")
for line in missing[:10]:
    print("   ", line)

# --- 2. Repair: rebuild index from HEAD, re-hash everything from disk --------
idx = os.path.join(ROOT, ".git", "index")
if os.path.exists(idx):
    os.remove(idx)
    print("index dropped")

r = sh("git", "reset")
print(f"git reset      -> rc={r.returncode} {r.stderr.strip()[:200]}")
r = sh("git", "add", "-A")
print(f"git add -A     -> rc={r.returncode} {r.stderr.strip()[:200]}")
if r.returncode != 0:
    print("FATAL: re-add failed — check .git/objects permissions/free space")
    sys.exit(1)

# --- 3. Force re-hash of any entry still pointing at a missing blob ----------
# (`git add -A` skips files whose stat cache matches, so a stale sha can survive
#  step 2; rm --cached + add forces a fresh blob write from disk.)


def missing_paths():
    ls2 = sh("git", "ls-files", "-s")
    ent = [l.split() for l in ls2.stdout.strip().splitlines() if l]
    sha_list = sorted({e[1] for e in ent})
    proc = subprocess.Popen(["git", "cat-file", "--batch-check"], cwd=ROOT,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate("\n".join(sha_list) + "\n")
    bad = {l.split()[0] for l in out.splitlines() if "missing" in l}
    return [e[3] for e in ent if e[1] in bad]


for attempt in range(3):
    bad_paths = missing_paths()
    if not bad_paths:
        print(f"index clean after force pass {attempt}")
        break
    print(f"force re-hash pass {attempt + 1}: {len(bad_paths)} stale entries")
    for path in bad_paths:
        sh("git", "rm", "--cached", "--", path)
        r = sh("git", "add", "--", path)
        print(f"   re-hashed {path} (rc={r.returncode})")

# --- 4. Verify ---------------------------------------------------------------
r = sh("git", "status", "--short")
lines = [l for l in r.stdout.splitlines() if l.strip()]
print(f"staged changes : {len(lines)}")
for l in lines[:8]:
    print("   ", l)

left = missing_paths()
print(f"entries still pointing at missing blobs: {len(left)}")
for p in left:
    print("   ", p)

r = sh("git", "fsck", "--connectivity-only")
fsck_bad = [l for l in r.stderr.splitlines() if "missing" in l or "invalid" in l]
print(f"fsck issues    : {len(fsck_bad)} (old-commit references; tip commit is what matters)")
for l in fsck_bad[:5]:
    print("   ", l)

ok = not left
print("REPAIR RESULT  : " + ("INDEX REBUILT + ALL BLOBS RESOLVED — ready to commit" if ok
                             else "STILL BROKEN — manual intervention needed"))
sys.exit(0 if ok else 1)
