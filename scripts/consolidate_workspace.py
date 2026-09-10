#!/usr/bin/env python3
"""Consolidate the durable /tmp/my-project workspace into /home/z/my-project.

Rules:
- Copy ALL code (.py), logs (.log/.txt), docs (.md), manifests (.json) from
  /tmp/my-project/scripts + download + root docs.
- Copy ALL submission CSVs (the shipped lineage v1a..v21b + probes + rescue).
- SKIP heavy binaries (.nc, .npz, .npy, .parquet, big model dumps) and the
  external-data dirs — those stay in /tmp/my-project (the compute workspace)
  and are documented in MASTER_HANDOFF §0 / README.
- Never overwrite a NEWER file in /home/z with an OLDER one from /tmp.
"""
import os, shutil, time

SRC = '/tmp/my-project'
DST = '/home/z/my-project'
SKIP_EXT = ('.nc', '.npz', '.npy', '.parquet')
SKIP_DIRS = {'skills', 'node_modules', '__pycache__', 'tool-results',
             'era5_extract', '15a_era5_extract', 'gdo_soil', 'gdo_spei',
             'gdo_twsa', 'auditB_cache', 'v10_logs', 'recovered', 'tmp_audit',
             '.git', 'db'}
ROOT_FILES = ['MASTER_HANDOFF.md', 'worklog.md', 'worklog_restored.md',
              'worklog_restored_backup.md', 'lb_now.html', 'README.md']

copied, skipped, kept_newer = [], [], []

def want(src_rel):
    if any(seg in SKIP_DIRS for seg in src_rel.split('/')):
        return False
    if src_rel.split('/')[0] == 'data':
        return False  # competition CSVs already in DST; external stays in /tmp
    ext = os.path.splitext(src_rel)[1].lower()
    if ext in SKIP_EXT:
        return False
    if src_rel.startswith('scripts/') and ext not in ('.py', '.log', '.txt', '.md', '.json'):
        return False
    return True

def cp(rel):
    s = os.path.join(SRC, rel)
    d = os.path.join(DST, rel)
    if os.path.isdir(s):
        os.makedirs(d, exist_ok=True)
        return
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.exists(d):
        st, dt = os.path.getmtime(s), os.path.getmtime(d)
        ss, ds = os.path.getsize(s), os.path.getsize(d)
        if dt >= st and ds == ss:
            return  # destination same/newer — keep
        if dt > st:
            kept_newer.append(rel)
            return
    shutil.copy2(s, d)
    copied.append(rel)

# root-level docs
for f in ROOT_FILES:
    if os.path.exists(os.path.join(SRC, f)):
        cp(f)

# scripts/ (code + logs + small text outputs only)
sdir = os.path.join(SRC, 'scripts')
for name in sorted(os.listdir(sdir)):
    rel = f'scripts/{name}'
    if not want(rel):
        skipped.append(rel)
        continue
    cp(rel)

# download/ (everything incl. all submission CSVs)
ddir = os.path.join(SRC, 'download')
for name in sorted(os.listdir(ddir)):
    rel = f'download/{name}'
    if not want(rel):
        skipped.append(rel)
        continue
    cp(rel)

# upload/ (small photos/handoffs)
udir = os.path.join(SRC, 'upload')
for name in sorted(os.listdir(udir)):
    rel = f'upload/{name}'
    if not want(rel):
        skipped.append(rel)
        continue
    cp(rel)

print(f"copied: {len(copied)}")
print(f"kept newer dst: {len(kept_newer)}")
print(f"skipped (heavy/bin): {len(skipped)}")
for r in kept_newer[:20]:
    print(f"  kept-dst-newer: {r}")
# verify the 4 critical files landed
for f in ['download/submission_v21a.csv', 'download/submission_v13b.csv',
          'download/submission_v12b.csv', 'scripts/build_v21_phaseC.py',
          'scripts/build_v20.py', 'download/submissions_manifest.md']:
    p = os.path.join(DST, f)
    print(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {f} "
          f"({os.path.getsize(p) if os.path.exists(p) else 0:,} B)")
