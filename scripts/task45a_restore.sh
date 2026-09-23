#!/bin/bash
# Task 45a: 11th split-brain recovery (EXECUTED 2026-09-10 ~12:05 UTC+8).
# /home had reverted to the OLD NESTED pre-Task-42 workspace; /tmp was
# freshest (Tasks 1-44). Restored /home from /tmp EXCLUDING 3.98G of
# external binaries (*.nc / era5 / gdo / gpcp caches — tombstoned lanes,
# disk-safe, never needed again). Result: 9/9 canonical md5 SYNC_OK.
# Kept for the record; re-runnable if another reset hits (idempotent).
set -euo pipefail
SRC=/tmp/my-project
DST=/home/z/my-project/my-project
rsync -a --delete \
  --exclude '.git' --exclude 'tool-results' \
  --exclude '*.nc' --exclude '*.npz' --exclude 'data/era5*' \
  --exclude 'scripts/gdo_*' --exclude 'scripts/15a_era5_extract' \
  --exclude 'data/*.zip' --exclude 'download/*.bundle' --exclude 'download/*.zip' \
  "$SRC/" "$DST/"
for f in worklog.md download/REPORT_DRAFT.md download/72H_PACKAGE.md \
         download/submissions_manifest.md download/FINAL2_SELECTION.md \
         scripts/task44_shap_codecarbon.py; do
  [ "$(md5sum "$SRC/$f" | cut -d' ' -f1)" = "$(md5sum "$DST/$f" | cut -d' ' -f1)" ] \
    && echo "SYNC_OK  $f" || { echo "FAIL $f"; exit 1; }
done
echo "RECOVERY VERIFIED"
